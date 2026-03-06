# editagent_script.py

import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory

def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    # Initialize the RepoEnv
    env = RepoEnv(env_args, logger=logger, backend=backend)
    # set agent args
    if use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
        )
    else:
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
        )
    agent_args.llm_name = llm_name
    if base_url:
        agent_args.llm_base_url = base_url

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    try:
        trajectory = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )
        return None

    # also get the gt outputs
    reward_calc_time = time.time()
    reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    reward_calc_time = time.time() - reward_calc_time
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    return trajectory.model_dump_json()


def runagent_passk(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=1.0,
    use_fn_calling: bool = True,
    backend: str = "docker",
    max_reward_calc_time: int = 300,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    sample_id: int = 0,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Run a single independent sample for pass@k evaluation.
    Each call creates its own env, runs the agent once, calculates reward, and returns.
    The sample_id is stored in exp_name for identification.
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    sample_exp_name = f"{exp_name}_s{sample_id}"
    logger_inst = setup_logging(
        name=f"{ds['docker_image'].replace('/', '_')}_s{sample_id}",
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}_s{sample_id}.log",
        console=True,
        level=INFO,
    )
    logger_inst.info(f"Starting pass@k sample {sample_id} on Docker image: {ds['docker_image']}")
    logger_inst.info(f"Using LLM: {llm_name}, temperature: {temperature}")

    assert scaffold in ["r2egym", "sweagent", "openhands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"

    # Each sample gets its own independent env
    env_args = EnvArgs(ds=ds)
    env = RepoEnv(env_args, logger=logger_inst, backend=backend)

    if use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
        )
    else:
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
        )
    agent_args.llm_name = llm_name
    if base_url:
        agent_args.llm_base_url = base_url

    agent = Agent(name="EditAgent", args=agent_args, logger=logger_inst)

    # Single run — no restarts, no iterations
    try:
        trajectory = agent.run(
            env,
            max_steps=max_steps,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            scaffold=scaffold,
            max_token_limit=max_tokens,
        )

        # Calculate reward for THIS sample
        reward_calc_time = time.time()
        reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
        reward_calc_time = time.time() - reward_calc_time

        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = sample_exp_name
        trajectory.reward_calc_time = reward_calc_time
        logger_inst.warning(f"pass@k sample {sample_id}: reward={reward}, calc_time={reward_calc_time:.2f}s")

        return trajectory.model_dump_json()
    except Exception as e:
        logger_inst.error(f"Error during agent run for sample {sample_id}: {e}")
        return None
    finally:
        env.close()


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
    n_samples: int = 1,
    # If True, process instances in reverse order.
    # Useful for two-node parallel pass@k collection: node A runs forward (default),
    # node B runs with --reverse_order so the two nodes start from opposite ends,
    # minimising overlap and maximising throughput early in the run.
    reverse_order: bool = False,
    # Comma-separated paths to completed JSONL files from other nodes.
    # Used for cross-node deduplication when collecting pass@k samples in parallel:
    # instances that already have >= n_samples entries across all nodes are skipped,
    # so no docker image is evaluated more times than requested.
    extra_jsonl: str = "",
    base_url: Optional[str] = None,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    # Load the dataset
    if dataset.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=dataset, split="train")
    else:
        ds = load_dataset(dataset, split=split)
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # get selected idxs
    selected_idx = range(start_idx, start_idx + k)
    ds_selected = [ds[i] for i in selected_idx]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            if n_samples > 1:
                # For pass@k: skip instances that already have n_samples entries
                from collections import Counter
                docker_counts = Counter(existing_dockers)
                ds_selected = [
                    ds_entry
                    for ds_entry in ds_selected
                    if docker_counts.get(ds_entry["docker_image"], 0) < n_samples
                ]
            else:
                ds_selected = [
                    ds_entry
                    for ds_entry in ds_selected
                    if ds_entry["docker_image"] not in existing_dockers
                ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    # Cross-node deduplication for pass@k collection:
    # When running pass@k across multiple nodes in parallel, each node may already
    # have partial results for some instances. extra_jsonl points to the output
    # JSONL(s) from other nodes so we can count existing samples per docker_image
    # and skip any instance that already has >= n_samples completed elsewhere.
    # (Only active when n_samples > 1; harmless no-op for standard pass@1 eval.)
    if extra_jsonl:
        from collections import Counter
        extra_dockers = []
        for extra_path in extra_jsonl.split(","):
            extra_path = extra_path.strip()
            if extra_path:
                try:
                    with open(extra_path) as ef:
                        for line in ef:
                            try:
                                extra_dockers.append(json.loads(line)["ds"]["docker_image"])
                            except:
                                pass
                    logger.info(f"Loaded {len(extra_dockers)} entries from extra JSONL: {extra_path}")
                except FileNotFoundError:
                    logger.warning(f"Extra JSONL not found: {extra_path}")
        if extra_dockers and n_samples > 1:
            extra_counts = Counter(extra_dockers)
            # Combine with existing counts from use_existing
            ds_selected = [
                ds_entry for ds_entry in ds_selected
                if extra_counts.get(ds_entry["docker_image"], 0) < n_samples
            ]

    if reverse_order:  # See parameter docstring above for rationale
        ds_selected = list(reversed(ds_selected))
        logger.info("Reversed instance order for processing.")

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    if n_samples > 1:
        # pass@k mode: run n_samples independent trials per instance
        logger.info(f"pass@k mode: n_samples={n_samples}, temperature={temperature}")
        logger.info(f"Total tasks: {len(ds_selected)} instances x {n_samples} samples = {len(ds_selected) * n_samples}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {}
            for ds_entry in ds_selected:
                for sample_id in range(n_samples):
                    future = executor.submit(
                        runagent_passk,
                        ds=ds_entry,
                        exp_name=exp_name,
                        max_steps=max_steps,
                        max_steps_absolute=max_steps_absolute,
                        llm_name=llm_name,
                        temperature=temperature,
                        use_fn_calling=use_fn_calling,
                        backend=backend,
                        max_reward_calc_time=max_reward_calc_time,
                        scaffold=scaffold,
                        max_tokens=max_tokens,
                        sample_id=sample_id,
                        base_url=base_url,
                    )
                    future_to_info[future] = (ds_entry["docker_image"], sample_id)

            with open(jsonl_file, "a") as f:
                for future in concurrent.futures.as_completed(future_to_info):
                    docker_image, sample_id = future_to_info[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with file_lock:
                                f.write(result + "\n")
                    except Exception as e:
                        logger.error(f"Exception for {docker_image} sample {sample_id}: {e}")
    else:
        # Original single-sample mode
        # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(
                    runagent,
                    ds=ds_entry,
                    exp_name=exp_name,
                    max_steps=max_steps,
                    num_restarts=num_restarts,
                    max_steps_absolute=max_steps_absolute,
                    llm_name=llm_name,
                    temperature=temperature,
                    use_fn_calling=use_fn_calling,
                    backend=backend,
                    max_reward_calc_time=max_reward_calc_time,
                    max_iterations=max_iterations,
                    scaffold=scaffold,
                    max_tokens=max_tokens,
                    base_url=base_url,
                ): ds_entry["docker_image"]
                for ds_entry in ds_selected
            }

            with open(jsonl_file, "a") as f:
                for future in concurrent.futures.as_completed(future_to_image):
                    docker_image = future_to_image[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with file_lock:
                                f.write(result + "\n")
                    except Exception as e:
                        logger.error(f"Exception for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_passk": runagent_passk,
            "runagent_multiple": runagent_multiple,
        }
    )
