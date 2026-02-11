#!/usr/bin/env python3
"""
R2E-Gym Data Collection: Run agent on 32 cached training samples
"""
import os
import json
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/claude/work/R2E-Gym/src')

from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

# Set vLLM base URL
os.environ['LLM_BASE_URL'] = 'http://localhost:8000/v1'

# Configuration
MAX_STEPS = 30
OUTPUT_DIR = Path('/home/claude/work/R2E-Gym/collected_data')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load 32 cached samples from training parquet
PARQUET_FILE = '/home/claude/work/rllm-origin/rllm/data/datasets/R2E_Gym_Subset/train_cached.parquet'

def load_samples():
    df = pd.read_parquet(PARQUET_FILE)
    samples = []
    for idx, row in df.iterrows():
        extra_info = row['extra_info']
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        
        # Build complete sample dict matching R2E-Gym format
        sample = {
            'repo_name': extra_info.get('repo_name', ''),
            'docker_image': extra_info.get('docker_image', ''),
            'commit_hash': extra_info.get('commit_hash', ''),
            'problem_statement': extra_info.get('problem_statement', ''),
            'prompt': row['prompt'].tolist() if hasattr(row['prompt'], 'tolist') else row['prompt'],
            'parsed_commit_content': extra_info.get('parsed_commit_content', '{}'),
            'execution_result_content': extra_info.get('execution_result_content', ''),
            'modified_files': extra_info.get('modified_files', []),
            'modified_entity_summaries': extra_info.get('modified_entity_summaries', []),
            'relevant_files': extra_info.get('relevant_files', []),
            'num_non_test_files': extra_info.get('num_non_test_files', 0),
            'num_non_test_func_methods': extra_info.get('num_non_test_func_methods', 0),
            'num_non_test_lines': extra_info.get('num_non_test_lines', 0),
            'expected_output_json': extra_info.get('expected_output_json', ''),
        }
        sample['instance_id'] = f"{sample['repo_name']}_{sample['commit_hash'][:8]}"
        samples.append(sample)
    return samples

def run_agent_on_sample(sample, idx, total):
    """Run R2E-Gym agent on a single sample"""
    print(f'\n{"="*60}')
    print(f'Sample {idx+1}/{total}: {sample["instance_id"]}')
    print(f'Repo: {sample["repo_name"]}')
    print(f'Docker: {sample["docker_image"][:60]}...')
    print(f'{"="*60}')
    
    start_time = time.time()
    result = {
        'instance_id': sample['instance_id'],
        'repo_name': sample['repo_name'],
        'docker_image': sample['docker_image'],
        'start_time': datetime.now().isoformat(),
    }
    
    try:
        # Create environment
        env_args = EnvArgs(ds=sample)
        env = RepoEnv(env_args)
        
        # Load agent config
        config_path = Path('/home/claude/work/R2E-Gym/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml')
        agent_args = AgentArgs.from_yaml(config_path)
        agent_args.llm_name = 'hosted_vllm//home/claude/work/rllm/models/Qwen3-32B'
        
        agent = Agent(name='EditingAgent', args=agent_args)
        
        # Run agent
        print(f'Starting agent run (max {MAX_STEPS} steps)...')
        output = agent.run(env, max_steps=MAX_STEPS, use_fn_calling=False)
        
        # Get results
        result['exit_reason'] = output.exit_reason
        result['steps'] = output.num_steps
        result['execution_time'] = output.total_time_traj

        # Calculate reward
        try:
            reward = env.runtime._calculate_reward()
            result['reward'] = reward
            print(f'Reward: {reward}')
        except Exception as e:
            result['reward'] = None
            result['reward_error'] = str(e)
            print(f'Reward error: {e}')

        # Save trajectory (handle numpy types in serialization)
        traj_file = OUTPUT_DIR / f'trajectory_{sample["instance_id"]}.json'
        traj_data = output.model_dump()
        with open(traj_file, 'w') as f:
            json.dump(traj_data, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
        result['trajectory_file'] = str(traj_file)

        print(f'Exit reason: {output.exit_reason}')
        print(f'Steps: {result["steps"]}')
        print(f'Time: {result["execution_time"]:.1f}s')
        
        # Close environment
        try:
            env.close()
        except:
            pass
            
    except Exception as e:
        import traceback
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        print(f'ERROR: {e}')
        print(traceback.format_exc())
    
    result['total_time'] = time.time() - start_time
    result['end_time'] = datetime.now().isoformat()
    
    return result

def main():
    print('='*60)
    print('R2E-Gym Data Collection - 32 Cached Samples')
    print(f'Started: {datetime.now().isoformat()}')
    print('='*60)
    
    # Load samples
    samples = load_samples()
    print(f'Loaded {len(samples)} samples from {PARQUET_FILE}')
    
    # Print sample repos
    repos = {}
    for s in samples:
        repos[s['repo_name']] = repos.get(s['repo_name'], 0) + 1
    print(f'Repos: {repos}')
    
    # Results tracking
    results = []
    summary_file = OUTPUT_DIR / 'collection_summary.json'
    
    # Run on each sample
    for idx, sample in enumerate(samples):
        result = run_agent_on_sample(sample, idx, len(samples))
        results.append(result)
        
        # Save intermediate results
        with open(summary_file, 'w') as f:
            json.dump({
                'total_samples': len(samples),
                'completed': len(results),
                'results': results,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
        
        # Print progress summary
        completed = len([r for r in results if 'error' not in r])
        rewards = [r.get('reward', 0) or 0 for r in results if 'error' not in r]
        positive = len([r for r in rewards if r > 0])
        print(f'\nProgress: {idx+1}/{len(samples)} | Completed: {completed} | Positive rewards: {positive}')
    
    # Final summary
    print('\n' + '='*60)
    print('COLLECTION COMPLETE')
    print('='*60)
    completed = len([r for r in results if 'error' not in r])
    rewards = [r.get('reward', 0) or 0 for r in results if 'error' not in r]
    positive = len([r for r in rewards if r > 0])
    print(f'Total samples: {len(samples)}')
    print(f'Completed: {completed}')
    print(f'Positive rewards: {positive} ({100*positive/len(samples):.1f}%)')
    print(f'Results saved to: {summary_file}')

if __name__ == '__main__':
    main()
