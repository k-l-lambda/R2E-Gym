"""
Convert local rich pipeline output to DatasetRow JSONL and Parquet format.

This script is self-contained — it defines DatasetRow inline to avoid
importing validate_docker_and_hf.py (which has heavy Docker dependencies).

Input files (per commit):
  - commit_data/rich/{commit_hash}.json            (ParsedCommit)
  - repos/rich_{commit_hash}/execution_result.json  (ExecutionResult)
  - repos/rich_{commit_hash}/syn_issue.json         (syn_issue + prompt)
  - repos/rich_{commit_hash}/expected_test_output.json

Output:
  - repo_datasets/rich.jsonl   (one DatasetRow per line)
  - repo_datasets/rich.parquet (same data in parquet format)
"""

import json
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.repo_analysis.execution_result_analysis import (
    ExecutionResult,
    CommitExecutionType,
)
from r2egym.repo_analysis.build_syn_issue import get_prompt


class DatasetRow(BaseModel):
    """Matches the schema in validate_docker_and_hf.py."""
    repo_name: str
    docker_image: str
    commit_hash: str
    parsed_commit_content: str
    execution_result_content: str
    modified_files: list[str]
    modified_entity_summaries: list[dict]
    relevant_files: list[str]
    num_non_test_files: int
    num_non_test_func_methods: int
    num_non_test_lines: int
    prompt: str
    problem_statement: str
    expected_output_json: str = ""


REPO_NAME = "rich"
REPOS_DIR = Path("repos")
COMMIT_DATA_DIR = Path("commit_data") / REPO_NAME
OUTPUT_JSONL = Path("repo_datasets") / f"{REPO_NAME}.jsonl"
OUTPUT_PARQUET = Path("repo_datasets") / f"{REPO_NAME}.parquet"


def find_repo_dirs():
    """Find all repos/rich_<commit_hash> directories with execution results."""
    pattern = str(REPOS_DIR / f"{REPO_NAME}_*")
    dirs = glob.glob(pattern)
    return [Path(d) for d in dirs if (Path(d) / "execution_result.json").exists()]


def convert_one(repo_dir: Path) -> DatasetRow | None:
    """Convert a single repo directory to a DatasetRow."""
    commit_hash = repo_dir.name.split(f"{REPO_NAME}_", 1)[1]
    docker_image = f"namanjain12/{REPO_NAME}_final:{commit_hash}"

    # Load ParsedCommit
    commit_file = COMMIT_DATA_DIR / f"{commit_hash}.json"
    if not commit_file.exists():
        print(f"  SKIP: No commit data for {commit_hash}")
        return None
    with open(commit_file, "r", encoding="utf-8") as f:
        parsed_commit = ParsedCommit(**json.load(f))

    # Load ExecutionResult
    exec_file = repo_dir / "execution_result.json"
    with open(exec_file, "r", encoding="utf-8") as f:
        execution_result = ExecutionResult(**json.load(f))

    # Check it's a good execution (NEW_COMMIT_BETTER)
    commit_exec_type, improved_fns = execution_result.is_good_exec()
    if commit_exec_type != CommitExecutionType.NEW_COMMIT_BETTER:
        print(f"  SKIP: {commit_hash} is {commit_exec_type.value}, not NEW_COMMIT_BETTER")
        return None

    # Load syn_issue
    syn_issue = ""
    syn_issue_file = repo_dir / "syn_issue.json"
    if syn_issue_file.exists():
        with open(syn_issue_file, "r", encoding="utf-8") as f:
            syn_issue_content = json.load(f)
            syn_issue = syn_issue_content.get("syn_issue", "")

    # Load expected test output
    expected_output = ""
    expected_output_file = repo_dir / "expected_test_output.json"
    if expected_output_file.exists():
        with open(expected_output_file, "r", encoding="utf-8") as f:
            expected_output = json.dumps(json.load(f), indent=4)
    else:
        expected_output = json.dumps(execution_result.new_commit_log_parse, indent=4)

    # Build prompt
    prompt = get_prompt(parsed_commit, execution_result)

    # For relevant_files, use non-test modified files
    # (full file_relevance_filter requires Docker containers)
    relevant_files = [
        f for f in parsed_commit.file_name_list
        if "test" not in f.lower()
    ]
    if not relevant_files:
        relevant_files = parsed_commit.file_name_list

    row = DatasetRow(
        repo_name=REPO_NAME,
        docker_image=docker_image,
        commit_hash=commit_hash,
        parsed_commit_content=parsed_commit.model_dump_json(indent=4),
        execution_result_content=execution_result.model_dump_json(indent=4),
        modified_files=parsed_commit.file_name_list,
        modified_entity_summaries=[
            entity.json_summary_dict() for entity in parsed_commit.edited_entities()
        ],
        relevant_files=relevant_files,
        num_non_test_files=parsed_commit.num_non_test_files,
        num_non_test_func_methods=(
            parsed_commit.num_function_entities(False)
            + parsed_commit.num_class_entities(False)
        ),
        num_non_test_lines=parsed_commit.num_non_test_edited_lines,
        prompt=prompt,
        problem_statement=syn_issue if syn_issue and syn_issue != "BLANK" else "",
        expected_output_json=expected_output,
    )

    return row


def main():
    repo_dirs = find_repo_dirs()
    print(f"Found {len(repo_dirs)} repo directories for {REPO_NAME}")

    rows = []
    for repo_dir in sorted(repo_dirs):
        print(f"Processing {repo_dir.name}...")
        row = convert_one(repo_dir)
        if row:
            rows.append(row)
            print(f"  OK: DatasetRow created for {row.commit_hash[:12]}")

    if not rows:
        print("No valid rows created. Exiting.")
        return

    # Write JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json(indent=None) + "\n")
    print(f"\nWritten {len(rows)} rows to {OUTPUT_JSONL}")

    # Convert to Parquet
    try:
        import pandas as pd
        records = [json.loads(row.model_dump_json()) for row in rows]
        df = pd.DataFrame(records)
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Written {len(rows)} rows to {OUTPUT_PARQUET}")
        print(f"\nParquet columns: {list(df.columns)}")
        print(f"Parquet shape: {df.shape}")
    except ImportError:
        print("\npandas/pyarrow not installed, skipping parquet conversion.")
        print("Install with: pip install pandas pyarrow")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for {REPO_NAME}:")
    print(f"  Total repo dirs found: {len(repo_dirs)}")
    print(f"  Valid DatasetRows:     {len(rows)}")
    for row in rows:
        has_issue = bool(row.problem_statement and row.problem_statement != "BLANK")
        print(
            f"  - {row.commit_hash[:12]}  "
            f"issue={'YES' if has_issue else 'NO (needs LLM)'}  "
            f"files={len(row.modified_files)}  "
            f"relevant={len(row.relevant_files)}"
        )
    print(f"\nNote: problem_statement is empty because no LLM server was")
    print(f"available during pipeline execution. Use recollect_issues.py")
    print(f"or an LLM API to fill it in.")


if __name__ == "__main__":
    main()
