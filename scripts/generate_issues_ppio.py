"""
Generate synthetic issues for rich DatasetRows using PPIO's OpenAI-compatible API.

Reads repo_datasets/rich.jsonl, calls GPT-5.2 via PPIO to generate problem_statement,
writes back to repo_datasets/rich.jsonl and repo_datasets/rich.parquet.

Usage:
    export PPIO_LLM_API_KEY="sk_..."
    python scripts/generate_issues_ppio.py

Or specify the key directly:
    PPIO_LLM_API_KEY="sk_..." python scripts/generate_issues_ppio.py
"""

import os
import json
import sys
from pathlib import Path

from openai import OpenAI

# --- Config ---
API_KEY = os.environ.get("PPIO_LLM_API_KEY", "")
BASE_URL = "https://api.ppinfra.com/v3/openai"
MODEL = "pa/gpt-5.2"
MAX_TOKENS = 12000
JSONL_PATH = Path("repo_datasets/rich.jsonl")
PARQUET_PATH = Path("repo_datasets/rich.parquet")


def extract_issue(model_output: str) -> str:
    """Extract issue text from [ISSUE]...[/ISSUE] tags."""
    if "[ISSUE]" in model_output:
        model_output = model_output.split("[ISSUE]")[1]
    return model_output.split("[/ISSUE]")[0].strip()


def main():
    if not API_KEY:
        print("ERROR: PPIO_LLM_API_KEY not set. Export it or set in ~/.bashrc")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Read existing data
    rows = []
    with open(JSONL_PATH, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows from {JSONL_PATH}")

    # Generate issues for rows with empty problem_statement
    for i, row in enumerate(rows):
        if row.get("problem_statement"):
            print(f"[{i+1}/{len(rows)}] {row['commit_hash'][:12]} — already has issue, skipping")
            continue

        prompt = row["prompt"]
        print(f"[{i+1}/{len(rows)}] {row['commit_hash'][:12]} — generating issue ({len(prompt)} chars)...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            model_output = response.choices[0].message.content or ""
            issue = extract_issue(model_output)

            row["problem_statement"] = issue
            print(f"  OK: generated {len(issue)} chars")
            print(f"  Preview: {issue[:200]}...")
            print(f"  Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Write JSONL
    with open(JSONL_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWritten {len(rows)} rows to {JSONL_PATH}")

    # Write Parquet
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(PARQUET_PATH, index=False)
        print(f"Written {len(rows)} rows to {PARQUET_PATH}")
    except ImportError:
        print("pandas not available, skipping parquet output")

    # Summary
    print(f"\n{'='*60}")
    for row in rows:
        has_issue = bool(row.get("problem_statement"))
        print(f"  {row['commit_hash'][:12]}  problem_statement={'YES' if has_issue else 'NO'}  "
              f"len={len(row.get('problem_statement', ''))}")


if __name__ == "__main__":
    main()
