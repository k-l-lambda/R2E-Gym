"""
Build Docker images for all NEW_COMMIT_BETTER rich commits.

Uses a two-stage approach:
1. rich_base:latest - pre-built image with all deps (built once)
2. Per-commit: just git checkout + copy r2e_tests (fast, ~5s each)
"""

import json
import subprocess
import os
import sys
import shutil
from pathlib import Path

REPO_NAME = "rich"
JSONL_PATH = Path("repo_datasets/rich.jsonl")
IMAGE_PREFIX = f"namanjain12/{REPO_NAME}_final"
BUILD_TIMEOUT = 120  # 2 minutes per commit (should take ~5-10s)

# Lightweight Dockerfile that extends the base image
PER_COMMIT_DOCKERFILE = """\
FROM rich_base:latest

ARG OLD_COMMIT

WORKDIR /testbed
RUN git checkout $OLD_COMMIT

COPY run_tests.sh /testbed/run_tests.sh
COPY r2e_tests /r2e_tests
"""


def get_existing_images():
    """Get set of commit hashes that already have Docker images."""
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Tag}}", IMAGE_PREFIX],
        capture_output=True, text=True
    )
    return set(result.stdout.strip().split("\n")) - {""}


def build_one(commit_hash: str, repo_dir: Path, build_ctx: Path) -> bool:
    """Build Docker image for a single commit."""
    image_tag = f"{IMAGE_PREFIX}:{commit_hash}"

    try:
        # Write Dockerfile
        (build_ctx / "Dockerfile").write_text(PER_COMMIT_DOCKERFILE)

        # Copy run_tests.sh
        run_tests_src = repo_dir / "run_tests.sh"
        if run_tests_src.exists():
            shutil.copy2(run_tests_src, build_ctx / "run_tests.sh")
        else:
            (build_ctx / "run_tests.sh").write_text(
                "#!/bin/bash\ncd /testbed && .venv/bin/python -m pytest -rA r2e_tests \"$@\"\n"
            )

        # Copy r2e_tests
        r2e_tests_dest = build_ctx / "r2e_tests"
        if r2e_tests_dest.exists():
            shutil.rmtree(r2e_tests_dest)
        r2e_tests_src = repo_dir / "r2e_tests"
        if r2e_tests_src.exists():
            shutil.copytree(r2e_tests_src, r2e_tests_dest)
        else:
            r2e_tests_dest.mkdir()

        # Build Docker image
        result = subprocess.run(
            [
                "docker", "build",
                "--build-arg", f"OLD_COMMIT={commit_hash}",
                "-t", image_tag,
                str(build_ctx)
            ],
            capture_output=True, text=True,
            timeout=BUILD_TIMEOUT
        )

        if result.returncode != 0:
            stderr = result.stderr[-300:] if result.stderr else ""
            stdout = result.stdout[-300:] if result.stdout else ""
            print(f"  BUILD FAILED: {stderr or stdout}", flush=True)
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {BUILD_TIMEOUT}s", flush=True)
        return False
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return False


def main():
    # Verify base image exists
    result = subprocess.run(
        ["docker", "images", "-q", "rich_base:latest"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print("ERROR: rich_base:latest not found. Build it first.")
        sys.exit(1)

    # Load commits from JSONL
    with open(JSONL_PATH) as f:
        rows = [json.loads(line) for line in f]
    print(f"Loaded {len(rows)} rows from {JSONL_PATH}")

    # Get existing images
    existing = get_existing_images()
    print(f"Found {len(existing)} existing Docker images")

    # Filter to commits needing builds
    to_build = []
    for row in rows:
        ch = row["commit_hash"]
        repo_dir = Path("repos") / f"{REPO_NAME}_{ch}"
        if ch not in existing and repo_dir.exists():
            to_build.append((ch, repo_dir))

    print(f"Need to build {len(to_build)} Docker images\n")

    if not to_build:
        print("All images already built!")
        return

    # Create lightweight build context
    build_ctx = Path("/tmp/rich_commit_build")
    build_ctx.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for i, (commit_hash, repo_dir) in enumerate(to_build):
        print(f"[{i+1}/{len(to_build)}] Building {commit_hash[:12]}...", end=" ", flush=True)
        if build_one(commit_hash, repo_dir, build_ctx):
            success += 1
            print(f"OK ({success}/{i+1})", flush=True)
        else:
            failed += 1
            print(f"FAIL ({failed} failures)", flush=True)

    print(f"\n{'='*60}")
    print(f"Done: {success} built, {failed} failed, {len(existing)} pre-existing")
    print(f"Total images: {success + len(existing)}")


if __name__ == "__main__":
    main()
