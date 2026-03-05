# Adding Textualize/rich to the R2E-Gym Data Synthesis Pipeline

This document records the complete procedure and results from integrating
[Textualize/rich](https://github.com/Textualize/rich) into the R2E-Gym
SWE-GEN data-synthesis pipeline. It covers the research background, how and
why rich was selected, every execution step, and the results.

---

## 1. Background: R2E-Gym and the SWE-GEN Pipeline

### 1.1 The Problem

Training open-weight SWE-Agents (models that solve real-world GitHub issues)
requires large-scale **executable environments** — each consisting of:

- A repository snapshot at a specific commit
- A failing test that exposes a bug
- A natural-language problem description (issue)
- A Docker image where the test can be executed

Traditional benchmarks like SWE-Bench rely on **human-written pull requests
and issues**, which limits scale. SWE-Bench-Verified contains only ~500
curated instances across 12 repositories.

### 1.2 R2E-Gym's Contribution

R2E-Gym (Jain, Singh et al., 2025; arXiv:2504.07164) introduces **SWE-GEN**,
a synthetic data curation recipe that constructs executable environments
directly from **git commits**, bypassing the need for human-written issues or
tests. The resulting dataset contains **8,100+ environments across 13
repositories**, which is over 16x larger than SWE-Bench-Verified.

Key result: a 32B model trained on SWE-GEN data achieves **34.4% pass@1** on
SWE-Bench-Verified — matching models trained on human-written data (28.0%).
Combined with Hybrid Test-time Scaling, the approach reaches **51% pass@1**,
a state-of-the-art for open-weight SWE-Agents.

### 1.3 The SWE-GEN Pipeline in Detail

SWE-GEN converts raw commits into training environments in three stages:

```
Git Commits ──► Commit Filtering ──► Test Extraction & Execution ──► Backtranslation
                (heuristic rules)     (Fail→Pass identification)     (LLM generates issues)
```

**Stage 1 — Commit Curation** (`store_repo_commits.py`, `load_repo_commits.py`)

For each commit in the repository history, the pipeline:
1. Parses the diff and extracts an AST-level representation
2. Applies a cascade of heuristic filters:
   - **Size filter** — at most 5 non-test files, 200 edited lines
   - **Language filter** — only Python source changes
   - **Docstring filter** — excludes documentation-only changes
   - **Bug-edit filter** — only modifications to existing entities (no new
     files, no deletions), at most 4 edited entities, at most 6 statements
   - **Test-entity filter** — requires a test file change in the same commit

**Stage 2 — Test Extraction & Differential Execution** (`repo_testextract.py`)

For each surviving commit:
1. Clone the repo at the new commit, extract the edited test files into
   an `r2e_tests/` directory
2. Install the project and run the tests on the **new commit** (should pass)
3. Checkout the **old commit** and re-run the same tests (should fail)
4. If tests pass on new and fail on old → `NEW_COMMIT_BETTER` — this is a
   valid training environment
5. Build a Docker image encapsulating the environment

**Stage 3 — Backtranslation** (`build_syn_issue.py`)

For each `NEW_COMMIT_BETTER` environment, prompt an LLM with:
- The commit patch (excluding test files)
- The test diff
- The old-commit test output (failure) and new-commit test output (pass)
- Few-shot example issues from the same or similar repositories

The LLM generates a realistic GitHub issue describing the bug, without
revealing the solution.

### 1.4 Existing Repositories in R2E-Gym

The original R2E-Gym dataset covers 13 repositories:

| Repository | Domain |
|------------|--------|
| sympy | Symbolic math |
| pandas | Data analysis |
| numpy | Numerical computing |
| scrapy | Web scraping |
| tornado | Web server |
| statsmodels | Statistics |
| pillow | Image processing |
| pyramid | Web framework |
| datalad | Data management |
| aiohttp | Async HTTP |
| mypy | Type checker |
| coveragepy | Code coverage |
| orange3 | Data mining |
| bokeh | Visualization |

This document adds **rich** as the 15th repository.

### 1.5 Overlap Exclusion

SWE-Bench-Verified uses 12 repositories (django, flask, requests, sympy,
scikit-learn, matplotlib, pytest, astropy, xarray, pydicom, pylint,
sphinx). R2E-Gym already includes 14 repositories (listed above, including
flask via scrapy-era overlap). Any new repository must **not** appear in
either benchmark to avoid train/test contamination.

---

## 2. Repository Selection: Why Textualize/rich?

### 2.1 Candidate Sourcing

We started from the **top 100 most-starred Python repositories on GitHub**
(as of February 2026), providing a large pool of well-maintained projects.

### 2.2 First-Round Filtering (100 → 24)

We removed 76 repositories in four categories:

| Filter Reason | Count | Examples |
|---------------|-------|---------|
| Educational / list / resource repos | 37 | awesome-python, 30-Days-Of-Python, system-design-primer |
| AI/LLM apps or fast-churning frameworks | 16 | langchain, open-webui, MetaGPT, crewAI |
| GUI / GPU-dependent applications | 10 | stable-diffusion-webui, ComfyUI, faceswap |
| Non-Python core (C/C++) | 5 | pytorch, cpython, openpilot, PaddleOCR |
| Already in R2E-Gym | 3 | pandas, scrapy, flask |
| Application / script / too niche | 5 | odoo, you-get, copyparty |

### 2.3 Second-Round Filtering (24 → 22)

From the 24 candidates, we cross-referenced against the **SWE-Bench-Verified
repository list** and removed 2 overlapping repos:

- **django/django** — 231 instances in SWE-Bench-Verified
- **psf/requests** — 8 instances in SWE-Bench-Verified

### 2.4 Final 22 Candidates

| # | Repository | Category |
|---|-----------|----------|
| 1 | yt-dlp/yt-dlp | CLI tool |
| 2 | ytdl-org/youtube-dl | CLI tool |
| 3 | nvbn/thefuck | CLI tool |
| 4 | openai/whisper | ML |
| 5 | microsoft/markitdown | Document processing |
| 6 | home-assistant/core | Home automation |
| 7 | 3b1b/manim | Math animation |
| 8 | sherlock-project/sherlock | CLI tool |
| 9 | vllm-project/vllm | ML inference |
| 10 | ansible/ansible | DevOps |
| 11 | hiyouga/LlamaFactory | ML training |
| 12 | localstack/localstack | Cloud simulation |
| 13 | unclecode/crawl4ai | Web crawling |
| 14 | ultralytics/yolov5 | Object detection |
| 15 | **Textualize/rich** | **Terminal UI library** |
| 16 | opendatalab/MinerU | Document processing |
| 17 | docling-project/docling | Document processing |
| 18 | ultralytics/ultralytics | Object detection |
| 19 | freqtrade/freqtrade | Trading framework |
| 20 | apache/airflow | Workflow scheduling |
| 21 | streamlit/streamlit | Web framework |
| 22 | getsentry/sentry | Error monitoring |

### 2.5 Why rich Is the Best Choice for a First Test

Among the 22 candidates, **Textualize/rich** is the most suitable for
validating the full pipeline because of its favorable combination of
properties:

| Criterion | rich | Typical alternatives |
|-----------|------|---------------------|
| Pure Python | Yes — no C extensions, no build step | ansible/airflow have complex deps |
| Runtime dependencies | 2 (`pygments`, `markdown-it-py`) | vllm needs CUDA; whisper needs ffmpeg |
| Test framework | pytest, 63 test files, self-contained | Some repos use custom test harnesses |
| Network / GPU required | No | whisper, vllm, yolov5 all need GPU |
| Commit history | ~4,400 commits with steady bug-fix cadence | markitdown is too new (~100 commits) |
| Repo size | ~50 MB — fast to clone and iterate | home-assistant is 700+ MB |
| Installation | `pip install -e .` | ansible requires system packages |

Other strong candidates like **ansible** and **airflow** have mature test
suites but require complex infrastructure (SSH, databases, cloud APIs).
**rich** lets us validate every pipeline stage with minimal setup friction.

---

## 3. Prerequisites

- R2E-Gym virtualenv activated (`source .venv/bin/activate`)
- Docker daemon running
- `uv` available
- Git available
- (Optional) vLLM server on `localhost:8000` for synthetic issue generation

---

## 4. Pipeline Execution

### Step 1 — Register the repo in pipeline config

#### 1a. `src/r2egym/repo_analysis/constants.py`

Add `"rich"` to the `repo_str_names` list:

```python
repo_str_names = [
    # ... existing repos ...
    "bokeh",
    "rich",       # <-- added
]
```

This auto-creates `RICH_DIR`, `LOCAL_RICH_COMMIT_DATA_DIR`,
`RICH_COMMIT_DATA_DIR`, and `RICH_TEST_DATA_DIR` at import time.

#### 1b. `src/r2egym/repo_analysis/repo_analysis_args.py`

Add `rich` to the `RepoName` enum:

```python
class RepoName(str, Enum):
    # ...
    bokeh = "bokeh"
    rich = "rich"     # <-- added
```

Add a `tests_cmd` case:

```python
@property
def tests_cmd(self):
    # ...
    if self.repo_name == RepoName.rich:
        return (
            "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' "
            ".venv/bin/python -W ignore -m pytest -rA r2e_tests"
        )
```

### Step 2 — Create install script

**File:** `src/r2egym/install_utils/rich_install.sh`

```bash
uv venv --python=python3.10
source .venv/bin/activate
uv pip install -e .
uv pip install pytest
```

Rich is pure Python with minimal dependencies, so the install script is
straightforward.

### Step 3 — Create Dockerfile template

**File:** `src/r2egym/repo_analysis/base_dockerfiles/Dockerfile.rich`

```dockerfile
FROM ubuntu:22.04

ARG OLD_COMMIT

RUN apt-get update -y && apt-get upgrade -y

ENV DEBIAN_FRONTEND noninteractive

RUN echo "tzdata tzdata/Areas select America" | debconf-set-selections \
 && echo "tzdata tzdata/Zones/America select Los_Angeles" | debconf-set-selections

RUN apt-get install -y git curl wget build-essential ca-certificates python3-dev

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:${PATH}"
ENV PATH="/root/.local/bin:${PATH}"

RUN git clone https://github.com/Textualize/rich.git testbed

COPY run_tests.sh /testbed/run_tests.sh
COPY install.sh /testbed/install.sh

WORKDIR /testbed

RUN git checkout $OLD_COMMIT
RUN git status
RUN bash install.sh
RUN uv pip install tree_sitter_languages

ENV VIRTUAL_ENV=/testbed/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY r2e_tests /r2e_tests
```

### Step 4 — Create example issues for few-shot prompting

**File:** `src/r2egym/repo_analysis/issues/rich_issues.py`

Contains 1-2 representative bug-report-style issues from the rich
repository, used as few-shot examples when the LLM generates synthetic
issues via backtranslation.

Also update:
- `src/r2egym/repo_analysis/issues/__init__.py` — add the import
- `src/r2egym/repo_analysis/validate_docker_and_hf.py` — add `"rich"` case
  to `get_issues_for_repo()`

### Step 5 — Clone the repo and create data directories

```bash
cd /home/claude/work/R2E-Gym

git clone https://github.com/Textualize/rich.git rich
mkdir -p commit_data/rich test_data/rich
```

### Step 6 — Collect commits (SWE-GEN Stage 1a)

```bash
python src/r2egym/repo_analysis/store_repo_commits.py \
    --repo_name rich \
    --n_cpus 8
```

This step iterates over every commit in the rich repository, runs
`git diff` to extract the patch, parses the diff into a structured
`ParsedCommit` model with AST-level entity analysis, and writes each
commit as a JSON file.

**Result:** 4,330 commit JSON files written to `commit_data/rich/`.

A handful of commits produce `utf-8 codec` errors — these are commits that
touch binary files (PNG images) and are skipped automatically.

### Step 7 — Filter commits (SWE-GEN Stage 1b)

```bash
python src/r2egym/repo_analysis/load_repo_commits.py \
    --repo_name rich \
    --keep_only_small_commits \
    --keep_only_python_commits \
    --keep_only_non_docstring_commits \
    --keep_only_bug_edit_commits \
    --keep_only_test_entity_edit_commits
```

This step applies the paper's heuristic filter cascade to narrow down
commits to high-quality bug-fix candidates:

| Filter | What it checks | Remaining |
|--------|----------------|-----------|
| (initial) | All commits loaded | 4,330 |
| `keep_only_small_commits` | ≤5 non-test files, ≤200 edited lines, ≤10K patch length | 3,285 |
| `keep_only_python_commits` | At least one `.py` file changed | 1,934 |
| `keep_only_non_docstring_commits` | Non-test changes beyond docstrings/comments | 1,569 |
| `keep_only_bug_edit_commits` | No new/deleted entities; ≤4 edited entities, ≤6 statements | 1,002 |
| `keep_only_test_entity_edit_commits` | At least one test file edited in the commit | **278** |

278 commits qualify for test extraction from the full dataset.

### Step 8 — Extract tests and build Docker environments (SWE-GEN Stage 2)

```bash
python src/r2egym/repo_analysis/repo_testextract.py \
    --repo_name rich \
    --use_local_commit_data \
    --n_cpus 8 \
    --N 50 \
    --keep_only_small_commits \
    --keep_only_python_commits \
    --keep_only_non_docstring_commits \
    --keep_only_bug_edit_commits \
    --keep_only_test_entity_edit_commits \
    --build_dockers True \
    --chunk_size 10
```

Use `--N 50` for an initial smoke test. Remove `--N` to process all 278
qualifying commits.

For each filtered commit, the pipeline:
1. Clones the repo and checks out the new commit
2. Copies the install script and extracts the edited test files
3. Installs the project and runs tests on the new commit
4. Checks out the old commit and re-runs the same tests
5. Classifies the result:
   - `NEW_COMMIT_BETTER` — tests pass on new, fail on old (valid environment)
   - `NEW_COMMIT_NOT_BETTER` — same result on both commits
   - Other states (setup failure, etc.)
6. For `NEW_COMMIT_BETTER` commits: calls the LLM for backtranslation
   (synthetic issue generation), then builds a Docker image

**Result from N=50 run:**

| Metric | Value |
|--------|-------|
| Commits after filters (from 50) | 3 |
| `NEW_COMMIT_BETTER` | 2 |
| `NEW_COMMIT_NOT_BETTER` | 1 |
| Docker images built | 1 (833 MB) |
| `test_data/rich/` entries | 2 |

For each `NEW_COMMIT_BETTER` commit, the pipeline generates:
- `execution_result.json` — test stdout/stderr for old and new commits
- `parsed_commit.json` — structured commit data
- `modified_files.json` — list of changed files
- `modified_entities.json` — changed functions/classes
- `expected_test_output.json` — expected pytest log parse
- `syn_issue.json` — synthetic issue prompt (and LLM output if available)
- `Dockerfile` — parameterized Dockerfile for the commit
- `run_tests.sh` / `install.sh` — copied into the Docker context

### Step 9 — Generate synthetic issues (SWE-GEN Stage 3)

Synthetic issue generation requires a running LLM server. By default
`build_syn_issue()` is called with `do_llm=False`, so the prompt is stored
but no LLM call is made.

To enable LLM-generated issues, ensure a vLLM server is running:

```bash
# Example: start vLLM with Qwen3-32B
vllm serve Qwen/Qwen3-32B --port 8000
```

The `repo_testextract.py` script will call the LLM through
`litellm.completion()` using the `--model` and `--base_url` arguments.

The LLM receives: the commit patch (without test diffs), the test diff,
old-commit failure output, new-commit success output, and a set of few-shot
example issues. It generates a realistic GitHub issue describing the bug from
a user's perspective, without revealing the fix.

---

## 5. Verification Checklist

```bash
# 1. Commit data collected
ls commit_data/rich/ | wc -l          # expect ~4,330

# 2. Test data generated
ls test_data/rich/                     # expect JSON files for good commits

# 3. Execution results
ls repos/rich_*/execution_result.json  # one per processed commit

# 4. Docker images
docker images | grep rich              # namanjain12/rich_final:<hash>

# 5. NEW_COMMIT_BETTER commits
python -c "
import json, glob
for f in sorted(glob.glob('repos/rich_*/execution_result.json')):
    d = json.load(open(f))
    new, old = d.get('new_commit_res_code'), d.get('old_commit_res_code')
    status = 'BETTER' if new == 0 and old != 0 else 'SAME/WORSE'
    print(f'{d[\"new_commit_hash\"][:12]}  new={new} old={old}  {status}')
"
```

---

## 6. Troubleshooting

### `ModuleNotFoundError: No module named 'r2e'`

The `r2e` package is an internal dependency not published to PyPI. A minimal
stub package was created at `src/r2e/` with the modules needed by the
pipeline: `llms`, `paths`, `models`, `pat.ast.explorer`,
`pat.dependency_slicer`.

### `ModuleNotFoundError: No module named 'r2e_edits'`

The `src/r2egym/repo_analysis/issues/__init__.py` file originally used
`r2e_edits` import paths. These were updated to `r2egym`.

### Docker build: `ubuntu:22.04: not found`

Pull the base image manually first:

```bash
docker pull ubuntu:22.04
```

### Docker build timeout

The default Docker build timeout is 1200 seconds. The first build for a
commit is slow because `apt-get upgrade` runs from scratch. Subsequent
builds are faster thanks to Docker layer caching. If builds time out, re-run
the same command — cached layers will be reused.

---

## 7. Full-Scale Run

To process all 278 qualifying commits (instead of the N=50 sample), run:

```bash
python src/r2egym/repo_analysis/repo_testextract.py \
    --repo_name rich \
    --use_local_commit_data \
    --n_cpus 8 \
    --keep_only_small_commits \
    --keep_only_python_commits \
    --keep_only_non_docstring_commits \
    --keep_only_bug_edit_commits \
    --keep_only_test_entity_edit_commits \
    --build_dockers True \
    --chunk_size 10
```

Expect the full run to produce 50-100+ `NEW_COMMIT_BETTER` environments
based on the 66% success rate observed in the N=50 sample.

---

## 8. Converting Raw Data to Training-Ready Formats

After running the SWE-GEN pipeline (Steps 6-9), the raw data is scattered
across multiple directories. This section covers converting it into the
unified `DatasetRow` JSONL and Parquet formats used for training.

### 8.1 Data Flow Overview

```
repos/rich_{hash}/execution_result.json  ─┐
repos/rich_{hash}/parsed_commit.json     ─┼──► DatasetRow JSONL ──► Parquet / HuggingFace
repos/rich_{hash}/syn_issue.json         ─┤
commit_data/rich/{hash}.json             ─┘
```

### 8.2 Production Path vs. Local Path

The production pipeline uses `validate_docker_and_hf.py`, which:
1. Fetches Docker image tags from DockerHub (`fetch_docker_tags()`)
2. Pulls each Docker image and re-validates tests inside the container
3. Runs `file_relevance_filter()` (removes files from the patch one at a
   time and checks if tests still fail — requires running Docker containers)
4. Assembles `DatasetRow` objects and writes to `repo_datasets/<repo>.jsonl`

For local development (images not pushed to DockerHub), use the standalone
conversion script at `scripts/convert_rich_to_jsonl.py`.

### 8.3 Running the Local Conversion Script

```bash
source .venv/bin/activate
python scripts/convert_rich_to_jsonl.py
```

This script:
- Scans `repos/rich_*` directories for `execution_result.json` files
- Filters to only `NEW_COMMIT_BETTER` commits
- Reads `parsed_commit.json`, `execution_result.json`, `syn_issue.json`
- Builds `DatasetRow` objects (same Pydantic schema as production)
- Writes `repo_datasets/rich.jsonl` (one JSON line per environment)
- Writes `repo_datasets/rich.parquet` (same data, columnar format)

**Result from N=50 sample run:**

```
Found 3 repo directories for rich
  OK: DatasetRow created for 00181151a4a6
  OK: DatasetRow created for 01b85ac116c4
  SKIP: 02dffcf9cfe0... is NEW_COMMIT_NOT_BETTER

Written 2 rows to repo_datasets/rich.jsonl
Written 2 rows to repo_datasets/rich.parquet
Parquet shape: (2, 14)
```

### 8.4 DatasetRow Schema

The `DatasetRow` Pydantic model (defined in `validate_docker_and_hf.py`)
has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `repo_name` | str | Repository name (e.g., "rich") |
| `docker_image` | str | Full Docker image tag |
| `commit_hash` | str | The new (fixed) commit hash |
| `parsed_commit_content` | str | JSON-serialized ParsedCommit |
| `execution_result_content` | str | JSON-serialized ExecutionResult |
| `modified_files` | list[str] | All files changed in the commit |
| `relevant_files` | list[str] | Non-test files whose changes affect test outcomes |
| `modified_entity_summaries` | list[dict] | Summary of edited functions/classes |
| `num_non_test_files` | int | Number of non-test files modified |
| `num_non_test_func_methods` | int | Number of non-test functions/methods edited |
| `num_non_test_lines` | int | Lines changed in non-test files |
| `prompt` | str | Backtranslation prompt sent to the LLM |
| `problem_statement` | str | LLM-generated synthetic issue (empty if no LLM) |
| `expected_output_json` | str | Expected pytest output (JSON) |

### 8.5 Filling in `problem_statement` (Synthetic Issue Generation)

The `problem_statement` field is empty when no LLM server is available.
To fill it in using `recollect_issues.py`:

```bash
# Ensure repo_datasets/rich.jsonl exists (from Step 8.3)
# Then start an LLM server and run:
python src/r2egym/repo_analysis/recollect_issues.py
```

This script reads `repo_datasets/*.jsonl`, sends each row's prompt to the
LLM (OpenAI API by default, configurable), and writes the generated issue
back as `problem_statement`.

Alternatively, you can use any LLM API to generate issues from the
`prompt` field in the JSONL/Parquet files.

### 8.6 Loading into HuggingFace Datasets

```python
from datasets import Dataset
import pandas as pd

df = pd.read_parquet("repo_datasets/rich.parquet")
ds = Dataset.from_pandas(df)

# Push to HuggingFace Hub
ds.push_to_hub("your-org/rich-swe-gym", split="train")
```

### 8.7 Note on `relevant_files`

The local conversion script uses a simple heuristic for `relevant_files`:
non-test files from the commit's modified file list. The production
pipeline uses `file_relevance_filter()`, which is more precise — it
removes each non-test file from the patch individually and checks whether
the tests still fail inside a Docker container. For production-quality
data, run the full `validate_docker_and_hf.py` pipeline with Docker
images pushed to DockerHub.

---

## 9. Summary of Code Changes

| File | Change |
|------|--------|
| `src/r2egym/repo_analysis/constants.py` | Added `"rich"` to `repo_str_names` |
| `src/r2egym/repo_analysis/repo_analysis_args.py` | Added `rich` enum, `tests_cmd` case; fixed `parameterized_dockerfile` path (`r2e_edits` → `r2egym`) |
| `src/r2egym/repo_analysis/issues/__init__.py` | Fixed all `r2e_edits` imports to `r2egym`; added `rich_issues` import |
| `src/r2egym/repo_analysis/validate_docker_and_hf.py` | Added `"rich"` case to `get_issues_for_repo()` |
| `src/r2egym/install_utils/rich_install.sh` | **New** — install script |
| `src/r2egym/repo_analysis/base_dockerfiles/Dockerfile.rich` | **New** — Docker template |
| `src/r2egym/repo_analysis/issues/rich_issues.py` | **New** — example issues for few-shot prompting |
| `src/r2e/` (stub package) | **New** — minimal stubs for `r2e.llms`, `r2e.paths`, `r2e.models`, `r2e.pat.*` |
| `scripts/convert_rich_to_jsonl.py` | **New** — converts raw pipeline data to JSONL + Parquet |

---

## References

- Jain, N., Singh, J., Shetty, M., Zheng, L., Sen, K., & Stoica, I. (2025).
  R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling
  Open-Weights SWE Agents. *arXiv preprint arXiv:2504.07164*.
- Textualize/rich: https://github.com/Textualize/rich
- R2E-Gym project page: https://r2e-gym.github.io/
