"""
Unit tests for R2E-Gym eval pipeline components.
Tests each stage independently to isolate issues before running full eval.

Usage:
    cd /home/claude/work/R2E-Gym
    source .venv/bin/activate
    export OPENAI_API_KEY="$PPIO_LLM_API_KEY"
    python scripts/test_eval_pipeline.py
"""
import os, sys, json, time, re

PPIO_BASE = "https://api.ppinfra.com/v3/openai"
MODEL = "openai/pa/gpt-5.2"
API_KEY = os.environ.get("OPENAI_API_KEY", "")


def test_1_dataset_loading():
    """Test: load local parquet via HuggingFace datasets"""
    print("=== Test 1: Dataset Loading ===")
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files="repo_datasets/rich.parquet", split="train")
    assert len(ds) == 2, f"Expected 2 rows, got {len(ds)}"
    assert "docker_image" in ds.column_names
    assert "problem_statement" in ds.column_names
    row = ds[0]
    assert row["repo_name"] == "rich"
    assert len(row["problem_statement"]) > 100, "problem_statement too short"
    print(f"  OK: {len(ds)} rows, columns={ds.column_names[:5]}...")
    return ds


def test_2_litellm_basic(api_key):
    """Test: litellm.completion() to PPIO API without extra kwargs"""
    print("=== Test 2: litellm Basic Call ===")
    import litellm
    litellm.api_key = api_key
    resp = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        api_base=PPIO_BASE,
        max_tokens=10,
        timeout=30,
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    print(f"  OK: response='{text}', tokens={resp.usage.total_tokens}")
    return True


def test_3_litellm_with_kwargs(api_key):
    """Test: litellm.completion() with the exact kwargs agent.py uses"""
    print("=== Test 3: litellm with agent.py kwargs (tool_choice=none, function_call=None) ===")
    import litellm
    litellm.api_key = api_key
    # These are the exact kwargs from agent.py when use_fn_calling=False
    kwargs = {
        "tool_choice": "none",
        "function_call": None,
        "temperature": 0.7,
    }
    resp = litellm.completion(
        model=MODEL,
        tools=None,
        messages=[
            {"role": "system", "content": "You are a programming agent. Respond with EXACTLY one function call in this format:\n<function=execute_bash>\n<parameter=command>your_command</parameter>\n</function>"},
            {"role": "user", "content": "List files in the current directory."},
        ],
        timeout=120,
        api_base=PPIO_BASE,
        **kwargs,
    )
    text = resp.choices[0].message.content
    has_function = "<function=" in text
    is_garbled = any(ord(c) > 0x4E00 for c in text[:50]) and "function" not in text[:50]
    print(f"  Response ({len(text)} chars): {text[:150]}...")
    print(f"  Contains <function=: {has_function}")
    print(f"  Looks garbled: {is_garbled}")
    if is_garbled:
        print(f"  *** LIKELY BUG: tool_choice='none' may cause garbled output via PPIO ***")
    return text, has_function, is_garbled


def test_4_litellm_no_extra_kwargs(api_key):
    """Test: same prompt but WITHOUT tool_choice/function_call kwargs"""
    print("=== Test 4: litellm WITHOUT tool_choice/function_call ===")
    import litellm
    litellm.api_key = api_key
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a programming agent. Respond with EXACTLY one function call in this format:\n<function=execute_bash>\n<parameter=command>your_command</parameter>\n</function>"},
            {"role": "user", "content": "List files in the current directory."},
        ],
        timeout=120,
        api_base=PPIO_BASE,
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    has_function = "<function=" in text
    is_garbled = any(ord(c) > 0x4E00 for c in text[:50]) and "function" not in text[:50]
    print(f"  Response ({len(text)} chars): {text[:150]}...")
    print(f"  Contains <function=: {has_function}")
    print(f"  Looks garbled: {is_garbled}")
    return text, has_function, is_garbled


def test_5_action_parsing():
    """Test: parse_response extracts correct action from model output"""
    print("=== Test 5: Action Parsing ===")
    pattern_action = re.compile(r"(?s)(<function=.*?</function>)")

    # Good output
    good = '<function=search>\n<parameter=search_term>FORCE_COLOR</parameter>\n<parameter=path>/testbed/rich/console.py</parameter>\n</function>'
    m = pattern_action.search(good)
    assert m, "Failed to match good output"
    print(f"  Good output: matched")

    # Garbled output
    bad = 'shell \u9ad8\u9891\u5f69\u5927\u53d1\u5feb\u4e09="json"'
    m2 = pattern_action.search(bad)
    assert m2 is None, "Should NOT match garbled output"
    print(f"  Garbled output: correctly rejected")

    # Think block bug
    think_output = '<think>I should use <function=search args={{"term":"x"}}></function></think>\n<function=search>\n<parameter=search_term>FORCE_COLOR</parameter>\n</function>'
    m3 = pattern_action.search(think_output)
    assert m3, "Failed to match output with think block"
    extracted = m3.group(1)
    has_parameter = "<parameter=" in extracted
    print(f"  Think block: extracted has <parameter=: {has_parameter}")
    if not has_parameter:
        print(f"  *** KNOWN BUG: regex extracts draft from <think> block ***")

    print(f"  OK")


def test_6_docker_image():
    """Test: Docker image exists locally"""
    print("=== Test 6: Docker Image ===")
    import subprocess
    result = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "namanjain12/rich_final"],
        capture_output=True, text=True
    )
    images = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    print(f"  Found {len(images)} rich Docker images")
    for img in images:
        print(f"    {img}")
    assert len(images) >= 1, "No rich Docker images found"
    print(f"  OK")


if __name__ == "__main__":
    os.chdir("/home/claude/work/R2E-Gym")

    print("\n" + "=" * 60)
    print("R2E-Gym Eval Pipeline Unit Tests")
    print("=" * 60 + "\n")

    results = {}

    # Test 1: Dataset
    try:
        ds = test_1_dataset_loading()
        results["dataset"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["dataset"] = f"FAIL: {e}"

    # Test 2-4: litellm
    if API_KEY:
        try:
            test_2_litellm_basic(API_KEY)
            results["litellm_basic"] = "PASS"
        except Exception as e:
            print(f"  FAIL: {e}")
            results["litellm_basic"] = f"FAIL: {e}"

        try:
            text3, has3, garbled3 = test_3_litellm_with_kwargs(API_KEY)
            if garbled3:
                results["litellm_kwargs"] = "FAIL: garbled output with tool_choice=none"
            elif has3:
                results["litellm_kwargs"] = "PASS"
            else:
                results["litellm_kwargs"] = "WARN: no <function= but not garbled"
        except Exception as e:
            print(f"  FAIL: {e}")
            results["litellm_kwargs"] = f"FAIL: {e}"

        try:
            text4, has4, garbled4 = test_4_litellm_no_extra_kwargs(API_KEY)
            if garbled4:
                results["litellm_clean"] = "FAIL: garbled even without kwargs"
            elif has4:
                results["litellm_clean"] = "PASS"
            else:
                results["litellm_clean"] = "WARN: no <function= but not garbled"
        except Exception as e:
            print(f"  FAIL: {e}")
            results["litellm_clean"] = f"FAIL: {e}"
    else:
        print("\n=== Skipping litellm tests (no OPENAI_API_KEY) ===")

    # Test 5: Parsing
    try:
        test_5_action_parsing()
        results["parsing"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["parsing"] = f"FAIL: {e}"

    # Test 6: Docker
    try:
        test_6_docker_image()
        results["docker"] = "PASS"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["docker"] = f"FAIL: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "+" if status == "PASS" else ("!" if "WARN" in str(status) else "-")
        print(f"  [{icon}] {name}: {status}")
