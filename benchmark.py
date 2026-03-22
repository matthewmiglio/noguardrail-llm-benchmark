"""
Dolphin Model Guardrail Benchmark
==================================
Runs a suite of test prompts against unguardrailed dolphin models
and records their responses for later analysis.

Usage:
    python benchmark.py                  # Run all models, all prompts
    python benchmark.py --model <id>     # Run a specific model only
    python benchmark.py --size small     # Run only models of a given size category
    python benchmark.py --format gguf    # Run only GGUF models
    python benchmark.py --list-models    # List available models
    python benchmark.py                  # Automatically skips already-completed prompts
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Enable fast HuggingFace downloads (5-10x faster) BEFORE importing huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config" / "config.json"
MODELS_PATH = SCRIPT_DIR / "config" / "models.json"
PROMPTS_PATH = SCRIPT_DIR / "config" / "prompts.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# Files to skip when downloading transformers models
IGNORE_PATTERNS = [
    "*.md",
    "*.txt",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    ".gitattributes",
    "original/",
    "training_args.bin",
]


def check_hf_transfer():
    """Check if hf_transfer is installed for fast downloads."""
    try:
        import hf_transfer  # noqa: F401
        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
            print("[OK] hf_transfer enabled - fast downloads active")
            return True
    except ImportError:
        pass

    print("[WARNING] hf_transfer not installed - downloads will be 5-10x slower!")
    print("  Install it with: pip install hf-transfer")
    print("  Or: poetry add hf-transfer")
    print()
    return False


def load_or_create_config():
    """Load config or create it on first run by prompting the user."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
            if config.get("model_download_dir"):
                return config
        except (json.JSONDecodeError, KeyError):
            pass

    print("=" * 60)
    print("  First-time setup")
    print("=" * 60)
    print()
    print("Models will be downloaded from HuggingFace and stored locally.")
    print("They can be very large (1GB - 30GB+ each).")
    print()

    drive = input("Enter the drive/directory to store models (e.g. D:/): ").strip()
    if not drive:
        drive = "D:/"
    drive = drive.replace("\\", "/")
    if not drive.endswith("/"):
        drive += "/"

    config = {
        "model_download_dir": drive + "dolphin_models",
        "max_tokens": 512,
        "temperature": 0.7,
        "n_ctx": 2048,
        "n_gpu_layers": -1,
        "system_prompt": ""
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {CONFIG_PATH}")
    print(f"Models will be downloaded to: {config['model_download_dir']}")
    print()
    return config


def load_models():
    """Load the models list from config."""
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return data["models"]


def load_prompts():
    """Load the test prompts from config."""
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    return data["prompts"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def find_gguf_file(repo_id, preferred_pattern):
    """Find the best GGUF file in a HuggingFace repo."""
    print(f"  Listing files in {repo_id}...")
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"  ERROR listing repo files: {e}")
        return None

    gguf_files = [f for f in files if f.endswith(".gguf")]

    if not gguf_files:
        print(f"  WARNING: No .gguf files found in {repo_id}")
        return None

    # Try preferred pattern first, then common good quantizations
    search_patterns = [preferred_pattern, "Q4_K_M", "q4_k_m", "Q4_K_S", "Q5_K_M", "Q8_0"]

    for pattern in search_patterns:
        if not pattern:
            continue
        for f in gguf_files:
            if pattern.lower() in f.lower():
                return f

    # Fall back to first available GGUF file
    print(f"  No preferred quant found, using: {gguf_files[0]}")
    return gguf_files[0]


def download_gguf_model(model_info, download_dir):
    """Download a single GGUF file from HuggingFace. Returns local file path."""
    repo_id = model_info["repo_id"]
    model_dir = Path(download_dir) / repo_id.replace("/", "_")

    # Check if already downloaded
    if model_dir.exists():
        existing = list(model_dir.rglob("*.gguf"))
        if existing:
            print(f"  Already downloaded: {existing[0].name}")
            return str(existing[0])

    filename = find_gguf_file(repo_id, model_info.get("filename_pattern", "Q4_K_M"))
    if not filename:
        return None

    print(f"  Downloading {filename} from {repo_id}...")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
        )
        print(f"  Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None


def download_transformers_model(model_info, download_dir):
    """Download a transformers model via snapshot_download. Returns local directory path."""
    repo_id = model_info["repo_id"]
    model_dir = Path(download_dir) / repo_id.replace("/", "_")

    # Check if already downloaded (look for config.json which all transformers models have)
    config_file = model_dir / "config.json"
    if config_file.exists():
        print(f"  Already downloaded: {model_dir.name}")
        return str(model_dir)

    print(f"  Downloading {repo_id} (transformers format)...")
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"  Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None


def download_model(model_info, download_dir):
    """Download a model based on its format. Returns local path."""
    fmt = model_info.get("format", "gguf")
    if fmt == "gguf":
        return download_gguf_model(model_info, download_dir)
    elif fmt == "transformers":
        return download_transformers_model(model_info, download_dir)
    else:
        print(f"  Unsupported format: {fmt}")
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gguf_model(model_path, config):
    """Load a GGUF model using llama-cpp-python."""
    from llama_cpp import Llama

    print(f"  Loading GGUF model: {Path(model_path).name}")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=config.get("n_ctx", 2048),
            n_gpu_layers=config.get("n_gpu_layers", -1),
            verbose=False,
        )
        return ("gguf", llm)
    except Exception as e:
        print(f"  ERROR loading GGUF model: {e}")
        return None


def load_transformers_model(model_path, config):
    """Load a transformers model using AutoModelForCausalLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading transformers model: {Path(model_path).name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        return ("transformers", (model, tokenizer))
    except Exception as e:
        print(f"  ERROR loading transformers model: {e}")
        return None


def load_model(model_info, model_path, config):
    """Load a model based on its format. Returns (format, model_obj) or None."""
    fmt = model_info.get("format", "gguf")
    if fmt == "gguf":
        return load_gguf_model(model_path, config)
    elif fmt == "transformers":
        return load_transformers_model(model_path, config)
    else:
        print(f"  Cannot load format: {fmt}")
        return None


def unload_model(loaded):
    """Free model memory."""
    if loaded is None:
        return
    fmt, obj = loaded
    del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_prompt_gguf(llm, prompt_text, config):
    """Run a prompt through a GGUF model."""
    system_prompt = config.get("system_prompt", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0.7),
    )
    return output["choices"][0]["message"]["content"]


def run_prompt_transformers(model_and_tokenizer, prompt_text, config):
    """Run a prompt through a transformers model."""
    import torch

    model, tokenizer = model_and_tokenizer
    system_prompt = config.get("system_prompt", "")

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    # Try chat template first, fall back to raw text
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.get("max_tokens", 512),
            temperature=config.get("temperature", 0.7),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def run_prompt(loaded, prompt_text, config):
    """Run a single prompt through a loaded model. Returns (response, status, elapsed)."""
    fmt, obj = loaded

    start = time.time()
    try:
        if fmt == "gguf":
            response = run_prompt_gguf(obj, prompt_text, config)
        elif fmt == "transformers":
            response = run_prompt_transformers(obj, prompt_text, config)
        else:
            response = ""
        status = "success"
    except Exception as e:
        response = ""
        status = f"error: {str(e)}"

    elapsed = time.time() - start
    return response, status, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "model_id", "model_repo", "model_params", "model_base", "model_format",
    "prompt_id", "category", "subcategory", "technique",
    "prompt", "response", "status", "time_seconds", "timestamp"
]


def init_results_csv(results_path):
    """Create a fresh results CSV with header row."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()


def write_result_row(results_path, row, max_retries=60, retry_delay=5):
    """Append a single result row to the CSV. Retries on drive disconnect."""
    for attempt in range(max_retries):
        try:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                writer.writerow(row)
            return
        except (FileNotFoundError, OSError) as e:
            if attempt < max_retries - 1:
                print(f"\n  [WARN] Write failed (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  [WARN] Drive may have disconnected. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"\n  [ERROR] Write failed after {max_retries} attempts. Giving up.")
                raise


def get_completed_pairs(results_path):
    """Read existing results CSV to find already-completed (model, prompt) pairs."""
    completed = set()
    if results_path.exists():
        try:
            with open(results_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "success":
                        completed.add((row["model_id"], row["prompt_id"]))
        except Exception:
            pass
    return completed


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def make_error_row(model_info, prompt_item, error_status):
    """Create an error result row."""
    return {
        "model_id": model_info["id"],
        "model_repo": model_info["repo_id"],
        "model_params": model_info["params"],
        "model_base": model_info.get("base", ""),
        "model_format": model_info.get("format", "gguf"),
        "prompt_id": prompt_item["id"],
        "category": prompt_item["category"],
        "subcategory": prompt_item.get("subcategory", ""),
        "technique": prompt_item.get("technique", "direct"),
        "prompt": prompt_item["prompt"],
        "response": "",
        "status": error_status,
        "time_seconds": 0,
        "timestamp": datetime.now().isoformat(),
    }


def run_benchmark(models, prompts, config, reset=False):
    """Run the full benchmark: all models x all prompts."""
    results_path = RESULTS_DIR / "benchmark_results.csv"

    if reset:
        print("Resetting: wiping existing results.")
        init_results_csv(results_path)
        completed = set()
    else:
        # Always check for already-completed (model, prompt) pairs
        completed = get_completed_pairs(results_path)
        if completed:
            print(f"Found {len(completed)} already-completed results in {results_path.name}.")
        if not results_path.exists():
            init_results_csv(results_path)

    download_dir = config["model_download_dir"]
    total_tasks = len(models) * len(prompts) - len(completed)
    print(f"\nBenchmark: {len(models)} models x {len(prompts)} prompts = {len(models) * len(prompts)} total")
    if completed:
        print(f"Skipping {len(completed)} already completed, {total_tasks} remaining")
    print()

    for model_info in models:
        model_id = model_info["id"]
        print(f"\n{'='*60}")
        print(f"Model: {model_id} ({model_info['params']}, {model_info.get('format', 'gguf')})")
        print(f"{'='*60}")

        # Check if all prompts already done for this model
        model_prompts_remaining = [
            p for p in prompts
            if (model_id, p["id"]) not in completed
        ]
        if not model_prompts_remaining:
            print("  All prompts already completed, skipping.")
            continue

        # Download
        model_path = download_model(model_info, download_dir)
        if not model_path:
            print(f"  SKIPPING {model_id}: download failed")
            for prompt_item in model_prompts_remaining:
                write_result_row(results_path, make_error_row(model_info, prompt_item, "error: download failed"))
            continue

        # Load
        loaded = load_model(model_info, model_path, config)
        if not loaded:
            print(f"  SKIPPING {model_id}: failed to load")
            for prompt_item in model_prompts_remaining:
                write_result_row(results_path, make_error_row(model_info, prompt_item, "error: load failed"))
            continue

        # Run prompts
        print(f"\n  Running {len(model_prompts_remaining)} prompts...")
        for prompt_item in tqdm(model_prompts_remaining, desc=f"  {model_id}", ncols=80):
            response, status, elapsed = run_prompt(loaded, prompt_item["prompt"], config)

            row = {
                "model_id": model_id,
                "model_repo": model_info["repo_id"],
                "model_params": model_info["params"],
                "model_base": model_info.get("base", ""),
                "model_format": model_info.get("format", "gguf"),
                "prompt_id": prompt_item["id"],
                "category": prompt_item["category"],
                "subcategory": prompt_item.get("subcategory", ""),
                "technique": prompt_item.get("technique", "direct"),
                "prompt": prompt_item["prompt"],
                "response": response,
                "status": status,
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
            write_result_row(results_path, row)

        # Free memory
        unload_model(loaded)
        print(f"  Done with {model_id}, model unloaded.")

    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to: {results_path}")
    print(f"{'='*60}")


def list_models_info(models):
    """Print a table of available models."""
    print(f"\n{'ID':<45} {'Params':<8} {'Format':<14} {'Size':<10} {'Base':<18} {'Enabled'}")
    print("-" * 130)
    for m in models:
        enabled = "yes" if m.get("enabled", True) else "NO"
        print(f"{m['id']:<45} {m['params']:<8} {m.get('format','gguf'):<14} {m['size_category']:<10} {m.get('base', ''):<18} {enabled}")
    enabled_count = sum(1 for m in models if m.get("enabled", True))
    print(f"\nTotal: {len(models)} models ({enabled_count} enabled, {len(models) - enabled_count} disabled)")


def main():
    parser = argparse.ArgumentParser(
        description="Dolphin Model Guardrail Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                     Run full benchmark (enabled models only)
  python benchmark.py --model dolphin-x1-8b  Run single model
  python benchmark.py --size small        Run only small models
  python benchmark.py --format gguf       Run only GGUF models
  python benchmark.py --reset             Wipe results and start fresh
  python benchmark.py --list-models       Show available models
  python benchmark.py --all               Include disabled models
        """,
    )
    parser.add_argument("--model", type=str, help="Run only this model (by id)")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large", "xlarge"],
                        help="Run only models of this size category")
    parser.add_argument("--format", type=str, choices=["gguf", "transformers"],
                        help="Run only models of this format")
    parser.add_argument("--all", action="store_true", help="Include disabled models")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--resume", action="store_true", help="(Deprecated, now default behavior) Skip already-completed prompts")
    parser.add_argument("--reset", action="store_true", help="Wipe existing results and start fresh")
    parser.add_argument("--max-tokens", type=int, help="Override max tokens per response")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--n-gpu-layers", type=int, help="Override GPU layers (-1 = all)")
    args = parser.parse_args()

    # Check for fast downloads
    check_hf_transfer()

    # Load models list
    models = load_models()

    if args.list_models:
        list_models_info(models)
        return

    # Filter to enabled models by default
    if not args.all:
        models = [m for m in models if m.get("enabled", True)]

    # Load or create config
    config = load_or_create_config()

    # Apply CLI overrides
    if args.max_tokens is not None:
        config["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.n_gpu_layers is not None:
        config["n_gpu_layers"] = args.n_gpu_layers

    # Filter models
    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not found. Use --list-models to see available models.")
            sys.exit(1)
    if args.size:
        models = [m for m in models if m["size_category"] == args.size]
        if not models:
            print(f"ERROR: No models found with size category '{args.size}'.")
            sys.exit(1)
    if args.format:
        models = [m for m in models if m.get("format", "gguf") == args.format]
        if not models:
            print(f"ERROR: No models found with format '{args.format}'.")
            sys.exit(1)

    # Load prompts
    prompts = load_prompts()

    print(f"Dolphin Guardrail Benchmark")
    print(f"Models: {len(models)}")
    print(f"Prompts: {len(prompts)}")
    print(f"Download dir: {config['model_download_dir']}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Temperature: {config['temperature']}")
    print(f"GPU layers: {config['n_gpu_layers']}")

    run_benchmark(models, prompts, config, reset=args.reset)


if __name__ == "__main__":
    main()
