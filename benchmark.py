#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import pandas as pd

import requests

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class APIKeyError(RuntimeError):
    """Raised when an API key is missing or invalid for OpenRouter calls."""


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_models_file(path: str) -> List[str]:
    """Read model slugs from a file (one slug per line). Skip blank lines and lines starting with '#'."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Models file not found: {path}")
    models: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
    return models

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    # Use timezone-aware UTC datetime and omit microseconds for a stable ISO timestamp
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

def load_tasks_csv(csv_path: str, max_rows: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    # Require exact header names (case-sensitive) in this order: ID,Task,Decomposition
    df = pd.read_csv(csv_path, sep=",")
    cols = [c.strip() for c in list(df.columns)]
    expected = ["ID", "Task", "Decomposition"]
    if cols != expected:
        raise ValueError(f"Input CSV must have exactly three columns with headers: {expected}. Found: {cols}")
    if max_rows and max_rows > 0:
        df = df.head(max_rows)
    records = df.to_dict(orient="records")
    # Return the Task column name so existing code continues to work
    return "Task", records

def build_full_prompt(base_prompt: str, request_text: str) -> List[Dict[str, str]]:
    # OpenRouter supports OpenAI-style chat format
    content = f"{base_prompt}\n\nCitizen request:\n{request_text}\n"
    return [{"role": "user", "content": content}]

def write_batch_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        headers = ["ID", "Task", "Decomposition", "Latency"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def call_openrouter(model: str, messages: List[Dict[str, str]], timeout: int = 60) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        # Raise a dedicated error so callers can exit immediately without retries
        raise APIKeyError("OPENROUTER_API_KEY is not set. export OPENROUTER_API_KEY=<your_key>.")
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Recommended attribution headers (optional)
    site = os.environ.get("OPENROUTER_SITE_URL")
    appname = os.environ.get("OPENROUTER_APP_NAME")
    if site:
        headers["HTTP-Referer"] = site
    if appname:
        headers["X-Title"] = appname

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "stream": False,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # OpenAI-style response
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # Best-effort fallback: return the whole JSON as text
        return str(data)

def run_models(
    models: List[str],
    base_prompt: str,
    tasks_col: str,
    rows: List[Dict[str, Any]],
    outdir: str,
    timeout: int = 60,
    retries: int = 2,
    sleep_s: float = 0.0,
) -> List[str]:
    written = []
    for idx, model in enumerate(models, start=1):
        results: List[Dict[str, Any]] = []
        for i, row in enumerate(rows, start=1):
            request_text = str(row.get(tasks_col, "")).strip()
            messages = build_full_prompt(base_prompt, request_text)

            attempt = 0
            start = time.time()
            last_err = None
            while attempt <= retries:
                try:
                    resp_text = call_openrouter(model, messages, timeout=timeout)
                    latency_ms = int((time.time() - start) * 1000)
                    results.append({
                        "ID": i,
                        "Task": request_text,
                        "Decomposition": resp_text,
                        "Latency": latency_ms
                    })
                    break
                except APIKeyError as e:
                    # API key missing/invalid: print to stdout and exit immediately.
                    print(str(e))
                    sys.exit(1)
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < retries:
                        time.sleep(1.0 * (attempt + 1))
                        attempt += 1
                        continue
                    else:
                        latency_ms = int((time.time() - start) * 1000)
                        results.append({
                            "ID": i,
                            "Task": request_text,
                            "Decomposition": f"[ERROR] {last_err}",
                            "Latency": latency_ms
                        })
                        break

            if sleep_s > 0:
                time.sleep(sleep_s)

        out_path = os.path.join(outdir, f"llm_{os.path.basename(model)}.csv")
        write_batch_csv(out_path, results)
        written.append(out_path)
        print(f"Wrote: {out_path}")

    return written

def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark multiple models via OpenRouter (single API key)")
    ap.add_argument("--prompt", default="PROMPT_1.txt", help="Path to base prompt file")
    ap.add_argument("--input", default="tasks.csv", help="Path to tasks CSV")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--max-rows", type=int, default=0, help="Max rows from input (0 = all)")
    ap.add_argument("--models-file", default="models.txt", help="Path to file with model slugs (one per line)")
    ap.add_argument("--models", default=None, help="Comma-separated model slugs to test (overrides/filters models-file). Accepts exact or substring matches.")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout per request in seconds")
    ap.add_argument("--retries", type=int, default=2, help="Retries on error")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    return ap.parse_args()

def load_models(args) -> List[str]:
    """Load and optionally filter models based on args (models-file and optional --models selectors)."""
    models_file = args.models_file
    try:
        models = read_models_file(models_file)
    except Exception as e:
        print(f"Error reading models file '{models_file}': {e}", file=sys.stderr)
        sys.exit(1)
    if not models:
        print(f"No models found in '{models_file}'. Provide at least one model slug.", file=sys.stderr)
        sys.exit(1)

    # Optional override/filter: --models "slug1,slug2" (exact or substring match)
    if args.models:
        selectors = [s.strip() for s in args.models.split(",") if s.strip()]
        if not selectors:
            print("No valid model selectors provided via --models.", file=sys.stderr)
            sys.exit(1)
        filtered = []
        for sel in selectors:
            matched = [m for m in models if m == sel or sel in m]
            if matched:
                filtered.extend(matched)
            else:
                print(f"Warning: selector '{sel}' did not match any model in {models_file}", file=sys.stderr)
        # preserve order and dedupe
        seen = set()
        models_filtered: List[str] = []
        for m in filtered:
            if m not in seen:
                seen.add(m)
                models_filtered.append(m)
        models = models_filtered
        if not models:
            print("No models matched the --models selectors.", file=sys.stderr)
            sys.exit(1)

    return models

def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    # Load and optionally filter models
    models = load_models(args)

    base_prompt = read_file(args.prompt)
    tasks_col, records = load_tasks_csv(args.input, max_rows=args.max_rows)

    print(f"Detected task column: '{tasks_col}' with {len(records)} rows.")
    print(f"Running models via OpenRouter: {', '.join(models)}")

    run_models(
        models=models,
        base_prompt=base_prompt,
        tasks_col=tasks_col,
        rows=records,
        outdir=args.outdir,
        timeout=args.timeout,
        retries=args.retries,
        sleep_s=args.sleep,
    )

    print("\nAll done.")

if __name__ == "__main__":
    main()
