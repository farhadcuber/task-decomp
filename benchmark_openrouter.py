#!/usr/bin/env python3
"""
Benchmark multiple LLMs via a SINGLE unified API: OpenRouter

Reads:
  - PROMPT_0.txt                 (base prompt/system instruction)
  - smart_city_tasks.csv         (tasks to evaluate; detects the first suitable text column)

Writes (one file per model):
  - out/llm_1.csv, out/llm_2.csv, ...

Requirements:
  - Python 3.9+
  - pip install requests pandas python-dotenv (pandas optional, but recommended)

Environment variables (required):
  - OPENROUTER_API_KEY=...

Optional environment variables (recommended by OpenRouter for attribution):
  - OPENROUTER_SITE_URL=https://yourdomain.example   # or any URL you control
  - OPENROUTER_APP_NAME=Your App Name

Endpoint:
  - Base URL: https://openrouter.ai/api/v1
  - Endpoint: /chat/completions  (OpenAI-compatible)

Example:
  python benchmark_openrouter.py \
    --prompt PROMPT_0.txt \
    --input smart_city_tasks.csv \
    --outdir out \
    --models openai/gpt-4o-mini anthropic/claude-3.5-sonnet google/gemini-1.5-pro mistralai/mistral-large qwen/qwen2.5-72b-instruct

Notes:
  * You can list ANY models you have access to on OpenRouter; slugs vary over time.
  * This script is vendor-agnostic: you only need ONE API key and ONE endpoint.
  * Temperature is set to 0.0 for determinism.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

import requests

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def detect_text_column(headers: List[str]) -> str:
    preferred = ["request", "task", "input", "prompt", "text"]
    for p in preferred:
        for h in headers:
            if h.strip().lower() == p:
                return h
    return headers[0]

def load_tasks_csv(csv_path: str, max_rows: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
    if pd is not None:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 0:
            raise ValueError("CSV has no columns.")
        text_col = detect_text_column(list(df.columns))
        if max_rows and max_rows > 0:
            df = df.head(max_rows)
        records = df.to_dict(orient="records")
        return text_col, records
    # Fallback manual CSV reader
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError("CSV has no header.")
        text_col = detect_text_column(headers)
        records: List[Dict[str, Any]] = []
        for row in reader:
            records.append(row)
            if max_rows and max_rows > 0 and len(records) >= max_rows:
                break
        return text_col, records

def build_full_prompt(base_prompt: str, request_text: str) -> List[Dict[str, str]]:
    # OpenRouter supports OpenAI-style chat format
    content = f"{base_prompt}\n\nCitizen request:\n{request_text}\n"
    return [{"role": "user", "content": content}]

def write_batch_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        headers = ["id", "request", "response", "latency_ms", "provider", "model", "timestamp"]
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
        raise RuntimeError("OPENROUTER_API_KEY is not set. Get one from https://openrouter.ai/")
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
                        "id": i,
                        "request": request_text,
                        "response": resp_text,
                        "latency_ms": latency_ms,
                        "provider": "openrouter",
                        "model": model,
                        "timestamp": now_iso(),
                    })
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempt < retries:
                        time.sleep(1.0 * (attempt + 1))
                        attempt += 1
                        continue
                    else:
                        latency_ms = int((time.time() - start) * 1000)
                        results.append({
                            "id": i,
                            "request": request_text,
                            "response": f"[ERROR] {last_err}",
                            "latency_ms": latency_ms,
                            "provider": "openrouter",
                            "model": model,
                            "timestamp": now_iso(),
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
    ap.add_argument("--input", default="smart_city_tasks.csv", help="Path to tasks CSV")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--max-rows", type=int, default=0, help="Max rows from input (0 = all)")
    ap.add_argument("--models", nargs="+", required=False, default=[
        # Provide example defaults; replace with any slugs you have access to.
        "openai/gpt-5",
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.5-pro",
        "mistralai/mistral-large",
        "meta-llama/llama-4-maverick",
    ], help="List of OpenRouter model slugs to benchmark")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout per request in seconds")
    ap.add_argument("--retries", type=int, default=2, help="Retries on error")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    base_prompt = read_file(args.prompt)
    tasks_col, records = load_tasks_csv(args.input, max_rows=args.max_rows)

    print(f"Detected task column: '{tasks_col}' with {len(records)} rows.")
    print(f"Running models via OpenRouter: {', '.join(args.models)}")


    written = run_models(
        models=args.models,
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
