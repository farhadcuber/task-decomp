#!/usr/bin/env python3
import argparse
import os
import sys
import re
import pandas as pd
from typing import List

def tokenize_plan(s: str) -> List[str]:
    """Tokenize a decomposition plan.
    - Plans are separated by semicolons (`;`).
    """
    if not isinstance(s, str):
        return []
    return [t.strip() for t in s.split(';') if t.strip()]

def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
    return dp[n][m]

def lcs_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    denom = max(len(a_tokens), len(b_tokens))
    if denom == 0:
        return 1.0
    return lcs_length(a_tokens, b_tokens) / denom

def read_tasks(tasks_path: str):
    try:
        tasks_df = pd.read_csv(tasks_path, sep=",")
    except Exception as e:
        print(f"[ERROR] Unable to read tasks file '{tasks_path}': {e}")
        sys.exit(2)

    expected_cols = ["ID", "Task", "Decomposition"]
    if list(tasks_df.columns[:3]) != expected_cols:
        print(f"[ERROR] Tasks CSV must start with headers ID,Task,Decomposition. Found: {tasks_df.columns.tolist()}")
        sys.exit(2)

    ids = tasks_df["ID"].astype(str).tolist()
    requests_list = tasks_df["Task"].astype(str).tolist()
    gt_list = tasks_df["Decomposition"].astype(str).tolist()
    return tasks_df, ids, requests_list, gt_list


def find_model_files(outputs_dir: str):
    if not os.path.isdir(outputs_dir):
        print(f"[ERROR] Outputs directory not found: {outputs_dir}")
        sys.exit(2)

    files = sorted([f for f in os.listdir(outputs_dir) if f.startswith("llm_") and f.endswith(".csv")])
    if not files:
        print(f"[ERROR] No llm_*.csv files found in {outputs_dir}")
        sys.exit(2)
    return files


def load_model_outputs(outputs_dir: str):
    """Load llm_*.csv files from outputs_dir and return
    (model_names, id_to_decomp_per_model, present_id_sets).

    This is a shared loader used by other scripts to keep parsing consistent.
    """
    if not os.path.isdir(outputs_dir):
        print(f"[ERROR] Outputs directory not found: {outputs_dir}")
        sys.exit(2)

    files = sorted([f for f in os.listdir(outputs_dir) if f.startswith("llm_") and f.endswith(".csv")])
    if not files:
        print(f"[ERROR] No llm_*.csv files found in {outputs_dir}")
        sys.exit(2)

    model_names = []
    id_to_decomp_per_model = []
    present_id_sets = []

    for fname in files:
        fpath = os.path.join(outputs_dir, fname)
        try:
            df = pd.read_csv(fpath, sep=",")
        except Exception as e:
            print(f"[WARN] Skipping {fname}: read error: {e}")
            continue

        if "ID" not in df.columns or "Decomposition" not in df.columns:
            print(f"[WARN] Skipping {fname}: missing 'ID' or 'Decomposition' column")
            continue

        model_name = fname[len("llm_"):-len(".csv")]
        if "model" in df.columns:
            nn = df["model"].dropna().astype(str)
            if not nn.empty and nn.iloc[0].strip():
                model_name = nn.iloc[0].strip()

        id_series = df["ID"].astype(str).tolist()
        dec_series = df["Decomposition"].astype(str).tolist()
        id_to_decomp = {}
        present_ids = set()
        for rid, dec in zip(id_series, dec_series):
            if rid not in id_to_decomp:
                id_to_decomp[rid] = dec
                present_ids.add(rid)

        model_names.append(model_name)
        id_to_decomp_per_model.append(id_to_decomp)
        present_id_sets.append(present_ids)

    return model_names, id_to_decomp_per_model, present_id_sets


def compute_sims_from_map(model_name: str, id_to_decomp: dict, ids: List[str], gt_list: List[str]):
    """Compute similarity list and present_ids from an ID->decomp mapping."""
    present_ids = set(id_to_decomp.keys())
    sims = []
    for idx, gt in zip(ids, gt_list):
        resp = id_to_decomp.get(idx, "")
        gt_tokens = tokenize_plan(gt)
        resp_tokens = tokenize_plan(resp)
        sim = lcs_similarity(gt_tokens, resp_tokens)
        sims.append(round(float(sim), 6))

    return model_name, sims, present_ids


def process_model_file(fpath: str, ids: List[str], gt_list: List[str]):
    try:
        df = pd.read_csv(fpath, sep=",")
    except Exception as e:
        print(f"[WARN] Skipping {os.path.basename(fpath)}: read error: {e}")
        return None

    if "ID" not in df.columns or "Decomposition" not in df.columns:
        print(f"[WARN] Skipping {os.path.basename(fpath)}: missing 'ID' or 'Decomposition' column")
        return None

    model_name = os.path.basename(fpath)[len("llm_"):-len(".csv")]
    if "model" in df.columns:
        nn = df["model"].dropna().astype(str)
        if not nn.empty and nn.iloc[0].strip():
            model_name = nn.iloc[0].strip()

    id_series = df["ID"].astype(str)
    dec_series = df["Decomposition"].astype(str)
    id_to_resp = {}
    present_ids = set()
    for rid, dec in zip(id_series.tolist(), dec_series.tolist()):
        if rid not in id_to_resp:
            id_to_resp[rid] = dec
            present_ids.add(rid)

    sims = []
    missing = 0
    for idx, gt in zip(ids, gt_list):
        resp = id_to_resp.get(idx, "")
        if resp == "":
            missing += 1
        gt_tokens = tokenize_plan(gt)
        resp_tokens = tokenize_plan(resp)
        sim = lcs_similarity(gt_tokens, resp_tokens)
        sims.append(round(float(sim), 6))

    if missing:
        print(f"[INFO] {os.path.basename(fpath)}: {missing} task IDs missing in model outputs (they will be excluded from final table if any model misses them).")

    return model_name, sims, present_ids


def format_and_write_wide(wide: pd.DataFrame, out_path: str):
    model_cols = [c for c in wide.columns if c != "ID"]
    wide_out = wide.copy()
    for c in model_cols:
        wide_out[c] = pd.to_numeric(wide_out[c], errors="coerce").fillna(0.0).map(lambda v: f"{v:.2f}")

    wide_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote wide similarity table to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="LCS similarity: wide CSV (ID + per-model similarities).")
    ap.add_argument("--tasks", default="tasks.csv", help="Tasks CSV with ID,Task,Decomposition")
    ap.add_argument("--outputs_dir", default="outputs", help="Directory containing llm_*.csv files")
    ap.add_argument("--out", default=None, help="Output CSV path (default: write 'similarity.csv' inside --outputs_dir)")
    args = ap.parse_args()

    tasks_df, ids, requests_list, gt_list = read_tasks(args.tasks)

    out_path = args.out if args.out else os.path.join(args.outputs_dir, "similarity.csv")

    # Load model outputs using shared loader
    model_names, id_to_decomp_per_model, present_id_sets = load_model_outputs(args.outputs_dir)

    # Compute sims for each model from the loaded maps
    model_results = []
    for model_name, id_to_decomp in zip(model_names, id_to_decomp_per_model):
        res = compute_sims_from_map(model_name, id_to_decomp, ids, gt_list)
        model_results.append(res)

    if not model_results:
        print("[ERROR] No valid model outputs processed. Exiting.")
        sys.exit(2)

    # Compute IDs present in ALL processed models (intersection)
    model_present_sets = [mrs[2] for mrs in model_results]
    intersection_ids = set(ids)
    for s in model_present_sets:
        intersection_ids &= s

    included_ids = [i for i in ids if i in intersection_ids]
    if not included_ids:
        print("[ERROR] No task IDs are present in all model outputs; nothing to compare.")
        sys.exit(0)

    # Build wide table only with included IDs (preserve tasks order)
    wide = pd.DataFrame({"ID": included_ids})
    numeric_avgs = {}
    for model_name, sims, _present in model_results:
        id_to_sim = dict(zip(ids, sims))
        col_vals = [id_to_sim[i] for i in included_ids]
        wide[model_name] = col_vals
        numeric_avgs[model_name] = sum(col_vals) / len(col_vals) if col_vals else float("nan")

    format_and_write_wide(wide, out_path)

    if numeric_avgs:
        print("\nAverage LCS Similarity by Model:")
        for c, avg in numeric_avgs.items():
            print(f"  {c:30s}  {avg:.4f}")

if __name__ == "__main__":
    main()
