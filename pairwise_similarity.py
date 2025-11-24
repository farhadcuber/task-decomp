#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from typing import List
from similarity import tokenize_plan, lcs_similarity, read_tasks, load_model_outputs


def compute_included_ids(ids: List[str], present_id_sets: List[set]) -> List[str]:
    """Return list of IDs (in original order) that are present in all provided sets."""
    common = set(ids)
    for s in present_id_sets:
        common &= s
    return [i for i in ids if i in common]


def process_task_id(idx: str, model_names: List[str], id_to_decomp_per_model: List[dict], ids: List[str], gt_list: List[str], outdir: str):
    """Build pairwise matrix for a single task ID, write TSV to outdir, and return best-row info dict."""
    # build token lists for each model for this ID
    model_token_lists = [tokenize_plan(id_to_decomp.get(idx, "")) for id_to_decomp in id_to_decomp_per_model]

    # build NxN similarity matrix
    matrix = []
    for a in range(len(model_names)):
        row_vals = []
        a_tokens = model_token_lists[a]
        for b in range(len(model_names)):
            b_tokens = model_token_lists[b]
            sim = 1.0 if a == b else lcs_similarity(a_tokens, b_tokens)
            row_vals.append(round(sim, 2))
        matrix.append(row_vals)

    df_mat = pd.DataFrame(matrix, columns=model_names)
    df_mat["sum"] = df_mat.sum(axis=1).round(2)

    best_idx = int(df_mat["sum"].idxmax())
    best_model = model_names[best_idx]
    best_sum = float(df_mat.loc[best_idx, "sum"])

    # similarity of best model to GT
    gt_decomp = dict(zip(ids, gt_list)).get(idx, "")
    best_tokens = model_token_lists[best_idx]
    sim_to_gt = round(lcs_similarity(best_tokens, tokenize_plan(gt_decomp)), 2)

    # write per-task matrix
    out_name = f"task_{idx}_pairwise_similarity.tsv"
    out_path = os.path.join(outdir, out_name)
    df_mat.to_csv(out_path, sep="\t", index=False, float_format="%.2f")
    print(f"Wrote {out_path}")

    return {
        "task_id": idx,
        "best_model": best_model,
        "row_sum": f"{best_sum:.2f}",
        "sim_to_gt": f"{sim_to_gt:.2f}",
    }


def write_summary_and_print_overall(best_rows: List[dict], outdir: str):
    """Write trimmed summary CSV and print overall average to stdout."""
    trim_df = pd.DataFrame([
        {"ID": r["task_id"], "Score": r["row_sum"], "Similarity": r["sim_to_gt"], "Model": r["best_model"]}
        for r in best_rows
    ], columns=["ID", "Score", "Similarity", "Model"]) if best_rows else pd.DataFrame(columns=["ID", "Score", "Similarity", "Model"])

    summary_path = os.path.join(outdir, "selected_model_similarity_to_ground_truth.csv")
    trim_df.to_csv(summary_path, index=False)
    print(f"Wrote selected-model summary to {summary_path}")

    sims = pd.to_numeric([r.get("sim_to_gt") for r in best_rows], errors="coerce")
    overall_avg = float(pd.Series(sims).astype(float).mean()) if len(sims) else 0.0
    print(f"Overall average similarity to GT: {overall_avg:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Pairwise similarity matrices (+row sums) and best-model-vs-GT summary.")
    ap.add_argument("--tasks", default="tasks.csv", help="Tasks CSV with  ID,Task,Decomposition")
    ap.add_argument("--outputs_dir", default="outputs", help="Directory containing llm_*.csv files")
    ap.add_argument("--outdir", default="pairwise_similarity", help="Output directory for per-task TSV matrices")
    args = ap.parse_args()

    # Read tasks
    tasks_df, ids, requests_list, gt_list = read_tasks(args.tasks)

    # Load model outputs
    if not os.path.isdir(args.outputs_dir):
        print(f"[ERROR] Outputs directory not found: {args.outputs_dir}")
        sys.exit(2)

    model_names, id_to_decomp_per_model, present_id_sets = load_model_outputs(args.outputs_dir)
    if len(model_names) < 2:
        print("[ERROR] Need at least two valid llm_*.csv files with 'ID' and 'Decomposition' columns.")
        sys.exit(2)

    included_ids = compute_included_ids(ids, present_id_sets)
    if not included_ids:
        print("[ERROR] No task IDs are present in all model outputs; nothing to compare.")
        sys.exit(0)

    outdir = os.path.join(args.outputs_dir, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    best_rows = []
    for idx in included_ids:
        best_info = process_task_id(idx, model_names, id_to_decomp_per_model, ids, gt_list, outdir)
        # enrich with request text if desired later
        best_info["request"] = dict(zip(ids, requests_list)).get(idx, "")
        best_rows.append(best_info)

    write_summary_and_print_overall(best_rows, outdir)


if __name__ == "__main__":
    main()
