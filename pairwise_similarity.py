#!/usr/bin/env python3
import argparse
import os
import sys
import random
import pandas as pd
from typing import List, Dict, Callable
from similarity import tokenize_plan, lcs_similarity, tfidf_similarity, sbert_similarity, read_tasks, load_model_outputs


def compute_included_ids(ids: List[str], present_id_sets: List[set]) -> List[str]:
    """Return list of IDs (in original order) that are present in all provided sets."""
    common = set(ids)
    for s in present_id_sets:
        common &= s
    return [i for i in ids if i in common]


def similarity_majority_voting(a_tokens: List[str], b_tokens: List[str]) -> float:
    """Exact match similarity: 1.0 if decompositions are exactly the same, 0.0 otherwise."""
    return 1.0 if a_tokens == b_tokens else 0.0


class LLM:
    """Represents an LLM with name, current reputation, and score history."""
    def __init__(self, name: str, initial_reputation: float = 4.0):
        self.name = name
        self.current_reputation = float(initial_reputation)
        self.scores: List[float] = []

    def update_reputation(self, score: float):
        self.scores.append(float(score))
        self.current_reputation = sum(self.scores) / len(self.scores)


class Oracle:
    """Runs pairwise similarity, selects best model per task using score * reputation, and tracks reputation history."""
    def __init__(self, model_names: List[str], id_to_decomp_per_model: List[Dict[str, str]], similarity_func: Callable = lcs_similarity):
        self.models: List[LLM] = [LLM(m) for m in model_names]
        self.model_names = model_names
        self.id_to_decomp_per_model = id_to_decomp_per_model
        self.similarity_func = similarity_func
        self.history: List[Dict[str, float]] = []  # per task, map model name -> reputation

    def _build_similarity_matrix(self, idx: str) -> pd.DataFrame:
        token_lists = [tokenize_plan(m.get(idx, "")) for m in self.id_to_decomp_per_model]
        matrix = []
        for a in range(len(self.models)):
            a_tokens = token_lists[a]
            row_vals = []
            for b in range(len(self.models)):
                b_tokens = token_lists[b]
                sim = 1.0 if a == b else self.similarity_func(a_tokens, b_tokens)
                row_vals.append(round(sim, 2))
            matrix.append(row_vals)
        df = pd.DataFrame(matrix, columns=self.model_names)
        df["sum"] = df.sum(axis=1).round(2)
        return df, token_lists

    def select_model(self, df_mat: pd.DataFrame) -> int:
        weighted = []
        for i, mdl in enumerate(self.models):
            score = float(df_mat.loc[i, "sum"])
            weighted.append(score * mdl.current_reputation)
        max_w = max(weighted)
        candidates = [i for i, w in enumerate(weighted) if w == max_w]
        return random.choice(candidates)

    def update_after_task(self, task_id: str):
        snapshot = {m.name: m.current_reputation for m in self.models}
        self.history.append({"ID": task_id, **snapshot})

    def run_task(self, idx: str, gt_decomp: str, outdir: str) -> dict:
        df_mat, token_lists = self._build_similarity_matrix(idx)
        best_idx = self.select_model(df_mat)
        best_model = self.models[best_idx]
        best_sum = float(df_mat.loc[best_idx, "sum"])
        
        # Update reputation for ALL models based on their scores
        for i, mdl in enumerate(self.models):
            score = float(df_mat.loc[i, "sum"])
            mdl.update_reputation(score)
        
        self.update_after_task(idx)

        # similarity to GT
        sim_to_gt = round(self.similarity_func(token_lists[best_idx], tokenize_plan(gt_decomp)), 2)

        # write per-task matrix
        out_name = f"task_{idx}_pairwise_similarity.tsv"
        out_path = os.path.join(outdir, out_name)
        df_mat.to_csv(out_path, sep="\t", index=False, float_format="%.2f")

        return {
            "task_id": idx,
            "best_model": best_model.name,
            "row_sum": f"{best_sum:.2f}",
            "sim_to_gt": f"{sim_to_gt:.2f}",
        }

    def run_all_tasks(self, included_ids: List[str], ids: List[str], gt_list: List[str], requests_list: List[str], outdir: str):
        best_rows = []
        for idx in included_ids:
            gt_decomp = dict(zip(ids, gt_list)).get(idx, "")
            best_info = self.run_task(idx, gt_decomp, outdir)
            # enrich with request text if desired later
            best_info["request"] = dict(zip(ids, requests_list)).get(idx, "")
            best_rows.append(best_info)

        self.write_summary(best_rows, outdir)


    def write_summary(self, best_rows: List[dict], outdir: str):
        """Write trimmed summary CSV, reputation history CSV, and print overall average to stdout."""
        trim_df = pd.DataFrame([
            {"ID": r["task_id"], "Score": r["row_sum"], "Similarity": r["sim_to_gt"], "Model": r["best_model"]}
            for r in best_rows
        ], columns=["ID", "Score", "Similarity", "Model"]) if best_rows else pd.DataFrame(columns=["ID", "Score", "Similarity", "Model"])

        summary_path = os.path.join(outdir, "selected_model_similarity_to_ground_truth.csv")
        trim_df.to_csv(summary_path, index=False)
        print(f"Wrote selected-model summary to {summary_path}")

        # Write reputation history
        reputation_rows = []
        for rep_entry in self.history:
            row = {"ID": rep_entry["ID"]}
            for model_name in self.model_names:
                row[model_name] = f"{rep_entry.get(model_name, 0.0):.4f}"
            reputation_rows.append(row)
        
        rep_df = pd.DataFrame(reputation_rows)
        reputation_path = os.path.join(outdir, "reputations.csv")
        rep_df.to_csv(reputation_path, index=False)
        print(f"Wrote reputation history to {reputation_path}")

        sims = pd.to_numeric([r.get("sim_to_gt") for r in best_rows], errors="coerce")
        overall_avg = float(pd.Series(sims).astype(float).mean()) if len(sims) else 0.0
        print(f"Overall average similarity to GT: {overall_avg:.4f}")


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pairwise similarity matrices (+row sums) and best-model-vs-GT summary.")
    ap.add_argument("--tasks", default="tasks.csv", help="Tasks CSV with  ID,Task,Decomposition")
    ap.add_argument("--outputs_dir", default="outputs", help="Directory containing llm_*.csv files")
    ap.add_argument("--outdir", default="pairwise_similarity", help="Output directory for per-task TSV matrices")
    return ap.parse_args()


def main():
    args = get_args()

    # Read tasks
    ids, requests_list, gt_list = read_tasks(args.tasks)

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

    # Define all similarity methods to run
    similarity_methods = [
        ("lcs", lcs_similarity),
        ("tfidf", tfidf_similarity),
        ("sbert", sbert_similarity),
        ("majority_voting", similarity_majority_voting),
    ]

    # Run for each similarity method
    for method_name, similarity_func in similarity_methods:
        print(f"\n{'='*60}")
        print(f"Running with {method_name.upper()} similarity method")
        print(f"{'='*60}\n")
        
        outdir = os.path.join(args.outputs_dir, f"pairwise_{method_name}")
        os.makedirs(outdir, exist_ok=True)

        # Initialize Oracle with models and similarity function
        oracle = Oracle(model_names, id_to_decomp_per_model, similarity_func)
        oracle.run_all_tasks(included_ids, ids, gt_list, requests_list, outdir)


if __name__ == "__main__":
    main()
