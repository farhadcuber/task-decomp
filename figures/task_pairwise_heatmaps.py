#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

METHODS = ["lcs", "sbert", "tfidf"]
# METHODS = ["lcs", "majority_voting", "sbert", "tfidf"]


def shorten_model_name(name: str) -> str:
    """Shorten model names for display."""
    name_map = {
        "gpt-4o-mini": "gpt",
        "claude-sonnet-4.5": "claude",
        "gemini-2.5-flash": "gemini",
        "mistral-large": "mistral",
    }
    return name_map.get(name, name)


def load_pairwise_matrix(experiment_dir: str, task_id: str, method: str):
    """Load the pairwise similarity matrix TSV for a method and task, dropping the 'sum' column."""
    tsv_path = os.path.join(experiment_dir, f"pairwise_{method}", f"task_{task_id}_pairwise_similarity.tsv")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(f"Missing file: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    # Remove 'sum' column if present
    if "sum" in df.columns:
        df = df.drop(columns=["sum"])  # square matrix remains

    # Ensure numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    # Row labels assumed to be in the same order as columns (construction in pairwise generator)
    labels = [shorten_model_name(col) for col in df.columns]
    return labels, df.values


def plot_heatmaps(experiment_dir: str, task_id: str, output: str | None):
    """Plot 1x3 heatmaps for three methods for the given experiment/task."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes = axes.flatten()

    # Define colormap from light blue (0) to pink (1) for better contrast
    cmap = LinearSegmentedColormap.from_list('blue_pink', ['#AEC6CF', '#FFB3BA'])
    
    method_display_names = {
        "lcs": "LCS",
        # "majority_voting": "Majority Voting",
        "sbert": "SBERT",
        "tfidf": "TF-IDF",
    }
    
    for idx, (ax, method) in enumerate(zip(axes, METHODS)):
        try:
            labels, mat = load_pairwise_matrix(experiment_dir, task_id, method)
        except Exception as e:
            ax.axis('off')
            ax.set_title(f"{method_display_names.get(method, method.upper())} (error)")
            ax.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=9, wrap=True)
            continue
        
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(method_display_names.get(method, method.upper()))
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Only show y-labels on the leftmost subplot
        if idx == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])

        # Draw grid lines for readability
        ax.set_xlim(-0.5, len(labels)-0.5)
        ax.set_ylim(len(labels)-0.5, -0.5)
        ax.set_aspect('equal')

        # Annotate values
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mat[i, j]
                if np.isnan(val):
                    text = ''
                else:
                    text = f"{val:.2f}"
                ax.text(j, i, text, ha='center', va='center', color='white' if val >= 0.6 else 'black', fontsize=8)

    fig.suptitle("Pairwise Similarity Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved heatmaps to {output}")
        
        # Also save as PDF for LaTeX
        pdf_output = os.path.splitext(output)[0] + '.pdf'
        plt.savefig(pdf_output, bbox_inches='tight')
        print(f"Saved PDF for LaTeX to {pdf_output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot 2x2 heatmaps of pairwise similarities for a given experiment and task.")
    ap.add_argument("experiment", help="Experiment folder (e.g., T1-claude, X1-claude, T2-gemini, ...)")
    ap.add_argument("task_id", help="Task ID (e.g., 3)")
    ap.add_argument("--output", "-o", default=None, help="Optional path to save the figure (PNG/PDF)")
    args = ap.parse_args()

    exp_dir = args.experiment
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] Experiment folder not found: {exp_dir}")
        sys.exit(2)

    # Default output path: figures/task{ID}_{EXPERIMENT_NAME}.png
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exp_name = os.path.basename(exp_dir)
        args.output = os.path.join(script_dir, f"task{args.task_id}_{exp_name}.png")

    plot_heatmaps(exp_dir, str(args.task_id), args.output)


if __name__ == "__main__":
    main()
