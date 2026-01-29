#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ["lcs", "majority_voting", "sbert", "tfidf"]

# Map from short names to full names
MODEL_NAME_MAP = {
    "gpt": "gpt-4o-mini",
    "claude": "claude-sonnet-4.5",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-large",
}

METHOD_DISPLAY_NAMES = {
    "lcs": "LCS",
    "majority_voting": "Majority Voting",
    "sbert": "SBERT",
    "tfidf": "TF-IDF",
}


def load_all_reputations(experiment_dir: str, method: str):
    """Load reputation data for all models in a method."""
    csv_path = os.path.join(experiment_dir, f"pairwise_{method}", "reputations.csv")
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if "ID" not in df.columns:
        raise ValueError(f"'ID' column not found in {csv_path}")
    
    task_ids = df["ID"].tolist()
    
    # Get all model columns (exclude ID)
    model_columns = [col for col in df.columns if col != "ID"]
    
    # Build a dict: full_model_name -> reputations
    reputations_dict = {}
    for col in model_columns:
        reputations_dict[col] = pd.to_numeric(df[col], errors="coerce").tolist()
    
    return task_ids, reputations_dict


def shorten_model_name(name: str) -> str:
    """Shorten model names for display."""
    name_map = {
        "gpt-4o-mini": "gpt",
        "claude-sonnet-4.5": "claude",
        "gemini-2.5-flash": "gemini",
        "mistral-large": "mistral",
    }
    return name_map.get(name, name)


def plot_reputations(experiment_dir: str, output: str | None, include_majority_voting: bool = False):
    """Plot reputation evolution across tasks for all models and methods in 1x4 or 1x3 grid."""
    # Filter methods based on flag
    methods_to_plot = [m for m in METHODS if m != "majority_voting" or include_majority_voting]
    
    num_plots = len(methods_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4.5))
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Define pastel colors matching latency.py - one per model
    colors = ['#AEC6CF', '#B4E7CE', '#FFD9B3', '#FFB3BA']
    
    # First pass: collect all reputation values to determine global y-range
    all_reputations = []
    for method in methods_to_plot:
        try:
            task_ids, reputations_dict = load_all_reputations(experiment_dir, method)
            for reputations in reputations_dict.values():
                all_reputations.extend([r for r in reputations if pd.notna(r)])
        except Exception:
            pass
    
    # Calculate global y-limits with some padding
    if all_reputations:
        y_min = min(all_reputations)
        y_max = max(all_reputations)
        y_padding = (y_max - y_min) * 0.05
        y_min = max(0, y_min - y_padding)  # Don't go below 0
        y_max = y_max + y_padding
    else:
        y_min, y_max = 0, 5  # Default range
    
    for idx, method in enumerate(methods_to_plot):
        ax = axes[idx]
        try:
            task_ids, reputations_dict = load_all_reputations(experiment_dir, method)
            
            # Sort models for consistent ordering
            model_names = sorted(reputations_dict.keys())
            
            # Plot each model with a different color
            for model_idx, model_full in enumerate(model_names):
                reputations = reputations_dict[model_full]
                model_short = shorten_model_name(model_full)
                
                ax.plot(range(len(task_ids)), reputations, 
                       color=colors[model_idx % len(colors)], 
                       marker='o', 
                       markersize=3,
                       linewidth=2,
                       label=model_short,
                       alpha=0.8)
            
            # Customize subplot
            ax.set_title(METHOD_DISPLAY_NAMES.get(method, method.upper()))
            ax.set_xlabel('Task Number', fontsize=10)
            
            # Only show y-label on leftmost subplot
            if idx == 0:
                ax.set_ylabel('Reputation', fontsize=10)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            
            # Set consistent y-limits across all subplots
            ax.set_ylim(y_min, y_max)
            
            # Set x-axis to show task indices
            if task_ids and len(task_ids) > 0:
                step = max(1, len(task_ids) // 5)
                tick_positions = range(0, len(task_ids), step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([task_ids[i] for i in tick_positions], fontsize=8)
            
        except Exception as e:
            ax.axis('off')
            ax.set_title(f"{METHOD_DISPLAY_NAMES.get(method, method.upper())} (error)")
            ax.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=9, wrap=True)
            continue
    
    fig.suptitle('Reputation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved reputation plot to {output}")
        
        # Also save as PDF for LaTeX
        pdf_output = os.path.splitext(output)[0] + '.pdf'
        plt.savefig(pdf_output, bbox_inches='tight')
        print(f"Saved PDF for LaTeX to {pdf_output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot reputation evolution for all models across different similarity methods in a 1x3 or 1x4 grid.")
    ap.add_argument("experiment", help="Experiment folder (e.g., T1-claude, X1-claude, T2-gemini, ...)")
    ap.add_argument("--output", "-o", default=None, help="Optional path to save the figure (PNG/PDF)")
    ap.add_argument("--include-majority-voting", action="store_true", help="Include Majority Voting method in the plot")
    args = ap.parse_args()

    exp_dir = args.experiment
    if not os.path.isdir(exp_dir):
        print(f"[ERROR] Experiment folder not found: {exp_dir}")
        sys.exit(2)

    # Default output path: figures/reputation-{experiment}.png
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exp_name = os.path.basename(exp_dir)
        args.output = os.path.join(script_dir, f"reputation-{exp_name}.png")

    plot_reputations(exp_dir, args.output, args.include_majority_voting)


if __name__ == "__main__":
    main()
