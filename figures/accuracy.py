#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_similarity_averages(outputs_dir: str):
    """Load similarity data and compute averages for each model."""
    if not os.path.isdir(outputs_dir):
        print(f"[ERROR] Outputs directory not found: {outputs_dir}")
        sys.exit(2)

    similarity_files = ["similarity_lcs.csv", "similarity_sbert.csv", "similarity_tfidf.csv"]
    methods = ["LCS", "SBERT", "TF-IDF"]
    
    # Dictionary to store average similarities
    # Structure: {model_name: {method: avg_similarity}}
    model_averages = {}
    
    for sim_file, method in zip(similarity_files, methods):
        fpath = os.path.join(outputs_dir, sim_file)
        if not os.path.exists(fpath):
            print(f"[WARN] File not found: {fpath}")
            continue
        
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] Error reading {sim_file}: {e}")
            continue
        
        # Get model columns (exclude ID column)
        model_cols = [col for col in df.columns if col.lower() != "id"]
        
        for model in model_cols:
            if model not in model_averages:
                model_averages[model] = {}
            
            # Compute average similarity for this model
            avg_sim = df[model].mean()
            model_averages[model][method] = avg_sim
    
    return model_averages, methods


def plot_grouped_bar(model_averages, methods, output_dir):
    """Create a grouped bar plot of average similarities."""
    # Use the same model order as files are loaded (not sorted)
    models = list(model_averages.keys())
    
    # Prepare data for plotting
    data = {method: [] for method in methods}
    for model in models:
        for method in methods:
            data[method].append(model_averages[model].get(method, 0))
    
    # Set up the plot with same size as latency.py
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(models) + 2))
    
    y = np.arange(len(models))
    height = 0.25
    
    # Define colors for each method (same as latency.py)
    colors = ['#AEC6CF', '#B4E7CE', '#FFD9B3', '#FFB3BA']
    
    # Plot horizontal bars for each method
    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = height * (i - 1)
        ax.barh(y + offset, data[method], height, label=method, color=color, alpha=0.7)
    
    # Customize plot
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Average Similarity', fontsize=12)
    ax.set_title('Average Similarity by Model and Method', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the figure in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "accuracy-x1-claude.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot to {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(script_dir, "accuracy-x1-claude.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")
    
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot grouped bar chart of average similarities across methods and models."
    )
    ap.add_argument(
        "--outputs_dir", 
        default="X1-claude", 
        help="Directory containing similarity CSV files (default: X1-claude)"
    )
    args = ap.parse_args()

    model_averages, methods = load_similarity_averages(args.outputs_dir)

    if not model_averages:
        print("[ERROR] No valid similarity data loaded. Exiting.")
        sys.exit(2)

    # Print summary statistics
    print("\nAverage Similarity by Model and Method:")
    for model in sorted(model_averages.keys()):
        print(f"\n  {model}:")
        for method in methods:
            avg = model_averages[model].get(method, 0)
            print(f"    {method:10s}: {avg:.4f}")

    plot_grouped_bar(model_averages, methods, args.outputs_dir)


if __name__ == "__main__":
    main()
