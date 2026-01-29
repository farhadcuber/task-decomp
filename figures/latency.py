#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_latencies(outputs_dir: str):
    """Load latency data from llm_*.csv files."""
    if not os.path.isdir(outputs_dir):
        print(f"[ERROR] Outputs directory not found: {outputs_dir}")
        sys.exit(2)

    files = sorted([f for f in os.listdir(outputs_dir) if f.startswith("llm_") and f.endswith(".csv")])
    if not files:
        print(f"[ERROR] No llm_*.csv files found in {outputs_dir}")
        sys.exit(2)

    model_names = []
    latency_data = []

    for fname in files:
        fpath = os.path.join(outputs_dir, fname)
        try:
            df = pd.read_csv(fpath, sep=",")
        except Exception as e:
            print(f"[WARN] Skipping {fname}: read error: {e}")
            continue

        if "Latency" not in df.columns:
            print(f"[WARN] Skipping {fname}: missing 'Latency' column")
            continue

        model_name = fname[len("llm_"):-len(".csv")]
        if "model" in df.columns:
            nn = df["model"].dropna().astype(str)
            if not nn.empty and nn.iloc[0].strip():
                model_name = nn.iloc[0].strip()

        latencies = df["Latency"].dropna().tolist()
        
        model_names.append(model_name)
        latency_data.append(latencies)

    return model_names, latency_data


def plot_violin(model_names, latency_data, output_path=None):
    """Create a violin plot of latencies for each model."""
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(model_names) + 2))  # More compact y axis
    
    # Define pastel colors
    colors = ['#AEC6CF', '#B4E7CE', '#FFD9B3', '#FFB3BA']
    
    # Create horizontal violin plot with smaller width
    parts = ax.violinplot(latency_data, positions=range(len(model_names)),
                          showmeans=False, showmedians=False, vert=False, widths=0.5)
    
    # Color each violin differently
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('none')
    
    # Show median and mean as ticks in the same color as the violin
    import numpy as np
    for i, latencies in enumerate(latency_data):
        if latencies:
            median = np.median(latencies)
            mean = np.mean(latencies)
            # Median: vertical line (tick)
            ax.plot([median, median], [i-0.18, i+0.18], color=colors[i % len(colors)], lw=4, solid_capstyle='round', label='Median' if i == 0 else None)
            # Mean: vertical line (tick, dashed)
            ax.plot([mean, mean], [i-0.18, i+0.18], color=colors[i % len(colors)], lw=2, ls='--', solid_capstyle='round', label='Mean' if i == 0 else None)
    
    # Add legend for mean/median (custom)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=colors[0], lw=4, solid_capstyle='round', label='Median'),
               Line2D([0], [0], color=colors[0], lw=2, ls='--', solid_capstyle='round', label='Mean')]
    ax.legend(handles=handles, loc='lower right', fontsize=10)
    
    # Customize appearance
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Latency Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_xlim(left=0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved violin plot to {output_path}\n")
        
        # Also save as PDF for LaTeX
        pdf_output = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_output, bbox_inches='tight')
        print(f"Saved PDF for LaTeX to {pdf_output}\n")
    else:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot violin plot of latencies for each model.")
    ap.add_argument("--outputs_dir", default="outputs", help="Directory containing llm_*.csv files")
    ap.add_argument("--output", default=None, help="Output image path (default: show plot)")
    args = ap.parse_args()

    model_names, latency_data = load_latencies(args.outputs_dir)

    if not model_names:
        print("[ERROR] No valid model data loaded. Exiting.")
        sys.exit(2)

    # Print summary statistics
    print("\nLatency Statistics by Model:")
    for name, latencies in zip(model_names, latency_data):
        if latencies:
            mean_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            print(f"  {name:30s}  Mean: {mean_lat:7.2f}ms  Min: {min_lat:7.2f}ms  Max: {max_lat:7.2f}ms")

    plot_violin(model_names, latency_data, args.output)


if __name__ == "__main__":
    main()
