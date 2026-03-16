"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data_loader import download_cmapss, load_fd001, add_rul_labels
from utils.helpers import set_seed

FIGURES_DIR = 'results/figures'

matplotlib.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def get_feature_cols(df):
    """Return all sensor and setting column names."""
    return [c for c in df.columns if c.startswith('s') or c.startswith('setting_')]


def variance_analysis(train_df, feature_cols, std_threshold=0.01):
    """Compute and plot standard deviation of all features.

    Args:
        train_df: Training DataFrame.
        feature_cols: List of feature column names.
        std_threshold: Threshold line to draw.

    Returns:
        stds: Series of standard deviations.
    """
    stds = train_df[feature_cols].std().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#d9534f' if v < std_threshold else '#5cb85c' for v in stds.values]
    ax.barh(range(len(stds)), stds.values, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(stds)))
    ax.set_yticklabels(stds.index, fontsize=9)
    ax.axvline(x=std_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold = {std_threshold}')
    ax.set_xlabel('Standard Deviation')
    ax.set_title('Feature Variance Analysis')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eda_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_variance.png")

    # Print summary
    removed = stds[stds < std_threshold].index.tolist()
    kept = stds[stds >= std_threshold].index.tolist()
    print(f"  Variance filter: {len(kept)} kept, {len(removed)} removed")
    print(f"  Removed (std < {std_threshold}): {removed}")
    return stds


def correlation_analysis(train_df, feature_cols):
    """Compute and plot Spearman correlation of all features with RUL.

    Args:
        train_df: Training DataFrame with RUL column.
        feature_cols: List of feature column names.

    Returns:
        correlations: Series of Spearman correlations with RUL.
    """
    correlations = train_df[feature_cols + ['RUL']].corr(method='spearman')['RUL'].drop('RUL')
    correlations = correlations.sort_values()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#5cb85c' if abs(v) > 0.1 else '#d9534f' for v in correlations.values]
    ax.barh(range(len(correlations)), correlations.values, color=colors,
            edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=9)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='|corr| = 0.1')
    ax.set_xlabel('Spearman Correlation with RUL')
    ax.set_title('Feature-RUL Correlation Analysis')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eda_correlation_rul.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_correlation_rul.png")

    # Print summary
    print("  Spearman correlations with RUL:")
    for feat in correlations.index:
        marker = '*' if abs(correlations[feat]) > 0.1 else ' '
        print(f"    {marker} {feat:>12s}: {correlations[feat]:+.4f}")
    return correlations


def sensor_comparison(train_df, informative, non_informative, engine_ids=None):
    """Plot side-by-side comparison of informative vs non-informative sensors.

    Args:
        train_df: Training DataFrame.
        informative: List of informative sensor names.
        non_informative: List of non-informative sensor names.
        engine_ids: List of engine IDs to plot.
    """
    if engine_ids is None:
        engine_ids = [1, 2, 3]

    n_info = len(informative)
    n_noninfo = len(non_informative)
    n_rows = max(n_info, n_noninfo)

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        # Left: informative sensors
        if i < n_info:
            sensor = informative[i]
            for eid in engine_ids:
                unit = train_df[train_df['unit_id'] == eid]
                axes[i, 0].plot(unit['cycle'], unit[sensor], alpha=0.7, label=f'Eng {eid}')
            axes[i, 0].set_ylabel(sensor)
            axes[i, 0].set_title(f'{sensor} (informative)' if i == 0 else '')
            axes[i, 0].legend(fontsize=7, loc='upper right')
        else:
            axes[i, 0].set_visible(False)

        # Right: non-informative sensors
        if i < n_noninfo:
            sensor = non_informative[i]
            for eid in engine_ids:
                unit = train_df[train_df['unit_id'] == eid]
                axes[i, 1].plot(unit['cycle'], unit[sensor], alpha=0.7, label=f'Eng {eid}')
            axes[i, 1].set_ylabel(sensor)
            axes[i, 1].set_title(f'{sensor} (non-informative)' if i == 0 else '')
            axes[i, 1].legend(fontsize=7, loc='upper right')
        else:
            axes[i, 1].set_visible(False)

    axes[0, 0].set_title('Informative Sensors', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Non-informative Sensors', fontsize=12, fontweight='bold')
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel('Cycle')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eda_sensor_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_sensor_comparison.png")


def feature_heatmap(train_df, selected_features):
    """Plot correlation heatmap among selected features.

    Args:
        train_df: Training DataFrame.
        selected_features: List of selected feature names.
    """
    corr_matrix = train_df[selected_features].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(selected_features)))
    ax.set_yticks(range(len(selected_features)))
    ax.set_xticklabels(selected_features, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(selected_features, fontsize=8)

    # Add correlation values
    for i in range(len(selected_features)):
        for j in range(len(selected_features)):
            val = corr_matrix.values[i, j]
            color = 'white' if abs(val) > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman Correlation')
    ax.set_title('Feature Correlation Matrix (Selected Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'eda_feature_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_feature_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis for C-MAPSS FD001')
    parser.add_argument('--data_dir', type=str, default='CMAPSSData', help='Data directory')
    parser.add_argument('--r_early', type=int, default=125, help='RUL clipping upper bound')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--std_threshold', type=float, default=0.01, help='Variance filter threshold')
    parser.add_argument('--corr_threshold', type=float, default=0.1, help='Correlation filter threshold')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    download_cmapss(args.data_dir)
    train_df, _, _ = load_fd001(args.data_dir)
    train_df = add_rul_labels(train_df, r_early=args.r_early)

    feature_cols = get_feature_cols(train_df)
    print(f"\nTotal candidate features: {len(feature_cols)}")
    print(f"  {feature_cols}")

    # === Analysis 1: Variance ===
    print(f"\n{'='*60}")
    print("Analysis 1: Feature Variance")
    print(f"{'='*60}")
    stds = variance_analysis(train_df, feature_cols, args.std_threshold)

    # === Analysis 2: Correlation with RUL ===
    print(f"\n{'='*60}")
    print("Analysis 2: Spearman Correlation with RUL")
    print(f"{'='*60}")
    correlations = correlation_analysis(train_df, feature_cols)

    # === Determine selected features ===
    variance_passed = stds[stds >= args.std_threshold].index.tolist()
    corr_passed = correlations[correlations.abs() > args.corr_threshold].index.tolist()
    selected = [f for f in variance_passed if f in corr_passed]

    print(f"\n{'='*60}")
    print("Feature Selection Summary")
    print(f"{'='*60}")
    print(f"  Stage 1 (variance > {args.std_threshold}): {len(variance_passed)} features")
    print(f"  Stage 2 (|corr| > {args.corr_threshold}): {len(selected)} features")
    print(f"  Final selected: {selected}")

    # Identify informative vs non-informative for comparison plot
    non_informative = [f for f in feature_cols if f not in selected]

    # === Analysis 3: Sensor Comparison ===
    print(f"\n{'='*60}")
    print("Analysis 3: Informative vs Non-informative Sensor Trends")
    print(f"{'='*60}")
    info_sample = selected[:4]
    noninfo_sample = non_informative[:4]
    sensor_comparison(train_df, info_sample, noninfo_sample)

    # === Analysis 4: Feature Correlation Heatmap ===
    print(f"\n{'='*60}")
    print("Analysis 4: Inter-feature Correlation Matrix")
    print(f"{'='*60}")
    feature_heatmap(train_df, selected)

    print(f"\n{'='*60}")
    print("EDA complete! All figures saved to results/figures/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
