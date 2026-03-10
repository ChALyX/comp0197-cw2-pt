"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

FIGURES_DIR = 'results/figures'


def _ensure_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_sensor_degradation(train_df, sensors=None, engine_ids=None):
    """Plot sensor degradation trends for selected engines.

    Args:
        train_df: Training DataFrame with sensor data and cycle column.
        sensors: List of sensor column names to plot (default: 4 key sensors).
        engine_ids: List of engine unit IDs to plot (default: first 4).
    """
    _ensure_dir()
    if sensors is None:
        sensors = ['s2', 's3', 's4', 's11']
    if engine_ids is None:
        engine_ids = [1, 2, 3, 4]

    fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 3 * len(sensors)),
                             sharex=False)
    if len(sensors) == 1:
        axes = [axes]

    for i, sensor in enumerate(sensors):
        for eid in engine_ids:
            unit_data = train_df[train_df['unit_id'] == eid]
            axes[i].plot(unit_data['cycle'], unit_data[sensor],
                         label=f'Engine {eid}', alpha=0.7)
        axes[i].set_ylabel(sensor)
        axes[i].legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Cycle')
    fig.suptitle('Sensor Degradation Trends', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sensor_degradation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: sensor_degradation.png")


def plot_rul_scatter(y_true, y_pred, title='RUL Prediction vs Ground Truth',
                     filename='rul_scatter.png'):
    """Scatter plot of predicted vs true RUL with diagonal reference line.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        title: Plot title.
        filename: Output filename.
    """
    _ensure_dir()
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Ideal')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_predictions_with_uncertainty(y_true, mu, total_std, n_samples=None,
                                     filename='predictions_uncertainty.png'):
    """Plot predictions with uncertainty bands (mu +/- 2*sigma).

    Args:
        y_true: True RUL values.
        mu: Predicted means.
        total_std: Total standard deviation.
        n_samples: Number of samples to show (None = all).
        filename: Output filename.
    """
    _ensure_dir()
    # Sort by true RUL for clearer visualization
    sort_idx = np.argsort(y_true)
    y_true_s = y_true[sort_idx]
    mu_s = mu[sort_idx]
    std_s = total_std[sort_idx]

    if n_samples is not None:
        step = max(1, len(y_true_s) // n_samples)
        y_true_s = y_true_s[::step]
        mu_s = mu_s[::step]
        std_s = std_s[::step]

    x = np.arange(len(y_true_s))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_true_s, 'k.', markersize=4, label='True RUL', alpha=0.7)
    ax.plot(x, mu_s, 'b-', linewidth=1, label='Predicted Mean')
    ax.fill_between(x, mu_s - 2 * std_s, mu_s + 2 * std_s,
                    alpha=0.3, color='blue', label='95% CI (±2σ)')
    ax.set_xlabel('Sample Index (sorted by true RUL)')
    ax.set_ylabel('RUL')
    ax.set_title('Predictions with Uncertainty')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_uncertainty_decomposition(y_true, mu, aleatoric_std, epistemic_std,
                                   filename='uncertainty_decomposition.png'):
    """Stacked area plot showing aleatoric and epistemic uncertainty components.

    Args:
        y_true: True RUL values.
        mu: Predicted means.
        aleatoric_std: Aleatoric standard deviation.
        epistemic_std: Epistemic standard deviation.
        filename: Output filename.
    """
    _ensure_dir()
    sort_idx = np.argsort(y_true)
    mu_s = mu[sort_idx]
    al_s = aleatoric_std[sort_idx]
    ep_s = epistemic_std[sort_idx]
    y_true_s = y_true[sort_idx]

    x = np.arange(len(mu_s))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y_true_s, 'k.', markersize=4, label='True RUL', alpha=0.5)
    ax.plot(x, mu_s, 'b-', linewidth=1, label='Predicted Mean')
    # Aleatoric band
    ax.fill_between(x, mu_s - al_s, mu_s + al_s,
                    alpha=0.4, color='orange', label='Aleatoric')
    # Epistemic band (on top of aleatoric)
    total_std = np.sqrt(al_s**2 + ep_s**2)
    ax.fill_between(x, mu_s - total_std, mu_s - al_s,
                    alpha=0.4, color='green', label='Epistemic')
    ax.fill_between(x, mu_s + al_s, mu_s + total_std,
                    alpha=0.4, color='green')
    ax.set_xlabel('Sample Index (sorted by true RUL)')
    ax.set_ylabel('RUL')
    ax.set_title('Uncertainty Decomposition')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_calibration(expected, actual, filename='calibration.png'):
    """Calibration plot: actual coverage vs expected coverage.

    Args:
        expected: Array of expected coverage probabilities.
        actual: Array of actual coverage probabilities.
        filename: Output filename.
    """
    _ensure_dir()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect Calibration')
    ax.plot(expected, actual, 'bo-', linewidth=2, markersize=6, label='Model')
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Actual Coverage')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_training_curves(train_losses, val_losses, filename='training_curves.png',
                         title='Training Curves'):
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        filename: Output filename.
        title: Plot title.
    """
    _ensure_dir()
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_sparsification(y_true, y_pred, total_std, filename='sparsification.png'):
    """Sparsification plot: RMSE as high-uncertainty samples are removed.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values.
        total_std: Total uncertainty (standard deviation).
        filename: Output filename.
    """
    _ensure_dir()
    # Sort by uncertainty (descending)
    sort_idx = np.argsort(-total_std)
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    n = len(y_true)
    fractions = np.linspace(0, 0.9, 20)
    rmses = []
    for frac in fractions:
        n_remove = int(frac * n)
        remaining_true = y_true_sorted[n_remove:]
        remaining_pred = y_pred_sorted[n_remove:]
        if len(remaining_true) > 0:
            rmse = np.sqrt(np.mean((remaining_true - remaining_pred) ** 2))
            rmses.append(rmse)
        else:
            rmses.append(0.0)

    # Oracle (sort by actual error)
    errors = np.abs(y_true - y_pred)
    oracle_idx = np.argsort(-errors)
    y_true_oracle = y_true[oracle_idx]
    y_pred_oracle = y_pred[oracle_idx]
    oracle_rmses = []
    for frac in fractions:
        n_remove = int(frac * n)
        remaining_true = y_true_oracle[n_remove:]
        remaining_pred = y_pred_oracle[n_remove:]
        if len(remaining_true) > 0:
            rmse = np.sqrt(np.mean((remaining_true - remaining_pred) ** 2))
            oracle_rmses.append(rmse)
        else:
            oracle_rmses.append(0.0)

    fig, ax = plt.subplots()
    ax.plot(fractions * 100, rmses, 'b-o', markersize=4, label='Model Uncertainty')
    ax.plot(fractions * 100, oracle_rmses, 'r--o', markersize=4, label='Oracle')
    ax.set_xlabel('Fraction of Data Removed (%)')
    ax.set_ylabel('RMSE')
    ax.set_title('Sparsification Plot')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_ablation_results(results_dict, filename='ablation.png'):
    """Bar chart comparing metrics across ablation experiments.

    Args:
        results_dict: Dict mapping experiment name to metrics dict.
            e.g. {'Prob LSTM': {'RMSE': 12.5, 'MAE': 9.8}, ...}
        filename: Output filename.
    """
    _ensure_dir()
    experiments = list(results_dict.keys())
    metrics = list(results_dict[experiments[0]].keys())
    n_metrics = len(metrics)
    n_experiments = len(experiments)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_experiments)
    colors = plt.cm.Set2(np.linspace(0, 1, n_experiments))

    for i, metric in enumerate(metrics):
        values = [results_dict[exp][metric] for exp in experiments]
        axes[i].bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(experiments, rotation=30, ha='right', fontsize=9)
        axes[i].set_title(metric)
        axes[i].set_ylabel(metric)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
