"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import argparse
import json
import os
import numpy as np
import torch

from utils.helpers import set_seed, get_device
from utils.data_loader import (download_cmapss, load_fd001, add_rul_labels,
                                select_features, create_test_sequences,
                                CMAPSSDataset)
from utils.metrics import (calc_rmse, calc_mae, calc_r2, calc_nasa_score,
                            calc_picp, calc_mpiw, calc_nll, calc_calibration,
                            calc_all_metrics)
from utils.visualization import (plot_rul_scatter, plot_predictions_with_uncertainty,
                                  plot_uncertainty_decomposition, plot_calibration,
                                  plot_sparsification, plot_ablation_results,
                                  plot_sensor_degradation)
from models.probabilistic_lstm import ProbabilisticLSTM
from models.deterministic_lstm import DeterministicLSTM
from torch.utils.data import DataLoader


def mc_predict(model, loader, device, T=50):
    """MC Dropout inference: run T forward passes with dropout active.

    Args:
        model: Probabilistic LSTM model.
        loader: DataLoader for test data.
        device: Torch device.
        T: Number of MC samples.

    Returns:
        pred_mean: Mean predictions, shape (N,).
        aleatoric_var: Aleatoric variance (mean of sigma^2), shape (N,).
        epistemic_var: Epistemic variance (variance of mu), shape (N,).
        total_var: Total variance, shape (N,).
    """
    model.train()  # Keep dropout active
    all_mus = []
    all_sigmas = []

    with torch.no_grad():
        for t in range(T):
            batch_mus = []
            batch_sigmas = []
            for X_batch, _ in loader:
                X_batch = X_batch.to(device)
                mu, sigma = model(X_batch)
                batch_mus.append(mu.cpu().numpy())
                batch_sigmas.append(sigma.cpu().numpy())
            all_mus.append(np.concatenate(batch_mus, axis=0))
            all_sigmas.append(np.concatenate(batch_sigmas, axis=0))

    # Stack: (T, N, 1)
    all_mus = np.stack(all_mus, axis=0)
    all_sigmas = np.stack(all_sigmas, axis=0)

    # Squeeze last dimension: (T, N)
    all_mus = all_mus.squeeze(-1)
    all_sigmas = all_sigmas.squeeze(-1)

    pred_mean = all_mus.mean(axis=0)           # (N,)
    aleatoric_var = (all_sigmas ** 2).mean(axis=0)  # (N,)
    epistemic_var = all_mus.var(axis=0)         # (N,)
    total_var = aleatoric_var + epistemic_var    # (N,)

    return pred_mean, aleatoric_var, epistemic_var, total_var


def deterministic_predict(model, loader, device):
    """Run deterministic model inference.

    Args:
        model: Deterministic LSTM model.
        loader: DataLoader for test data.
        device: Torch device.

    Returns:
        predictions: numpy array of shape (N,).
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions, axis=0).squeeze(-1)


def print_metrics_table(metrics_dict, title='Metrics'):
    """Print metrics in a formatted table.

    Args:
        metrics_dict: Dictionary of metric name -> value.
        title: Table title.
    """
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for name, value in metrics_dict.items():
        print(f"  {name:<20s}: {value:>10.4f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RUL prediction models on C-MAPSS FD001')
    parser.add_argument('--mc_samples', type=int, default=100, help='MC Dropout samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='CMAPSSData', help='Data directory')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs('results/figures', exist_ok=True)

    # === Load checkpoints ===
    prob_ckpt = torch.load('saved_models/prob_lstm_best.pth', weights_only=False)
    det_ckpt = torch.load('saved_models/det_lstm_best.pth', weights_only=False)

    hp = prob_ckpt['hyperparams']
    scaler = prob_ckpt['scaler']
    selected_features = prob_ckpt['selected_features']

    # === Prepare test data ===
    download_cmapss(args.data_dir)
    train_df, test_df, rul_df = load_fd001(args.data_dir)
    train_df = add_rul_labels(train_df, r_early=hp['r_early'])

    # Scale test features using saved scaler
    test_df[selected_features] = scaler.transform(test_df[selected_features])

    test_seqs, test_labels = create_test_sequences(
        test_df, rul_df, selected_features, seq_len=hp['seq_len']
    )
    test_dataset = CMAPSSDataset(test_seqs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # === Load models ===
    prob_model = ProbabilisticLSTM(
        input_dim=hp['input_dim'],
        hidden_dim=hp['hidden_dim'],
        num_layers=hp['num_layers'],
        dropout=hp['dropout'],
    ).to(device)
    prob_model.load_state_dict(prob_ckpt['model_state_dict'])

    det_model = DeterministicLSTM(
        input_dim=hp['input_dim'],
        hidden_dim=hp['hidden_dim'],
        num_layers=hp['num_layers'],
        dropout=hp['dropout'],
    ).to(device)
    det_model.load_state_dict(det_ckpt['model_state_dict'])

    # === MC Dropout inference (Probabilistic model) ===
    print(f"Running MC Dropout inference (T={args.mc_samples})...")
    pred_mean, aleatoric_var, epistemic_var, total_var = mc_predict(
        prob_model, test_loader, device, T=args.mc_samples
    )

    aleatoric_std = np.sqrt(aleatoric_var)
    epistemic_std = np.sqrt(epistemic_var)
    total_std = np.sqrt(total_var)

    # === Deterministic model inference ===
    print("Running deterministic model inference...")
    det_preds = deterministic_predict(det_model, test_loader, device)

    y_true = test_labels

    # === Compute metrics ===
    # Probabilistic model metrics
    prob_metrics = calc_all_metrics(
        y_true, pred_mean, mu=pred_mean, sigma=aleatoric_std, total_std=total_std
    )
    print_metrics_table(prob_metrics, 'Probabilistic LSTM (MC Dropout)')

    # Deterministic model metrics
    det_metrics = {
        'RMSE': float(calc_rmse(y_true, det_preds)),
        'MAE': float(calc_mae(y_true, det_preds)),
        'R2': float(calc_r2(y_true, det_preds)),
        'NASA_Score': float(calc_nasa_score(det_preds, y_true)),
    }
    print_metrics_table(det_metrics, 'Deterministic LSTM (Baseline)')

    # === Generate all plots ===
    print("\nGenerating plots...")

    # 1. Sensor degradation
    plot_sensor_degradation(train_df)

    # 2. RUL scatter plots
    plot_rul_scatter(y_true, pred_mean,
                     title='Probabilistic LSTM: Predicted vs True RUL',
                     filename='rul_scatter_prob.png')
    plot_rul_scatter(y_true, det_preds,
                     title='Deterministic LSTM: Predicted vs True RUL',
                     filename='rul_scatter_det.png')

    # 3. Predictions with uncertainty
    plot_predictions_with_uncertainty(y_true, pred_mean, total_std)

    # 4. Uncertainty decomposition
    plot_uncertainty_decomposition(y_true, pred_mean, aleatoric_std, epistemic_std)

    # 5. Calibration plot
    expected, actual = calc_calibration(y_true, pred_mean, total_std)
    plot_calibration(expected, actual)

    # 6. Sparsification plot
    plot_sparsification(y_true, pred_mean, total_std)

    # 7. Ablation: Deterministic vs Probabilistic
    ablation_results = {
        'Det. LSTM': {'RMSE': det_metrics['RMSE'], 'MAE': det_metrics['MAE']},
        'Prob. LSTM\n(MC Dropout)': {'RMSE': prob_metrics['RMSE'], 'MAE': prob_metrics['MAE']},
    }
    plot_ablation_results(ablation_results, filename='ablation_det_vs_prob.png')

    # === Save metrics to JSON ===
    all_metrics = {
        'probabilistic_lstm': prob_metrics,
        'deterministic_lstm': det_metrics,
        'uncertainty_stats': {
            'mean_aleatoric_std': float(np.mean(aleatoric_std)),
            'mean_epistemic_std': float(np.mean(epistemic_std)),
            'mean_total_std': float(np.mean(total_std)),
        }
    }

    metrics_path = 'results/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Figures saved to results/figures/")
    print(f"Metrics saved to {metrics_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
