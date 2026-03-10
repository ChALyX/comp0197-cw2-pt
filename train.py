"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import argparse
import os
import time
import torch
import torch.nn as nn

from utils.helpers import set_seed, get_device, EarlyStopping
from utils.data_loader import prepare_data
from utils.visualization import plot_training_curves
from models.probabilistic_lstm import ProbabilisticLSTM
from models.deterministic_lstm import DeterministicLSTM


def train_one_epoch(model, loader, optimizer, loss_fn, device, probabilistic=True):
    """Train model for one epoch.

    Args:
        model: PyTorch model.
        loader: Training DataLoader.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Torch device.
        probabilistic: If True, model outputs (mu, sigma); otherwise single value.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        if probabilistic:
            mu, sigma = model(X_batch)
            loss = loss_fn(mu, y_batch, sigma ** 2)
        else:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, loss_fn, device, probabilistic=True):
    """Evaluate model on validation set.

    Args:
        model: PyTorch model.
        loader: Validation DataLoader.
        loss_fn: Loss function.
        device: Torch device.
        probabilistic: If True, model outputs (mu, sigma).

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if probabilistic:
                mu, sigma = model(X_batch)
                loss = loss_fn(mu, y_batch, sigma ** 2)
            else:
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model(model, train_loader, val_loader, device, args,
                probabilistic=True, model_name='model'):
    """Full training loop with early stopping and LR scheduling.

    Args:
        model: PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Torch device.
        args: Training hyperparameters (namespace).
        probabilistic: If True, use GaussianNLLLoss; otherwise MSELoss.
        model_name: Name for logging and saving.

    Returns:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    if probabilistic:
        loss_fn = nn.GaussianNLLLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=args.patience)

    train_losses = []
    val_losses = []

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, probabilistic
        )
        val_loss = validate(model, val_loader, loss_fn, device, probabilistic)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        improved = early_stopping.step(val_loss, model)
        marker = ' *' if improved else ''

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}{marker}")

        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Best val loss: {early_stopping.best_loss:.4f}")

    # Restore best model weights
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train RUL prediction models on C-MAPSS FD001')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--r_early', type=int, default=125, help='RUL clipping upper bound')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='CMAPSSData', help='Data directory')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    print("Preparing data...")
    data = prepare_data(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        r_early=args.r_early,
        batch_size=args.batch_size,
    )

    input_dim = data['input_dim']
    train_loader = data['train_loader']
    val_loader = data['val_loader']

    # === Train Probabilistic LSTM ===
    prob_model = ProbabilisticLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    prob_train_losses, prob_val_losses = train_model(
        prob_model, train_loader, val_loader, device, args,
        probabilistic=True, model_name='Probabilistic LSTM'
    )

    # Save probabilistic model checkpoint
    prob_checkpoint = {
        'model_state_dict': prob_model.state_dict(),
        'scaler': data['scaler'],
        'selected_features': data['selected_features'],
        'hyperparams': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'seq_len': args.seq_len,
            'r_early': args.r_early,
        }
    }
    torch.save(prob_checkpoint, 'saved_models/prob_lstm_best.pth')
    print("Saved: saved_models/prob_lstm_best.pth")

    # Save training curves
    plot_training_curves(
        prob_train_losses, prob_val_losses,
        filename='training_curves_prob.png',
        title='Probabilistic LSTM Training Curves'
    )

    # === Train Deterministic LSTM (Baseline) ===
    det_model = DeterministicLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    det_train_losses, det_val_losses = train_model(
        det_model, train_loader, val_loader, device, args,
        probabilistic=False, model_name='Deterministic LSTM (Baseline)'
    )

    # Save deterministic model checkpoint
    det_checkpoint = {
        'model_state_dict': det_model.state_dict(),
        'scaler': data['scaler'],
        'selected_features': data['selected_features'],
        'hyperparams': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'seq_len': args.seq_len,
            'r_early': args.r_early,
        }
    }
    torch.save(det_checkpoint, 'saved_models/det_lstm_best.pth')
    print("Saved: saved_models/det_lstm_best.pth")

    plot_training_curves(
        det_train_losses, det_val_losses,
        filename='training_curves_det.png',
        title='Deterministic LSTM Training Curves'
    )

    print("\n" + "=" * 60)
    print("Training complete! Models saved to saved_models/")
    print("Training curves saved to results/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
