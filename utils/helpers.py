"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import torch
import numpy as np
import random
import copy


def set_seed(seed=42):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return CPU device (CPU-only environment)."""
    return torch.device('cpu')


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_model_state = None

    def step(self, val_loss, model=None):
        """Check if validation loss improved.

        Args:
            val_loss: Current validation loss.
            model: Model to save state dict from if improved.

        Returns:
            True if improved (should save model), False otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = copy.deepcopy(model.state_dict())
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
