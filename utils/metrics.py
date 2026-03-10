"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm


def calc_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calc_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calc_r2(y_true, y_pred):
    """Calculate R-squared score."""
    return r2_score(y_true, y_pred)


def calc_nasa_score(y_pred, y_true):
    """NASA asymmetric scoring function.

    Late predictions (overestimating RUL) are penalized more heavily
    than early predictions (underestimating RUL).

    Args:
        y_pred: Predicted RUL values.
        y_true: True RUL values.

    Returns:
        NASA score (lower is better).
    """
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1))


def calc_picp(y_true, mu, total_std, z=1.96):
    """Prediction Interval Coverage Probability (95%).

    Args:
        y_true: True values.
        mu: Predicted means.
        total_std: Total standard deviation (sqrt of total uncertainty).
        z: Z-score for confidence level (1.96 for 95%).

    Returns:
        Coverage probability (fraction of true values within prediction interval).
    """
    lower = mu - z * total_std
    upper = mu + z * total_std
    return np.mean((y_true >= lower) & (y_true <= upper))


def calc_mpiw(total_std, z=1.96):
    """Mean Prediction Interval Width (95%).

    Args:
        total_std: Total standard deviation.
        z: Z-score for confidence level.

    Returns:
        Average width of the prediction interval.
    """
    return np.mean(2 * z * total_std)


def calc_nll(y_true, mu, sigma):
    """Negative Log-Likelihood under Gaussian assumption.

    Args:
        y_true: True values.
        mu: Predicted means.
        sigma: Predicted standard deviations.

    Returns:
        Mean NLL value.
    """
    return np.mean(0.5 * (np.log(sigma**2 + 1e-8) + (y_true - mu)**2 / (sigma**2 + 1e-8)))


def calc_calibration(y_true, mu, total_std, num_bins=10):
    """Compute calibration curve: expected vs actual coverage at various confidence levels.

    Args:
        y_true: True values.
        mu: Predicted means.
        total_std: Total standard deviation.
        num_bins: Number of confidence levels to evaluate.

    Returns:
        expected: Array of expected coverage probabilities.
        actual: Array of actual coverage probabilities.
    """
    expected = np.linspace(0.05, 0.95, num_bins)
    actual = []
    for p in expected:
        z = norm.ppf((1 + p) / 2)
        lower = mu - z * total_std
        upper = mu + z * total_std
        actual.append(np.mean((y_true >= lower) & (y_true <= upper)))
    return expected, np.array(actual)


def calc_all_metrics(y_true, y_pred, mu=None, sigma=None, total_std=None):
    """Calculate all RUL prediction and uncertainty metrics.

    Args:
        y_true: True RUL values.
        y_pred: Predicted RUL values (mean predictions).
        mu: Predicted means (for uncertainty metrics).
        sigma: Predicted aleatoric std (for NLL).
        total_std: Total std (for PICP/MPIW).

    Returns:
        Dictionary of all computed metrics.
    """
    metrics = {
        'RMSE': float(calc_rmse(y_true, y_pred)),
        'MAE': float(calc_mae(y_true, y_pred)),
        'R2': float(calc_r2(y_true, y_pred)),
        'NASA_Score': float(calc_nasa_score(y_pred, y_true)),
    }
    if mu is not None and total_std is not None:
        metrics['PICP_95'] = float(calc_picp(y_true, mu, total_std))
        metrics['MPIW_95'] = float(calc_mpiw(total_std))
    if mu is not None and sigma is not None:
        metrics['NLL'] = float(calc_nll(y_true, mu, sigma))
    return metrics
