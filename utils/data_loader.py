"""
GenAI Usage Statement:
This code was developed with the assistance of Claude (Anthropic) as a coding assistant.
The AI tool was used for code structuring, debugging, and implementation guidance.
All outputs were manually reviewed and verified for technical correctness by the authors.
"""

import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# Column names for C-MAPSS data (26 columns, space-separated, no header)
COLUMNS = (['unit_id', 'cycle'] +
           [f'setting_{i}' for i in range(1, 4)] +
           [f's{i}' for i in range(1, 22)])


def download_cmapss(data_dir='CMAPSSData'):
    """Download and extract C-MAPSS dataset using urllib.

    Tries two URLs, then falls back to a local zip file.

    Args:
        data_dir: Directory name where extracted data will reside.
    """
    if os.path.exists(data_dir) and os.path.isfile(os.path.join(data_dir, 'train_FD001.txt')):
        print(f"Data already exists at {data_dir}")
        return

    urls = [
        'https://data.nasa.gov/download/fd5v-kuh6/application%2Fzip',
        'https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip',
    ]
    zip_path = 'CMAPSSData.zip'

    if os.path.exists(zip_path):
        print(f"Found {zip_path}, extracting...")
    else:
        downloaded = False
        for url in urls:
            try:
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, zip_path)
                downloaded = True
                print("Download complete.")
                break
            except Exception as e:
                print(f"Failed: {e}")
        if not downloaded:
            raise RuntimeError(
                "Could not download C-MAPSS data. "
                "Please place CMAPSSData.zip in the project root manually."
            )

    # Extract outer zip to a temp directory to avoid polluting the project root
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp_dir)

        # Check if the outer zip directly contains data files
        if os.path.isfile(os.path.join(tmp_dir, 'train_FD001.txt')):
            os.makedirs(data_dir, exist_ok=True)
            for fname in os.listdir(tmp_dir):
                shutil.move(os.path.join(tmp_dir, fname),
                            os.path.join(data_dir, fname))
        else:
            # Handle nested zip (e.g. outer contains a folder with CMAPSSData.zip inside)
            nested_zip = None
            for root, dirs, files in os.walk(tmp_dir):
                for f in files:
                    if f == 'CMAPSSData.zip':
                        nested_zip = os.path.join(root, f)
                        break
                if nested_zip:
                    break

            if nested_zip:
                print(f"Found nested zip, extracting to {data_dir}...")
                os.makedirs(data_dir, exist_ok=True)
                with zipfile.ZipFile(nested_zip, 'r') as z2:
                    z2.extractall(data_dir)
            else:
                raise RuntimeError(
                    "Could not find data files in the downloaded zip. "
                    f"Please manually extract data to {data_dir}/."
                )

    # Clean up the downloaded zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    if not os.path.exists(os.path.join(data_dir, 'train_FD001.txt')):
        raise RuntimeError(
            f"Could not find train_FD001.txt in {data_dir}. "
            "Please check the extracted data structure."
        )
    print("Data extracted successfully.")


def load_fd001(data_dir='CMAPSSData'):
    """Load FD001 subset files using pandas.

    Args:
        data_dir: Path to extracted CMAPSSData directory.

    Returns:
        train_df, test_df, rul_df: DataFrames for training, test, and RUL ground truth.
    """
    train_df = pd.read_csv(
        os.path.join(data_dir, 'train_FD001.txt'),
        sep=r'\s+', header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, 'test_FD001.txt'),
        sep=r'\s+', header=None, names=COLUMNS
    )
    rul_df = pd.read_csv(
        os.path.join(data_dir, 'RUL_FD001.txt'),
        sep=r'\s+', header=None, names=['RUL']
    )
    return train_df, test_df, rul_df


def add_rul_labels(df, r_early=125):
    """Compute piece-wise linear RUL labels for training data.

    Args:
        df: Training DataFrame with unit_id and cycle columns.
        r_early: RUL clipping upper bound.

    Returns:
        DataFrame with added 'RUL' column.
    """
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=r_early)
    df.drop('max_cycle', axis=1, inplace=True)
    return df


def select_features(train_df, threshold=0.01):
    """Select features by removing near-constant columns.

    Args:
        train_df: Training DataFrame.
        threshold: Minimum standard deviation to keep a feature.

    Returns:
        List of selected feature column names.
    """
    # Only consider sensor and setting columns (exclude unit_id, cycle, RUL)
    feature_cols = [c for c in train_df.columns
                    if c.startswith('s') or c.startswith('setting_')]
    stds = train_df[feature_cols].std()
    selected = stds[stds > threshold].index.tolist()
    print(f"Selected {len(selected)} features: {selected}")
    return selected


def create_sequences(df, feature_cols, seq_len=30, label_col='RUL'):
    """Create sliding window sequences for each engine unit.

    Args:
        df: DataFrame with features and RUL labels.
        feature_cols: List of feature column names.
        seq_len: Sliding window length.
        label_col: Name of the label column.

    Returns:
        sequences: numpy array of shape (N, seq_len, num_features).
        labels: numpy array of shape (N,).
    """
    sequences = []
    labels = []
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        features = unit_data[feature_cols].values
        rul = unit_data[label_col].values
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])
            labels.append(rul[i + seq_len - 1])
    return np.array(sequences), np.array(labels)


def create_test_sequences(test_df, rul_df, feature_cols, seq_len=30):
    """Create test sequences: last seq_len timesteps per engine, with zero-padding if needed.

    Args:
        test_df: Test DataFrame.
        rul_df: Ground truth RUL DataFrame.
        feature_cols: List of feature column names.
        seq_len: Sequence length.

    Returns:
        sequences: numpy array of shape (num_engines, seq_len, num_features).
        labels: numpy array of shape (num_engines,).
    """
    sequences = []
    labels = rul_df['RUL'].values
    num_features = len(feature_cols)

    for unit_id in test_df['unit_id'].unique():
        unit_data = test_df[test_df['unit_id'] == unit_id]
        features = unit_data[feature_cols].values
        if len(features) >= seq_len:
            sequences.append(features[-seq_len:])
        else:
            # Zero-pad on the left
            pad_len = seq_len - len(features)
            padded = np.zeros((pad_len, num_features))
            sequences.append(np.vstack([padded, features]))

    return np.array(sequences), labels


class CMAPSSDataset(Dataset):
    """PyTorch Dataset for C-MAPSS sliding window sequences."""

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: numpy array of shape (N, seq_len, num_features).
            labels: numpy array of shape (N,).
        """
        self.sequences = torch.FloatTensor(np.array(sequences, copy=True))
        self.labels = torch.FloatTensor(np.array(labels, copy=True)).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_data(data_dir='CMAPSSData', seq_len=30, r_early=125, batch_size=64,
                 val_ratio=0.2):
    """Full data preparation pipeline.

    Args:
        data_dir: Path to C-MAPSS data directory.
        seq_len: Sliding window length.
        r_early: RUL clipping upper bound.
        batch_size: Batch size for DataLoaders.
        val_ratio: Fraction of engines used for validation.

    Returns:
        Dictionary containing DataLoaders, scaler, feature list, and test ground truth.
    """
    # Download if needed
    download_cmapss(data_dir)

    # Load data
    train_df, test_df, rul_df = load_fd001(data_dir)

    # Add RUL labels to training data
    train_df = add_rul_labels(train_df, r_early=r_early)

    # Feature selection
    selected_features = select_features(train_df)

    # Train/val split by engine unit_id
    all_units = train_df['unit_id'].unique()
    np.random.shuffle(all_units)
    n_val = int(len(all_units) * val_ratio)
    val_units = all_units[:n_val]
    train_units = all_units[n_val:]

    train_split = train_df[train_df['unit_id'].isin(train_units)].copy()
    val_split = train_df[train_df['unit_id'].isin(val_units)].copy()

    print(f"Train engines: {len(train_units)}, Val engines: {len(val_units)}")

    # Fit scaler on training data
    scaler = MinMaxScaler()
    train_split[selected_features] = scaler.fit_transform(
        train_split[selected_features]
    )
    val_split[selected_features] = scaler.transform(
        val_split[selected_features]
    )
    test_df[selected_features] = scaler.transform(
        test_df[selected_features]
    )

    # Create sequences
    train_seqs, train_labels = create_sequences(
        train_split, selected_features, seq_len=seq_len
    )
    val_seqs, val_labels = create_sequences(
        val_split, selected_features, seq_len=seq_len
    )
    test_seqs, test_labels = create_test_sequences(
        test_df, rul_df, selected_features, seq_len=seq_len
    )

    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}, "
          f"Test sequences: {len(test_seqs)}")

    # Create DataLoaders
    train_dataset = CMAPSSDataset(train_seqs, train_labels)
    val_dataset = CMAPSSDataset(val_seqs, val_labels)
    test_dataset = CMAPSSDataset(test_seqs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'selected_features': selected_features,
        'input_dim': len(selected_features),
        'test_labels': test_labels,
        'train_df': train_df,
    }
