# src/fusion/fusion_model.py: Multimodal feature fusion and projection module for Layer 3.

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import AppConfig

logger = logging.getLogger(__name__)


class FusionProjector(nn.Module):
    """PyTorch module to project concatenated multimodal features into a unified vector space.

    Architecture: Linear(input_dim -> hidden_dim) + ReLU + Dropout
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input tensor of shape (N, input_dim) to (N, hidden_dim)."""
        return self.dropout(self.relu(self.linear(x)))


def extract_lstm_features(
    cfg: AppConfig, deal_ids: List[int] = None
) -> Dict[int, np.ndarray]:
    """Runs the trained/validated LSTM encoder from Phase 3 on each deal's message sequence.

    Args:
        cfg: The parsed app configuration.
        deal_ids: Optional list of deal IDs to process. If None, processes all deals.

    Returns:
        A dictionary mapping deal_id (int) to a 128-dimensional trajectory vector (numpy array).
    """
    from src.features.sequence_dataset import (
        DealSequenceDataset,
        collate_padded_sequences,
    )
    from src.features.sequence_model import LSTMSequenceEncoder
    from torch.utils.data import DataLoader

    # Set seed for reproducible initialization of the encoder weights
    torch.manual_seed(cfg.model.lstm_seed)

    # Instantiate the LSTM sequence encoder (Stream C)
    encoder = LSTMSequenceEncoder(
        input_size=386,  # 384 SBERT + 2 temporal features
        hidden_size=cfg.model.lstm_hidden_size,
        num_layers=cfg.model.lstm_num_layers,
        dropout=cfg.model.lstm_dropout,
        use_attention=True,
    )
    encoder.eval()

    deals_dir = cfg.data.processed_deals_dir
    if not os.path.exists(deals_dir):
        logger.error(f"Deals directory does not exist: {deals_dir}")
        raise FileNotFoundError(f"Missing deals directory: {deals_dir}")

    # Determine files to process
    if deal_ids is not None:
        deal_files = [f"{d_id}.json" for d_id in deal_ids]
    else:
        deal_files = [f for f in os.listdir(deals_dir) if f.endswith(".json")]

    dataset = DealSequenceDataset(
        deals_dir=deals_dir,
        sbert_model_name=cfg.features.sbert_model_name,
        batch_size=cfg.features.batch_size,
        deal_files=deal_files,
    )

    loader = DataLoader(
        dataset,
        batch_size=max(1, min(cfg.features.batch_size, len(dataset))),
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )

    lstm_features = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["embeddings"]
            lengths = batch["lengths"]
            batch_ids = batch["deal_ids"]

            trajectory = encoder(x, lengths)  # Shape: (batch_size, 128)
            for d_id, vector in zip(batch_ids.tolist(), trajectory.cpu().numpy()):
                lstm_features[int(d_id)] = vector

    return lstm_features


def align_and_concatenate_features(
    tab_df: pd.DataFrame,
    text_df: pd.DataFrame,
    lstm_features: Dict[int, np.ndarray],
    exclude_velocities: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    """Asserts identical deal_id sets across all three streams, aligns, and concatenates them.

    Args:
        tab_df: Tabular features DataFrame with deal_id as index.
        text_df: Text features DataFrame with deal_id as index.
        lstm_features: Dictionary of LSTM sequential features indexed by deal_id.
        exclude_velocities: If True, velocity_* stage features are excluded from concatenation.

    Returns:
        A tuple containing:
            - A 2D numpy array of aligned and concatenated features.
            - A list of the corresponding sorted deal_ids.

    Raises:
        ValueError: If the deal_id sets across streams do not match exactly.
    """
    tab_ids = set(tab_df.index)
    text_ids = set(text_df.index)
    lstm_ids = set(lstm_features.keys())

    # Assert that all three streams have identical deal_id sets
    if not (tab_ids == text_ids == lstm_ids):
        all_ids = tab_ids.union(text_ids).union(lstm_ids)
        missing_in_tab = all_ids - tab_ids
        missing_in_text = all_ids - text_ids
        missing_in_lstm = all_ids - lstm_ids

        err_msg = "Mismatched deal_ids found across feature streams:\n"
        if missing_in_tab:
            err_msg += (
                f"- Missing in tabular features: {sorted(list(missing_in_tab))}\n"
            )
        if missing_in_text:
            err_msg += f"- Missing in text features: {sorted(list(missing_in_text))}\n"
        if missing_in_lstm:
            err_msg += f"- Missing in LSTM features: {sorted(list(missing_in_lstm))}\n"

        logger.error(err_msg)
        raise ValueError(err_msg)

    # Sort deal_ids for deterministic alignment
    sorted_deal_ids = sorted(list(tab_ids))

    concatenated_list = []
    for d_id in sorted_deal_ids:
        # Stream A: 384-dim SBERT embedding
        sbert_emb = np.array(text_df.loc[d_id, "sbert_embedding"], dtype=np.float32)

        # Stream A: other metadata columns (topics, sentiment, hedge)
        other_text_cols = [c for c in text_df.columns if c != "sbert_embedding"]
        other_text_vals = text_df.loc[d_id, other_text_cols].values.astype(np.float32)

        # Stream B: tabular features (~11-dim or ~6-dim if velocities are excluded)
        tab_cols = [
            c
            for c in tab_df.columns
            if not (exclude_velocities and c.startswith("velocity_"))
        ]
        tab_vals = tab_df.loc[d_id, tab_cols].values.astype(np.float32)

        # Stream C: LSTM output (128-dim)
        lstm_val = lstm_features[d_id].astype(np.float32)

        # Concatenate them
        combined = np.concatenate([sbert_emb, other_text_vals, tab_vals, lstm_val])
        concatenated_list.append(combined)

    return np.vstack(concatenated_list), sorted_deal_ids


def project_features(concatenated_arr: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Projects concatenated multimodal features to a unified 256-dimensional vector space.

    Args:
        concatenated_arr: 2D numpy array of shape (N, input_dim).
        cfg: The parsed app configuration.

    Returns:
        A 2D numpy array of shape (N, 256) containing projected deal vectors.
    """
    # Set seed for reproducible projection weights
    torch.manual_seed(cfg.model.lstm_seed)

    input_dim = concatenated_arr.shape[1]
    projector = FusionProjector(
        input_dim=input_dim,
        hidden_dim=cfg.model.fusion_hidden_dim,
        dropout=cfg.model.fusion_dropout,
    )
    projector.eval()

    tensor_in = torch.tensor(concatenated_arr, dtype=torch.float32)
    with torch.no_grad():
        tensor_out = projector(tensor_in)

    return tensor_out.numpy()


def get_fused_deal_vectors(
    cfg: AppConfig,
    exclude_velocities: bool = False,
) -> Tuple[np.ndarray, List[int], pd.DataFrame, pd.DataFrame, Dict[int, np.ndarray]]:
    """Loads all streams, runs alignment assertions, concatenates, and projects them.

    Args:
        cfg: The parsed app configuration.
        exclude_velocities: If True, velocity_* features are excluded from fusion.

    Returns:
        A tuple of (fused_features_ndarray, sorted_deal_ids, tab_df, text_df, lstm_feats).
    """
    logger.info("Loading tabular features...")
    tab_df = pd.read_parquet(cfg.data.processed_features_path)

    logger.info("Loading text features...")
    text_df = pd.read_parquet(cfg.features.processed_text_features_path)

    logger.info("Extracting LSTM sequential features (Stream C)...")
    lstm_feats = extract_lstm_features(cfg)

    logger.info("Aligning and concatenating feature streams...")
    concatenated, sorted_deal_ids = align_and_concatenate_features(
        tab_df, text_df, lstm_feats, exclude_velocities=exclude_velocities
    )

    logger.info(
        f"Projecting features from dim {concatenated.shape[1]} to {cfg.model.fusion_hidden_dim}..."
    )
    fused_features = project_features(concatenated, cfg)

    return fused_features, sorted_deal_ids, tab_df, text_df, lstm_feats
