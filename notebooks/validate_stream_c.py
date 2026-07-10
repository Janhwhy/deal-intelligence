# notebooks/validate_stream_c.py: Diagnostic validation probe for Stream C sequential model.

import json
import logging
import os
import random
from typing import List

import numpy as np

from src.config import load_config
from src.features.sequence_dataset import EMBEDDING_DIM
from src.features.sequence_model import HAS_TORCH

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_deal_examples(deals_dir: str, deal_files: List[str]) -> None:
    """Sanity prints 5 example won and 5 example lost deal timelines."""
    won_examples = []
    lost_examples = []

    for filename in sorted(deal_files):
        filepath = os.path.join(deals_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        outcome = data.get("outcome")
        emails = [e for e in data.get("events", []) if e.get("type") == "email"]
        msg_count = len(emails)
        snippet = "No communication."
        if msg_count > 0:
            snippet = emails[0].get("content", "")[:100].replace("\n", " ") + "..."

        example_info = {
            "deal_id": data.get("deal_id"),
            "msg_count": msg_count,
            "snippet": snippet,
        }

        if outcome == "won":
            if len(won_examples) < 5:
                won_examples.append(example_info)
        else:
            if len(lost_examples) < 5:
                lost_examples.append(example_info)

        if len(won_examples) >= 5 and len(lost_examples) >= 5:
            break

    logger.info("=== Example 'Won' Deal Timelines ===")
    for ex in won_examples:
        logger.info(
            f"Deal ID: {ex['deal_id']} | Messages: {ex['msg_count']} | Snippet: {ex['snippet']}"
        )

    logger.info("=== Example 'Lost' Deal Timelines ===")
    for ex in lost_examples:
        logger.info(
            f"Deal ID: {ex['deal_id']} | Messages: {ex['msg_count']} | Snippet: {ex['snippet']}"
        )


def _train_one_epoch(model, train_loader, optimizer, criterion, device, train_size):
    """Runs one training epoch and returns average weighted loss."""
    import torch

    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        embeddings = batch["embeddings"].to(device)
        lengths = batch["lengths"].to(device)
        outcomes = batch["outcomes"].to(device)
        logits = model(embeddings, lengths).squeeze(-1)
        loss = criterion(logits, outcomes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(outcomes)
    return total_loss / train_size if train_size > 0 else 0.0


def _evaluate(model, loader, criterion, device, dataset_size):
    """Runs evaluation and returns (avg_loss, predictions, labels)."""
    import torch

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            lengths = batch["lengths"].to(device)
            outcomes = batch["outcomes"].to(device)
            logits = model(embeddings, lengths).squeeze(-1)
            loss = criterion(logits, outcomes)
            total_loss += loss.item() * len(outcomes)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(outcomes.cpu().numpy())
    avg_loss = total_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss, np.array(all_preds), np.array(all_labels)


def _count_won_lost(deals_dir: str, filenames: List[str]):
    """Counts won and lost outcomes in a list of deal JSON files.

    Returns:
        Tuple of (won_count, lost_count).
    """
    won = 0
    lost = 0
    for filename in filenames:
        filepath = os.path.join(deals_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("outcome") == "won":
            won += 1
        else:
            lost += 1
    return won, lost


def run_diagnostic_validation() -> None:
    """Loads actual dataset timelines, trains a tiny linear probe on LSTM outputs,
    and validates classification signal.
    """
    logger.info("Initializing Stream C standalone validation...")

    if not HAS_TORCH:
        logger.warning(
            "PyTorch is not installed in this environment. Skipping diagnostic LSTM training.\n"
            "This is normal when running locally on Python 3.14 where PyTorch wheels are not yet available.\n"
            "The full verification suite and LSTM validation will execute automatically on the CI runner."
        )
        return

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from src.features.sequence_dataset import (
        DealSequenceDataset,
        collate_padded_sequences,
    )
    from src.features.sequence_model import LSTMSequenceEncoder
    from src.features.text_features import HAS_SBERT

    # Load configuration
    cfg = load_config()
    model_cfg = cfg.model
    features_cfg = cfg.features

    # Set seed for reproducibility
    torch.manual_seed(model_cfg.lstm_seed)
    np.random.seed(model_cfg.lstm_seed)
    random.seed(model_cfg.lstm_seed)

    # Check for existing processed deals directory
    deals_dir = cfg.data.processed_deals_dir

    if not os.path.exists(deals_dir):
        raise FileNotFoundError(
            f"No processed deals directory found at '{deals_dir}'. "
            "Please run the ingestion pipeline (poetry run python src/ingestion/pipeline.py) first."
        )

    deal_files = sorted([f for f in os.listdir(deals_dir) if f.endswith(".json")])
    num_deals = len(deal_files)

    if num_deals == 0:
        raise FileNotFoundError(
            f"No real deal timeline JSON files found in '{deals_dir}'. "
            "Please run the ingestion pipeline first."
        )

    logger.info(f"Loaded {num_deals} real deal timeline files from '{deals_dir}'.")

    # Print example deal timelines
    print_deal_examples(deals_dir, deal_files)

    # Explicitly print the SBERT embedding source used
    embedding_source = (
        f"Real SBERT model ('{features_cfg.sbert_model_name}')"
        if HAS_SBERT
        else "Fallback HashingVectorizer"
    )
    logger.info(f"Embedding Source Used: {embedding_source}")
    logger.info(
        f"Input Dimensionality: {EMBEDDING_DIM} "
        f"(SBERT 384 + 3 temporal features: time_delta, is_external_sender, content_length)"
    )

    # Enforce sample size warning
    if num_deals < 100:
        logger.warning(
            f"WARNING: The sample size ({num_deals} deals) is too small (< 100) for a statistically meaningful signal check."
        )

    # Shuffle files to partition by deal_id BEFORE loading to prevent leakage
    shuffled_files = list(deal_files)
    random.shuffle(shuffled_files)

    train_len = int(0.7 * len(shuffled_files))
    train_files = shuffled_files[:train_len]
    val_files = shuffled_files[train_len:]

    # Count classes in train and validation sets
    train_won, train_lost = _count_won_lost(deals_dir, train_files)
    val_won, val_lost = _count_won_lost(deals_dir, val_files)

    logger.info(
        f"Split sizes (leakage-free by deal_id): Train={len(train_files)}, Validation={len(val_files)}"
    )
    logger.info(f"Train Split Class Counts: {train_won} Won, {train_lost} Lost")
    logger.info(f"Validation Split Class Counts: {val_won} Won, {val_lost} Lost")

    # Load pre-split Train/Val datasets
    train_set = DealSequenceDataset(
        deals_dir=deals_dir,
        sbert_model_name=features_cfg.sbert_model_name,
        batch_size=features_cfg.batch_size,
        deal_files=train_files,
    )
    val_set = DealSequenceDataset(
        deals_dir=deals_dir,
        sbert_model_name=features_cfg.sbert_model_name,
        batch_size=features_cfg.batch_size,
        deal_files=val_files,
    )

    batch_size = min(8, len(train_set)) if len(train_set) > 0 else 1
    val_batch_size = min(8, len(val_set)) if len(val_set) > 0 else 1

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_padded_sequences,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )

    # Define probe model: LSTM (with attention) + linear head
    # Uses EMBEDDING_DIM (387) — SBERT 384 + 3 temporal features
    class LinearProbeModel(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, dropout: float):
            super().__init__()
            self.encoder = LSTMSequenceEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=dropout,
                use_attention=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            trajectory = self.encoder(x, lengths)
            logits = self.fc(trajectory)
            return logits

    model = LinearProbeModel(
        input_size=EMBEDDING_DIM,
        hidden_size=model_cfg.lstm_hidden_size,
        dropout=model_cfg.lstm_dropout,
    )

    # Device allocation
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    logger.info(f"Using device for training: {device}")
    model = model.to(device)

    # Confirm parameter optimization status
    lstm_trainable = any(p.requires_grad for p in model.encoder.parameters())
    fc_trainable = any(p.requires_grad for p in model.fc.parameters())
    logger.info(
        f"Optimization Status: LSTM weights trainable = {lstm_trainable} | "
        f"Linear head weights trainable = {fc_trainable} | "
        f"Attention pooling = {model.encoder.use_attention}"
    )
    logger.info(
        "CONFIRMATION: LSTM Encoder (with attention) + Linear head trained jointly end-to-end."
    )

    # Class weighting: inverse frequency for imbalanced setting
    pos_weight_val = train_lost / train_won if train_won > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    logger.info(f"BCE Class Weighting (pos_weight): {pos_weight_val:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )

    # ── Early stopping setup ──────────────────────────────────────────────────
    MAX_EPOCHS = 30
    PATIENCE = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    logger.info(
        f"Training LSTM probe (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})..."
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        avg_train_loss = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, len(train_set)
        )
        avg_val_loss, _, _ = _evaluate(
            model, val_loader, criterion, device, len(val_set)
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} — Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            import copy

            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {PATIENCE} consecutive epochs)."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best model weights for final evaluation.")

    # ── Final evaluation ──────────────────────────────────────────────────────
    _, all_preds, all_labels = _evaluate(
        model, val_loader, criterion, device, len(val_set)
    )

    # Metrics
    binary_preds = (all_preds > 0.5).astype(float)
    accuracy = np.mean(binary_preds == all_labels) if len(all_labels) > 0 else 0.0

    val_total = len(all_labels)
    val_baseline_accuracy = max(val_won, val_lost) / val_total if val_total > 0 else 0.0

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.5

    logger.info("=== Validation Signal Check Results ===")
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(
        f"Validation Majority-Class Baseline Accuracy: {val_baseline_accuracy:.4f}"
    )
    logger.info(f"Validation AUC:      {auc:.4f}")

    logger.info(
        "SUMMARY: Stream C's standalone predictive power against independently-sampled "
        "synthetic labels is not the primary validation criterion for this "
        "architecture. Stream C's true contribution will be evaluated via "
        "ablation in Phase 8 (does including Stream C improve the fused "
        "model's performance on the causal-category task versus Streams A+B "
        "alone), not via standalone binary classification against a label it "
        "has no guaranteed causal relationship to."
    )


if __name__ == "__main__":
    run_diagnostic_validation()
