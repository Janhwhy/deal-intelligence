# src/models/train_phase4.py: End-to-end training pipeline for Phase 4.

import json
import logging
import os
from collections import Counter

# Set environment variables to prevent silent OpenMP duplicate library crashes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from src.config import load_config
from src.fusion import (
    align_and_concatenate_features,
    extract_lstm_features,
    project_features,
)
from src.models.outcome_classifier import OutcomeClassifier
from src.models.time_to_close import TimeToCloseRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_deal_targets(deals_dir: str) -> pd.DataFrame:
    """Computes outcomes and days_to_close regression targets for all deals.

    Args:
        deals_dir: Path to the directory containing deal timeline JSON files.

    Returns:
        DataFrame containing target columns, indexed by deal_id.
    """
    targets = []
    for filename in sorted(os.listdir(deals_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(deals_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                d = json.load(f)

            deal_id = int(d["deal_id"])
            outcome = d.get("outcome")
            close_date_str = d.get("close_date")
            first_event_str = d["events"][0]["timestamp"]

            # Outcome target: 1 if won, 0 otherwise (lost and open are treated as 0)
            outcome_target = 1.0 if outcome == "won" else 0.0

            # Days to close target: close_date - creation_date (first event timestamp)
            days_to_close = None
            if close_date_str is not None:
                days_to_close = (
                    pd.to_datetime(close_date_str) - pd.to_datetime(first_event_str)
                ).total_seconds() / 86400.0

            # Use close_date if available, else first event timestamp as a proxy for temporal sorting
            sort_date = (
                pd.to_datetime(close_date_str)
                if close_date_str is not None
                else pd.to_datetime(first_event_str)
            )

            targets.append(
                {
                    "deal_id": deal_id,
                    "outcome": outcome_target,
                    "days_to_close": days_to_close,
                    "sort_date": sort_date,
                }
            )

    df = pd.DataFrame(targets)
    df.set_index("deal_id", inplace=True)
    return df


def main() -> None:
    """Runs the end-to-end training, evaluation, and serialization pipeline."""
    logger.info("Loading configuration...")
    cfg = load_config()

    # 1. Load raw streams and extract LSTM features once
    logger.info("Executing multimodal feature fusion (Layer 3)...")
    logger.info("Loading tabular features...")
    tab_df = pd.read_parquet(cfg.data.processed_features_path)

    logger.info("Loading text features...")
    text_df = pd.read_parquet(cfg.features.processed_text_features_path)

    logger.info("Extracting LSTM sequential features (Stream C)...")
    lstm_feats = extract_lstm_features(cfg)

    # Outcome classification fusion (includes velocities)
    logger.info(
        "Aligning and concatenating feature streams for outcome classification..."
    )
    concatenated_clf, sorted_deal_ids_clf = align_and_concatenate_features(
        tab_df, text_df, lstm_feats, exclude_velocities=False
    )
    fused_features_clf = project_features(concatenated_clf, cfg)
    fused_df_clf = pd.DataFrame(fused_features_clf, index=sorted_deal_ids_clf)

    # Time-to-close regression fusion (excludes velocities to prevent target representation leak)
    logger.info(
        "Aligning and concatenating feature streams for regression (excluding velocities)..."
    )
    concatenated_reg, sorted_deal_ids_reg = align_and_concatenate_features(
        tab_df, text_df, lstm_feats, exclude_velocities=True
    )
    fused_features_reg = project_features(concatenated_reg, cfg)
    fused_df_reg = pd.DataFrame(fused_features_reg, index=sorted_deal_ids_reg)

    # 2. Compute targets for all deals
    logger.info("Computing deal target labels...")
    targets_df = compute_deal_targets(cfg.data.processed_deals_dir)

    # Assert deal alignment
    assert set(fused_df_clf.index) == set(targets_df.index), (
        f"Mismatched deal IDs between fused classification features and targets! "
        f"Fused: {len(fused_df_clf)}, Targets: {len(targets_df)}"
    )
    assert set(fused_df_reg.index) == set(targets_df.index), (
        f"Mismatched deal IDs between fused regression features and targets! "
        f"Fused: {len(fused_df_reg)}, Targets: {len(targets_df)}"
    )

    # Join target dataframe to keep sorting and targets aligned
    aligned_df_clf = fused_df_clf.join(
        targets_df[["outcome", "days_to_close", "sort_date"]]
    )
    aligned_df_clf.sort_values(by="sort_date", inplace=True)

    aligned_df_reg = fused_df_reg.join(
        targets_df[["outcome", "days_to_close", "sort_date"]]
    )
    aligned_df_reg.sort_values(by="sort_date", inplace=True)

    sorted_deals_info = [{"deal_id": d_id} for d_id in aligned_df_clf.index]

    # 3. Perform or load Temporal Split
    splits_path = "data/processed/splits.json"
    if os.path.exists(splits_path):
        logger.info(f"Loading splits from existing file: {splits_path}")
        with open(splits_path, "r") as f:
            splits = json.load(f)
        train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    else:
        logger.info("Computing new temporal split...")
        ratios = cfg.model.train_val_test_ratios or [0.7, 0.15, 0.15]
        train_ratio, val_ratio, test_ratio = ratios

        n = len(sorted_deals_info)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train_ids = [d["deal_id"] for d in sorted_deals_info[:train_end]]
        val_ids = [d["deal_id"] for d in sorted_deals_info[train_end:val_end]]
        test_ids = [d["deal_id"] for d in sorted_deals_info[val_end:]]

        # Save splits
        os.makedirs(os.path.dirname(splits_path), exist_ok=True)
        splits = {"train": train_ids, "val": val_ids, "test": test_ids}
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)
        logger.info(f"Successfully saved splits to {splits_path}")

    logger.info(
        f"Temporal Split Counts: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}"
    )

    # Feature columns lists
    feature_cols_clf = [
        c
        for c in aligned_df_clf.columns
        if c not in ["outcome", "days_to_close", "sort_date"]
    ]
    feature_cols_reg = [
        c
        for c in aligned_df_reg.columns
        if c not in ["outcome", "days_to_close", "sort_date"]
    ]

    # Split datasets for outcome classification
    train_data_clf = aligned_df_clf.loc[train_ids]
    val_data_clf = aligned_df_clf.loc[val_ids]

    X_train_clf = train_data_clf[feature_cols_clf].values
    X_val_clf = val_data_clf[feature_cols_clf].values
    y_train_outcome = train_data_clf["outcome"].values
    y_val_outcome = val_data_clf["outcome"].values

    # Split datasets for regression
    train_data_reg = aligned_df_reg.loc[train_ids]
    val_data_reg = aligned_df_reg.loc[val_ids]

    train_reg_data = train_data_reg[train_data_reg["days_to_close"].notna()]
    val_reg_data = val_data_reg[val_data_reg["days_to_close"].notna()]

    X_train_reg = train_reg_data[feature_cols_reg].values
    X_val_reg = val_reg_data[feature_cols_reg].values
    y_train_reg = train_reg_data["days_to_close"].values
    y_val_reg = val_reg_data["days_to_close"].values

    logger.info(
        f"Regression split counts (closed deals only): Train={len(train_reg_data)}, Val={len(val_reg_data)}"
    )

    # 4. Train Outcome Classifier
    outcome_clf = OutcomeClassifier(params=cfg.train.outcome_classifier_params)
    outcome_clf.fit(X_train_clf, y_train_outcome)

    # Evaluate Outcome Classifier
    val_outcome_preds = outcome_clf.predict(X_val_clf)
    val_outcome_probs = outcome_clf.predict_proba(X_val_clf)

    val_acc = accuracy_score(y_val_outcome, val_outcome_preds)
    if len(np.unique(y_val_outcome)) > 1:
        val_auc = roc_auc_score(y_val_outcome, val_outcome_probs)
    else:
        val_auc = 0.5

    # Baseline: Majority class from train split
    majority_class = Counter(y_train_outcome).most_common(1)[0][0]
    baseline_preds = np.full_like(y_val_outcome, majority_class)
    baseline_acc = accuracy_score(y_val_outcome, baseline_preds)

    logger.info("=== Outcome Classifier Validation Metrics ===")
    logger.info(f"Validation Accuracy:      {val_acc:.4f}")
    logger.info(f"Majority-Class Baseline:  {baseline_acc:.4f}")
    logger.info(f"Validation ROC AUC Score: {val_auc:.4f}")

    # 5. Train Time-to-Close Regressor
    if len(train_reg_data) > 0:
        regressor = TimeToCloseRegressor(params=cfg.train.time_to_close_params)
        regressor.fit(X_train_reg, y_train_reg)

        # Evaluate Regressor
        val_reg_preds = regressor.predict(X_val_reg)
        val_mae = mean_absolute_error(y_val_reg, val_reg_preds)
        val_mse = mean_squared_error(y_val_reg, val_reg_preds)
        val_rmse = np.sqrt(val_mse)

        # Evaluate Naive Mean Baseline (Training set mean)
        train_mean = np.mean(y_train_reg)
        naive_preds = np.full_like(y_val_reg, train_mean)
        naive_mae = mean_absolute_error(y_val_reg, naive_preds)
        naive_rmse = np.sqrt(mean_squared_error(y_val_reg, naive_preds))

        logger.info("=== Time-to-Close Regressor Validation Metrics ===")
        logger.info(f"XGBoost MAE:               {val_mae:.4f} days")
        logger.info(f"XGBoost RMSE:              {val_rmse:.4f} days")
        logger.info(f"Naive Mean Baseline MAE:   {naive_mae:.4f} days")
        logger.info(f"Naive Mean Baseline RMSE:  {naive_rmse:.4f} days")
    else:
        logger.warning(
            "No regression training data available (no closed deals in train split). Skipping regression head."
        )
        regressor = None

    # 6. Save trained model checkpoints
    models_dir = "data/processed/models"
    os.makedirs(models_dir, exist_ok=True)

    outcome_model_path = os.path.join(models_dir, "outcome_classifier.pkl")
    outcome_clf.save(outcome_model_path)

    if regressor is not None:
        reg_model_path = os.path.join(models_dir, "time_to_close.pkl")
        regressor.save(reg_model_path)

    logger.info("Phase 4 training pipeline successfully completed end-to-end!")


if __name__ == "__main__":
    main()
