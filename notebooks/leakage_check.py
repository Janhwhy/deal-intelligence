# notebooks/leakage_check.py: Trivial baseline classifier to check for behavioral label leakage.

import json
import logging
import os
import random
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_structural_features(
    deals_dir: str, filenames: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts only two trivial scalar features per deal:
    1. total_message_count
    2. has_external_reply (binary: 1.0 if at least one email sender is not @enron.com, else 0.0)

    And returns (features, labels).
    """
    X = []
    y = []

    for filename in filenames:
        filepath = os.path.join(deals_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        outcome = 1.0 if data.get("outcome") == "won" else 0.0
        emails = [e for e in data.get("events", []) if e.get("type") == "email"]

        msg_count = len(emails)

        # Check if at least one email is sent by a non-enron address
        has_external_reply = 0.0
        for email in emails:
            sender = (email.get("metadata", {}).get("sender") or "").lower()
            if sender and "@" in sender:
                domain = sender.split("@")[-1].strip()
                if domain != "enron.com":
                    has_external_reply = 1.0
                    break

        X.append([float(msg_count), has_external_reply])
        y.append(outcome)

    return np.array(X), np.array(y)


def run_leakage_check() -> None:
    """Trains a trivial logistic regression model on message count and has_external_reply
    to check for label leakage.
    """
    logger.info("Starting Trivial Baseline Classifier (Leakage Sanity Check)...")

    cfg = load_config()
    deals_dir = cfg.data.processed_deals_dir

    if not os.path.exists(deals_dir):
        raise FileNotFoundError(
            f"No processed deals directory found at '{deals_dir}'. "
            "Please run the ingestion pipeline first."
        )

    deal_files = sorted([f for f in os.listdir(deals_dir) if f.endswith(".json")])
    num_deals = len(deal_files)

    if num_deals == 0:
        raise FileNotFoundError("No processed deals files found.")

    logger.info(f"Loaded {num_deals} processed deal files.")

    # Replicate the exact same split as validate_stream_c.py
    # Set seed for reproducibility using the config's lstm_seed
    random.seed(cfg.model.lstm_seed)
    shuffled_files = list(deal_files)
    random.shuffle(shuffled_files)

    train_len = int(0.7 * len(shuffled_files))
    train_files = shuffled_files[:train_len]
    val_files = shuffled_files[train_len:]

    # Extract features & labels
    X_train, y_train = extract_structural_features(deals_dir, train_files)
    X_val, y_val = extract_structural_features(deals_dir, val_files)

    # Class weighting to handle imbalance, mirroring pos_weight
    train_lost = np.sum(y_train == 0.0)
    train_won = np.sum(y_train == 1.0)

    logger.info(f"Train Split: {train_won} Won, {train_lost} Lost")
    logger.info(
        f"Validation Split: {np.sum(y_val == 1.0)} Won, {np.sum(y_val == 0.0)} Lost"
    )

    # Train trivial logistic regression classifier
    # balanced class_weight handles the class imbalance automatically
    clf = LogisticRegression(class_weight="balanced", random_state=cfg.model.lstm_seed)
    clf.fit(X_train, y_train)

    # Evaluate
    train_preds_prob = clf.predict_proba(X_train)[:, 1]
    val_preds_prob = clf.predict_proba(X_val)[:, 1]
    val_preds = clf.predict(X_val)

    train_auc = roc_auc_score(y_train, train_preds_prob)
    val_auc = roc_auc_score(y_val, val_preds_prob)
    val_acc = accuracy_score(y_val, val_preds)

    val_baseline_acc = max(np.mean(y_val == 0.0), np.mean(y_val == 1.0))

    logger.info("=== Trivial Baseline Classifier Results ===")
    logger.info(f"Train AUC:       {train_auc:.4f}")
    logger.info(f"Validation AUC:  {val_auc:.4f}")
    logger.info(f"Validation Acc:  {val_acc:.4f}")
    logger.info(f"Majority Baseline Acc: {val_baseline_acc:.4f}")
    logger.info("===========================================")


if __name__ == "__main__":
    run_leakage_check()
