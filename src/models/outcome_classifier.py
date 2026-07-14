# src/models/outcome_classifier.py: XGBoost outcome classifier for predicting Won/Lost deal state.

import logging
import os
import pickle
from typing import Any, Dict

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class OutcomeClassifier:
    """Wrapper around XGBClassifier to predict P(win) for a deal based on a fused vector representation.

    Consumes 256-dimensional unified deal vectors from Layer 3 and yields win probability.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """Initializes the outcome classifier wrapper.

        Args:
            params: Dictionary of hyperparameters for the XGBClassifier.
        """
        self.params = params or {}
        if "random_state" not in self.params:
            self.params["random_state"] = 42
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "logloss"

        self.model = XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the XGBoost outcome classifier.

        Args:
            X: Input fused vectors of shape (num_deals, feature_dim).
            y: Binary outcomes (1.0 for won, 0.0 for lost).
        """
        logger.info(f"Training XGBoost outcome classifier on shape {X.shape}...")
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability of winning the deal.

        Args:
            X: Input fused vectors of shape (num_deals, feature_dim).

        Returns:
            Numpy array of probabilities of class 1 (Won), shape (num_deals,).
        """
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary outcome (1 for won, 0 for lost).

        Args:
            X: Input fused vectors of shape (num_deals, feature_dim).

        Returns:
            Numpy array of binary decisions (0 or 1), shape (num_deals,).
        """
        return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Saves the trained model to a pickle file.

        Args:
            filepath: Target file path to write to.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Successfully saved OutcomeClassifier model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "OutcomeClassifier":
        """Loads a saved OutcomeClassifier model.

        Args:
            filepath: Path to the serialized pickle file.

        Returns:
            An instance of OutcomeClassifier.
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded OutcomeClassifier model from {filepath}")
        return model
