# src/models/time_to_close.py: XGBoost regressor for predicting days-to-close for a deal.

import logging
import os
import pickle
from typing import Any, Dict

import numpy as np
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class TimeToCloseRegressor:
    """Wrapper around XGBRegressor to predict days-to-close based on a fused vector representation.

    Consumes 256-dimensional unified deal vectors from Layer 3 and yields predicted days to close.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """Initializes the time-to-close regressor wrapper.

        Args:
            params: Dictionary of hyperparameters for the XGBRegressor.
        """
        self.params = params or {}
        if "random_state" not in self.params:
            self.params["random_state"] = 42
        if "eval_metric" not in self.params:
            self.params["eval_metric"] = "rmse"

        self.model = XGBRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the XGBoost time-to-close regressor.

        Args:
            X: Input fused vectors of shape (num_deals, feature_dim).
            y: Regression target (days to close).
        """
        logger.info(f"Training XGBoost time-to-close regressor on shape {X.shape}...")
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts days to close for the input deal features.

        Args:
            X: Input fused vectors of shape (num_deals, feature_dim).

        Returns:
            Numpy array of predicted days to close, shape (num_deals,).
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
        logger.info(f"Successfully saved TimeToCloseRegressor model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TimeToCloseRegressor":
        """Loads a saved TimeToCloseRegressor model.

        Args:
            filepath: Path to the serialized pickle file.

        Returns:
            An instance of TimeToCloseRegressor.
        """
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded TimeToCloseRegressor model from {filepath}")
        return model
