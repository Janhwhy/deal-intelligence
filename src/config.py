# src/config.py: Configuration utility using OmegaConf.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    enron_raw_dir: str
    hubspot_deals_csv: str
    hubspot_companies_csv: str
    hubspot_contacts_csv: str
    deal_linker_seed: int
    subject_similarity_threshold: float
    time_proximity_window_days: int
    processed_deals_dir: str
    processed_features_path: str
    deal_relevance_keywords_path: str = (
        "src/ingestion/resources/deal_relevance_keywords.txt"
    )
    max_deals_debug: Optional[int] = None


@dataclass
class ModelConfig:
    lstm_hidden_size: int
    lstm_num_layers: int
    lstm_seed: int
    lstm_dropout: float
    fusion_dropout: float = 0.1
    fusion_hidden_dim: int = 256
    train_val_test_ratios: List[float] = field(
        default_factory=lambda: [0.7, 0.15, 0.15]
    )


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    outcome_classifier_params: Dict[str, Any] = field(default_factory=dict)
    time_to_close_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturesConfig:
    sbert_model_name: str
    roberta_sentiment_model_name: str
    bertopic_min_topic_size: int
    batch_size: int
    hedge_words_resource_path: str
    processed_text_features_path: str
    bertopic_model_dir: str


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    features: FeaturesConfig


def load_config(config_dir: str = None) -> AppConfig:
    """Loads and merges configuration YAML files from the configs directory.

    Args:
        config_dir: Directory containing data.yaml, model.yaml, train.yaml, and features.yaml.
                    Defaults to the root 'configs/' directory.

    Returns:
        An instance of AppConfig populated with configuration values.
    """
    if config_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(base_dir, "configs")

    # Load component configurations
    data_path = os.path.join(config_dir, "data.yaml")
    model_path = os.path.join(config_dir, "model.yaml")
    train_path = os.path.join(config_dir, "train.yaml")
    features_path = os.path.join(config_dir, "features.yaml")

    data_cfg = OmegaConf.load(data_path)
    model_cfg = OmegaConf.load(model_path)
    train_cfg = OmegaConf.load(train_path)
    features_cfg = OmegaConf.load(features_path)

    # Merge into the base structure
    loaded_cfg = OmegaConf.create(
        {
            "data": data_cfg,
            "model": model_cfg,
            "train": train_cfg,
            "features": features_cfg,
        }
    )

    # Validate configuration against the dataclass schema
    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, loaded_cfg)

    # Convert structure to Python dataclass objects
    return OmegaConf.to_object(merged)
