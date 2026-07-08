# tests/test_ingestion/test_pipeline.py: End-to-end pipeline integration tests.

import os
import sys
from unittest.mock import patch

import pandas as pd

from src.config import load_config
from src.features.tabular_features import build_tabular_features
from src.ingestion.pipeline import run_ingestion_pipeline


def test_pipeline_and_feature_extraction_end_to_end(mock_data_dir, mock_app_config):
    """Integration test checking end-to-end execution of pipeline and feature extraction."""
    # Set up a temporary config directory
    config_dir = mock_data_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write configs targeting our mock data paths
    data_cfg = mock_app_config.data
    data_yaml_content = (
        f"enron_raw_dir: {data_cfg.enron_raw_dir}\n"
        f"hubspot_deals_csv: {data_cfg.hubspot_deals_csv}\n"
        f"hubspot_companies_csv: {data_cfg.hubspot_companies_csv}\n"
        f"hubspot_contacts_csv: {data_cfg.hubspot_contacts_csv}\n"
        f"deal_linker_seed: {data_cfg.deal_linker_seed}\n"
        f"subject_similarity_threshold: {data_cfg.subject_similarity_threshold}\n"
        f"time_proximity_window_days: {data_cfg.time_proximity_window_days}\n"
        f"processed_deals_dir: {data_cfg.processed_deals_dir}\n"
        f"processed_features_path: {data_cfg.processed_features_path}\n"
    )
    (config_dir / "data.yaml").write_text(data_yaml_content)
    (config_dir / "model.yaml").write_text(
        "lstm_hidden_size: 128\nlstm_num_layers: 1\n"
    )
    (config_dir / "train.yaml").write_text("batch_size: 32\nlearning_rate: 0.001\n")
    features_cfg = mock_app_config.features
    features_yaml_content = (
        f"sbert_model_name: {features_cfg.sbert_model_name}\n"
        f"roberta_sentiment_model_name: {features_cfg.roberta_sentiment_model_name}\n"
        f"bertopic_min_topic_size: {features_cfg.bertopic_min_topic_size}\n"
        f"batch_size: {features_cfg.batch_size}\n"
        f"hedge_words_resource_path: {features_cfg.hedge_words_resource_path}\n"
        f"processed_text_features_path: {features_cfg.processed_text_features_path}\n"
        f"bertopic_model_dir: {features_cfg.bertopic_model_dir}\n"
    )
    (config_dir / "features.yaml").write_text(features_yaml_content)

    # Run the ingestion pipeline by mocking sys.argv to pass the temporary config_dir
    with patch.object(sys, "argv", ["pipeline.py", str(config_dir)]):
        run_ingestion_pipeline()

    # Verify that the deal timeline JSON was successfully created
    deal_json_path = os.path.join(data_cfg.processed_deals_dir, "1.json")
    assert os.path.exists(deal_json_path)

    # Load configuration
    cfg = load_config(config_dir=str(config_dir))

    # Run feature extraction on the generated timelines
    df = build_tabular_features(cfg.data)

    # Verify that the features DataFrame was created, saved to parquet, and contains the deal
    assert not df.empty
    assert os.path.exists(data_cfg.processed_features_path)

    # Read the parquet file back and assert structure
    read_df = pd.read_parquet(data_cfg.processed_features_path)
    assert len(read_df) == 1
    assert 1 in read_df.index
    assert "touches" in read_df.columns
    assert "days_since_last_reply" in read_df.columns
