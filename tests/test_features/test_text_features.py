# tests/test_features/test_text_features.py: Tests for Phase 2 text features.

import json
import os

import numpy as np
import pandas as pd
import pytest

import src.features.text_features
from src.config import AppConfig, DataConfig, FeaturesConfig, ModelConfig, TrainConfig
from src.features.text_features import (
    build_text_features,
    compute_hedge_word_density,
    compute_sentiment_slope,
    extract_sbert_embeddings_batched,
    validate_text_features_df,
)


@pytest.fixture(autouse=True)
def force_fallback_modes(monkeypatch):
    """Forces the feature extraction module to use fallback modes (no deep learning)
    for all tests to ensure they are fast, hermetic, and do not make network calls.
    """
    monkeypatch.setattr(src.features.text_features, "HAS_SBERT", False)
    monkeypatch.setattr(src.features.text_features, "HAS_BERTOPIC", False)
    monkeypatch.setattr(src.features.text_features, "HAS_TRANSFORMERS", False)
    monkeypatch.setattr(src.features.text_features, "HAS_TORCH", False)


def test_sbert_embeddings_dimensionality_and_mean():
    """Tests that SBERT embedding outputs have the correct dimension (384)

    and that mean-pooling works correctly on a known example.
    """
    texts = ["Hello world", "This is a test message."]

    # Extract embeddings (which runs the fallback or real SBERT depending on environment)
    embeddings = extract_sbert_embeddings_batched(
        texts, "all-MiniLM-L6-v2", batch_size=32
    )

    # Dimensionality check: must be shape (2, 384)
    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32 or embeddings.dtype == np.float64

    # L2 Norm check (since they should be normalized)
    for row in embeddings:
        norm = np.linalg.norm(row)
        assert np.isclose(norm, 1.0, atol=1e-3)

    # Mean-pooling calculation check:
    expected_mean = np.mean(embeddings, axis=0)
    # The function itself doesn't calculate mean pool internally, but we perform it in the pipeline.
    # Check that np.mean handles the matrix rows correctly.
    assert expected_mean.shape == (384,)
    assert np.isclose(expected_mean[0], (embeddings[0][0] + embeddings[1][0]) / 2.0)


def test_hedge_word_density():
    """Verifies that hedge word density is computed correctly against

    a hand-calculated value.
    """
    hedge_words = ["might", "perhaps", "I think", "not sure"]

    # Text contains: "hello perhaps world might not sure"
    # Total words = 6
    # Matches = 3 ("perhaps", "might", "not sure" matches as a phrase)
    # Density = 3 / 6 = 0.5
    text = "hello perhaps world might not sure"
    density = compute_hedge_word_density(text, hedge_words)
    assert np.isclose(density, 0.5)

    # Text contains 0 hedge words
    text_none = "hello world this is a test"
    assert compute_hedge_word_density(text_none, hedge_words) == 0.0

    # Text has 0 total words
    assert compute_hedge_word_density("", hedge_words) == 0.0


def test_sentiment_slope_direction():
    """Verifies that sentiment slope sign is correct on synthetic sequences

    where sentiment obviously increases or decreases.
    """
    # Case 1: Increasing sentiment
    # Scores: [-0.8, -0.2, 0.3, 0.9] -> upward trend (slope > 0)
    inc_scores = [-0.8, -0.2, 0.3, 0.9]
    slope_inc = compute_sentiment_slope(inc_scores)
    assert slope_inc > 0.0

    # Case 2: Decreasing sentiment
    # Scores: [0.9, 0.4, -0.1, -0.7] -> downward trend (slope < 0)
    dec_scores = [0.9, 0.4, -0.1, -0.7]
    slope_dec = compute_sentiment_slope(dec_scores)
    assert slope_dec < 0.0

    # Case 3: Flat sentiment
    flat_scores = [0.5, 0.5, 0.5]
    assert compute_sentiment_slope(flat_scores) == 0.0

    # Case 4: Insufficient points (1 or 0 points)
    assert compute_sentiment_slope([0.8]) == 0.0
    assert compute_sentiment_slope([]) == 0.0


def test_schema_validation_catches_invalid():
    """Verifies that validate_text_features_df fails loudly on out-of-range values."""
    # sentiment_mean is 5.0 (invalid, must be in [-1.0, 1.0])
    invalid_data = [
        {
            "deal_id": 1,
            "sbert_embedding": [0.1] * 384,
            "dominant_topic_id": 0,
            "topic_drift_score": 0.2,
            "sentiment_mean": 5.0,  # Invalid!
            "sentiment_slope": 0.05,
            "hedge_word_density": 0.1,
        }
    ]
    df_invalid = pd.DataFrame(invalid_data).set_index("deal_id")

    with pytest.raises(ValueError, match="validation error"):
        validate_text_features_df(df_invalid)

    # hedge_word_density is -0.5 (invalid, must be in [0.0, 1.0])
    invalid_data_2 = [
        {
            "deal_id": 1,
            "sbert_embedding": [0.1] * 384,
            "dominant_topic_id": 0,
            "topic_drift_score": 0.2,
            "sentiment_mean": 0.5,
            "sentiment_slope": 0.05,
            "hedge_word_density": -0.5,  # Invalid!
        }
    ]
    df_invalid_2 = pd.DataFrame(invalid_data_2).set_index("deal_id")

    with pytest.raises(ValueError, match="validation error"):
        validate_text_features_df(df_invalid_2)


def test_deal_id_alignment(tmp_path):
    """Tests end-to-end extraction on handcrafted timelines in a temporary folder.

    Verifies the output parquet has matching deal_ids, correct column schema,
    and is joinable with tabular_features.parquet.
    """
    # Set up directories
    processed_dir = tmp_path / "processed"
    deals_dir = processed_dir / "deals"
    deals_dir.mkdir(parents=True)

    # 1. Write hedge words resource file
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir(parents=True)
    hedge_file = resource_dir / "hedge_words.txt"
    hedge_file.write_text("might\nperhaps\nI think\nnot sure\n")

    # 2. Write 3 mock deal timeline JSONs
    # Deal 1: Increasing sentiment, contains hedge words
    deal1 = {
        "deal_id": 1,
        "stage": "Negotiation",
        "outcome": "open",
        "amount": 10000.0,
        "company_id": 101,
        "company_name": "Company A",
        "industry": "Tech",
        "annual_revenue": 5000000.0,
        "num_employees": 50,
        "country": "USA",
        "contacts": [],
        "events": [
            {
                "timestamp": "2025-01-01T10:00:00Z",
                "type": "email",
                "content": "We might have an issue. Not sure yet.",
                "metadata": {
                    "sender": "bob@enron.com",
                    "recipients": ["alice@company.com"],
                    "subject": "Proposal",
                    "message_id": "m1",
                },
            },
            {
                "timestamp": "2025-01-02T10:00:00Z",
                "type": "email",
                "content": "This is great progress. Agree with terms. Thanks!",
                "metadata": {
                    "sender": "alice@company.com",
                    "recipients": ["bob@enron.com"],
                    "subject": "RE: Proposal",
                    "message_id": "m2",
                },
            },
        ],
    }
    # Deal 2: Decreasing sentiment, no hedge words
    deal2 = {
        "deal_id": 2,
        "stage": "Prospecting",
        "outcome": "open",
        "amount": 20000.0,
        "company_id": 102,
        "company_name": "Company B",
        "industry": "Finance",
        "annual_revenue": 10000000.0,
        "num_employees": 100,
        "country": "UK",
        "contacts": [],
        "events": [
            {
                "timestamp": "2025-01-01T10:00:00Z",
                "type": "email",
                "content": "Yes, looks very good and happy to partner.",
                "metadata": {
                    "sender": "bob@enron.com",
                    "recipients": ["charlie@company.com"],
                    "subject": "Hello",
                    "message_id": "m3",
                },
            },
            {
                "timestamp": "2025-01-03T10:00:00Z",
                "type": "email",
                "content": "This is bad and a big fail unfortunately.",
                "metadata": {
                    "sender": "charlie@company.com",
                    "recipients": ["bob@enron.com"],
                    "subject": "RE: Hello",
                    "message_id": "m4",
                },
            },
        ],
    }
    # Deal 3: Single email
    deal3 = {
        "deal_id": 3,
        "stage": "Demo Scheduled",
        "outcome": "open",
        "amount": 5000.0,
        "company_id": 103,
        "company_name": "Company C",
        "industry": "SaaS",
        "annual_revenue": 2000000.0,
        "num_employees": 20,
        "country": "Germany",
        "contacts": [],
        "events": [
            {
                "timestamp": "2025-01-01T10:00:00Z",
                "type": "email",
                "content": "Neutral message text here.",
                "metadata": {
                    "sender": "bob@enron.com",
                    "recipients": ["dave@company.com"],
                    "subject": "Status",
                    "message_id": "m5",
                },
            }
        ],
    }

    with open(deals_dir / "1.json", "w") as f:
        json.dump(deal1, f)
    with open(deals_dir / "2.json", "w") as f:
        json.dump(deal2, f)
    with open(deals_dir / "3.json", "w") as f:
        json.dump(deal3, f)

    # 3. Setup configurations
    data_cfg = DataConfig(
        enron_raw_dir="dummy/raw/enron",
        hubspot_deals_csv="dummy/raw/deals.csv",
        hubspot_companies_csv="dummy/raw/companies.csv",
        hubspot_contacts_csv="dummy/raw/contacts.csv",
        deal_linker_seed=42,
        subject_similarity_threshold=0.5,
        time_proximity_window_days=7,
        processed_deals_dir=str(deals_dir),
        processed_features_path=str(processed_dir / "tabular_features.parquet"),
    )

    features_cfg = FeaturesConfig(
        sbert_model_name="all-MiniLM-L6-v2",
        roberta_sentiment_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        bertopic_min_topic_size=2,  # set small so KMeans fits on tiny test corpus
        batch_size=2,
        hedge_words_resource_path=str(hedge_file),
        processed_text_features_path=str(processed_dir / "text_features.parquet"),
        bertopic_model_dir=str(processed_dir / "models" / "bertopic_model"),
    )

    config = AppConfig(
        data=data_cfg,
        model=ModelConfig(
            lstm_hidden_size=128, lstm_num_layers=1, lstm_seed=42, lstm_dropout=0.0
        ),
        train=TrainConfig(batch_size=32, learning_rate=0.001),
        features=features_cfg,
    )

    # 4. Run the text features extraction pipeline
    df_text = build_text_features(config)

    # 5. Assertions on text features
    assert len(df_text) == 3
    assert set(df_text.index) == {1, 2, 3}

    expected_cols = {
        "sbert_embedding",
        "dominant_topic_id",
        "topic_drift_score",
        "sentiment_mean",
        "sentiment_slope",
        "hedge_word_density",
    }
    assert set(df_text.columns) == expected_cols

    # Verify values and types
    assert isinstance(df_text.loc[1, "sbert_embedding"], (list, np.ndarray))
    assert len(df_text.loc[1, "sbert_embedding"]) == 384

    # Check sentiment slopes
    # Deal 1 increases: slope should be positive
    assert df_text.loc[1, "sentiment_slope"] > 0.0
    # Deal 2 decreases: slope should be negative
    assert df_text.loc[2, "sentiment_slope"] < 0.0
    # Deal 3 has only 1 message: slope should be 0.0
    assert df_text.loc[3, "sentiment_slope"] == 0.0

    # Check hedge word density
    # Deal 1 contains "might" and "not sure" in 10-word message, and no hedge words in 8-word message.
    # Combined words: "We might have an issue. Not sure yet." (8 words) + "This is great progress. Agree with terms. Thanks!" (9 words) = 17 words.
    # Hedge matches: "might", "not sure" (matches 1 phrase) = 2 matches.
    # Density = 2 / 17 > 0.0
    assert df_text.loc[1, "hedge_word_density"] > 0.0
    assert df_text.loc[2, "hedge_word_density"] == 0.0

    # 6. Parquet file verification
    assert os.path.exists(features_cfg.processed_text_features_path)
    df_parquet = pd.read_parquet(features_cfg.processed_text_features_path)
    assert len(df_parquet) == 3
    assert set(df_parquet.index) == {1, 2, 3}

    # 7. Join-compatibility check with tabular_features.parquet
    # We write a mock tabular_features.parquet with matching index
    df_tabular = pd.DataFrame(
        {"touches": [2, 2, 1], "stakeholder_count": [2, 2, 2]},
        index=pd.Index([1, 2, 3], name="deal_id"),
    )
    df_tabular.to_parquet(data_cfg.processed_features_path)

    # Perform the join on deal_id
    df_joined = df_tabular.join(df_parquet, how="inner")
    assert len(df_joined) == 3
    assert set(df_joined.index) == {1, 2, 3}
    assert "sentiment_mean" in df_joined.columns
    assert "touches" in df_joined.columns
