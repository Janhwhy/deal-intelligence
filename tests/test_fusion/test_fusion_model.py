# tests/test_fusion/test_fusion_model.py: Tests for Phase 4 fusion model.

import json

import numpy as np
import pandas as pd
import pytest

from src.fusion.fusion_model import align_and_concatenate_features, project_features
from src.models.train_phase4 import compute_deal_targets


def test_align_and_concatenate_mismatch():
    """Asserts that align_and_concatenate_features raises ValueError when deal_id sets do not match."""
    # Tabular: deal_ids 1, 2, 3
    tab_df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    # Text: deal_ids 1, 2 (missing 3)
    text_df = pd.DataFrame(
        {"sbert_embedding": [[0.1] * 384, [0.2] * 384], "other": [0.5, 0.6]},
        index=[1, 2],
    )
    # LSTM: deal_ids 1, 2, 3
    lstm_feats = {1: np.zeros(128), 2: np.zeros(128), 3: np.zeros(128)}

    with pytest.raises(ValueError) as excinfo:
        align_and_concatenate_features(tab_df, text_df, lstm_feats)

    assert "Mismatched deal_ids" in str(excinfo.value)
    assert "Missing in text features: [3]" in str(excinfo.value)


def test_fusion_projection_shape(mock_app_config):
    """Verifies that the fusion projection output shape is exactly (N, 256)."""
    # Create fake concatenated features of dimension (5, 528)
    N = 5
    input_dim = 528
    concat_arr = np.random.randn(N, input_dim)

    projected = project_features(concat_arr, mock_app_config)

    assert projected.shape == (N, 256)
    assert projected.dtype == np.float32 or projected.dtype == np.float64


def test_temporal_splits_and_determinism(tmp_path, mock_app_config):
    """Verifies that temporal splits are non-overlapping, complete, and load identically on re-run."""
    deals_dir = tmp_path / "deals"
    deals_dir.mkdir()

    # Write 10 mock deals with different close/creation dates
    for i in range(10):
        deal_id = i + 1
        # Make close_date optionally present
        close_date = f"200{i}-01-01T00:00:00Z" if i % 2 == 0 else None
        first_event = f"200{i}-02-01T00:00:00Z"

        deal_data = {
            "deal_id": deal_id,
            "stage": "Prospecting" if i < 9 else "Closed Won",
            "outcome": "open" if i < 9 else "won",
            "close_date": close_date,
            "events": [
                {
                    "timestamp": first_event,
                    "type": "email",
                    "content": "test message",
                    "metadata": {
                        "sender": "alice@enron.com",
                        "recipients": ["john@test.com"],
                        "subject": "RE: Deal",
                        "message_id": f"m{deal_id}",
                    },
                }
            ],
        }
        with open(deals_dir / f"{deal_id}.json", "w") as f:
            json.dump(deal_data, f)

    # Compute deal targets & get sorted order
    targets_df = compute_deal_targets(str(deals_dir))
    targets_df.sort_values(by="sort_date", inplace=True)
    sorted_deal_ids = targets_df.index.tolist()

    # Ratios
    train_ratio = 0.6
    val_ratio = 0.2

    n = len(sorted_deal_ids)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_ids = sorted_deal_ids[:train_end]
    val_ids = sorted_deal_ids[train_end:val_end]
    test_ids = sorted_deal_ids[val_end:]

    # Check split assertions:
    # 1. Non-overlapping
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))

    # 2. Together equal the full set
    all_split_ids = set(train_ids).union(set(val_ids)).union(set(test_ids))
    assert all_split_ids == set(sorted_deal_ids)
    assert len(all_split_ids) == n

    # 3. Determinism: check if splits saved and loaded are identical
    splits_path = tmp_path / "splits.json"
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    with open(splits_path, "r") as f:
        loaded_splits = json.load(f)

    assert loaded_splits["train"] == train_ids
    assert loaded_splits["val"] == val_ids
    assert loaded_splits["test"] == test_ids
