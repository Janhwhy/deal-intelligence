# tests/test_features/test_sequence_model.py: Tests for Phase 3 sequential model.

import json
import os

import pytest

import src.features.text_features
from src.features.sequence_model import HAS_TORCH


@pytest.fixture(autouse=True)
def force_fallback_modes(monkeypatch):
    """Forces the text feature extraction module to use fallback modes.
    This ensures that the dataset tests run instantly without downloading large models.
    """
    monkeypatch.setattr(src.features.text_features, "HAS_SBERT", False)
    monkeypatch.setattr(src.features.text_features, "HAS_BERTOPIC", False)
    monkeypatch.setattr(src.features.text_features, "HAS_TRANSFORMERS", False)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment.")
def test_lstm_encoder_shape():
    """Checks that the LSTM encoder produces trajectory vectors of correct shape (batch, 128)."""
    import torch

    from src.features.sequence_model import LSTMSequenceEncoder

    encoder = LSTMSequenceEncoder(input_size=384, hidden_size=128, num_layers=1)

    # Create a batch of padded sequences: batch_size=3, max_seq_len=5, input_size=384
    batch_size = 3
    max_seq_len = 5
    x = torch.randn(batch_size, max_seq_len, 384)
    lengths = torch.tensor([5, 2, 1], dtype=torch.long)

    output = encoder(x, lengths)

    assert output.shape == (batch_size, 128)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment.")
def test_padding_leakage():
    """Verifies that padding + packing prevents leakage.
    The output for a given sequence must be identical regardless of what other
    longer sequences it is batched with.
    """
    import torch

    from src.features.sequence_model import LSTMSequenceEncoder

    torch.manual_seed(42)
    encoder = LSTMSequenceEncoder(input_size=384, hidden_size=128, num_layers=1)
    encoder.eval()

    # Sequence A: length 2
    # Sequence B: length 4
    seq_a = torch.randn(1, 2, 384)
    seq_b = torch.randn(1, 4, 384)

    # 1. Run Sequence A alone
    with torch.no_grad():
        out_a_alone = encoder(seq_a, torch.tensor([2], dtype=torch.long))

    # 2. Run Sequence A and B batched together with padding
    padded_x = torch.zeros(2, 4, 384)
    padded_x[0, :2] = seq_a[0]
    padded_x[1, :4] = seq_b[0]
    lengths = torch.tensor([2, 4], dtype=torch.long)

    with torch.no_grad():
        out_batched = encoder(padded_x, lengths)

    # The first element in the batch corresponds to sequence A
    out_a_batched = out_batched[0:1]

    # Debug print statements
    print("DEBUG - out_a_alone:", out_a_alone)
    print("DEBUG - out_a_batched:", out_a_batched)
    print("DEBUG - out_batched:", out_batched)
    print(
        "DEBUG - unsorted_indices:",
        (
            encoder.last_unsorted_indices
            if hasattr(encoder, "last_unsorted_indices")
            else "N/A"
        ),
    )
    print(
        "DEBUG - sorted_indices:",
        (
            encoder.last_sorted_indices
            if hasattr(encoder, "last_sorted_indices")
            else "N/A"
        ),
    )

    # Verify no padding leakage (must be identical)
    assert torch.allclose(out_a_alone, out_a_batched, atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment.")
def test_lstm_determinism():
    """Verifies that same inputs and same seed produce identical outputs across runs."""
    import torch

    from src.features.sequence_model import LSTMSequenceEncoder

    # Set seed and create first encoder
    torch.manual_seed(123)
    encoder1 = LSTMSequenceEncoder(input_size=384, hidden_size=128, num_layers=1)

    # Set seed and create second encoder
    torch.manual_seed(123)
    encoder2 = LSTMSequenceEncoder(input_size=384, hidden_size=128, num_layers=1)

    x = torch.randn(2, 3, 384)
    lengths = torch.tensor([3, 2], dtype=torch.long)

    with torch.no_grad():
        out1 = encoder1(x, lengths)
        out2 = encoder2(x, lengths)

    assert torch.allclose(out1, out2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed in this environment.")
def test_single_message_deal(tmp_path):
    """Verifies Dataset and DataLoader handle a deal timeline with only 1 message (seq len 1)."""
    from torch.utils.data import DataLoader

    from src.features.sequence_dataset import (
        DealSequenceDataset,
        collate_padded_sequences,
    )

    deals_dir = tmp_path / "deals"
    deals_dir.mkdir()

    # Write a single message deal timeline
    deal_data = {
        "deal_id": 99,
        "company_id": 990,
        "company_name": "Single Message Corp",
        "amount": 50000.0,
        "stage": "Closed Won",
        "outcome": "won",
        "close_date": "2026-07-08T12:00:00Z",
        "industry": "Tech",
        "annual_revenue": 1000000.0,
        "num_employees": 50,
        "country": "USA",
        "contacts": [],
        "events": [
            {
                "type": "email",
                "timestamp": "2026-07-08T10:00:00Z",
                "metadata": {
                    "message_id": "<msg-99-1@enron.com>",
                    "sender": "sender@enron.com",
                    "recipients": ["buyer@external.com"],
                    "subject": "Initial Pitch",
                },
                "content": "Let's kick off the contract discussion.",
            }
        ],
    }

    with open(os.path.join(deals_dir, "99.json"), "w", encoding="utf-8") as f:
        json.dump(deal_data, f, indent=2)

    # Initialize dataset
    dataset = DealSequenceDataset(
        deals_dir=str(deals_dir), sbert_model_name="all-MiniLM-L6-v2"
    )
    assert len(dataset) == 1

    item = dataset[0]
    assert item["deal_id"] == 99
    assert item["seq_len"] == 1
    assert item["embeddings"].shape == (1, 384)

    # Wrap in dataloader
    loader = DataLoader([item], batch_size=1, collate_fn=collate_padded_sequences)
    batch = next(iter(loader))

    assert batch["embeddings"].shape == (1, 1, 384)
    assert batch["lengths"].tolist() == [1]
    assert batch["deal_ids"].tolist() == [99]
    assert batch["outcomes"].tolist() == [1.0]
