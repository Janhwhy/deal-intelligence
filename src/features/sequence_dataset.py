# src/features/sequence_dataset.py: PyTorch Dataset & DataLoader collate helper for Stream C.

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.ingestion.timeline_builder import DealTimelineModel

logger = logging.getLogger(__name__)

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from torch.utils.data import Dataset
else:
    # Fallback to plain object if PyTorch is not available (e.g. locally in Python 3.14)
    class Dataset:
        pass


# Number of temporal features appended per message to SBERT embedding
TEMPORAL_FEATURE_DIM = 2
# Total input dimensionality (SBERT 384 + 2 temporal)
SBERT_DIM = 384
EMBEDDING_DIM = SBERT_DIM + TEMPORAL_FEATURE_DIM  # 386


def _compute_temporal_features(emails: list) -> np.ndarray:
    """Computes two temporal/structural features per email message.

    These features encode observable deal-process signals that SBERT alone
    cannot capture because they depend on *when* and *how long* each message
    was, not just its content.

    NOTE: ``is_external_sender`` was intentionally removed.  The behavioral
    label ("won" = at least one domain-switch occurred) is derived from sender
    domain identity, so including an ``is_external_sender`` flag as a feature
    would allow the model to trivially recover the label without reading
    content (feature-label leakage, AUC → 0.97).  The model must instead
    learn from content and timing patterns.

    1. **time_delta_norm**: Hours elapsed since the previous message, clipped
       at 30 days and scaled to [0, 1].  Quickening cadence (low delta) often
       signals active negotiation; long silences signal stagnation / ghosting.

    2. **content_length_norm**: Normalised character count of the message body
       (clipped at 5,000 chars, scaled to [0, 1]).  Longer, more detailed
       replies from either party correlate with genuine interest; very short
       replies or boilerplate correlate with disengagement.

    Args:
        emails: List of EmailEventModel objects in chronological order.

    Returns:
        np.ndarray of shape (len(emails), 2), dtype float32.
    """
    n = len(emails)
    features = np.zeros((n, TEMPORAL_FEATURE_DIM), dtype=np.float32)

    MAX_HOURS = 30 * 24  # 30 days ceiling for normalisation
    MAX_CHARS = 5_000

    for i, email in enumerate(emails):
        # 1. Time delta from previous message (hours)
        if i == 0:
            delta_hours = 0.0
        else:
            delta = (email.timestamp - emails[i - 1].timestamp).total_seconds() / 3600.0
            delta_hours = max(0.0, delta)
        features[i, 0] = min(delta_hours / MAX_HOURS, 1.0)

        # 2. Content length (normalised)
        content_len = len(email.content or "")
        features[i, 1] = min(content_len / MAX_CHARS, 1.0)

    return features


class DealSequenceDataset(Dataset):
    """Dataset that loads deal timelines, extracts per-message SBERT embeddings
    concatenated with 3 temporal/structural features, and returns sequence
    representations.

    Each item in the dataset is a tensor of shape (seq_len, 387):
      - dims 0:384   → SBERT sentence embedding of message content
      - dim  384     → normalised inter-message time delta (hours / 720)
      - dim  385     → is_external_sender flag (0 or 1)
      - dim  386     → normalised message content length (chars / 5000)
    """

    def __init__(
        self,
        deals_dir: str,
        sbert_model_name: str,
        batch_size: int = 32,
        deal_files: Optional[List[str]] = None,
    ):
        self.deals_dir = deals_dir
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size

        if not os.path.exists(deals_dir):
            raise FileNotFoundError(f"Deals directory does not exist: {deals_dir}")

        # Load all timelines or use provided list of file names
        if deal_files is not None:
            self.deal_files = sorted(deal_files)
        else:
            self.deal_files = sorted(
                [f for f in os.listdir(deals_dir) if f.endswith(".json")]
            )
        self.data_items: List[Dict[str, Any]] = []
        self._load_and_process_timelines()

    def _load_and_process_timelines(self) -> None:
        """Crawl all timelines, build a corpus of all emails, extract embeddings,
        compute temporal features, and segment them back into chronological
        sequences per deal.
        """
        all_emails_texts: List[str] = []
        self.deal_ids: List[int] = []
        self.outcomes: List[float] = []
        self.deal_email_counts: List[int] = []
        # Store email event objects so we can compute temporal features later
        all_email_events: List[Any] = []
        # Track start index in all_email_events for each deal
        deal_event_slices: List[tuple] = []

        for filename in self.deal_files:
            filepath = os.path.join(self.deals_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            timeline = DealTimelineModel(**data)

            emails = [e for e in timeline.events if e.type == "email"]
            if not emails:
                # Sentinel: use a placeholder text so SBERT still runs
                emails_texts = ["No communication."]
                # No real event objects for placeholder — use None list
                placeholder = True
            else:
                emails_texts = [email.content for email in emails]
                placeholder = False

            start_idx = len(all_email_events)
            self.deal_ids.append(timeline.deal_id)
            self.outcomes.append(1.0 if timeline.outcome == "won" else 0.0)
            self.deal_email_counts.append(len(emails_texts))

            for text in emails_texts:
                all_emails_texts.append(text)

            if not placeholder:
                for email in emails:
                    all_email_events.append(email)
                deal_event_slices.append((start_idx, start_idx + len(emails), False))
            else:
                # Placeholder: no event objects
                deal_event_slices.append((start_idx, start_idx, True))

        if not all_emails_texts:
            return

        # Run SBERT per-message extraction in a single batched operation
        from src.features.text_features import extract_sbert_embeddings_per_message

        all_embeddings = extract_sbert_embeddings_per_message(
            all_emails_texts, self.sbert_model_name, self.batch_size
        )

        # Segment the embeddings back to each deal sequence and concatenate
        # temporal features
        curr_emb_idx = 0
        for i, count in enumerate(self.deal_email_counts):
            sbert_emb = all_embeddings[curr_emb_idx : curr_emb_idx + count]
            curr_emb_idx += count

            event_start, event_end, is_placeholder = deal_event_slices[i]

            if is_placeholder:
                # Placeholder: temporal features are all zeros
                temporal_feats = np.zeros((1, TEMPORAL_FEATURE_DIM), dtype=np.float32)
            else:
                events_slice = all_email_events[event_start:event_end]
                temporal_feats = _compute_temporal_features(events_slice)

            # Concatenate: (seq_len, 384) + (seq_len, 3) → (seq_len, 387)
            combined = np.concatenate([sbert_emb, temporal_feats], axis=1)

            if HAS_TORCH:
                deal_emb_tensor = torch.tensor(combined, dtype=torch.float32)
                outcome_tensor = torch.tensor(self.outcomes[i], dtype=torch.float32)
            else:
                deal_emb_tensor = combined
                outcome_tensor = self.outcomes[i]

            self.data_items.append(
                {
                    "deal_id": self.deal_ids[i],
                    "outcome": outcome_tensor,
                    "embeddings": deal_emb_tensor,
                    "seq_len": count,
                }
            )

    def __len__(self) -> int:
        return len(self.data_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data_items[idx]


def collate_padded_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function to pad variable-length deal message sequences.

    Args:
        batch: List of dataset items, each being a dict containing:
               - "embeddings": Tensor/array of shape (seq_len, 387)
               - "deal_id": int
               - "outcome": float tensor/value
               - "seq_len": int

    Returns:
        Dict containing:
        - "embeddings": Padded tensor of shape (batch_size, max_seq_len, 387)
        - "lengths": LongTensor of shape (batch_size,) containing actual sequence lengths
        - "deal_ids": LongTensor of shape (batch_size,) containing deal identifiers
        - "outcomes": Tensor of shape (batch_size,) containing outcome labels
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. collate_padded_sequences cannot be executed."
        )

    from torch.nn.utils.rnn import pad_sequence

    # Sort the batch by sequence length in descending order (optional, but good practice)
    batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)

    embeddings = [item["embeddings"] for item in batch]
    deal_ids = torch.tensor([item["deal_id"] for item in batch], dtype=torch.long)
    outcomes = torch.tensor([item["outcome"] for item in batch], dtype=torch.float32)
    lengths = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)

    # Pad the sequence of embeddings
    # shape: (batch_size, max_seq_len, 387)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    return {
        "embeddings": padded_embeddings,
        "lengths": lengths,
        "deal_ids": deal_ids,
        "outcomes": outcomes,
    }
