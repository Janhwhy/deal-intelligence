# src/features/sequence_dataset.py: PyTorch Dataset & DataLoader collate helper for Stream C.

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List

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


class DealSequenceDataset(Dataset):
    """Dataset that loads deal timelines, extracts per-message SBERT embeddings,

    and returns sequence representations.
    """

    def __init__(self, deals_dir: str, sbert_model_name: str, batch_size: int = 32):
        self.deals_dir = deals_dir
        self.sbert_model_name = sbert_model_name
        self.batch_size = batch_size

        if not os.path.exists(deals_dir):
            raise FileNotFoundError(f"Deals directory does not exist: {deals_dir}")

        # Load all timelines
        self.deal_files = sorted(
            [f for f in os.listdir(deals_dir) if f.endswith(".json")]
        )
        self.data_items: List[Dict[str, Any]] = []
        self._load_and_process_timelines()

    def _load_and_process_timelines(self) -> None:
        """Crawl all timelines, build a corpus of all emails, extract embeddings,

        and segment them back into chronological sequences per deal.
        """
        all_emails: List[str] = []
        self.deal_ids: List[int] = []
        self.outcomes: List[float] = []
        self.deal_email_counts: List[int] = []

        for filename in self.deal_files:
            filepath = os.path.join(self.deals_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            timeline = DealTimelineModel(**data)

            emails = [e for e in timeline.events if e.type == "email"]
            if not emails:
                emails_texts = ["No communication."]
            else:
                emails_texts = [email.content for email in emails]

            self.deal_ids.append(timeline.deal_id)
            self.outcomes.append(1.0 if timeline.outcome == "won" else 0.0)
            self.deal_email_counts.append(len(emails_texts))

            for text in emails_texts:
                all_emails.append(text)

        if not all_emails:
            return

        # Run SBERT per-message extraction in a single batched operation
        from src.features.text_features import extract_sbert_embeddings_per_message

        all_embeddings = extract_sbert_embeddings_per_message(
            all_emails, self.sbert_model_name, self.batch_size
        )

        # Segment the embeddings back to each deal sequence
        curr_idx = 0
        for i, count in enumerate(self.deal_email_counts):
            deal_emb = all_embeddings[curr_idx : curr_idx + count]
            curr_idx += count

            if HAS_TORCH:
                deal_emb_tensor = torch.tensor(deal_emb, dtype=torch.float32)
                outcome_tensor = torch.tensor(self.outcomes[i], dtype=torch.float32)
            else:
                deal_emb_tensor = deal_emb
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
               - "embeddings": Tensor/array of shape (seq_len, 384)
               - "deal_id": int
               - "outcome": float tensor/value
               - "seq_len": int

    Returns:
        Dict containing:
        - "embeddings": Padded tensor of shape (batch_size, max_seq_len, 384)
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
    # shape: (batch_size, max_seq_len, 384)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    return {
        "embeddings": padded_embeddings,
        "lengths": lengths,
        "deal_ids": deal_ids,
        "outcomes": outcomes,
    }
