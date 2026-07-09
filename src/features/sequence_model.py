# src/features/sequence_model.py: LSTM encoder module for Stream C.

import importlib.util
import logging

logger = logging.getLogger(__name__)

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class LSTMSequenceEncoder(nn.Module):
        """LSTM-based sequential encoder to model the trajectory of events in a deal.

        This module consumes variable-length sequences of per-message feature vectors
        (SBERT embedding + temporal features, default 387-dimensional) for a deal and
        encodes them into a fixed-size trajectory vector.

        Pooling modes
        -------------
        - ``use_attention=False`` (default): Returns the final hidden state h_n[-1].
          Emphasises the final state of the conversation, which is useful when the
          last few messages are the most diagnostic.
        - ``use_attention=True``: Learns a soft attention weight over all hidden
          states and returns the attention-weighted mean.  This lets the model
          focus on the most predictive messages anywhere in the sequence rather
          than relying solely on the final state.  Preferred when sequences contain
          important signals in the middle (e.g. the moment the external party first
          replied).
        """

        def __init__(
            self,
            input_size: int = 386,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
            use_attention: bool = True,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_attention = use_attention

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            if use_attention:
                # Single-layer attention scorer: maps each hidden state to a scalar
                # weight, then softmax across the sequence to get a convex combination.
                self.attention_scorer = nn.Linear(hidden_size, 1, bias=False)

        def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            """Encodes a batch of padded sequences into deal trajectory vectors.

            Args:
                x: Padded sequence tensor of shape (batch_size, max_seq_len, input_size).
                lengths: Tensor of actual sequence lengths, shape (batch_size,).

            Returns:
                Trajectory tensor of shape (batch_size, hidden_size).
            """
            # Pack → avoids computing over padded zeros
            packed_input = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # output shape (packed), h_n shape: (num_layers, batch_size, hidden_size)
            packed_output, (h_n, _) = self.lstm(packed_input)

            if not self.use_attention:
                # Final hidden state from last LSTM layer
                return h_n[-1]  # (batch_size, hidden_size)

            # --- Attention pooling ---
            # Unpack to get all hidden states, padding positions zeroed
            # all_hidden: (batch_size, max_seq_len, hidden_size)
            all_hidden, _ = pad_packed_sequence(packed_output, batch_first=True)

            # Score each position: (batch_size, max_seq_len, 1)
            scores = self.attention_scorer(all_hidden)

            # Build padding mask so softmax ignores padded positions
            batch_size, max_len, _ = all_hidden.shape
            # mask: True at valid positions, False at padding
            mask = torch.arange(max_len, device=lengths.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            # Set padded positions to -inf before softmax
            scores = scores.squeeze(-1)  # (batch_size, max_seq_len)
            scores = scores.masked_fill(~mask, float("-inf"))

            # Normalised attention weights
            attn_weights = torch.softmax(scores, dim=1)  # (batch_size, max_seq_len)

            # Weighted sum of hidden states
            # (batch_size, 1, max_seq_len) @ (batch_size, max_seq_len, hidden_size)
            # → (batch_size, hidden_size)
            context = torch.bmm(attn_weights.unsqueeze(1), all_hidden).squeeze(1)
            return context  # (batch_size, hidden_size)

else:

    class LSTMSequenceEncoder:
        """Stub class for LSTMSequenceEncoder when PyTorch is not installed."""

        def __init__(
            self,
            input_size: int = 386,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
            use_attention: bool = True,
        ):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_attention = use_attention

        def __call__(self, x, lengths):
            raise ImportError(
                "PyTorch is not installed. LSTMSequenceEncoder cannot be executed."
            )
