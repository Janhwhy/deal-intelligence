# src/features/sequence_model.py: LSTM encoder module for Stream C.

import importlib.util
import logging

logger = logging.getLogger(__name__)

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class LSTMSequenceEncoder(nn.Module):
        """LSTM-based sequential encoder to model the trajectory of events in a deal.

        This module consumes variable-length sequences of per-message SBERT embeddings
        (384-dimensional) for a deal and encodes them into a fixed-size 128-dimensional
        trajectory vector.
        """

        def __init__(
            self,
            input_size: int = 384,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            """Encodes batch of padded sequences of embeddings.

            Args:
                x: Padded sequence tensor of shape (batch_size, max_seq_len, input_size).
                lengths: Tensor containing actual lengths of each sequence in the batch.

            Returns:
                The final hidden state representing the deal's trajectory vector of shape (batch_size, hidden_size).

            Design Decision (Final Hidden State vs. Pooling):
            We extract the final hidden state of the LSTM (from the last valid timestep of each sequence)
            as the deal's trajectory vector.
            - Final Hidden State: Captures the final sequential state after processing the entire chronological
              order of events. It naturally emphasizes the final state of the conversation (e.g., whether it
              ended positively or went quiet), which is highly indicative of HubSpot deal outcomes.
            - Alternatives (e.g., mean-pooling hidden states, attention-pooling): Mean-pooling can dilute
              temporally late signals by averaging them with early/irrelevant messages. Attention-pooling
              could learn to focus on specific key messages but adds model complexity. We use the final hidden
              state as a simple, standard starting point and document this choice as worth revisiting.
            """
            # Pack the padded sequence to skip computation on padded steps.
            # We pass enforce_sorted=False so that PyTorch handles sorting internally.
            # This is critical to avoid waste of compute resources and gradient corruption
            # due to processing padded zero vectors!
            packed_input = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # Output h_n shape: (num_layers, batch_size, hidden_size)
            _, (h_n, _) = self.lstm(packed_input)

            # Extract the final hidden state from the last layer
            # shape after select: (batch_size, hidden_size)
            final_state = h_n[-1]

            # Restore original batch order using packed_input.unsorted_indices.
            # When enforce_sorted=False, PyTorch sorts sequences internally;
            # LSTM output hidden states (h_n) remain sorted and must be unsorted manually.
            unsorted_indices = packed_input.unsorted_indices
            self.last_unsorted_indices = unsorted_indices
            self.last_sorted_indices = packed_input.sorted_indices

            final_state = final_state.index_select(
                0, unsorted_indices.to(final_state.device)
            )

            return final_state

else:

    class LSTMSequenceEncoder:
        """Stub class for LSTMSequenceEncoder when PyTorch is not installed."""

        def __init__(
            self,
            input_size: int = 384,
            hidden_size: int = 128,
            num_layers: int = 1,
            dropout: float = 0.0,
        ):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers

        def __call__(self, x, lengths):
            raise ImportError(
                "PyTorch is not installed. LSTMSequenceEncoder cannot be executed."
            )
