# src/fusion/__init__.py: Package initialization for multimodal data fusion modules.

from .fusion_model import (
    FusionProjector,
    align_and_concatenate_features,
    extract_lstm_features,
    get_fused_deal_vectors,
    project_features,
)

__all__ = [
    "FusionProjector",
    "extract_lstm_features",
    "align_and_concatenate_features",
    "project_features",
    "get_fused_deal_vectors",
]
