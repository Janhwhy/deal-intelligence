# src/features/tabular_features.py: Tabular feature extraction and validation for deal timelines.

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.config import DataConfig
from src.ingestion.timeline_builder import DealTimelineModel

logger = logging.getLogger(__name__)

ALL_STAGES = [
    "Prospecting",
    "Demo Scheduled",
    "Negotiation",
    "Closed Won",
    "Closed Lost",
]


# --- Pydantic Schema for Feature Validation ---


class TabularFeaturesRow(BaseModel):
    deal_id: int
    touches: int = Field(..., ge=0)
    response_latency_avg: Optional[float] = Field(None, ge=0.0)
    response_latency_median: Optional[float] = Field(None, ge=0.0)
    engagement_asymmetry: Optional[float] = Field(None, ge=0.0)
    velocity_prospecting: Optional[float] = Field(None, ge=0.0)
    velocity_demo_scheduled: Optional[float] = Field(None, ge=0.0)
    velocity_negotiation: Optional[float] = Field(None, ge=0.0)
    velocity_closed_won: Optional[float] = Field(None, ge=0.0)
    velocity_closed_lost: Optional[float] = Field(None, ge=0.0)
    stakeholder_count: int = Field(..., ge=0)
    days_since_last_reply: Optional[float] = Field(None, ge=0.0)


# --- Feature Extraction Functions ---


def compute_deal_features(
    timeline: DealTimelineModel, as_of: Optional[datetime] = None
) -> Dict[str, Any]:
    """Computes tabular features for a single deal timeline as of a reference point.

    All computations only use events with timestamp <= as_of to prevent future leakage.

    Args:
        timeline: The validated DealTimelineModel.
        as_of: Optional reference timestamp. Defaults to now.

    Returns:
        A dictionary of computed features.
    """
    if as_of is None:
        as_of = datetime.now(timeline.events[0].timestamp.tzinfo)

    # Filter events to prevent leakage
    filtered_events = [e for e in timeline.events if e.timestamp <= as_of]

    # Email events only
    email_events = [e for e in filtered_events if e.type == "email"]

    # Touches: count of message events
    touches = len(email_events)

    # Engagement asymmetry & Response latency
    response_latency_avg = None
    response_latency_median = None
    engagement_asymmetry = None

    if touches > 0:
        seller_count = 0
        buyer_count = 0
        latencies = []

        last_side: Optional[str] = None
        last_ts: Optional[datetime] = None

        for email in email_events:
            sender = email.metadata.sender
            is_seller = sender.endswith("@enron.com")
            side = "seller" if is_seller else "buyer"

            if is_seller:
                seller_count += 1
            else:
                buyer_count += 1

            # Latency between opposite parties
            if last_side is not None and last_side != side and last_ts is not None:
                diff_days = (email.timestamp - last_ts).total_seconds() / 86400.0
                if diff_days >= 0:
                    latencies.append(diff_days)

            last_side = side
            last_ts = email.timestamp

        # Compute asymmetry ratio
        if seller_count == 0:
            engagement_asymmetry = float("inf") if buyer_count > 0 else 1.0
        else:
            engagement_asymmetry = buyer_count / seller_count

        # Compute latency metrics
        if latencies:
            response_latency_avg = float(np.mean(latencies))
            response_latency_median = float(np.median(latencies))

    # Stage Velocity
    stage_durations = {stage: 0.0 for stage in ALL_STAGES}
    stage_events = [e for e in filtered_events if e.type == "stage_change"]
    sorted_stages = sorted(stage_events, key=lambda x: x.timestamp)

    if sorted_stages:
        for i in range(len(sorted_stages)):
            current_event = sorted_stages[i]
            current_stage = current_event.metadata.to_stage
            start_ts = current_event.timestamp

            # End of stage is the next transition, the close date, or as_of (whichever comes first)
            if i < len(sorted_stages) - 1:
                end_ts = sorted_stages[i + 1].timestamp
            elif timeline.close_date is not None and timeline.close_date <= as_of:
                end_ts = timeline.close_date
            else:
                end_ts = as_of

            duration_days = max(0.0, (end_ts - start_ts).total_seconds() / 86400.0)
            stage_durations[current_stage] += duration_days

    # Stakeholder count: distinct participants in filtered email thread
    participants = set()
    for email in email_events:
        participants.add(email.metadata.sender)
        participants.update(email.metadata.recipients)
    stakeholder_count = len(participants)

    # Days since last reply
    days_since_last_reply = None
    if email_events:
        last_email_ts = email_events[-1].timestamp
        days_since_last_reply = max(
            0.0, (as_of - last_email_ts).total_seconds() / 86400.0
        )

    # Consolidate features
    features = {
        "deal_id": timeline.deal_id,
        "touches": touches,
        "response_latency_avg": response_latency_avg,
        "response_latency_median": response_latency_median,
        "engagement_asymmetry": engagement_asymmetry,
        "velocity_prospecting": stage_durations.get("Prospecting", 0.0),
        "velocity_demo_scheduled": stage_durations.get("Demo Scheduled", 0.0),
        "velocity_negotiation": stage_durations.get("Negotiation", 0.0),
        "velocity_closed_won": stage_durations.get("Closed Won", 0.0),
        "velocity_closed_lost": stage_durations.get("Closed Lost", 0.0),
        "stakeholder_count": stakeholder_count,
        "days_since_last_reply": days_since_last_reply,
    }

    return features


def validate_features_df(df: pd.DataFrame) -> None:
    """Validates types and value ranges of the features dataframe using Pydantic.

    Args:
        df: The tabular features dataframe.
    """
    records = df.reset_index().to_dict(orient="records")
    for r in records:
        # Convert NaN values to None for Pydantic validation compatibility
        cleaned = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        try:
            TabularFeaturesRow(**cleaned)
        except Exception as e:
            logger.error(
                f"Feature validation failed for deal row: {cleaned}. Error: {e}"
            )
            raise ValueError(f"Feature dataframe validation error: {e}")


def build_tabular_features(
    data_config: DataConfig, as_of: Optional[datetime] = None
) -> pd.DataFrame:
    """Loads all deal timeline JSON files, extracts tabular features, and saves them to parquet.

    Args:
        data_config: Loaded data configurations containing path inputs/outputs.
        as_of: Optional reference timestamp. Defaults to now.

    Returns:
        The validated feature DataFrame.
    """
    deals_dir = data_config.processed_deals_dir
    if not os.path.exists(deals_dir):
        logger.error(f"Processed deals directory does not exist: {deals_dir}")
        raise FileNotFoundError(f"Missing processed deals directory: {deals_dir}")

    timeline_files = [f for f in os.listdir(deals_dir) if f.endswith(".json")]
    if not timeline_files:
        logger.warning(f"No deal timeline files found in {deals_dir}")
        return pd.DataFrame()

    feature_rows = []
    for filename in timeline_files:
        filepath = os.path.join(deals_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            timeline = DealTimelineModel(**data)
            row = compute_deal_features(timeline, as_of)
            feature_rows.append(row)
        except Exception as e:
            logger.error(f"Failed to process timeline features for {filepath}: {e}")
            raise

    # Convert to DataFrame
    df = pd.DataFrame(feature_rows)
    if not df.empty:
        df.set_index("deal_id", inplace=True)
        # Validate data quality and ranges
        validate_features_df(df)

        # Save to Parquet format
        os.makedirs(os.path.dirname(data_config.processed_features_path), exist_ok=True)
        df.to_parquet(data_config.processed_features_path, engine="pyarrow")
        logger.info(
            f"Successfully saved {len(df)} deal feature rows to {data_config.processed_features_path}"
        )
    else:
        logger.warning("Empty features dataframe computed.")

    return df
