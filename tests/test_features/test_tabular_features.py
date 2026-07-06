# tests/test_features/test_tabular_features.py: Tests for the tabular_features module.

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.ingestion.timeline_builder import (
    DealTimelineModel,
    ContactModel,
    EmailEventModel,
    EmailMetadataModel,
    StageChangeEventModel,
    StageChangeMetadataModel,
)
from src.features.tabular_features import compute_deal_features, validate_features_df


def _build_test_timeline() -> DealTimelineModel:
    """Helper to build a handcrafted deal timeline for precise calculations."""
    return DealTimelineModel(
        deal_id=42,
        stage="Negotiation",
        outcome="open",
        amount=100000.0,
        close_date=None,
        company_id=10,
        company_name="Test Company",
        industry="SaaS",
        annual_revenue=1000000.0,
        num_employees=10,
        country="USA",
        contacts=[
            ContactModel(
                contact_id=100,
                first_name="John",
                last_name="Doe",
                email="john@test.com",
                phone="123",
                job_title="CEO",
            )
        ],
        events=[
            # Day 1: Email 1 (Seller to Buyer)
            EmailEventModel(
                timestamp=datetime(2001, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                type="email",
                content="Proposal",
                metadata=EmailMetadataModel(
                    sender="alice@enron.com",
                    recipients=["john@test.com"],
                    subject="Deal",
                    message_id="m1",
                ),
            ),
            # Day 1: Stage Change to Prospecting
            StageChangeEventModel(
                timestamp=datetime(2001, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                type="stage_change",
                content="Prospecting",
                metadata=StageChangeMetadataModel(
                    from_stage=None, to_stage="Prospecting"
                ),
            ),
            # Day 2: Email 2 (Buyer reply to Seller) - 1 day later
            EmailEventModel(
                timestamp=datetime(2001, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
                type="email",
                content="Let's talk",
                metadata=EmailMetadataModel(
                    sender="john@test.com",
                    recipients=["alice@enron.com"],
                    subject="RE: Deal",
                    message_id="m2",
                ),
            ),
            # Day 2: Stage Change to Demo Scheduled
            StageChangeEventModel(
                timestamp=datetime(2001, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
                type="stage_change",
                content="Demo Scheduled",
                metadata=StageChangeMetadataModel(
                    from_stage="Prospecting", to_stage="Demo Scheduled"
                ),
            ),
            # Day 3: Email 3 (Seller reply to Buyer) - 1 day later
            EmailEventModel(
                timestamp=datetime(2001, 1, 3, 10, 0, 0, tzinfo=timezone.utc),
                type="email",
                content="Confirmed",
                metadata=EmailMetadataModel(
                    sender="alice@enron.com",
                    recipients=["john@test.com"],
                    subject="RE: Deal",
                    message_id="m3",
                ),
            ),
            # Day 3: Stage Change to Negotiation
            StageChangeEventModel(
                timestamp=datetime(2001, 1, 3, 10, 0, 0, tzinfo=timezone.utc),
                type="stage_change",
                content="Negotiation",
                metadata=StageChangeMetadataModel(
                    from_stage="Demo Scheduled", to_stage="Negotiation"
                ),
            ),
        ],
    )


def test_handcalculated_feature_values():
    """Verifies that features are computed exactly as hand-calculated at a specific reference date."""
    timeline = _build_test_timeline()

    # We evaluate "as of" Day 2, 12:00:00 UTC (two hours after email 2 / stage change 2, and before day 3 events)
    as_of = datetime(2001, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    features = compute_deal_features(timeline, as_of=as_of)

    # 1. Touches: Email 1 and Email 2 are within range. Email 3 is in future.
    assert features["touches"] == 2

    # 2. Response Latency:
    # Transition from Email 1 (seller) to Email 2 (buyer) took exactly 1.0 day.
    # Latencies list: [1.0]
    assert features["response_latency_avg"] == 1.0
    assert features["response_latency_median"] == 1.0

    # 3. Engagement Asymmetry:
    # 1 Buyer email (john@test.com), 1 Seller email (alice@enron.com).
    # Ratio = 1.0 / 1.0 = 1.0
    assert features["engagement_asymmetry"] == 1.0

    # 4. Stakeholder Count:
    # alice@enron.com and john@test.com = 2 stakeholders
    assert features["stakeholder_count"] == 2

    # 5. Days since last reply:
    # as_of is 2001-01-02 12:00:00. Last email is Email 2 at 2001-01-02 10:00:00.
    # Difference = 2 hours = 2/24 = 0.083333... days.
    assert np.isclose(features["days_since_last_reply"], 2.0 / 24.0)

    # 6. Stage Velocities:
    # - Prospecting: from 2001-01-01 10:00:00 to 2001-01-02 10:00:00 = exactly 1.0 day.
    # - Demo Scheduled: from 2001-01-02 10:00:00 to as_of (2001-01-02 12:00:00) = 2 hours = 0.083333 days.
    # - Negotiation: Not yet reached (0.0).
    assert np.isclose(features["velocity_prospecting"], 1.0)
    assert np.isclose(features["velocity_demo_scheduled"], 2.0 / 24.0)
    assert features["velocity_negotiation"] == 0.0


def test_no_future_leakage():
    """Verifies that adding future events to the timeline does not change features computed 'as of' past date."""
    timeline = _build_test_timeline()

    # Reference point: Day 2, 12:00:00 UTC
    as_of = datetime(2001, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    # Features computed with the full timeline (which contains Day 3 events)
    features_with_future = compute_deal_features(timeline, as_of=as_of)

    # Now we strip out Day 3 events manually to simulate the historical timeline as of Day 2
    timeline_past_only = _build_test_timeline()
    timeline_past_only.events = timeline_past_only.events[:-2]  # Remove Day 3 email and stage change

    features_past_only = compute_deal_features(timeline_past_only, as_of=as_of)

    # The features computed as of Day 2 must be IDENTICAL, proving zero leakage from Day 3 events
    for key in features_past_only:
        assert features_with_future[key] == features_past_only[key]


def test_dataframe_validation_error():
    """Verifies that validate_features_df fails loudly on out-of-range value."""
    # Create invalid data (touches is negative)
    invalid_data = [{
        "deal_id": 1,
        "touches": -1,  # Invalid (must be >= 0)
        "response_latency_avg": 1.0,
        "response_latency_median": 1.0,
        "engagement_asymmetry": 1.0,
        "velocity_prospecting": 1.0,
        "velocity_demo_scheduled": 0.0,
        "velocity_negotiation": 0.0,
        "velocity_closed_won": 0.0,
        "velocity_closed_lost": 0.0,
        "stakeholder_count": 2,
        "days_since_last_reply": 0.0,
    }]
    df = pd.DataFrame(invalid_data).set_index("deal_id")
    
    import pytest
    with pytest.raises(ValueError):
        validate_features_df(df)
