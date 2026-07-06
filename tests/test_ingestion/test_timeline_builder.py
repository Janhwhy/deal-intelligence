# tests/test_ingestion/test_timeline_builder.py: Tests for the timeline_builder module.

import json
import os

from src.ingestion.deal_linker import link_deals
from src.ingestion.email_parser import crawl_enron_emails
from src.ingestion.timeline_builder import (
    DealTimelineModel,
    build_deal_timeline,
    save_deal_timeline,
)


def test_timeline_validation_and_saving(mock_data_dir, mock_app_config):
    """Verifies timelines can be built, validated via Pydantic, saved to JSON, and read back."""
    enron_raw_dir = os.path.join(mock_data_dir, "raw", "enron")
    emails = list(crawl_enron_emails(enron_raw_dir))
    deals = link_deals(emails, mock_app_config.data)

    assert (
        len(deals) == 1
    )  # Our test emails share a domain pair/subject group, so they form 1 deal
    deal_data = deals[0]

    # Build timeline model (validates with Pydantic)
    timeline = build_deal_timeline(deal_data)
    assert isinstance(timeline, DealTimelineModel)
    assert timeline.deal_id == 1
    assert len(timeline.events) > 0

    # Save to disk
    out_dir = mock_app_config.data.processed_deals_dir
    file_path = save_deal_timeline(timeline, out_dir)
    assert os.path.exists(file_path)

    # Read back and re-validate
    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    # Re-validate via Pydantic
    re_validated = DealTimelineModel(**loaded_data)
    assert re_validated.deal_id == timeline.deal_id
    assert len(re_validated.events) == len(timeline.events)
