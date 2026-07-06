# tests/test_ingestion/test_deal_linker.py: Tests for the deal_linker module.

import os

from src.ingestion.deal_linker import link_deals
from src.ingestion.email_parser import crawl_enron_emails


def test_deal_linker_determinism(mock_data_dir, mock_app_config):
    """Verifies that running the deal linker twice with the same inputs yields identical results."""
    enron_raw_dir = os.path.join(mock_data_dir, "raw", "enron")
    emails = list(crawl_enron_emails(enron_raw_dir))

    # Run deal linking twice
    deals_first = link_deals(emails, mock_app_config.data)
    deals_second = link_deals(emails, mock_app_config.data)

    # Must find the same number of deals
    assert len(deals_first) == len(deals_second)
    assert len(deals_first) > 0

    for d1, d2 in zip(deals_first, deals_second):
        assert d1["deal_id"] == d2["deal_id"]
        assert d1["stage"] == d2["stage"]
        assert d1["outcome"] == d2["outcome"]
        assert d1["amount"] == d2["amount"]
        assert d1["company_id"] == d2["company_id"]
        assert d1["company_name"] == d2["company_name"]
        assert d1["close_date"] == d2["close_date"]

        # Verify emails list
        assert len(d1["emails"]) == len(d2["emails"])
        for e1, e2 in zip(d1["emails"], d2["emails"]):
            assert e1["message_id"] == e2["message_id"]

        # Verify contacts list
        assert len(d1["contacts"]) == len(d2["contacts"])
        for c1, c2 in zip(d1["contacts"], d2["contacts"]):
            assert c1["contact_id"] == c2["contact_id"]
            assert c1["email"] == c2["email"]

        # Verify stage transitions list
        assert len(d1["stage_transitions"]) == len(d2["stage_transitions"])
        for t1, t2 in zip(d1["stage_transitions"], d2["stage_transitions"]):
            assert t1["timestamp"] == t2["timestamp"]
            assert t1["from_stage"] == t2["from_stage"]
            assert t1["to_stage"] == t2["to_stage"]
