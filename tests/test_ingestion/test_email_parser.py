# tests/test_ingestion/test_email_parser.py: Tests for the email_parser module.

import os
from datetime import datetime

from src.ingestion.email_parser import clean_body, crawl_enron_emails, parse_email_file


def test_clean_body_strips_original_message():
    """Verifies that clean_body correctly removes Outlook style quoted replies."""
    raw_body = (
        "Let's finalize this deal.\n"
        "\n"
        "Thanks,\n"
        "John\n"
        " -----Original Message-----\n"
        "From: sales@external.com\n"
        "Sent: Monday, December 10, 2001 10:00 AM\n"
        "To: john.doe@enron.com\n"
        "Subject: RE: Price sheet\n"
        "\n"
        "Here are the prices.\n"
    )
    cleaned = clean_body(raw_body)
    assert cleaned == "Let's finalize this deal."


def test_clean_body_strips_signatures():
    """Verifies that clean_body strips standard salutations and phone signatures."""
    body_with_sig = (
        "We agree to the pricing terms.\n"
        "\n"
        "Best regards,\n"
        "Jane Smith\n"
        "VP Sales, Global Corp\n"
        "Phone: 555-0199\n"
    )
    cleaned = clean_body(body_with_sig)
    assert cleaned == "We agree to the pricing terms."


def test_clean_body_strips_inline_quotes():
    """Verifies that clean_body ignores lines prefixed with '>'."""
    body_with_quotes = (
        "This is my new comment.\n"
        "> Older quote line 1\n"
        "> Older quote line 2\n"
        "Hope this helps."
    )
    cleaned = clean_body(body_with_quotes)
    # The cleaned body should skip the lines with '>'
    assert "This is my new comment." in cleaned
    assert "Hope this helps." in cleaned
    assert "Older quote line" not in cleaned


def test_parse_email_file(mock_data_dir):
    """Verifies parse_email_file correctly extracts fields from a raw email file."""
    # Path to Email 1 created in mock_data_dir fixture
    email_path = os.path.join(
        mock_data_dir, "raw", "enron", "maildir", "lay-k", "inbox", "1."
    )

    parsed = parse_email_file(email_path)
    assert parsed is not None
    assert parsed["message_id"] == "<msg-1-lay@enron.com>"
    assert parsed["sender"] == "kenneth.lay@enron.com"
    assert "alice@acme.com" in parsed["recipients"]
    assert parsed["subject"] == "Partnership Proposal"
    assert isinstance(parsed["timestamp"], datetime)
    assert (
        parsed["cleaned_body"]
        == "Hi Alice,\n\nHere is our business proposal for Acme SaaS."
    )


def test_crawl_enron_emails(mock_data_dir):
    """Verifies crawl_enron_emails crawls the directory tree and yields all valid emails."""
    enron_raw_dir = os.path.join(mock_data_dir, "raw", "enron")
    emails = list(crawl_enron_emails(enron_raw_dir))

    # We wrote two emails in conftest.py
    assert len(emails) == 2
    senders = {e["sender"] for e in emails}
    assert "kenneth.lay@enron.com" in senders
    assert "alice@acme.com" in senders
