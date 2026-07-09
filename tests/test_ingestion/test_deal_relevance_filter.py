# tests/test_ingestion/test_deal_relevance_filter.py: Tests for the content-relevance filter functions.

import pytest

from src.ingestion.deal_linker import (
    is_deal_relevant,
    is_external_email,
    load_deal_relevance_keywords,
)

# ---------------------------------------------------------------------------
# Tests for is_external_email
# ---------------------------------------------------------------------------


def test_is_external_email_cross_domain():
    """Cross-domain email (enron.com -> acme.com) should be marked external."""
    assert is_external_email("alice@enron.com", ["bob@acme.com"]) is True


def test_is_external_email_same_domain():
    """Internal-only email (enron.com -> enron.com) should NOT be external."""
    assert is_external_email("alice@enron.com", ["bob@enron.com"]) is False


def test_is_external_email_mixed_recipients():
    """Email with one internal and one external recipient should be external."""
    assert (
        is_external_email("alice@enron.com", ["bob@enron.com", "carol@buyer.com"])
        is True
    )


def test_is_external_email_no_at_sign_sender():
    """Sender without '@' should not be treated as external."""
    assert is_external_email("noemail", ["bob@acme.com"]) is False


def test_is_external_email_empty_recipients():
    """Email with no recipients should not be external."""
    assert is_external_email("alice@enron.com", []) is False


def test_is_external_email_case_insensitive_domain():
    """Domain comparison should be case-insensitive."""
    assert is_external_email("alice@ENRON.COM", ["bob@ENRON.COM"]) is False
    assert is_external_email("alice@ENRON.COM", ["bob@ACME.COM"]) is True


# ---------------------------------------------------------------------------
# Tests for is_deal_relevant
# ---------------------------------------------------------------------------


@pytest.fixture
def deal_keywords():
    return ["proposal", "contract", "pricing", "quote", "rfp"]


def test_is_deal_relevant_keyword_in_subject(deal_keywords):
    """Email with keyword in subject should be deal-relevant."""
    email = {"subject": "RE: Pricing Proposal for Q3", "content": "See attached."}
    assert is_deal_relevant(email, deal_keywords) is True


def test_is_deal_relevant_keyword_in_body(deal_keywords):
    """Email with keyword in body should be deal-relevant."""
    email = {
        "subject": "Follow Up",
        "content": "Please review the attached contract details.",
    }
    assert is_deal_relevant(email, deal_keywords) is True


def test_is_deal_relevant_no_keyword(deal_keywords):
    """Email with no keyword in subject or body should NOT be deal-relevant."""
    email = {
        "subject": "Team lunch today",
        "content": "Let's meet at noon. See you there!",
    }
    assert is_deal_relevant(email, deal_keywords) is False


def test_is_deal_relevant_case_insensitive(deal_keywords):
    """Keyword matching should be case-insensitive."""
    email = {"subject": "RFP Response", "content": "Attached is our formal QUOTE."}
    assert is_deal_relevant(email, deal_keywords) is True


def test_is_deal_relevant_empty_content(deal_keywords):
    """Email with empty content but keyword in subject should still match."""
    email = {"subject": "New Contract Draft", "content": ""}
    assert is_deal_relevant(email, deal_keywords) is True


def test_is_deal_relevant_empty_email(deal_keywords):
    """Email with both empty subject and content should not match."""
    email = {"subject": "", "content": ""}
    assert is_deal_relevant(email, deal_keywords) is False


def test_is_deal_relevant_missing_keys(deal_keywords):
    """Email dict missing 'subject' and 'content' keys should not crash and return False."""
    email = {}
    assert is_deal_relevant(email, deal_keywords) is False


# ---------------------------------------------------------------------------
# Tests for load_deal_relevance_keywords
# ---------------------------------------------------------------------------


def test_load_deal_relevance_keywords_from_file(tmp_path):
    """Keywords should be loaded correctly from a valid file."""
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text("proposal\n# comment line\ncontract\n\npricing\n")
    keywords = load_deal_relevance_keywords(str(keyword_file))
    assert "proposal" in keywords
    assert "contract" in keywords
    assert "pricing" in keywords
    # Comments and blank lines should be excluded
    assert "# comment line" not in keywords
    assert "" not in keywords


def test_load_deal_relevance_keywords_fallback_on_missing_file():
    """Missing file should trigger fallback list without crashing."""
    keywords = load_deal_relevance_keywords("/nonexistent/path/keywords.txt")
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    # Fallback list should include core sales keywords
    assert "proposal" in keywords


def test_load_deal_relevance_keywords_empty_file(tmp_path):
    """Empty file should trigger fallback list."""
    keyword_file = tmp_path / "empty_keywords.txt"
    keyword_file.write_text("")
    keywords = load_deal_relevance_keywords(str(keyword_file))
    assert isinstance(keywords, list)
    assert len(keywords) > 0
