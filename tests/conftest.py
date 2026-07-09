# tests/conftest.py: Shared fixtures for testing the ingestion and features modules.

import pytest

from src.config import AppConfig, DataConfig, FeaturesConfig, ModelConfig, TrainConfig


@pytest.fixture
def mock_data_dir(tmp_path):
    """Creates a temporary workspace with mock raw data files."""
    # Create directories
    raw_dir = tmp_path / "raw"
    enron_dir = raw_dir / "enron" / "maildir"
    hubspot_dir = raw_dir / "hubspot"

    enron_dir.mkdir(parents=True)
    hubspot_dir.mkdir(parents=True)

    # 1. Write HubSpot mock CSVs
    deals_data = (
        "deal_id,deal_name,amount,stage,close_date,company_id,contact_id\n"
        "1,Deal 1,150000.0,Closed Won,2025-01-01,10,100\n"
        "2,Deal 2,250000.0,Negotiation,2025-02-01,20,200\n"
    )
    companies_data = (
        "company_id,company_name,industry,annual_revenue,num_employees,country\n"
        "10,Acme SaaS,SaaS,5000000.0,42,United States\n"
        "20,Global Fintech,Fintech,12000000.0,150,United Kingdom\n"
    )
    contacts_data = (
        "contact_id,first_name,last_name,email,phone,job_title,company_id\n"
        "100,Alice,Green,alice@acme.com,111-222,CEO,10\n"
        "200,Bob,Brown,bob@global.com,333-442,VP Sales,20\n"
    )

    (hubspot_dir / "deals.csv").write_text(deals_data)
    (hubspot_dir / "companies.csv").write_text(companies_data)
    (hubspot_dir / "contacts.csv").write_text(contacts_data)

    # 2. Write Fake Enron emails
    # Employee: lay-k
    lay_inbox = enron_dir / "lay-k" / "inbox"
    lay_inbox.mkdir(parents=True)

    # Email 1: From seller to buyer (start of thread)
    email1 = (
        "Message-ID: <msg-1-lay@enron.com>\n"
        "Date: Mon, 10 Dec 2001 08:00:00 -0800 (PST)\n"
        "From: kenneth.lay@enron.com\n"
        "To: alice@acme.com\n"
        "Subject: Partnership Proposal\n"
        "\n"
        "Hi Alice,\n"
        "\n"
        "Here is our business proposal for Acme SaaS.\n"
        "\n"
        "Thanks,\n"
        "Ken Lay\n"
    )
    (lay_inbox / "1.").write_text(email1)

    # Email 2: From buyer to seller (reply)
    email2 = (
        "Message-ID: <msg-2-alice@acme.com>\n"
        "Date: Tue, 11 Dec 2001 09:30:00 -0800 (PST)\n"
        "From: alice@acme.com\n"
        "To: kenneth.lay@enron.com\n"
        "Subject: RE: Partnership Proposal\n"
        "\n"
        "Hi Ken,\n"
        "\n"
        "Thanks for the proposal. I will review it today.\n"
        "\n"
        "Regards,\n"
        "Alice Green\n"
        " -----Original Message-----\n"
        "From: \tkenneth.lay@enron.com\n"
        "Sent:\tMon, 10 Dec 2001 08:00:00 -0800 (PST)\n"
        "To:\talice@acme.com\n"
        "Subject:\tPartnership Proposal\n"
    )
    (lay_inbox / "2.").write_text(email2)

    return tmp_path


@pytest.fixture
def mock_app_config(mock_data_dir):
    """Provides a validated AppConfig with mock data folder paths."""
    raw_dir = mock_data_dir / "raw"
    processed_dir = mock_data_dir / "processed"
    processed_deals_dir = processed_dir / "deals"
    processed_features_path = processed_dir / "tabular_features.parquet"

    # Make processed folders
    processed_deals_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        enron_raw_dir=str(raw_dir / "enron"),
        hubspot_deals_csv=str(raw_dir / "hubspot" / "deals.csv"),
        hubspot_companies_csv=str(raw_dir / "hubspot" / "companies.csv"),
        hubspot_contacts_csv=str(raw_dir / "hubspot" / "contacts.csv"),
        deal_linker_seed=42,
        subject_similarity_threshold=0.5,
        time_proximity_window_days=7,
        processed_deals_dir=str(processed_deals_dir),
        processed_features_path=str(processed_features_path),
        deal_relevance_keywords_path="src/ingestion/resources/deal_relevance_keywords.txt",
    )

    features_cfg = FeaturesConfig(
        sbert_model_name="all-MiniLM-L6-v2",
        roberta_sentiment_model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        bertopic_min_topic_size=10,
        batch_size=32,
        hedge_words_resource_path="src/features/resources/hedge_words.txt",
        processed_text_features_path=str(processed_dir / "text_features.parquet"),
        bertopic_model_dir=str(processed_dir / "models" / "bertopic_model"),
    )

    return AppConfig(
        data=data_cfg,
        model=ModelConfig(
            lstm_hidden_size=128, lstm_num_layers=1, lstm_seed=42, lstm_dropout=0.0
        ),
        train=TrainConfig(batch_size=32, learning_rate=0.001),
        features=features_cfg,
    )
