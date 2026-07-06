# tests/test_config.py: Verifies config loading and overrides.

from src.config import load_config


def test_config_override_with_custom_yaml(tmp_path):
    """Proves that editing the configuration YAML files changes the loaded configuration
    object properties without modification to any Python source files.
    """
    # Create a temporary configuration directory structure
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    # Write custom configurations to temporary yaml files
    data_yaml = configs_dir / "data.yaml"
    model_yaml = configs_dir / "model.yaml"
    train_yaml = configs_dir / "train.yaml"

    data_yaml.write_text(
        "enron_raw_dir: 'custom/enron/path'\n"
        "hubspot_deals_csv: 'custom/deals.csv'\n"
        "hubspot_companies_csv: 'custom/companies.csv'\n"
        "hubspot_contacts_csv: 'custom/contacts.csv'\n"
        "deal_linker_seed: 42\n"
        "subject_similarity_threshold: 0.5\n"
        "time_proximity_window_days: 7\n"
        "processed_deals_dir: 'custom/deals'\n"
        "processed_features_path: 'custom/features.parquet'\n"
    )
    model_yaml.write_text("lstm_hidden_size: 256\nlstm_num_layers: 3\n")
    train_yaml.write_text("batch_size: 64\nlearning_rate: 0.05\n")

    # Load configuration from the custom folder path
    cfg = load_config(config_dir=str(configs_dir))

    # Verify that the loaded settings reflect the changes made in YAML
    assert cfg.data.enron_raw_dir == "custom/enron/path"
    assert cfg.data.hubspot_deals_csv == "custom/deals.csv"
    assert cfg.data.hubspot_companies_csv == "custom/companies.csv"
    assert cfg.data.hubspot_contacts_csv == "custom/contacts.csv"
    assert cfg.data.deal_linker_seed == 42
    assert cfg.data.subject_similarity_threshold == 0.5
    assert cfg.data.time_proximity_window_days == 7
    assert cfg.data.processed_deals_dir == "custom/deals"
    assert cfg.data.processed_features_path == "custom/features.parquet"
    assert cfg.model.lstm_hidden_size == 256
    assert cfg.model.lstm_num_layers == 3
    assert cfg.train.batch_size == 64
    assert cfg.train.learning_rate == 0.05
