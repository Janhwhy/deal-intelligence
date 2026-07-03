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
        "raw_enron_path: 'custom/enron/path'\nraw_hubspot_path: 'custom/hubspot/path'\n"
    )
    model_yaml.write_text("lstm_hidden_size: 256\nlstm_num_layers: 3\n")
    train_yaml.write_text("batch_size: 64\nlearning_rate: 0.05\n")

    # Load configuration from the custom folder path
    cfg = load_config(config_dir=str(configs_dir))

    # Verify that the loaded settings reflect the changes made in YAML
    assert cfg.data.raw_enron_path == "custom/enron/path"
    assert cfg.data.raw_hubspot_path == "custom/hubspot/path"
    assert cfg.model.lstm_hidden_size == 256
    assert cfg.model.lstm_num_layers == 3
    assert cfg.train.batch_size == 64
    assert cfg.train.learning_rate == 0.05
