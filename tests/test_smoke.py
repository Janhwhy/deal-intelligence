# tests/test_smoke.py: Verification of imports and configuration loading.

from src.config import load_config


def test_imports():
    """Asserts that all subpackages under src/ are importable."""
    import src.eval
    import src.explainability
    import src.features
    import src.fusion
    import src.ingestion
    import src.models

    assert src.ingestion is not None
    assert src.features is not None
    assert src.fusion is not None
    assert src.models is not None
    assert src.explainability is not None
    assert src.eval is not None


def test_config_load():
    """Asserts that the config loader parses files."""
    cfg = load_config()
    assert cfg is not None
    assert cfg.data.enron_raw_dir == "data/raw/enron"
    assert cfg.data.hubspot_deals_csv == "data/raw/hubspot/deals.csv"
    assert cfg.model.lstm_hidden_size == 128
    assert cfg.model.lstm_num_layers == 1
    assert cfg.model.lstm_seed == 42
    assert cfg.model.lstm_dropout == 0.0
    assert cfg.train.batch_size == 32
    assert cfg.train.learning_rate == 0.001
