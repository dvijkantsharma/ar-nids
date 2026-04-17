from ar_nids.config import load_config


def test_load_config_has_expected_defaults():
    config = load_config("configs/default.yaml")
    assert config.feature_count == 80
    assert config.adversarial.steps == 10
    assert config.num_classes == 6
