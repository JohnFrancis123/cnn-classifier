from src.config import Config, load_config


def test_load_config_uses_defaults_for_empty_file(tmp_path): #Empty YAML should fall back to defaults.
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_config(config_path) == Config()


def test_load_config_overrides_values(tmp_path): #Explicit YAML values should override defaults.
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("epochs: 3\nbatch_size: 8\n", encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.epochs == 3
    assert cfg.batch_size == 8
    assert cfg.dataset == "cifar10"