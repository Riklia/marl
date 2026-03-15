import pytest
import sys
import json
from pathlib import Path

from run_train import main


@pytest.fixture
def mock_config_path(tmp_path):
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "game": {
            "size": 3,
            "n_landmarks": 1,
            "n_clues": 1,
            "n_questions": 1,
            "max_moves": 2,
            "history_len": 1,
            "instant_reward_multiplier": 1.0,
            "end_reward_multiplier": 1.0
        },
        "training": {
            "training_epochs": 1,
            "n_episodes": 1,
            "batch_size": 1,
            "alpha": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "n_epochs": 1,
            "seed": 123
        }
    }

    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


def test_run_train(mock_config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_train.py",
            "--config",
            str(mock_config_path),
            "--output-root",
            str(tmp_path),
        ],
    )

    main()

    cfg = json.loads(mock_config_path.read_text())
    out_dir = Path(cfg["output_dir"])

    assert out_dir.exists()
    assert any(out_dir.rglob("*.pkl")), "Stats or model files missing"
    assert any(out_dir.rglob("*.png")), "Plots missing"