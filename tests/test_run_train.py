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


@pytest.fixture
def sender_navigation_config_path(tmp_path):
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "game": {
            "size": 3,
            "n_landmarks": 1,
            "n_clues": 1,
            "n_questions": 0,
            "max_moves": 2,
            "history_len": 1,
            "instant_reward_multiplier": 0.0,
            "end_reward_multiplier": 0.0,
            "sender_shaping_multiplier": 1.0,
        },
        "training": {
            "training_epochs": 1,
            "n_episodes": 2,
            "batch_size": 2,
            "alpha": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "n_epochs": 1,
        },
        "agents": {
            "sender": {"kind": "ppo"},
            "receiver": {"kind": "copycat"},
        },
    }
    config_path = tmp_path / "sender_nav_config.json"
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


@pytest.fixture
def sender_navigation_convergence_config_path(tmp_path):
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "game": {
            "size": 3,
            "n_landmarks": 1,
            "n_clues": 1,
            "n_questions": 0,
            "max_moves": 10,
            "history_len": 1,
            "instant_reward_multiplier": 0.0,
            "end_reward_multiplier": 1.0,
            "sender_shaping_multiplier": 1.0,
        },
        "training": {
            "training_epochs": 1,
            "n_episodes": 1000,
            "batch_size": 64,
            "alpha": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "policy_clip": 0.2,
            "n_epochs": 4,
            "learn_interval": 64,
        },
        "agents": {
            "sender": {"kind": "ppo"},
            "receiver": {"kind": "copycat"},
        },
        "seeds": {"env_seed": 42, "sender_seed": 42},
    }
    config_path = tmp_path / "convergence_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


def test_run_train_sender_navigation(sender_navigation_config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_train.py",
            "--config",
            str(sender_navigation_config_path),
            "--output-root",
            str(tmp_path),
        ],
    )

    main()

    cfg = json.loads(sender_navigation_config_path.read_text())
    out_dir = Path(cfg["output_dir"])

    assert out_dir.exists()
    assert any(out_dir.rglob("*.pkl")), "Stats or model files missing"


def test_sender_navigation_converges(sender_navigation_convergence_config_path, tmp_path, monkeypatch):
    """Sender with allow_clue_on_landmark should learn to place the clue on the landmark,
    driving performance toward 1.0. Asserts that final mean performance is well above random (~0)."""
    run_name = "convergence_test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_train.py",
            "--config", str(sender_navigation_convergence_config_path),
            "--output-root", str(tmp_path),
            "--seed", "42",
            "--run-name", run_name,
            "--no-plots",
        ],
    )

    main()

    cfg = json.loads(sender_navigation_convergence_config_path.read_text())
    summary_path = Path(cfg["output_dir"]) / run_name / "summary.json"
    summary = json.loads(summary_path.read_text())
    final_perf = summary["stages"][0]["final_mean_performance_last_50"]

    assert final_perf > 0.5, (
        f"Sender navigation did not converge with allow_clue_on_landmark=True: "
        f"final mean performance over last 50 episodes = {final_perf:.3f}, expected > 0.5"
    )