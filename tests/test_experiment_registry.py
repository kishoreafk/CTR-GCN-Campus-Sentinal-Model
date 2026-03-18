"""Tests for utils/experiment_registry.py"""
import pytest
import json
from pathlib import Path
from utils.experiment_registry import ExperimentRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a fresh registry in a temp directory."""
    return ExperimentRegistry(str(tmp_path / "registry.json"))


def test_register_creates_entry(registry):
    run_id = registry.register(
        run_id="test_run_001",
        phase="phase1",
        config={"lr_backbone": 1e-4, "batch_size": 64},
        class_names=["eat", "walk"],
        class_ids=[17, 49],
    )
    assert run_id == "test_run_001"

    # Verify it's in the file
    data = json.loads(registry.path.read_text())
    assert len(data) == 1
    assert data[0]["run_id"] == "test_run_001"
    assert data[0]["status"] == "running"


def test_update_final(registry):
    registry.register("run_002", "phase2",
                      config={}, class_names=["eat"], class_ids=[17])
    registry.update_final(
        run_id="run_002",
        best_mAP=0.7512,
        per_class_AP={"eat": 0.7512},
        best_epoch=25,
        total_epochs=50,
        checkpoint_path="/path/to/best.pth",
    )

    data = json.loads(registry.path.read_text())
    assert data[0]["status"] == "completed"
    assert data[0]["best_mAP"] == 0.7512
    assert data[0]["best_epoch"] == 25
    assert data[0]["finished_at"] is not None


def test_find_best_run(registry):
    # Register two runs with different mAP
    registry.register("run_a", "phase1", config={},
                      class_names=["eat", "walk"], class_ids=[17, 49])
    registry.update_final("run_a", 0.65, {"eat": 0.7, "walk": 0.6},
                         10, 50, "/a/best.pth")

    registry.register("run_b", "phase1", config={},
                      class_names=["eat", "walk"], class_ids=[17, 49])
    registry.update_final("run_b", 0.80, {"eat": 0.85, "walk": 0.75},
                         20, 50, "/b/best.pth")

    best = registry.find_best_run()
    assert best["run_id"] == "run_b"
    assert best["best_mAP"] == 0.80


def test_find_best_run_with_filters(registry):
    registry.register("run_1", "phase1", config={},
                      class_names=["eat"], class_ids=[17])
    registry.update_final("run_1", 0.90, {}, 5, 10, "/x.pth")

    registry.register("run_2", "phase2", config={},
                      class_names=["walk"], class_ids=[49])
    registry.update_final("run_2", 0.95, {}, 5, 10, "/y.pth")

    # Filter by phase
    best = registry.find_best_run(phase="phase1")
    assert best["run_id"] == "run_1"

    # Filter by class names
    best = registry.find_best_run(class_names=["walk"])
    assert best["run_id"] == "run_2"

    # No match
    best = registry.find_best_run(class_names=["nonexistent"])
    assert best is None


def test_find_best_run_no_completed(registry):
    registry.register("run_x", "phase1", config={},
                      class_names=["eat"], class_ids=[17])
    # Don't update_final — still "running"
    assert registry.find_best_run() is None


def test_print_summary(registry, capsys):
    registry.register("run_p", "phase1", config={},
                      class_names=["eat", "walk", "run", "jump"],
                      class_ids=[17, 49, 50, 51])
    registry.update_final("run_p", 0.55, {}, 3, 10, "/z.pth")

    registry.print_summary(last_n=5)
    captured = capsys.readouterr().out
    assert "run_p" in captured
    assert "0.5500" in captured


def test_multiple_registries_same_file(tmp_path):
    path = str(tmp_path / "shared.json")
    r1 = ExperimentRegistry(path)
    r1.register("from_r1", "phase1", config={},
                class_names=["a"], class_ids=[1])

    r2 = ExperimentRegistry(path)
    r2.register("from_r2", "phase2", config={},
                class_names=["b"], class_ids=[2])

    data = json.loads(Path(path).read_text())
    assert len(data) == 2
