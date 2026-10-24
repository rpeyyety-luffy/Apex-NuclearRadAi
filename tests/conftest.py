import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_training  # noqa: E402

SMALL_RUNS = {"train": 4, "validation": 2, "test": 1}


@pytest.fixture(scope="session")
def small_bundle(tmp_path_factory):
    """A small model trained into a temporary folder (the real models/ folder is not touched)."""
    directory = tmp_path_factory.mktemp("model")
    bundle = model_training.load_or_train(retrain=True, directory=directory, runs=SMALL_RUNS, n_estimators=40)
    bundle["_directory"] = directory
    return bundle
