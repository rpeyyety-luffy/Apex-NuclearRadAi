"""Smoke test of the Streamlit dashboard (uses the saved model, training it once if missing)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "main.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_dashboard_runs_and_buttons_work():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert len(at.tabs) == 7
    _button(at, "Predict").click().run()
    assert not at.exception
    _button(at, "Simulate response").click().run()
    assert not at.exception
    at.selectbox[0].set_value("Stuck")       # scenario selector in tab B
    _button(at, "Generate scenario").click().run()
    assert not at.exception
    assert at.session_state.run.config.scenario == "Stuck"
    assert "detection" not in at.session_state      # old prediction discarded with the old run
    _button(at, "Run batch test").click().run()
    assert not at.exception
    assert len(at.session_state.batch["table"]) == 3 * 11
