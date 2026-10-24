import numpy as np
import pandas as pd

import batch_testing as bt
import fault_detection as fd
import fault_response as fr
import model_training
import nuclearrad as nr


def test_bundle_contents_and_split_bookkeeping(small_bundle):
    meta = small_bundle["meta"]
    assert meta["version"] == model_training.model_version()
    assert meta["n_runs"] == {"train": 4 * len(nr.SIM_SCENARIOS), "validation": 2 * len(nr.SIM_SCENARIOS),
                              "test": 1 * len(nr.SIM_SCENARIOS)}
    assert 1 <= small_bundle["min_flagged"] <= max(model_training.MIN_FLAGGED_CANDIDATES)
    assert set(small_bundle["classes"]) <= set(nr.CLASSES)
    assert len(small_bundle["test_table"]) == meta["n_runs"]["test"]
    assert set(small_bundle["test_table"]["split"]) == {"test"}


def test_saved_model_is_loaded_not_retrained(small_bundle, monkeypatch):
    directory = small_bundle["_directory"]
    monkeypatch.setattr(model_training, "train", lambda **k: (_ for _ in ()).throw(AssertionError("retrained")))
    loaded = model_training.load_or_train(directory=directory)
    assert loaded["meta"]["trained_at"] == small_bundle["meta"]["trained_at"]


def test_version_mismatch_forces_retraining(small_bundle, monkeypatch):
    monkeypatch.setattr(model_training, "model_version", lambda: "something-else")
    assert model_training.load(small_bundle["_directory"]) is None


def test_prediction_output(small_bundle):
    run = nr.simulate(nr.ScenarioConfig(scenario="Stuck", seed=123, fault_sensor=3))
    det = fd.predict_run(small_bundle, run.readings, run.config.warmup)
    assert det["proba"].shape == run.readings.shape + (len(nr.CLASSES),)
    np.testing.assert_allclose(det["proba"][run.config.warmup:].sum(axis=2), 1.0)
    assert det["category"] in nr.RUN_CATEGORIES or "+" in det["category"]
    for s in det["sensors"]:
        assert 0 <= s["confidence"] <= 1 and s["fault"] in nr.INSTRUMENT_FAULTS


def test_prediction_ignores_truth_and_future(small_bundle):
    run = nr.simulate(nr.ScenarioConfig(scenario="Drift (upward)", seed=8))
    a = fd.predict_observations(small_bundle, run.readings, 40)
    future = run.readings.copy()
    future[150:] += 50
    b = fd.predict_observations(small_bundle, future, 40)
    np.testing.assert_allclose(a[:150], b[:150], rtol=0, atol=1e-12)   # parallel tree sums differ by 1e-16


def _detection(T, S, sensors, warmup=40, process=False):
    flags = np.zeros((T, S), bool)
    for s, start in sensors.items():
        flags[start:, s] = True
    found = [dict(sensor=s, name=nr.NAMES[s], fault="Drift", flagged_samples=T - start, first_detection=start,
                  confidence=0.9) for s, start in sensors.items()]
    return dict(category="Drift", fault_present=bool(found), process_change=process, sensors=found,
                sample_flags=flags, notes=[])


def test_correction_uses_only_trusted_peers():
    rng = np.random.default_rng(0)
    x = 100 * (1 + 0.01 * rng.standard_normal((120, 5)))
    x[60:, 0] *= 1.3                    # sensor 0 drifts
    x[60:, 1] = 500                     # sensor 1 is badly wrong and already detected
    corr = fr.apply_correction(x, _detection(120, 5, {0: 60, 1: 60}), 40)
    est = corr["corrected"][60:, 0]
    assert np.all(np.abs(est - 100) < 5)    # sensor 1's 500 was never used


def test_insufficient_peers_marks_estimate_unavailable():
    x = np.full((100, 3), 100.0)
    corr = fr.apply_correction(x, _detection(100, 3, {0: 50, 1: 50}), 40)
    assert corr["unavailable"][50:, 0].all() and not corr["estimated"][:, 0].any()
    np.testing.assert_array_equal(corr["corrected"][:, 0], x[:, 0])      # raw kept, nothing invented
    assert any("trusted peers" in w for w in corr["warnings"])


def test_process_change_is_not_corrected():
    x = np.full((100, 5), 100.0)
    det = _detection(100, 5, {}, process=True)
    det["category"] = "Process change"
    corr = fr.apply_correction(x, det, 40)
    assert not corr["estimated"].any()
    np.testing.assert_array_equal(corr["corrected"], x)
    rec = fr.recommend(det)
    assert not rec["correction_recommended"]


def test_batch_testing_table(small_bundle):
    result = bt.run_batch(small_bundle, runs_per_scenario=1, scenarios=["Normal", "Stuck", "Real leak"])
    table = result["table"]
    assert len(table) == 3
    for col in ["run_id", "expected_category", "predicted_category", "detection_outcome",
                "detection_delay_samples", "raw_mae", "corrected_mae", "correction_status",
                "rule_category", "model_confidence_uncalibrated"]:
        assert col in table.columns
    normal = table[table["scenario"] == "Normal"].iloc[0]
    assert pd.isna(normal["detection_delay_samples"])
    assert bt.by_category(table)["runs"].sum() == 3


def test_rule_adapter_does_not_read_not_a_leak_as_a_leak():
    run = nr.simulate(nr.ScenarioConfig(scenario="Drift (upward)", seed=3))
    rules = fd.rule_based_run(run.readings, run.config.warmup, 11)
    assert not rules["process_change"]
    assert rules["category"] == "Drift"


import sensor_health as sh  # noqa: E402


def _det_from(bundle, scenario, **kw):
    run = nr.simulate(nr.ScenarioConfig(scenario=scenario, **kw))
    return run, fd.predict_run(bundle, run.readings, run.config.warmup, run.names)


def test_health_scores_are_bounded_and_low_for_the_faulty_sensor(small_bundle):
    run, det = _det_from(small_bundle, "Stuck", seed=21, fault_sensor=2)
    scores = sh.health_timeline(det)
    assert scores.shape == run.readings.shape
    assert np.all((scores >= 0) & (scores <= 100))
    assert scores[-1, 2] < scores[-1, [0, 1, 3, 4]].min()


def test_health_never_looks_past_t(small_bundle):
    run, det = _det_from(small_bundle, "Drift (upward)", seed=4)
    scores = sh.health_timeline(det)
    altered = dict(det, proba=det["proba"].copy())
    altered["proba"][150:] = 0
    np.testing.assert_array_equal(scores[:150], sh.health_timeline(altered)[:150])


def test_healthy_sensors_need_no_maintenance(small_bundle):
    run, det = _det_from(small_bundle, "Normal", seed=5)
    assert all(h["action"] in ("NONE", "MONITOR") for h in sh.assess_health(det, run.readings, 40))


def test_maintenance_table():
    assert sh.maintenance_action("Stuck", {})[0] == "REPLACE"
    assert sh.maintenance_action("Dropout", {"recent_missing_share": 0.8})[:2] == ("REPAIR", "Immediate")
    assert sh.maintenance_action("Drift", {"offset": 0.1})[0] == "RECALIBRATE"
    assert sh.maintenance_action("Drift", {"offset": 0.4})[0] == "REPLACE"
    assert sh.maintenance_action("Noise", {"noise_ratio": 4})[0] == "REPAIR"
    assert sh.maintenance_action("Noise", {"noise_ratio": 12})[0] == "REPLACE"
    assert sh.maintenance_action("Spike", {"spike_events": 1})[0] == "INSPECT"
    assert sh.maintenance_action("Spike", {"spike_events": 5})[0] == "REPAIR"


def test_expected_actions_come_from_ground_truth():
    big = nr.simulate(nr.ScenarioConfig(scenario="Drift (upward)", seed=1, magnitude=0.5))
    small = nr.simulate(nr.ScenarioConfig(scenario="Drift (upward)", seed=1, magnitude=0.1))
    assert sh.expected_actions(big) == {big.config.fault_sensor: "REPLACE"}
    assert sh.expected_actions(small) == {small.config.fault_sensor: "RECALIBRATE"}
    assert sh.expected_actions(nr.simulate(nr.ScenarioConfig(scenario="Real leak", seed=1))) == {}


def test_drifting_sensor_is_recalibrated_or_replaced(small_bundle):
    run, det = _det_from(small_bundle, "Drift (upward)", seed=9, fault_sensor=1, magnitude=0.5)
    h = sh.assess_health(det, run.readings, 40)[1]
    assert h["fault"] == "Drift" and h["action"] == "REPLACE" and h["severity"]["offset"] >= sh.DRIFT_REPLACE
