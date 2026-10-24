import numpy as np
import pytest

import nuclearrad as nr
from dataset_generator import SPLIT_SEED_BASE, BATCH_SEED_BASE, build_dataset, generate_runs
from feature_engineering import FEATURE_COLUMNS, compute_features


@pytest.mark.parametrize("scenario", list(nr.SIM_SCENARIOS))
def test_every_scenario_generates_valid_run(scenario):
    run = nr.simulate(nr.ScenarioConfig(scenario=scenario, seed=3))
    T, S = run.readings.shape
    assert (T, S) == (nr.N_SAMPLES, nr.N_SENSORS)
    assert run.truth.shape == run.labels.shape == (T, S)
    assert set(np.unique(run.labels)) <= set(nr.CLASSES)
    assert np.isfinite(run.truth).all()
    assert (run.labels[: run.config.warmup] == "Normal").all()


def test_same_config_same_run_and_different_seed_differs():
    a = nr.simulate(nr.ScenarioConfig(scenario="Noise", seed=11))
    b = nr.simulate(nr.ScenarioConfig(scenario="Noise", seed=11))
    c = nr.simulate(nr.ScenarioConfig(scenario="Noise", seed=12))
    np.testing.assert_array_equal(a.readings, b.readings)
    assert a.run_id == b.run_id != c.run_id
    assert not np.allclose(a.readings, c.readings)


@pytest.mark.parametrize("scenario", ["Spike (positive)", "Drift (upward)", "Stuck", "Noise", "Dropout",
                                      "Two faulty sensors"])
def test_instrument_faults_never_change_truth(scenario):
    cfg = nr.ScenarioConfig(scenario=scenario, seed=5)
    run = nr.simulate(cfg)
    healthy = nr.simulate(nr.ScenarioConfig(scenario="Normal", seed=5))
    np.testing.assert_array_equal(run.truth, healthy.truth)
    assert run.fault_sensors
    for s in run.fault_sensors:
        window = run.labels[cfg.fault_start:cfg.fault_end, s]
        assert np.isin(window, nr.INSTRUMENT_FAULTS).any()


def test_real_leak_changes_truth_and_is_not_an_instrument_fault():
    run = nr.simulate(nr.ScenarioConfig(scenario="Real leak", seed=2, magnitude=0.6))
    assert run.fault_sensors == []
    assert (run.labels == "Process change").sum() > 0
    assert run.truth[-1].mean() > run.truth[0].mean() * 1.3


def test_real_leak_plus_drift_has_both():
    run = nr.simulate(nr.ScenarioConfig(scenario="Real leak + Drift", seed=2, fault_sensor=1))
    assert run.fault_sensors == [1]
    assert (run.labels == "Process change").any() and (run.labels[:, 1] == "Drift").any()


def test_fault_window_and_dropout():
    cfg = nr.ScenarioConfig(scenario="Dropout", seed=1, fault_start=90, duration=50, magnitude=0.3)
    run = nr.simulate(cfg)
    missing = np.flatnonzero(np.isnan(run.readings[:, cfg.fault_sensor]))
    assert missing.size == 15 and missing.min() >= 90 and missing.max() < 140
    assert not np.isnan(np.delete(run.readings, cfg.fault_sensor, axis=1)).any()


@pytest.mark.parametrize("kwargs", [dict(scenario="Unknown"), dict(fault_start=5), dict(noise=0.5),
                                    dict(scenario="Drift (upward)", magnitude=5.0), dict(fault_sensor=9),
                                    dict(scenario="Two faulty sensors", fault_sensor=1, second_sensor=1)])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        nr.simulate(nr.ScenarioConfig(**kwargs))


def test_split_seed_ranges_are_disjoint():
    runs = {s: generate_runs(s, 2) for s in SPLIT_SEED_BASE}
    ids = {s: {r.run_id for r in rs} for s, rs in runs.items()}
    seeds = {s: {r.config.seed for r in rs} for s, rs in runs.items()}
    assert not ids["train"] & ids["validation"] and not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert not seeds["train"] & seeds["test"]
    batch = generate_runs("batch", 1, base=BATCH_SEED_BASE)
    assert not {r.config.seed for r in batch} & set().union(*seeds.values())


def test_features_are_causal_and_finite():
    run = nr.simulate(nr.ScenarioConfig(scenario="Dropout", seed=4))
    feats = compute_features(run.readings, run.config.warmup)
    assert feats.shape == run.readings.shape + (len(FEATURE_COLUMNS),)
    assert np.isfinite(feats).all()
    changed = run.readings.copy()
    changed[120:] *= 1.5                      # alter only the future
    feats2 = compute_features(changed, run.config.warmup)
    np.testing.assert_array_equal(feats[:120], feats2[:120])


def test_dataset_contains_no_ground_truth_columns():
    runs = generate_runs("train", 1, scenarios=["Drift (upward)", "Normal"])
    X, y, meta = build_dataset(runs)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert "healthy_reference" not in FEATURE_COLUMNS and "fault_type" not in FEATURE_COLUMNS
    assert set(y) <= set(nr.CLASSES) and len(meta) == len(y)


def test_long_frame_has_required_fields():
    frame = nr.simulate(nr.ScenarioConfig(scenario="Stuck", seed=1)).to_long_frame()
    for col in ["run_id", "scenario_id", "time_step", "sensor_id", "healthy_reference", "observed_reading",
                "fault_type", "fault_present", "affected_sensor", "fault_start", "fault_end", "magnitude",
                "noise", "seed"]:
        assert col in frame.columns


def test_original_demo_functions_still_work():
    x, truth = nr.generate("Drift", seed=7)
    res = nr.assess(x)
    assert res["fault"].shape == x.shape
    assert set(nr.summarise(res, truth)) == {"first_alarm", "first_ai", "err_before", "err_after"}
