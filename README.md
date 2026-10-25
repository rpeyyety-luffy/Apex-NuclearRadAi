# Apex-NuclearRadAi

NuclearRad AI monitoring console — a Streamlit training simulation of a five-sensor radiation watch.

## Run locally (Python prototype)

NuclearRad AI is a **research and training prototype on synthetic data**. It is not a
safety-certified system, has not been validated on real plant instruments, and never controls
equipment. All metrics it shows come from its own simulations.

Windows Command Prompt (Python 3.10 or newer):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python model_training.py
streamlit run main.py
```

Other commands (same window, with the virtual environment active):

```bat
python -m pytest
python batch_testing.py --runs-per-scenario 5
python model_training.py --retrain
```

On macOS / Linux use `source .venv/bin/activate` instead of `.venv\Scripts\activate`.

`python model_training.py` trains the model once (about 30 seconds) and saves it in `models/`;
the dashboard and the batch script load that file instead of retraining. If no compatible model is
saved (for example after the features or scikit-learn version change), it is trained automatically.

### Dashboard tabs (`main.py`)

| Tab | What it does |
|---|---|
| A · Overview | Reactor schematic, sensor cards and status from the original rule-based monitor |
| B · Scenario simulator | Choose scenario, seed, noise, sensor, start, duration and magnitude; **Generate scenario**; download the labelled run |
| C · Sensor monitoring | Peer comparison, trust over time and per-sensor timelines (rule-based monitor) |
| D · Model prediction | **Predict**: fault present, category, affected sensor, uncalibrated vote share; ML vs rule-based detector; health score of every sensor over time |
| E · Fault-tolerance response | **Simulate response**: recommended action, suggested maintenance per faulty sensor, and a peer-based correction with raw vs corrected MAE |
| F · Evaluation | Validation and held-out test metrics, confusion matrix, per-class report |
| G · Automated testing | **Run batch test** on fresh seeds; results table and CSV download |

### Pipeline

| File | Role |
|---|---|
| `nuclearrad.py` | Original demo (`generate`, `assess`, `forecast`, `summarise`) plus the configurable simulator (`ScenarioConfig`, `simulate`) |
| `feature_engineering.py` | 15 causal per-sensor features (past and present readings only) |
| `dataset_generator.py` | Labelled runs; train / validation / test split **by run**, from disjoint seed ranges |
| `model_training.py` | Random Forest; thresholds tuned on validation; test split evaluated once; save / load |
| `fault_detection.py` | Per-sample classification and run-level verdict; same aggregation applied to the rule-based detector |
| `fault_response.py` | Recommended responses and the simulated peer-based correction |
| `sensor_health.py` | Health score and condition per sensor, and a suggested maintenance action (replace, repair, recalibrate, inspect) |
| `evaluation.py` | Metric formulas (N/A where a denominator is zero) |
| `batch_testing.py` | Batch evaluation and CSV export (`results/batch_results.csv`) |

### Sensor health and maintenance

The model gives each sensor a **health score** from 0 to 100 at every sample: 100 × (1 − the mean
model probability of an instrument fault over the last 20 samples). Like the vote share, it is not
calibrated. Each sensor gets a condition: Healthy, Watch (not confirmed yet), Degraded, Faulty or
Failed (stuck, or signal lost). A confirmed fault gets a suggested maintenance action, chosen from
the fault type the model predicted and the severity measured in the readings:

| Predicted fault | Suggested action |
|---|---|
| Stuck | REPLACE (immediate) |
| Dropout | REPAIR cable, connector, power or logger (immediate when most recent readings are missing) |
| Drift | RECALIBRATE below a 30% offset from the peers; REPLACE at or above it |
| Noise | REPAIR shielding, grounding or supply below 8× normal scatter; REPLACE above it |
| Spike | INSPECT connectors for fewer than 4 spike events; REPAIR from 4 on |

These limits are illustrative values for this simulator, not limits of any real instrument. The
batch test checks each suggestion against the action the simulation's ground truth calls for
(`maintenance_correct` in the CSV) and counts runs where work was suggested on a healthy sensor.

Scenarios: Normal, Spike (positive / negative), Drift (upward / downward), Stuck, Noise, Dropout,
Two faulty sensors, Real leak (a real plant-wide change, *not* a sensor fault) and Real leak + Drift.

Design rules the code follows:

- The healthy reference ("truth") is generated separately from the readings and is used only for
  labels and evaluation, never as a model input or by the correction.
- Features at sample *t* use readings up to *t* only (a test checks that changing future readings
  does not change past features or predictions).
- The correction never uses a sensor as a reference after it has been detected as faulty; with
  fewer than two trusted peers it marks the estimate unavailable and keeps the raw reading.
- A real leak is reported and escalated, never "corrected".
- Model confidence is the Random Forest vote share. It is **not calibrated**.

Metrics: accuracy = (TP+TN)/(TP+TN+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN),
specificity = TN/(TN+FP), FPR = FP/(FP+TN), MAE = mean |reference − estimate|,
improvement = (MAE_raw − MAE_corrected)/MAE_raw × 100 (N/A when MAE_raw is 0), detection delay =
first detection − fault start in samples (missed detections are counted, not given a delay).
Positive class = an instrument fault; Normal and Real leak runs are negatives.

The single-page sensor view still runs with `streamlit run sensor_charts.py`.

## Deploy on Vercel

Vercel cannot host a Streamlit server, so `index.html` runs the app in the browser with
[stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide). `vercel.json` publishes it as a
static site — import the repo in Vercel and deploy with no extra settings. The first load takes a
few seconds while Python loads in the browser.

## Export report (PDF)

On the data monitor page (`index.html`), load the demo data or a file, then use the buttons above
**Assessment and suggested actions**:

- **Export report (PDF)** opens the browser's print dialog with the report; choose *Save as PDF*.
- **Print view** opens the same report in a new tab, with its own *Print / Save as PDF* button.

The report covers whatever is currently loaded (the uploaded file, or the demo with its selected
scenario, which is listed under *Filters applied*). It contains a header (title, generation time,
time range, data source), an executive summary with sensor counts by trust level, a sensor table
(type, location, last reading, last seen, verification status, trust level and score, flags), three
charts (sensors by trust level, abnormal readings by type, and reading trends for the flagged or
lowest-scoring sensors) and prioritised recommended actions. Every page has a footer with the page
number and generation time. Everything is built in the browser; no data is sent anywhere.

| File | Role |
|---|---|
| `report_data.js` | Collects the report data from the analysis (no rendering, no rules) |
| `report_rules.js` | Trust levels and the High / Medium / Low action rules |
| `report_render.js` | Turns report data into a print-optimised HTML page with static SVG charts |
| `sensors.js` | Sensor names, types and locations, shared with the 3D model |

**Trust levels** (first match from the top): *Offline* when the last 5% of readings (at least 5) are
missing; *Untrusted* when the detector reports the sensor as faulty; *Needs review* when it is on
watch or at risk; otherwise *Trusted*. The score shown next to the level is the detector's health
score (0–100).

**Customising the recommendations**: edit `report_rules.js`. `SETTINGS` holds the limits (offline
tail, the health score below which a healthy sensor gets a "monitor" note, the missing-data share
that triggers a logging check). `ACTION_RULES` is an ordered list; each rule has an `id`, a
`priority` (High, Medium or Low, possibly depending on the data), a `when` test that returns the
evidence text or `null`, and a `next` step. Rules that share a `group` are exclusive per sensor
(only the first match is used), so a sensor marked for replacement does not also get a
recalibration action. The rules are deterministic and use no external service.

Tests:

```bash
node ml/test_report.js        # trust levels, every action rule, aggregation on all labelled recordings, edge cases
node ml/test_report_pdf.js    # renders real PDFs with headless Chrome and checks they are valid
node ml/test_report_pdf.js --out sample_report.pdf   # also keeps a sample PDF
```

The PDF smoke test needs Chrome or Chromium (set `CHROME_PATH` if it is not found); without one it
prints `SKIPPED` and exits with code 2.

## Sensor fault detector

`fault_detector.html` lets anyone upload sensor readings (Excel or CSV, one column per sensor) and
see which sensors are faulty, with a graph per sensor and recommended fixes. It runs entirely in
the browser (`detector.js`), so nothing is uploaded to a server.

Each sensor is compared with an ideal healthy run (built into `detector.js`; users can supply their
own ideal file) and with the other sensors, so a change that moves every sensor together is
reported as a possible real radiation rise rather than a sensor fault. Open
`fault_detector.html#demo` to see it on demo data.

### Machine-learned fault models

`faultml.js` holds six logistic-regression models, trained on labelled recordings in `ml/data`:

- **Drift**: measures how far each sensor's gap to its peers has moved from its normal gap (the
  healthy start of the file) and turns six features (size, speed and persistence of that move)
  into a drift probability per reading.
- **Dropout**: looks at how often, how recently and for how long a sensor has lost its signal, and
  rates whether it is dropping out intermittently, also in the stretches between gaps. A single
  missed sample is treated as a logger hiccup, not a failing sensor.
- **Noise**: measures how rough a sensor's readings are from one reading to the next, compared
  with its own normal roughness and with its peers'; it must flag 5 readings in a row.
- **Stuck**: measures how long a sensor has repeated exactly the same value and how flat its recent
  readings are while its peers keep moving.
- **Spike**: measures how far each reading jumps from the sensor's recent level while its peers stay
  put, and how often that has happened lately, and rates whether the sensor keeps producing spikes.
  A single spike is treated as a one-off glitch.
- **Leak**: one rating for the whole plant: whether all sensors are falling together, as in a real
  coolant leak. It is reported as a plant event, separately from any sensor's own fault.

The file name tells the trainer the fault type (`drift*.csv`, `dropout*.csv`, `noise*.csv`, `stuck*.csv`, `spike*.csv`;
`realleak*_<fault>.csv` is a real leak plus that sensor fault; `normal_*.csv` is healthy). Labelled files need `Anomaly_Active` and `Faulty_Sensor` columns (and `Real_Leak` for leaks), as in the supplied
sets. Retrain after adding recordings:

```bash
node ml/train_models.js
```

This rewrites the model block in `faultml.js` and the leave-one-file-out results in
`ml/model_report.md`.
