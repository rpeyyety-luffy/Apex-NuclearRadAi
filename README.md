# Apex-NuclearRadAi

NuclearRad AI monitoring console — a Streamlit training simulation of a five-sensor radiation watch.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Deploy on Vercel

Vercel cannot host a Streamlit server, so `index.html` runs the app in the browser with
[stlite](https://github.com/whitphx/stlite) (Streamlit on Pyodide). `vercel.json` publishes it as a
static site — import the repo in Vercel and deploy with no extra settings. The first load takes a
few seconds while Python loads in the browser.

## Sensor fault detector

`fault_detector.html` lets anyone upload sensor readings (Excel or CSV, one column per sensor) and
see which sensors are faulty, with a graph per sensor and recommended fixes. It runs entirely in
the browser (`detector.js`), so nothing is uploaded to a server.

Each sensor is compared with an ideal healthy run (built into `detector.js`; users can supply their
own ideal file) and with the other sensors, so a change that moves every sensor together is
reported as a possible real radiation rise rather than a sensor fault. Open
`fault_detector.html#demo` to see it on demo data.

### Machine-learned fault models

`faultml.js` holds two logistic-regression models, trained on labelled recordings in `ml/data`:

- **Drift**: measures how far each sensor's gap to its peers has moved from its normal gap (the
  healthy start of the file) and turns six features (size, speed and persistence of that move)
  into a drift probability per reading.
- **Dropout**: looks at how often, how recently and for how long a sensor has lost its signal, and
  rates whether it is dropping out intermittently, also in the stretches between gaps. A single
  missed sample is treated as a logger hiccup, not a failing sensor.

The file name tells the trainer the fault type (`drift*.csv`, `dropout*.csv`; `normal_*.csv` is
healthy). Labelled files need `Anomaly_Active` and `Faulty_Sensor` columns, as in the supplied
sets. Retrain after adding recordings:

```bash
node ml/train_models.js
```

This rewrites the model block in `faultml.js` and the leave-one-file-out results in
`ml/model_report.md`.
