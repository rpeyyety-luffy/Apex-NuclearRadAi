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
