# Fault model evaluation

Each file was scored by a model trained on the other files (leave-one-file-out).
- **Detected after**: seconds from the labelled fault start until the faulty sensor stays flagged for 10 readings in a row.
- **Covered**: share of the faulty sensor's readings from the fault start on that are flagged.
- **False flags**: flags on readings that are not this fault (other sensors, other files, healthy stretches).
- **Rules** is the detector without the models.

## Drift model

| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |
|---|---|---|---|---|---|---|---|
| drift1.csv | 1 | 25 s | 100% | 0 / 2155 | 50 s | 97% | 0 / 2155 |
| drift2.csv | 3 | 24 s | 100% | 0 / 2215 | 43 s | 99% | 0 / 2215 |
| drift3.csv | 2 | 28 s | 100% | 0 / 2192 | 68 s | 91% | 0 / 2192 |
| drift4.csv | 1 | 30 s | 100% | 0 / 2201 | 58 s | 94% | 0 / 2201 |
| drift5.csv | 4 | 24 s | 100% | 0 / 2242 | 53 s | 95% | 0 / 2242 |
| drift6.csv | 1 | 32 s | 100% | 0 / 2116 | 72 s | 92% | 0 / 2116 |
| drift7.csv | 1 | 33 s | 100% | 0 / 2195 | 64 s | 92% | 0 / 2195 |
| drift8.csv | 4 | 32 s | 100% | 0 / 2176 | 61 s | 95% | 0 / 2176 |
| dropout1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 4 / 6120 |

Weights (standardised features |z15|, |z60|, away60, beyond2, persist, scatter): 0.753, 0.714, 0.620, 1.293, 0.217, 0.050; bias -4.037; threshold 0.500

## Dropout model

| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |
|---|---|---|---|---|---|---|---|
| drift1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| drift8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout1.csv | 1 | 0 s | 100% | 0 / 2164 | 0 s | 24% | 0 / 2164 |
| dropout2.csv | 1 | 0 s | 100% | 0 / 2243 | 0 s | 23% | 0 / 2243 |
| dropout3.csv | 5 | 0 s | 100% | 0 / 2238 | 0 s | 19% | 0 / 2238 |
| dropout4.csv | 1 | 0 s | 100% | 0 / 2341 | 0 s | 33% | 0 / 2341 |
| dropout5.csv | 3 | 0 s | 100% | 0 / 2231 | 0 s | 25% | 0 / 2231 |
| dropout6.csv | 5 | 0 s | 100% | 0 / 2186 | 0 s | 18% | 0 / 2186 |
| dropout7.csv | 2 | 0 s | 100% | 0 / 2188 | 0 s | 22% | 0 / 2188 |
| dropout8.csv | 1 | 0 s | 100% | 0 / 2174 | 0 s | 23% | 0 / 2174 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 0 / 6120 |

Weights (standardised features missing now, missing 30, missing 120, gaps 300, longest gap 300, since gap): 0.521, 0.684, 1.205, -0.149, 1.892, 0.243; bias -4.015; threshold 0.500

Trained on: drift1.csv, drift2.csv, drift3.csv, drift4.csv, drift5.csv, drift6.csv, drift7.csv, drift8.csv, dropout1.csv, dropout2.csv, dropout3.csv, dropout4.csv, dropout5.csv, dropout6.csv, dropout7.csv, dropout8.csv, normal_radiation.csv
