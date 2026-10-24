# Drift model evaluation

Each file was scored by a model trained on the other files (leave-one-file-out).
"Detected after" = seconds from the labelled drift start until the faulty sensor stays flagged for 10 readings in a row.
"False flags" = drift flags on readings that were not drifting.

| File | Faulty sensor | Drift starts | ML: detected after | ML: false flags | Rules: detected after | Rules: false flags |
|---|---|---|---|---|---|---|
| drift1.csv | 1 | 206 s | 25 s | 0 / 2155 | 50 s | 0 / 2155 |
| drift2.csv | 3 | 266 s | 23 s | 0 / 2215 | 43 s | 0 / 2215 |
| drift3.csv | 2 | 243 s | 28 s | 0 / 2192 | 68 s | 0 / 2192 |
| drift4.csv | 1 | 252 s | 30 s | 0 / 2201 | 58 s | 0 / 2201 |
| drift5.csv | 4 | 293 s | 24 s | 0 / 2242 | 53 s | 0 / 2242 |
| drift6.csv | 1 | 167 s | 32 s | 0 / 2116 | 72 s | 0 / 2116 |
| drift7.csv | 1 | 246 s | 33 s | 0 / 2195 | 64 s | 0 / 2195 |
| drift8.csv | 4 | 227 s | 32 s | 0 / 2176 | 61 s | 0 / 2176 |
| normal_radiation.csv | - | - s | n/a | 0 / 6120 | n/a | 4 / 6120 |

Final model trained on: drift1.csv, drift2.csv, drift3.csv, drift4.csv, drift5.csv, drift6.csv, drift7.csv, drift8.csv, normal_radiation.csv

Weights (standardised features |z15|, |z60|, away60, beyond2, persist, scatter): 0.864, 0.826, 0.752, 1.353, 0.323, 0.117; bias -3.780; threshold 0.500
