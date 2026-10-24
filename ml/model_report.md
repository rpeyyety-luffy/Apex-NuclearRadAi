# Fault model evaluation

Each file was scored by a model trained on the other files (leave-one-file-out).
- **Detected after**: seconds from the labelled fault start until the faulty sensor stays flagged for 10 readings in a row.
- **Covered**: share of the faulty sensor's readings from the fault start on that are flagged.
- **False flags**: flags on readings that are not this fault (other sensors, other files, healthy stretches).
- **Rules** is the detector without the models.

## Drift model

| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |
|---|---|---|---|---|---|---|---|
| drift1.csv | 1 | 30 s | 100% | 0 / 2155 | 50 s | 97% | 0 / 2155 |
| drift2.csv | 3 | 33 s | 100% | 0 / 2215 | 43 s | 99% | 0 / 2215 |
| drift3.csv | 2 | 37 s | 100% | 0 / 2192 | 68 s | 91% | 0 / 2192 |
| drift4.csv | 1 | 36 s | 100% | 0 / 2201 | 58 s | 94% | 0 / 2201 |
| drift5.csv | 4 | 28 s | 100% | 0 / 2242 | 53 s | 95% | 0 / 2242 |
| drift6.csv | 1 | 38 s | 100% | 0 / 2116 | 72 s | 92% | 0 / 2116 |
| drift7.csv | 1 | 29 s | 99% | 0 / 2195 | 64 s | 92% | 0 / 2195 |
| drift8.csv | 4 | 38 s | 100% | 0 / 2176 | 61 s | 95% | 0 / 2176 |
| dropout1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 4 / 6120 |
| realleak1_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak2_noise.csv | - | - | - | 35 / 2550 | - | - | 0 / 2550 |
| realleak3_drift.csv | 5 | 42 s | 99% | 0 / 2181 | 75 s | 90% | 0 / 2181 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | - | - | - | 88 / 2550 | - | - | 0 / 2550 |
| realleak6_noise.csv | - | - | - | 1 / 2550 | - | - | 0 / 2550 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features |z15|, |z60|, away60, beyond2, persist, scatter): 0.399, 0.201, 0.530, 1.514, 0.510, -1.806; bias -4.386; threshold 0.677

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
| noise1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 0 / 6120 |
| realleak1_dropout.csv | 4 | 0 s | 100% | 0 / 2268 | 0 s | 23% | 0 / 2268 |
| realleak2_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak3_drift.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak4_dropout.csv | 1 | 0 s | 100% | 0 / 2216 | 0 s | 23% | 0 / 2216 |
| realleak5_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak6_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak7_dropout.csv | 3 | 0 s | 100% | 0 / 2339 | 0 s | 34% | 0 / 2339 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features missing now, missing 30, missing 120, gaps 300, longest gap 300, since gap): 0.494, 0.688, 1.162, -0.154, 1.728, 0.215; bias -4.083; threshold 0.500

## Noise model

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
| dropout1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| dropout8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise1.csv | 5 | 22 s | 99% | 0 / 2147 | 3 s | 100% | 0 / 2147 |
| noise2.csv | 5 | 9 s | 99% | 0 / 2229 | 0 s | 100% | 0 / 2229 |
| noise3.csv | 3 | 10 s | 100% | 0 / 2122 | 2 s | 100% | 0 / 2122 |
| noise4.csv | 3 | 12 s | 99% | 0 / 2141 | 0 s | 100% | 0 / 2141 |
| noise5.csv | 4 | 16 s | 98% | 0 / 2144 | 2 s | 100% | 0 / 2144 |
| noise6.csv | 4 | 16 s | 99% | 0 / 2100 | 2 s | 100% | 0 / 2100 |
| noise7.csv | 4 | 13 s | 99% | 0 / 2188 | 1 s | 100% | 0 / 2188 |
| noise8.csv | 4 | 14 s | 99% | 0 / 2221 | 0 s | 100% | 0 / 2221 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 0 / 6120 |
| realleak1_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak2_noise.csv | 3 | 13 s | 99% | 0 / 2243 | 3 s | 100% | 0 / 2243 |
| realleak3_drift.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | 2 | 21 s | 98% | 0 / 2216 | 5 s | 100% | 0 / 2216 |
| realleak6_noise.csv | 4 | 20 s | 97% | 0 / 2216 | 7 s | 100% | 0 / 2216 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features rough8, rough30, rough60, jumps30, rough30 vs peers): 0.845, 0.244, 0.100, 1.455, 0.191; bias -4.311; threshold 0.857

## Leak model

| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |
|---|---|---|---|---|---|---|---|
| drift1.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift2.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift3.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift4.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift5.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift6.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift7.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| drift8.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout1.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout2.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout3.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout4.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout5.csv | - | - | - | 28 / 510 | - | - | 0 / 510 |
| dropout6.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout7.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| dropout8.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise1.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise2.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise3.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise4.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise5.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise6.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise7.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| noise8.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| normal_radiation.csv | - | - | - | 0 / 1224 | - | - | 0 / 1224 |
| realleak1_dropout.csv | all | 10 s | 100% | 0 / 137 | missed | 0% | 0 / 137 |
| realleak2_noise.csv | all | 15 s | 98% | 0 / 203 | missed | 0% | 0 / 203 |
| realleak3_drift.csv | all | 15 s | 99% | 0 / 141 | missed | 0% | 0 / 141 |
| realleak4_dropout.csv | all | 13 s | 99% | 0 / 130 | missed | 0% | 0 / 130 |
| realleak5_noise.csv | all | 17 s | 98% | 0 / 176 | missed | 0% | 0 / 176 |
| realleak6_noise.csv | all | 18 s | 98% | 0 / 176 | missed | 0% | 0 / 176 |
| realleak7_dropout.csv | all | 18 s | 98% | 0 / 71 | missed | 0% | 0 / 71 |
| realleak8_spike.csv | all | 19 s | 98% | 0 / 60 | missed | 0% | 0 / 60 |

Weights (standardised features fall15, fall60, slope60, sensors falling): 1.253, 0.179, 1.079, 2.064; bias -2.986; threshold 0.672

Trained on: drift1.csv, drift2.csv, drift3.csv, drift4.csv, drift5.csv, drift6.csv, drift7.csv, drift8.csv, dropout1.csv, dropout2.csv, dropout3.csv, dropout4.csv, dropout5.csv, dropout6.csv, dropout7.csv, dropout8.csv, noise1.csv, noise2.csv, noise3.csv, noise4.csv, noise5.csv, noise6.csv, noise7.csv, noise8.csv, normal_radiation.csv, realleak1_dropout.csv, realleak2_noise.csv, realleak3_drift.csv, realleak4_dropout.csv, realleak5_noise.csv, realleak6_noise.csv, realleak7_dropout.csv, realleak8_spike.csv
