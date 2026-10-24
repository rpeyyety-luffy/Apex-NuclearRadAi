# Fault model evaluation

Each file was scored by a model trained on the other files (leave-one-file-out).
- **Detected after**: seconds from the labelled fault start until the faulty sensor stays flagged for 10 readings in a row.
- **Covered**: share of the faulty sensor's readings from the fault start on that are flagged.
- **False flags**: flags on readings that are not this fault (other sensors, other files, healthy stretches).
- **Rules** is the detector without the models.

## Drift model

| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |
|---|---|---|---|---|---|---|---|
| drift1.csv | 1 | 38 s | 100% | 0 / 2155 | 50 s | 97% | 0 / 2155 |
| drift2.csv | 3 | 40 s | 100% | 0 / 2215 | 43 s | 99% | 0 / 2215 |
| drift3.csv | 2 | 48 s | 97% | 0 / 2192 | 68 s | 91% | 0 / 2192 |
| drift4.csv | 1 | 47 s | 98% | 0 / 2201 | 58 s | 94% | 0 / 2201 |
| drift5.csv | 4 | 45 s | 99% | 0 / 2242 | 53 s | 95% | 0 / 2242 |
| drift6.csv | 1 | 43 s | 99% | 0 / 2116 | 72 s | 92% | 0 / 2116 |
| drift7.csv | 1 | 48 s | 97% | 0 / 2195 | 64 s | 92% | 0 / 2195 |
| drift8.csv | 4 | 48 s | 98% | 0 / 2176 | 61 s | 95% | 0 / 2176 |
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
| realleak2_noise.csv | - | - | - | 33 / 2550 | - | - | 0 / 2550 |
| realleak3_drift.csv | 5 | 55 s | 95% | 0 / 2181 | 75 s | 90% | 0 / 2181 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | - | - | - | 8 / 2550 | - | - | 0 / 2550 |
| realleak6_noise.csv | - | - | - | 1 / 2550 | - | - | 0 / 2550 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak_maxdanger.csv | - | - | - | 0 / 2550 | - | - | 91 / 2550 |
| spike1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck1.csv | - | - | - | 51 / 2550 | - | - | 0 / 2550 |
| stuck2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck3.csv | - | - | - | 44 / 2550 | - | - | 0 / 2550 |
| stuck4.csv | - | - | - | 19 / 2550 | - | - | 0 / 2550 |
| stuck5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck7.csv | - | - | - | 23 / 2550 | - | - | 0 / 2550 |
| stuck8.csv | - | - | - | 15 / 2550 | - | - | 0 / 2550 |

Weights (standardised features |z15|, |z60|, away60, beyond2, persist, scatter): 1.087, -0.256, 1.907, 1.167, 0.622, -1.959; bias -9.370; threshold 0.952

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
| realleak_maxdanger.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features missing now, missing 30, missing 120, gaps 300, longest gap 300, since gap): 0.436, 0.128, 0.567, 0.203, 2.122, -0.089; bias -7.661; threshold 0.500

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
| noise1.csv | 5 | 23 s | 99% | 0 / 2147 | 3 s | 100% | 0 / 2147 |
| noise2.csv | 5 | 11 s | 95% | 0 / 2229 | 0 s | 100% | 0 / 2229 |
| noise3.csv | 3 | 10 s | 100% | 0 / 2122 | 2 s | 100% | 0 / 2122 |
| noise4.csv | 3 | 11 s | 100% | 0 / 2141 | 0 s | 100% | 0 / 2141 |
| noise5.csv | 4 | 16 s | 98% | 0 / 2144 | 2 s | 100% | 0 / 2144 |
| noise6.csv | 4 | 16 s | 99% | 0 / 2100 | 2 s | 100% | 0 / 2100 |
| noise7.csv | 4 | 14 s | 99% | 0 / 2188 | 1 s | 100% | 0 / 2188 |
| noise8.csv | 4 | 14 s | 99% | 0 / 2221 | 0 s | 100% | 0 / 2221 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 0 / 6120 |
| realleak1_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak2_noise.csv | 3 | 16 s | 98% | 0 / 2243 | 3 s | 100% | 0 / 2243 |
| realleak3_drift.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | 2 | 21 s | 97% | 0 / 2216 | 5 s | 100% | 0 / 2216 |
| realleak6_noise.csv | 4 | 20 s | 97% | 0 / 2216 | 7 s | 100% | 0 / 2216 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak_maxdanger.csv | - | - | - | 1464 / 2550 | - | - | 2244 / 2550 |
| spike1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike8.csv | - | - | - | 1 / 2550 | - | - | 0 / 2550 |
| stuck1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features rough8, rough30, rough60, jumps30, rough30 vs peers): 1.789, -0.406, -0.191, 1.107, 2.276; bias -7.814; threshold 0.882

## Stuck model

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
| noise1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| noise8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| normal_radiation.csv | - | - | - | 2 / 6120 | - | - | 0 / 6120 |
| realleak1_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak2_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak3_drift.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak6_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak_maxdanger.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| spike8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck1.csv | 4 | 3 s | 100% | 0 / 2136 | 0 s | 100% | 0 / 2136 |
| stuck2.csv | 1 | 3 s | 100% | 0 / 2214 | 0 s | 100% | 0 / 2214 |
| stuck3.csv | 2 | 3 s | 100% | 0 / 2114 | 0 s | 100% | 0 / 2114 |
| stuck4.csv | 1 | 3 s | 100% | 0 / 2166 | 0 s | 100% | 0 / 2166 |
| stuck5.csv | 3 | 3 s | 100% | 0 / 2113 | 0 s | 100% | 0 / 2113 |
| stuck6.csv | 2 | 3 s | 100% | 0 / 2229 | 0 s | 100% | 0 / 2229 |
| stuck7.csv | 4 | 3 s | 100% | 0 / 2144 | 0 s | 100% | 0 / 2144 |
| stuck8.csv | 2 | 3 s | 100% | 0 / 2247 | 0 s | 100% | 0 / 2247 |

Weights (standardised features same run, flat30, rough15, peers rough15): 3.035, 1.064, -0.379, -0.152; bias -4.994; threshold 0.500

## Spike model

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
| noise1.csv | - | - | - | 4 / 2550 | - | - | 0 / 2550 |
| noise2.csv | - | - | - | 2 / 2550 | - | - | 0 / 2550 |
| noise3.csv | - | - | - | 6 / 2550 | - | - | 0 / 2550 |
| noise4.csv | - | - | - | 8 / 2550 | - | - | 0 / 2550 |
| noise5.csv | - | - | - | 3 / 2550 | - | - | 0 / 2550 |
| noise6.csv | - | - | - | 4 / 2550 | - | - | 0 / 2550 |
| noise7.csv | - | - | - | 1 / 2550 | - | - | 0 / 2550 |
| noise8.csv | - | - | - | 3 / 2550 | - | - | 0 / 2550 |
| normal_radiation.csv | - | - | - | 0 / 6120 | - | - | 0 / 6120 |
| realleak1_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak2_noise.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak3_drift.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak4_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak5_noise.csv | - | - | - | 2 / 2550 | - | - | 0 / 2550 |
| realleak6_noise.csv | - | - | - | 2 / 2550 | - | - | 304 / 2550 |
| realleak7_dropout.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| realleak8_spike.csv | 2 | missed | 11% | 0 / 2170 | 22 s | 94% | 0 / 2170 |
| realleak_maxdanger.csv | - | - | - | 93 / 2550 | - | - | 271 / 2550 |
| spike1.csv | 2 | missed | 18% | 0 / 2274 | 1 s | 100% | 0 / 2274 |
| spike2.csv | 4 | missed | 3% | 0 / 2156 | 34 s | 91% | 0 / 2156 |
| spike3.csv | 4 | 295 s | 7% | 0 / 2105 | 1 s | 100% | 0 / 2105 |
| spike4.csv | 1 | missed | 18% | 0 / 2294 | 1 s | 100% | 0 / 2294 |
| spike5.csv | 3 | missed | 9% | 0 / 2204 | 1 s | 100% | 0 / 2204 |
| spike6.csv | 5 | 278 s | 11% | 0 / 2187 | 1 s | 100% | 0 / 2187 |
| spike7.csv | 4 | missed | 9% | 0 / 2120 | 90 s | 79% | 0 / 2120 |
| spike8.csv | 4 | 140 s | 27% | 0 / 2260 | 1 s | 100% | 0 / 2260 |
| stuck1.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck2.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck3.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck4.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck5.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck6.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck7.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |
| stuck8.csv | - | - | - | 0 / 2550 | - | - | 0 / 2550 |

Weights (standardised features jump now, peers jump now, spikes 60, spikes 300, since spike, rough30): -1.262, -0.442, -2.873, 2.046, -3.391, -0.802; bias -3.322; threshold 0.999

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
| dropout5.csv | - | - | - | 5 / 510 | - | - | 0 / 510 |
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
| realleak2_noise.csv | all | 14 s | 99% | 0 / 203 | missed | 0% | 0 / 203 |
| realleak3_drift.csv | all | 15 s | 99% | 0 / 141 | missed | 0% | 0 / 141 |
| realleak4_dropout.csv | all | 14 s | 99% | 0 / 130 | missed | 0% | 0 / 130 |
| realleak5_noise.csv | all | 17 s | 98% | 0 / 176 | missed | 0% | 0 / 176 |
| realleak6_noise.csv | all | 18 s | 98% | 0 / 176 | missed | 0% | 0 / 176 |
| realleak7_dropout.csv | all | 17 s | 98% | 0 / 71 | missed | 0% | 0 / 71 |
| realleak8_spike.csv | all | 19 s | 98% | 0 / 60 | missed | 0% | 0 / 60 |
| realleak_maxdanger.csv | all | 8 s | 100% | 0 / 30 | missed | 0% | 0 / 30 |
| spike1.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike2.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike3.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike4.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike5.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike6.csv | - | - | - | 39 / 510 | - | - | 0 / 510 |
| spike7.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| spike8.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck1.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck2.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck3.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck4.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck5.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck6.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck7.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |
| stuck8.csv | - | - | - | 0 / 510 | - | - | 0 / 510 |

Weights (standardised features fall15, fall60, slope60, sensors falling): 2.697, -0.904, 1.533, 1.738; bias -4.453; threshold 0.760

Trained on: drift1.csv, drift2.csv, drift3.csv, drift4.csv, drift5.csv, drift6.csv, drift7.csv, drift8.csv, dropout1.csv, dropout2.csv, dropout3.csv, dropout4.csv, dropout5.csv, dropout6.csv, dropout7.csv, dropout8.csv, noise1.csv, noise2.csv, noise3.csv, noise4.csv, noise5.csv, noise6.csv, noise7.csv, noise8.csv, normal_radiation.csv, realleak1_dropout.csv, realleak2_noise.csv, realleak3_drift.csv, realleak4_dropout.csv, realleak5_noise.csv, realleak6_noise.csv, realleak7_dropout.csv, realleak8_spike.csv, realleak_maxdanger.csv, spike1.csv, spike2.csv, spike3.csv, spike4.csv, spike5.csv, spike6.csv, spike7.csv, spike8.csv, stuck1.csv, stuck2.csv, stuck3.csv, stuck4.csv, stuck5.csv, stuck6.csv, stuck7.csv, stuck8.csv
