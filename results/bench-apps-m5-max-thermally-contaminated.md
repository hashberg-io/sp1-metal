# apps tier — Darwin 25.6.0 arm64 — git 7f8cd97 — 2026-08-28T08:59:26
144 timed runs, 24 measured points, 0 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| btc-headers | large | baseline | 21,201,218 | 6 | 99.03 | 17.93 | 5.52x | 102.99 | 41.20 |
| btc-headers | small | baseline | 2,809,378 | 2 | 18.65 | 5.50 | 3.39x | 25.55 | 11.98 |
| btc-headers | standard | baseline | 11,469,570 | 4 | 51.68 | 9.67 | 5.34x | 61.46 | 30.82 |
| btc-schnorr | large | baseline | 1,990,817 | 5 | 35.25 | 11.63 | 3.03x | 36.04 | 15.14 |
| btc-schnorr | small | baseline | 340,142 | 5 | 16.59 | 4.31 | 3.85x | 17.31 | 7.16 |
| btc-schnorr | standard | baseline | 1,072,389 | 5 | 25.23 | 7.66 | 3.29x | 27.00 | 11.07 |
| eth-bls-aggregate | large | baseline | 1,115,444 | 5 | 27.46 | 10.15 | 2.70x | 31.59 | 13.40 |
| eth-bls-aggregate | small | baseline | 382,102 | 5 | 16.69 | 5.45 | 3.07x | 19.04 | 8.27 |
| eth-bls-aggregate | standard | baseline | 627,352 | 5 | 20.29 | 7.19 | 2.82x | 24.53 | 10.16 |
| eth-bls-verify | large | baseline | 41,077,462 | 17 | 288.38 | 83.01 | 3.47x | 92.48 | 39.52 |
| eth-bls-verify | small | baseline | 25,323,904 | 11 | 170.74 | 54.59 | 3.13x | 84.06 | 40.50 |
| eth-bls-verify | standard | baseline | 30,557,611 | 13 | 208.59 | 64.71 | 3.22x | 84.48 | 39.66 |
| eth-ecrecover | large | baseline | 2,129,292 | 6 | 42.08 | 14.29 | 2.94x | 43.21 | 16.63 |
| eth-ecrecover | small | baseline | 294,246 | 6 | 19.95 | 4.02 | 4.96x | 17.06 | 7.82 |
| eth-ecrecover | standard | baseline | 1,083,329 | 6 | 28.98 | 9.51 | 3.05x | 27.75 | 12.13 |
| eth-groth16-inputs | large | baseline | 1,654,078 | 5 | 35.61 | 14.22 | 2.50x | 41.46 | 17.80 |
| eth-groth16-inputs | small | baseline | 322,930 | 5 | 16.32 | 5.20 | 3.14x | 19.02 | 8.08 |
| eth-groth16-inputs | standard | baseline | 890,356 | 5 | 24.22 | 8.49 | 2.85x | 33.29 | 12.67 |
| eth-mpt | large | baseline | 2,496,379 | 4 | 50.10 | 19.34 | 2.59x | 65.04 | 24.77 |
| eth-mpt | small | baseline | 628,015 | 3 | 16.07 | 5.83 | 2.76x | 28.90 | 11.18 |
| eth-mpt | standard | baseline | 1,250,809 | 3 | 25.13 | 9.59 | 2.62x | 46.70 | 17.27 |
| eth-pairing-check | large | baseline | 29,905,266 | 16 | 265.48 | 78.51 | 3.38x | 93.08 | 41.40 |
| eth-pairing-check | small | baseline | 12,360,698 | 8 | 112.20 | 40.34 | 2.78x | 93.03 | 32.42 |
| eth-pairing-check | standard | baseline | 19,866,992 | 12 | 177.67 | 59.83 | 2.97x | 100.04 | 38.76 |

speedup across 24 points: median 3.06x, range 2.50x - 5.52x
  best  btc-headers/large 5.52x
  worst eth-groth16-inputs/large 2.50x

Every run verified its proof.
