# apps tier — Darwin 25.6.0 arm64 — git 2cbb19a — 2026-08-28T12:45:31
144 timed runs, 24 measured points, 0 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| btc-headers | large | baseline | 21,201,218 | 6 | 94.60 | 15.79 | 5.99x | 94.77 | 42.87 |
| btc-headers | small | baseline | 2,809,378 | 2 | 18.73 | 5.49 | 3.41x | 25.21 | 12.05 |
| btc-headers | standard | baseline | 11,469,570 | 4 | 51.25 | 9.60 | 5.34x | 60.91 | 31.39 |
| btc-schnorr | large | baseline | 1,990,817 | 5 | 31.83 | 10.44 | 3.05x | 34.73 | 15.49 |
| btc-schnorr | small | baseline | 340,142 | 5 | 13.89 | 3.92 | 3.54x | 16.79 | 7.30 |
| btc-schnorr | standard | baseline | 1,072,389 | 5 | 22.13 | 7.07 | 3.13x | 27.87 | 10.64 |
| eth-bls-aggregate | large | baseline | 1,115,444 | 5 | 27.51 | 9.64 | 2.85x | 31.25 | 13.36 |
| eth-bls-aggregate | small | baseline | 382,102 | 5 | 16.25 | 5.17 | 3.14x | 19.05 | 8.12 |
| eth-bls-aggregate | standard | baseline | 627,352 | 5 | 20.00 | 6.63 | 3.02x | 24.64 | 10.01 |
| eth-bls-verify | large | baseline | 41,077,462 | 17 | 279.56 | 84.13 | 3.32x | 90.07 | 38.86 |
| eth-bls-verify | small | baseline | 25,323,904 | 11 | 163.61 | 48.44 | 3.38x | 84.30 | 39.00 |
| eth-bls-verify | standard | baseline | 30,557,611 | 13 | 205.34 | 62.56 | 3.28x | 84.78 | 38.45 |
| eth-ecrecover | large | baseline | 2,129,292 | 6 | 36.22 | 11.42 | 3.17x | 42.00 | 17.51 |
| eth-ecrecover | small | baseline | 294,246 | 6 | 14.89 | 3.93 | 3.79x | 16.94 | 7.41 |
| eth-ecrecover | standard | baseline | 1,083,329 | 6 | 23.87 | 7.16 | 3.34x | 27.83 | 12.74 |
| eth-groth16-inputs | large | baseline | 1,654,078 | 5 | 34.47 | 12.75 | 2.70x | 41.35 | 16.47 |
| eth-groth16-inputs | small | baseline | 322,930 | 5 | 15.07 | 4.77 | 3.16x | 19.15 | 8.09 |
| eth-groth16-inputs | standard | baseline | 890,356 | 5 | 24.21 | 8.08 | 3.00x | 34.55 | 12.07 |
| eth-mpt | large | baseline | 2,496,379 | 4 | 40.14 | 13.98 | 2.87x | 65.49 | 23.77 |
| eth-mpt | small | baseline | 628,015 | 3 | 15.37 | 5.43 | 2.83x | 28.78 | 11.00 |
| eth-mpt | standard | baseline | 1,250,809 | 3 | 23.90 | 8.97 | 2.67x | 46.70 | 17.26 |
| eth-pairing-check | large | baseline | 29,905,266 | 16 | 248.41 | 77.92 | 3.19x | 100.72 | 42.45 |
| eth-pairing-check | small | baseline | 12,360,698 | 8 | 104.99 | 34.88 | 3.01x | 87.92 | 34.50 |
| eth-pairing-check | standard | baseline | 19,866,992 | 12 | 168.00 | 52.20 | 3.22x | 94.44 | 37.29 |

speedup across 24 points: median 3.17x, range 2.67x - 5.99x
  best  btc-headers/large 5.99x
  worst eth-mpt/standard 2.67x

Every run verified its proof.
