# apps tier — Darwin 25.6.0 arm64 — git 93f08ac — 2026-08-27T17:23:57
144 timed runs, 24 measured points, 0 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| btc-headers | large | baseline | 21,201,218 | 6 | 102.76 | 17.88 | 5.75x | 89.85 | 43.00 |
| btc-headers | small | baseline | 2,809,378 | 2 | 20.76 | 6.04 | 3.44x | 25.41 | 12.52 |
| btc-headers | standard | baseline | 11,469,570 | 4 | 58.76 | 10.59 | 5.55x | 60.90 | 30.66 |
| btc-schnorr | large | baseline | 1,990,817 | 5 | 35.37 | 12.43 | 2.85x | 35.60 | 16.35 |
| btc-schnorr | small | baseline | 340,142 | 5 | 14.92 | 4.18 | 3.56x | 16.75 | 7.42 |
| btc-schnorr | standard | baseline | 1,072,389 | 5 | 24.63 | 7.95 | 3.10x | 27.01 | 10.85 |
| eth-bls-aggregate | large | baseline | 1,115,444 | 5 | 30.08 | 11.23 | 2.68x | 30.92 | 14.25 |
| eth-bls-aggregate | small | baseline | 382,102 | 5 | 17.50 | 5.79 | 3.03x | 19.22 | 7.83 |
| eth-bls-aggregate | standard | baseline | 627,352 | 5 | 21.80 | 7.68 | 2.84x | 24.23 | 10.20 |
| eth-bls-verify | large | baseline | 41,077,462 | 17 | 286.65 | 93.98 | 3.05x | 104.86 | 42.11 |
| eth-bls-verify | small | baseline | 25,323,904 | 11 | 174.35 | 56.47 | 3.09x | 84.18 | 40.81 |
| eth-bls-verify | standard | baseline | 30,557,611 | 13 | 210.54 | 69.64 | 3.02x | 86.63 | 38.61 |
| eth-ecrecover | large | baseline | 2,129,292 | 6 | 40.38 | 13.97 | 2.89x | 42.02 | 18.86 |
| eth-ecrecover | small | baseline | 294,246 | 6 | 16.01 | 4.34 | 3.69x | 17.18 | 7.88 |
| eth-ecrecover | standard | baseline | 1,083,329 | 6 | 26.71 | 8.51 | 3.14x | 27.69 | 12.28 |
| eth-groth16-inputs | large | baseline | 1,654,078 | 5 | 38.29 | 15.43 | 2.48x | 41.34 | 18.52 |
| eth-groth16-inputs | small | baseline | 322,930 | 5 | 16.93 | 5.17 | 3.27x | 18.94 | 7.97 |
| eth-groth16-inputs | standard | baseline | 890,356 | 5 | 26.92 | 9.56 | 2.82x | 34.22 | 12.90 |
| eth-mpt | large | baseline | 2,496,379 | 4 | 44.42 | 16.52 | 2.69x | 62.38 | 25.32 |
| eth-mpt | small | baseline | 628,015 | 3 | 16.86 | 6.28 | 2.68x | 28.20 | 11.34 |
| eth-mpt | standard | baseline | 1,250,809 | 3 | 26.45 | 10.65 | 2.48x | 46.52 | 17.48 |
| eth-pairing-check | large | baseline | 29,905,266 | 16 | 260.81 | 90.51 | 2.88x | 99.68 | 39.68 |
| eth-pairing-check | small | baseline | 12,360,698 | 8 | 112.78 | 40.29 | 2.80x | 87.48 | 33.44 |
| eth-pairing-check | standard | baseline | 19,866,992 | 12 | 176.83 | 60.36 | 2.93x | 84.52 | 35.91 |

speedup across 24 points: median 2.98x, range 2.48x - 5.75x
  best  btc-headers/large 5.75x
  worst eth-groth16-inputs/large 2.48x

Every run verified its proof.
