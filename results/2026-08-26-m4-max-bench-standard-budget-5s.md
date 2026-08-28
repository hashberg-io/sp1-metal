# standard tier — Darwin 25.6.0 arm64 — git e92b927 — 2026-08-26T21:53:26
204 timed runs, 34 measured points, 1 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 89.96 | 15.47 | 5.82x | 92.00 | 46.89 |
| bits | small | baseline | 1,754,793 | 2 | 11.96 | 3.64 | 3.29x | 18.32 | 8.28 |
| bits | standard | baseline | 14,004,793 | 3 | 49.30 | 10.00 | 4.93x | 63.73 | 27.28 |
| bn254 | large | baseline | 13,800 | 3 | 7.78 | 2.43 | 3.20x | 12.77 | 6.10 |
| bn254 | small | baseline | 7,050 | 3 | 7.45 | 2.35 | 3.17x | 12.41 | 6.30 |
| bn254 | standard | baseline | 9,300 | 3 | 7.62 | 2.31 | 3.29x | 12.46 | 6.19 |
| branch | large | baseline | 35,781,952 | 6 | 107.23 | 19.26 | 5.57x | 96.12 | 53.02 |
| branch | small | baseline | 2,259,444 | 2 | 13.20 | 3.89 | 3.40x | 18.56 | 9.07 |
| branch | standard | baseline | 18,005,165 | 4 | 56.81 | 10.22 | 5.56x | 59.55 | 32.72 |
| divrem | large | baseline | 36,004,793 | 7 | 140.49 | 32.51 | 4.32x | 100.16 | 52.68 |
| divrem | small | baseline | 2,254,793 | 2 | 16.22 | 5.45 | 2.97x | 23.53 | 11.07 |
| divrem | standard | baseline | 17,999,993 | 4 | 72.70 | 16.41 | 4.43x | 75.63 | 42.12 |
| fib | large | baseline | 36,004,809 | 6 | 112.05 | 22.94 | 4.88x | 103.06 | 53.76 |
| fib | small | baseline | 2,254,809 | 2 | 13.64 | 3.91 | 3.49x | 18.26 | 9.60 |
| fib | small | compressed | 2,254,809 | 5 | 27.85 | 9.41 | 2.96x | 23.45 | 9.87 |
| fib | standard | baseline | 18,004,809 | 4 | 59.12 | 10.82 | 5.46x | 62.35 | 34.40 |
| keccak | large | baseline | 60,893 | 3 | 19.89 | 8.78 | 2.27x | 41.87 | 15.39 |
| keccak | small | baseline | 11,893 | 3 | 8.95 | 2.73 | 3.28x | 15.93 | 6.72 |
| keccak | standard | baseline | 32,893 | 3 | 13.26 | 5.21 | 2.55x | 27.37 | 9.64 |
| mem | large | baseline | 16,087,222 | 3 | 52.02 | 10.44 | 4.98x | 60.25 | 27.62 |
| mem | small | baseline | 1,087,222 | 2 | 10.05 | 3.19 | 3.15x | 14.08 | 6.54 |
| mem | standard | baseline | 8,087,222 | 2 | 31.25 | 8.56 | 3.65x | 42.08 | 16.33 |
| mulh | large | baseline | 36,004,812 | 6 | 117.24 | 23.02 | 5.09x | 108.63 | 64.63 |
| mulh | small | baseline | 2,254,812 | 2 | 14.37 | 4.26 | 3.38x | 19.25 | 10.92 |
| mulh | standard | baseline | 18,000,000 | 4 | 61.75 | 11.72 | 5.27x | 69.82 | 41.64 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.72 | 2.36 | 3.28x | 12.73 | 5.98 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.50 | 2.26 | 3.32x | 12.40 | 6.03 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.59 | 2.26 | 3.35x | 12.19 | 6.07 |
| sha256 | large | baseline | 2,085,041 | 4 | 61.35 | 19.83 | 3.09x | 69.62 | 38.29 |
| sha256 | small | baseline | 265,041 | 2 | 14.10 | 12.39 | 1.14x | 19.20 | 10.35 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.51 | 12.81 | 2.77x | 43.88 | 22.61 |
| uint256 | large | baseline | 14,809 | 3 | 7.62 | 2.32 | 3.29x | 12.93 | 6.41 |
| uint256 | small | baseline | 7,309 | - | 7.38 | 2.27 | 3.25x | 12.95 | 6.06 |
| uint256 | standard | baseline | 9,809 | 3 | 7.42 | 2.31 | 3.20x | 12.67 | 6.26 |

speedup across 34 points: median 3.31x, range 1.14x - 5.82x
  best  bits/large 5.82x
  worst sha256/small 1.14x

1 FAILED runs (correctness or timeout):
  uint256/small gpu: timeout after 900.0s
