# standard tier — Darwin 25.6.0 arm64 — git 33fea39 — 2026-08-26T15:25:31
204 timed runs, 34 measured points, 1 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 92.65 | 31.51 | 2.94x | 92.66 | 48.03 |
| bits | small | baseline | 1,754,793 | 2 | 12.01 | 4.59 | 2.62x | 18.34 | 8.40 |
| bits | standard | baseline | 14,004,793 | 3 | 50.98 | 15.71 | 3.25x | 63.17 | 28.48 |
| bn254 | large | baseline | 13,800 | 3 | 7.91 | 2.34 | 3.38x | 12.77 | 6.14 |
| bn254 | small | baseline | 7,050 | 3 | 7.47 | 2.19 | 3.42x | 12.50 | 6.32 |
| bn254 | standard | baseline | 9,300 | 3 | 7.67 | 2.22 | 3.46x | 12.50 | 6.39 |
| branch | large | baseline | 35,781,952 | 6 | 108.45 | 31.57 | 3.43x | 95.87 | 53.37 |
| branch | small | baseline | 2,259,444 | 2 | 13.45 | 5.06 | 2.66x | 18.58 | 9.13 |
| branch | standard | baseline | 18,005,165 | 4 | 58.06 | 12.51 | 4.64x | 59.79 | 34.14 |
| divrem | large | baseline | 36,004,793 | 7 | 140.91 | 36.36 | 3.88x | 100.25 | 54.96 |
| divrem | small | baseline | 2,254,793 | 2 | 16.02 | 5.92 | 2.71x | 23.62 | 10.84 |
| divrem | standard | baseline | 17,999,993 | 4 | 72.53 | 17.43 | 4.16x | 76.38 | 39.36 |
| fib | large | baseline | 36,004,809 | 6 | 125.94 | 45.26 | 2.78x | 74.44 | 22.99 |
| fib | small | baseline | 2,254,809 | 2 | 13.79 | 4.48 | 3.08x | 18.29 | 9.67 |
| fib | small | compressed | 2,254,809 | 5 | 27.97 | 8.93 | 3.13x | 22.87 | 9.88 |
| fib | standard | baseline | 18,004,809 | 4 | 59.68 | 11.18 | 5.34x | 62.05 | 35.29 |
| keccak | large | baseline+ZEROCHECK_MAX_NODES=8000 | 60,893 | 3 | 19.81 | 8.96 | 2.21x | 42.27 | 15.76 |
| keccak | small | baseline+ZEROCHECK_MAX_NODES=8000 | 11,893 | 3 | 8.90 | 2.66 | 3.34x | 16.01 | 6.97 |
| keccak | standard | baseline+ZEROCHECK_MAX_NODES=8000 | 32,893 | 3 | 13.25 | 5.39 | 2.46x | 26.38 | 9.76 |
| mem | large | baseline | 16,087,222 | 3 | 53.47 | 16.58 | 3.23x | 60.22 | 27.05 |
| mem | small | baseline | 1,087,222 | 2 | 9.98 | 3.76 | 2.65x | 14.07 | 6.60 |
| mem | standard | baseline | 8,087,222 | 2 | 32.07 | 11.26 | 2.85x | 42.07 | 16.37 |
| mulh | large | baseline | 36,004,812 | 6 | 118.62 | 29.26 | 4.05x | 108.20 | 62.92 |
| mulh | small | baseline | 2,254,812 | 2 | 14.13 | 4.66 | 3.03x | 19.33 | 11.31 |
| mulh | standard | baseline | 18,000,000 | 4 | 61.62 | 11.60 | 5.31x | 69.58 | 41.73 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.70 | 2.26 | 3.40x | 12.75 | 5.96 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.45 | 2.12 | 3.51x | 12.44 | 6.44 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.54 | 2.18 | 3.46x | 12.34 | 6.31 |
| sha256 | large | baseline | 2,085,041 | 4 | 61.69 | 12.78 | 4.83x | 68.18 | 37.02 |
| sha256 | small | baseline | 265,041 | 2 | 14.20 | 4.99 | 2.84x | 19.21 | 10.61 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.35 | 8.72 | 4.05x | 43.96 | 21.81 |
| uint256 | large | baseline | 14,809 | 3 | 7.65 | 2.22 | 3.45x | 12.92 | 6.40 |
| uint256 | small | baseline | 7,309 | 3 | 7.49 | 2.06 | 3.64x | 12.85 | 6.21 |
| uint256 | standard | baseline | 9,809 | 3 | 7.76 | 2.22 | 3.50x | 12.67 | 6.34 |

speedup across 34 points: median 3.39x, range 2.21x - 5.34x
  best  fib/standard 5.34x
  worst keccak/large 2.21x

1 FAILED runs (correctness or timeout):
  keccak/standard gpu: timeout after 900.1s
