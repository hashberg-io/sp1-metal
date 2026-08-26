# standard tier — Darwin 25.6.0 arm64 — git 6d15661 — 2026-08-24T13:39:38
198 timed runs, 33 measured points, 5 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 90.90 | 17.07 | 5.32x | 91.57 | 48.73 |
| bits | small | baseline | 1,754,793 | 2 | 11.92 | 3.36 | 3.55x | 18.36 | 8.59 |
| bits | standard | baseline | 14,004,793 | 3 | 49.45 | 9.16 | 5.40x | 64.33 | 26.54 |
| bn254 | large | baseline | 13,800 | 3 | 7.84 | 2.15 | 3.65x | 12.72 | 6.30 |
| bn254 | small | baseline | 7,050 | 3 | 7.55 | 2.03 | 3.72x | 12.40 | 6.15 |
| bn254 | standard | baseline | 9,300 | 3 | 7.68 | 2.09 | 3.67x | 12.30 | 6.39 |
| branch | large | baseline | 35,781,952 | 6 | 107.31 | 19.39 | 5.54x | 95.67 | 51.49 |
| branch | small | baseline | 2,259,444 | 2 | 13.11 | 3.54 | 3.71x | 18.55 | 9.24 |
| branch | standard | baseline | 18,005,165 | 4 | 57.00 | 9.94 | 5.73x | 59.43 | 32.33 |
| divrem | large | baseline | 36,004,793 | 7 | 141.56 | 29.41 | 4.81x | 100.53 | 54.16 |
| divrem | small | baseline | 2,254,793 | 2 | 16.12 | 5.01 | 3.22x | 23.52 | 10.49 |
| divrem | standard | baseline | 17,999,993 | 4 | 72.05 | 14.13 | 5.10x | 76.20 | 41.64 |
| fib | large | baseline | 36,004,809 | 6 | 111.11 | 22.71 | 4.89x | 103.25 | 54.41 |
| fib | small | baseline | 2,254,809 | 2 | 14.01 | 3.71 | 3.78x | 18.25 | 9.62 |
| fib | standard | baseline | 18,004,809 | 4 | 59.57 | 10.45 | 5.70x | 62.21 | 33.31 |
| keccak | large | baseline+ZEROCHECK_MAX_NODES=8000 | 60,893 | - | 19.85 | - | - | 42.33 | - |
| keccak | small | baseline+ZEROCHECK_MAX_NODES=8000 | 11,893 | 3 | 9.00 | 2.67 | 3.38x | 15.95 | 6.64 |
| keccak | standard | baseline+ZEROCHECK_MAX_NODES=8000 | 32,893 | 3 | 13.36 | 5.31 | 2.52x | 27.25 | 10.22 |
| mem | large | baseline | 16,087,222 | 3 | 51.73 | 10.04 | 5.15x | 64.72 | 27.68 |
| mem | small | baseline | 1,087,222 | 2 | 9.89 | 2.96 | 3.34x | 14.11 | 6.69 |
| mem | standard | baseline | 8,087,222 | 2 | 31.28 | 8.14 | 3.84x | 42.08 | 16.33 |
| mulh | large | baseline | 36,004,812 | 6 | 116.61 | 23.06 | 5.06x | 108.96 | 65.53 |
| mulh | small | baseline | 2,254,812 | 2 | 14.20 | 3.90 | 3.64x | 19.22 | 11.32 |
| mulh | standard | baseline | 18,000,000 | 4 | 61.88 | 11.19 | 5.53x | 69.52 | 36.58 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.75 | 2.08 | 3.74x | 12.85 | 6.19 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.53 | 2.06 | 3.65x | 12.41 | 6.12 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.59 | 2.07 | 3.66x | 12.49 | 6.30 |
| sha256 | large | baseline | 2,085,041 | 4 | 60.79 | 11.19 | 5.43x | 69.15 | 38.34 |
| sha256 | small | baseline | 265,041 | 2 | 14.23 | 4.05 | 3.51x | 19.15 | 10.68 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.88 | 8.54 | 4.20x | 44.22 | 21.93 |
| uint256 | large | baseline | 14,809 | 3 | 7.65 | 2.10 | 3.64x | 12.89 | 6.42 |
| uint256 | small | baseline | 7,309 | 3 | 7.48 | 2.07 | 3.62x | 12.88 | 6.23 |
| uint256 | standard | baseline | 9,809 | 3 | 7.51 | 2.08 | 3.61x | 12.65 | 6.29 |

speedup across 32 points: median 3.73x, range 2.52x - 5.73x
  best  branch/standard 5.73x
  worst keccak/standard 2.52x

5 FAILED runs (correctness or timeout):
  keccak/standard gpu: timeout after 900.0513370037079s
  keccak/large gpu: timeout after 899.9202136993408s
