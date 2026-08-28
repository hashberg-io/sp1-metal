# full tier — Darwin 25.6.0 arm64 — git b6b89a7 — 2026-08-28T02:55:47
366 timed runs, 64 measured points, 6 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 86.30 | 14.45 | 5.97x | 92.16 | 50.65 |
| bits | small | baseline | 1,754,793 | 2 | 11.36 | 3.15 | 3.61x | 18.33 | 8.56 |
| bits | standard | baseline | 14,004,793 | 3 | 43.88 | 8.83 | 4.97x | 67.47 | 27.01 |
| bn254 | large | baseline | 13,800 | 3 | 7.88 | 2.18 | 3.61x | 12.71 | 6.19 |
| bn254 | small | baseline | 7,050 | 3 | 7.61 | 2.03 | 3.75x | 12.06 | 6.19 |
| bn254 | standard | baseline | 9,300 | 3 | 7.72 | 2.11 | 3.67x | 12.25 | 6.18 |
| bn254 | standard | nodebound=1e6 | 9,300 | - | 7.54 | - | - | 12.20 | - |
| branch | large | baseline | 35,781,952 | 6 | 103.21 | 17.99 | 5.74x | 96.11 | 52.58 |
| branch | small | baseline | 2,259,444 | 2 | 12.56 | 3.40 | 3.69x | 18.60 | 9.38 |
| branch | standard | baseline | 18,005,165 | 4 | 50.45 | 9.10 | 5.55x | 61.58 | 32.84 |
| divrem | large | baseline | 36,004,793 | 7 | 134.04 | 25.50 | 5.26x | 99.57 | 52.12 |
| divrem | small | baseline | 2,254,793 | 2 | 14.75 | 4.79 | 3.08x | 23.55 | 10.55 |
| divrem | standard | baseline | 17,999,993 | 4 | 63.72 | 12.86 | 4.96x | 75.17 | 43.43 |
| fib | large | baseline | 36,004,809 | 6 | 106.73 | 18.99 | 5.62x | 103.96 | 57.02 |
| fib | small | baseline | 2,254,809 | 2 | 12.84 | 3.48 | 3.69x | 18.19 | 9.89 |
| fib | small | compressed | 2,254,809 | 5 | 25.99 | 7.03 | 3.70x | 24.61 | 10.15 |
| fib | standard | ablate=ENCODE | 18,004,809 | 4 | - | 11.63 | - | - | 50.44 |
| fib | standard | ablate=FIRST_LAYER | 18,004,809 | 4 | - | 10.67 | - | - | 34.91 |
| fib | standard | ablate=JAGGED | 18,004,809 | 4 | - | 14.45 | - | - | 44.15 |
| fib | standard | ablate=MERKLE | 18,004,809 | 4 | - | 23.50 | - | - | 53.77 |
| fib | standard | ablate=TRANSITION | 18,004,809 | 4 | - | 12.24 | - | - | 40.48 |
| fib | standard | ablate=ZEROCHECK | 18,004,809 | 4 | - | 15.14 | - | - | 35.70 |
| fib | standard | baseline | 18,004,809 | 4 | 55.07 | 10.11 | 5.45x | 62.40 | 34.13 |
| fib | standard | permits=1 | 18,004,809 | 4 | 64.10 | 16.50 | 3.89x | 43.71 | 22.66 |
| fib | standard | permits=2 | 18,004,809 | 4 | 55.76 | 10.97 | 5.08x | 59.68 | 23.89 |
| fib | standard | permits=8 | 18,004,809 | 4 | 52.87 | 9.24 | 5.72x | 62.58 | 34.53 |
| fib | standard | shard=16MiB | 18,004,809 | 4 | 54.57 | 9.53 | 5.73x | 63.21 | 34.25 |
| fib | standard | shard=1MiB | 18,004,809 | 4 | 52.74 | 9.43 | 5.60x | 61.65 | 33.49 |
| fib | standard | shard=2MiB | 18,004,809 | 4 | 52.75 | 9.52 | 5.54x | 62.81 | 33.33 |
| fib | standard | shard=8MiB | 18,004,809 | 4 | 52.98 | 9.38 | 5.65x | 62.32 | 34.11 |
| keccak | large | baseline | 60,893 | 3 | 17.56 | 7.88 | 2.23x | 41.84 | 15.90 |
| keccak | small | baseline | 11,893 | 3 | 8.87 | 2.50 | 3.55x | 16.03 | 7.02 |
| keccak | standard | baseline | 32,893 | 3 | 12.61 | 4.80 | 2.63x | 26.42 | 9.88 |
| matmul | small | baseline | 2,302,299 | 2 | 12.15 | 3.34 | 3.64x | 18.37 | 8.69 |
| matmul | standard | baseline | 18,078,317 | 3 | 46.25 | 9.42 | 4.91x | 65.75 | 26.87 |
| mem | large | baseline | 16,087,222 | 3 | 46.11 | 9.01 | 5.12x | 61.16 | 27.40 |
| mem | small | baseline | 1,087,222 | 2 | 9.67 | 2.82 | 3.43x | 14.05 | 6.67 |
| mem | standard | baseline | 8,087,222 | 2 | 27.92 | 7.26 | 3.85x | 42.07 | 16.41 |
| merkle | small | baseline | 307,275 | 2 | 8.94 | 3.10 | 2.89x | 13.43 | 6.68 |
| merkle | standard | baseline | 1,199,194 | 2 | 15.94 | 5.12 | 3.11x | 24.40 | 11.21 |
| mulh | large | baseline | 36,004,812 | 6 | 108.41 | 21.22 | 5.11x | 107.86 | 66.96 |
| mulh | small | baseline | 2,254,812 | 2 | 12.91 | 3.66 | 3.52x | 19.38 | 11.41 |
| mulh | standard | baseline | 18,000,000 | 4 | 55.40 | 10.63 | 5.21x | 69.04 | 37.92 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.81 | 2.05 | 3.81x | 12.76 | 6.03 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.67 | 1.99 | 3.86x | 12.25 | 6.12 |
| secp256k1 | small | compressed | 7,066 | 7 | 23.73 | 5.92 | 4.01x | 21.56 | 10.21 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.71 | 2.03 | 3.80x | 12.32 | 6.05 |
| secp256k1 | standard | nodebound=1e6 | 9,316 | - | 7.42 | - | - | 12.23 | - |
| sha256 | large | baseline | 2,085,041 | 4 | 54.43 | 10.30 | 5.28x | 68.93 | 37.34 |
| sha256 | small | baseline | 265,041 | 2 | 13.19 | 3.92 | 3.36x | 19.21 | 10.58 |
| sha256 | small | compressed | 265,041 | 5 | 25.55 | 7.41 | 3.45x | 24.99 | 10.84 |
| sha256 | standard | baseline | 1,045,041 | 3 | 30.67 | 7.68 | 3.99x | 43.83 | 21.84 |
| sha256 | standard | permits=1 | 1,045,041 | 3 | 36.82 | 11.15 | 3.30x | 43.83 | 23.49 |
| sha256 | standard | permits=2 | 1,045,041 | 3 | 32.84 | 8.41 | 3.91x | 44.27 | 22.55 |
| sha256 | standard | permits=8 | 1,045,041 | 3 | 31.56 | 7.68 | 4.11x | 43.91 | 22.05 |
| sha256 | standard | shard=16MiB | 1,045,041 | 3 | 31.72 | 7.73 | 4.10x | 43.86 | 21.69 |
| sha256 | standard | shard=1MiB | 1,045,041 | 3 | 30.89 | 7.67 | 4.03x | 44.02 | 21.81 |
| sha256 | standard | shard=2MiB | 1,045,041 | 3 | 31.45 | 7.62 | 4.13x | 44.23 | 22.16 |
| sha256 | standard | shard=8MiB | 1,045,041 | 3 | 31.50 | 7.79 | 4.04x | 43.93 | 21.77 |
| sort | small | baseline | 2,198,832 | 2 | 12.38 | 3.50 | 3.54x | 18.38 | 8.37 |
| sort | standard | baseline | 21,603,702 | 4 | 59.67 | 11.75 | 5.08x | 67.33 | 27.68 |
| uint256 | large | baseline | 14,809 | 3 | 7.84 | 2.19 | 3.59x | 12.92 | 6.12 |
| uint256 | small | baseline | 7,309 | 3 | 7.68 | 2.01 | 3.82x | 12.64 | 6.00 |
| uint256 | standard | baseline | 9,809 | 3 | 7.67 | 2.06 | 3.72x | 12.67 | 6.07 |

speedup across 56 points: median 3.90x, range 2.23x - 5.97x
  best  bits/large 5.97x
  worst keccak/large 2.23x

6 FAILED runs (correctness or timeout):
  secp256k1/standard gpu: timeout after 900.0s
  bn254/standard gpu: timeout after 900.1s
