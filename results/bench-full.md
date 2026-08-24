# full tier — Darwin 25.6.0 arm64 — git 8b88c6e — 2026-08-24T17:06:46
349 timed runs, 62 measured points, 9 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 91.23 | 17.29 | 5.28x | 92.13 | 45.61 |
| bits | small | baseline | 1,754,793 | 2 | 12.16 | 3.36 | 3.62x | 18.27 | 8.54 |
| bits | standard | baseline | 14,004,793 | 3 | 50.48 | 9.73 | 5.19x | 66.30 | 27.66 |
| bn254 | large | baseline | 13,800 | 3 | 7.84 | 2.17 | 3.61x | 12.73 | 6.01 |
| bn254 | small | baseline | 7,050 | 3 | 7.57 | 2.09 | 3.62x | 12.42 | 6.08 |
| bn254 | standard | baseline | 9,300 | 3 | 7.65 | 2.10 | 3.64x | 12.32 | 6.36 |
| bn254 | standard | nodebound=1e6 | 9,300 | - | 7.63 | - | - | 12.52 | - |
| branch | large | baseline | 35,781,952 | 6 | 107.59 | 19.36 | 5.56x | 95.85 | 54.28 |
| branch | small | baseline | 2,259,444 | 2 | 13.29 | 3.58 | 3.71x | 18.63 | 9.32 |
| branch | standard | baseline | 18,005,165 | 4 | 56.91 | 9.96 | 5.71x | 60.04 | 33.11 |
| divrem | large | baseline | 36,004,793 | 7 | 140.42 | 30.72 | 4.57x | 99.97 | 54.66 |
| divrem | small | baseline | 2,254,793 | 2 | 16.25 | 4.99 | 3.26x | 23.51 | 10.72 |
| divrem | standard | baseline | 17,999,993 | 4 | 72.05 | 14.44 | 4.99x | 75.20 | 43.86 |
| fib | large | baseline | 36,004,809 | 6 | 115.04 | 23.53 | 4.89x | 103.85 | 56.81 |
| fib | small | baseline | 2,254,809 | 2 | 14.11 | 3.69 | 3.83x | 18.22 | 9.71 |
| fib | small | compressed | 2,254,809 | 5 | 27.62 | - | - | 23.45 | - |
| fib | standard | ablate=ENCODE | 18,004,809 | 4 | - | 12.25 | - | - | 48.46 |
| fib | standard | ablate=FIRST_LAYER | 18,004,809 | 4 | - | 11.15 | - | - | 34.69 |
| fib | standard | ablate=JAGGED | 18,004,809 | 4 | - | 14.57 | - | - | 46.77 |
| fib | standard | ablate=MERKLE | 18,004,809 | 4 | - | 23.85 | - | - | 54.03 |
| fib | standard | ablate=TRANSITION | 18,004,809 | 4 | - | 13.13 | - | - | 39.68 |
| fib | standard | ablate=ZEROCHECK | 18,004,809 | 4 | - | 17.32 | - | - | 35.33 |
| fib | standard | baseline | 18,004,809 | 4 | 59.68 | 10.98 | 5.44x | 61.95 | 35.29 |
| fib | standard | permits=1 | 18,004,809 | 4 | 70.68 | 17.83 | 3.96x | 43.72 | 22.63 |
| fib | standard | permits=2 | 18,004,809 | 4 | 61.88 | 12.06 | 5.13x | 59.47 | 24.54 |
| fib | standard | permits=8 | 18,004,809 | 4 | 59.13 | 10.50 | 5.63x | 62.38 | 33.21 |
| fib | standard | shard=16MiB | 18,004,809 | 4 | 58.85 | 10.29 | 5.72x | 62.73 | 33.45 |
| fib | standard | shard=1MiB | 18,004,809 | 4 | 58.66 | 10.69 | 5.49x | 61.88 | 33.01 |
| fib | standard | shard=2MiB | 18,004,809 | 4 | 58.88 | 10.61 | 5.55x | 63.16 | 32.79 |
| fib | standard | shard=8MiB | 18,004,809 | 4 | 59.27 | 10.32 | 5.74x | 61.41 | 34.04 |
| keccak | large | baseline+ZEROCHECK_MAX_NODES=8000 | 60,893 | 3 | 19.97 | 8.86 | 2.25x | 42.25 | 15.89 |
| keccak | small | baseline+ZEROCHECK_MAX_NODES=8000 | 11,893 | 3 | 9.00 | 2.69 | 3.35x | 15.88 | 6.94 |
| keccak | standard | baseline+ZEROCHECK_MAX_NODES=8000 | 32,893 | 3 | 13.46 | 5.30 | 2.54x | 27.26 | 9.91 |
| matmul | small | baseline | 2,302,299 | 2 | 13.01 | 3.58 | 3.64x | 18.42 | 8.66 |
| matmul | standard | baseline | 18,078,317 | 3 | 52.46 | 10.11 | 5.19x | 61.84 | 27.01 |
| mem | large | baseline | 16,087,222 | 3 | 53.25 | 10.46 | 5.09x | 60.34 | 28.83 |
| mem | small | baseline | 1,087,222 | 2 | 10.03 | 2.95 | 3.40x | 14.11 | 6.72 |
| mem | standard | baseline | 8,087,222 | 2 | 32.33 | 7.99 | 4.05x | 42.06 | 16.52 |
| merkle | small | baseline | 307,275 | 2 | 9.19 | 3.10 | 2.97x | 13.43 | 6.41 |
| merkle | standard | baseline | 1,199,194 | 2 | 17.81 | 5.42 | 3.29x | 24.46 | 11.16 |
| mulh | large | baseline | 36,004,812 | 6 | 117.34 | 25.86 | 4.54x | 110.47 | 63.11 |
| mulh | small | baseline | 2,254,812 | 2 | 13.75 | 3.88 | 3.55x | 19.24 | 11.29 |
| mulh | standard | baseline | 18,000,000 | 4 | 61.67 | 11.60 | 5.32x | 69.74 | 41.71 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.81 | 2.09 | 3.73x | 12.76 | 6.18 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.54 | 2.05 | 3.68x | 12.47 | 6.12 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.58 | 2.12 | 3.57x | 12.48 | 6.24 |
| secp256k1 | standard | nodebound=1e6 | 9,316 | - | 7.63 | - | - | 12.28 | - |
| sha256 | large | baseline | 2,085,041 | 4 | 61.13 | 11.33 | 5.40x | 69.45 | 36.48 |
| sha256 | small | baseline | 265,041 | 2 | 14.20 | 4.07 | 3.49x | 19.20 | 10.67 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.09 | 8.55 | 4.10x | 43.87 | 21.96 |
| sha256 | standard | permits=1 | 1,045,041 | 3 | 41.55 | 11.94 | 3.48x | 43.97 | 23.45 |
| sha256 | standard | permits=2 | 1,045,041 | 3 | 36.69 | 9.38 | 3.91x | 44.09 | 23.79 |
| sha256 | standard | permits=8 | 1,045,041 | 3 | 35.19 | 8.49 | 4.14x | 43.85 | 21.99 |
| sha256 | standard | shard=16MiB | 1,045,041 | 3 | 36.03 | 8.50 | 4.24x | 44.22 | 21.69 |
| sha256 | standard | shard=1MiB | 1,045,041 | 3 | 35.31 | 8.60 | 4.11x | 43.87 | 22.10 |
| sha256 | standard | shard=2MiB | 1,045,041 | 3 | 35.48 | 8.51 | 4.17x | 44.01 | 22.08 |
| sha256 | standard | shard=8MiB | 1,045,041 | 3 | 35.31 | 8.60 | 4.11x | 43.72 | 21.61 |
| sort | small | baseline | 2,198,832 | 2 | 12.95 | 3.61 | 3.59x | 18.38 | 8.33 |
| sort | standard | baseline | 21,603,702 | 4 | 64.84 | 11.87 | 5.46x | 66.72 | 28.58 |
| uint256 | large | baseline | 14,809 | 3 | 7.68 | 2.09 | 3.68x | 12.92 | 6.32 |
| uint256 | small | baseline | 7,309 | 3 | 7.45 | 2.03 | 3.67x | 12.87 | 6.25 |
| uint256 | standard | baseline | 9,809 | 3 | 7.53 | 2.07 | 3.64x | 12.69 | 6.37 |

speedup across 53 points: median 4.05x, range 2.25x - 5.74x
  best  fib/standard 5.74x
  worst keccak/large 2.25x

9 FAILED runs (correctness or timeout):
  keccak/standard gpu: timeout after 900.1s
  keccak/large gpu: timeout after 900.1s
  secp256k1/standard gpu: timeout after 900.1s
  bn254/standard gpu: timeout after 900.0s
