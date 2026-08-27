# full tier — Darwin 25.6.0 arm64 — git 2b0b267 — 2026-08-27T11:24:13
366 timed runs, 64 measured points, 6 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 89.98 | 15.08 | 5.97x | 92.58 | 47.57 |
| bits | small | baseline | 1,754,793 | 2 | 11.98 | 3.37 | 3.56x | 18.35 | 8.57 |
| bits | standard | baseline | 14,004,793 | 3 | 49.42 | 9.41 | 5.25x | 67.71 | 27.04 |
| bn254 | large | baseline | 13,800 | 3 | 7.79 | 2.22 | 3.51x | 12.74 | 6.09 |
| bn254 | small | baseline | 7,050 | 3 | 7.47 | 2.12 | 3.52x | 12.47 | 6.15 |
| bn254 | standard | baseline | 9,300 | 3 | 7.62 | 2.17 | 3.52x | 12.28 | 6.05 |
| bn254 | standard | nodebound=1e6 | 9,300 | - | 7.61 | - | - | 12.46 | - |
| branch | large | baseline | 35,781,952 | 6 | 107.97 | 18.91 | 5.71x | 95.24 | 53.05 |
| branch | small | baseline | 2,259,444 | 2 | 13.31 | 3.62 | 3.68x | 18.63 | 9.34 |
| branch | standard | baseline | 18,005,165 | 4 | 57.12 | 9.83 | 5.81x | 60.43 | 33.76 |
| divrem | large | baseline | 36,004,793 | 7 | 146.36 | 28.30 | 5.17x | 100.93 | 54.63 |
| divrem | small | baseline | 2,254,793 | 2 | 16.69 | 5.16 | 3.24x | 23.55 | 10.76 |
| divrem | standard | baseline | 17,999,993 | 4 | 75.00 | 14.58 | 5.14x | 76.11 | 41.32 |
| fib | large | baseline | 36,004,809 | 6 | 112.14 | 19.74 | 5.68x | 103.07 | 55.45 |
| fib | small | baseline | 2,254,809 | 2 | 13.69 | 3.65 | 3.75x | 18.28 | 9.76 |
| fib | small | compressed | 2,254,809 | 5 | 27.90 | 9.03 | 3.09x | 23.59 | 10.11 |
| fib | standard | ablate=ENCODE | 18,004,809 | 4 | - | 12.24 | - | - | 49.45 |
| fib | standard | ablate=FIRST_LAYER | 18,004,809 | 4 | - | 11.51 | - | - | 33.74 |
| fib | standard | ablate=JAGGED | 18,004,809 | 4 | - | 14.32 | - | - | 42.74 |
| fib | standard | ablate=MERKLE | 18,004,809 | 4 | - | 23.86 | - | - | 56.01 |
| fib | standard | ablate=TRANSITION | 18,004,809 | 4 | - | 12.74 | - | - | 40.30 |
| fib | standard | ablate=ZEROCHECK | 18,004,809 | 4 | - | 17.19 | - | - | 35.23 |
| fib | standard | baseline | 18,004,809 | 4 | 59.07 | 10.33 | 5.72x | 61.93 | 33.29 |
| fib | standard | permits=1 | 18,004,809 | 4 | 73.62 | 18.47 | 3.99x | 43.84 | 22.54 |
| fib | standard | permits=2 | 18,004,809 | 4 | 61.88 | 12.42 | 4.98x | 57.62 | 24.64 |
| fib | standard | permits=8 | 18,004,809 | 4 | 59.11 | 10.47 | 5.64x | 62.90 | 33.28 |
| fib | standard | shard=16MiB | 18,004,809 | 4 | 63.15 | 10.79 | 5.85x | 63.53 | 32.29 |
| fib | standard | shard=1MiB | 18,004,809 | 4 | 59.14 | 10.54 | 5.61x | 61.82 | 33.35 |
| fib | standard | shard=2MiB | 18,004,809 | 4 | 60.46 | 10.91 | 5.54x | 61.80 | 33.30 |
| fib | standard | shard=8MiB | 18,004,809 | 4 | 61.84 | 10.76 | 5.75x | 61.87 | 35.34 |
| keccak | large | baseline | 60,893 | 3 | 19.90 | 9.23 | 2.16x | 42.03 | 15.97 |
| keccak | small | baseline | 11,893 | 3 | 8.97 | 2.81 | 3.19x | 15.90 | 7.02 |
| keccak | standard | baseline | 32,893 | 3 | 13.25 | 5.59 | 2.37x | 26.45 | 10.06 |
| matmul | small | baseline | 2,302,299 | 2 | 12.82 | 3.62 | 3.54x | 18.42 | 8.61 |
| matmul | standard | baseline | 18,078,317 | 3 | 52.13 | 10.02 | 5.20x | 61.86 | 26.96 |
| mem | large | baseline | 16,087,222 | 3 | 52.17 | 10.06 | 5.18x | 60.81 | 27.17 |
| mem | small | baseline | 1,087,222 | 2 | 9.84 | 3.07 | 3.20x | 14.11 | 6.69 |
| mem | standard | baseline | 8,087,222 | 2 | 31.55 | 8.06 | 3.92x | 42.07 | 16.47 |
| merkle | small | baseline | 307,275 | 2 | 9.25 | 3.24 | 2.85x | 13.41 | 6.34 |
| merkle | standard | baseline | 1,199,194 | 2 | 17.81 | 5.57 | 3.20x | 24.44 | 11.09 |
| mulh | large | baseline | 36,004,812 | 6 | 118.81 | 21.81 | 5.45x | 107.21 | 69.91 |
| mulh | small | baseline | 2,254,812 | 2 | 14.22 | 3.86 | 3.68x | 19.33 | 11.45 |
| mulh | standard | baseline | 18,000,000 | 4 | 61.41 | 11.35 | 5.41x | 69.52 | 41.36 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.71 | 2.20 | 3.51x | 12.76 | 5.99 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.48 | 2.11 | 3.54x | 12.43 | 6.19 |
| secp256k1 | small | compressed | 7,066 | 7 | 25.33 | 8.73 | 2.90x | 24.09 | 11.15 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.60 | 2.12 | 3.59x | 12.29 | 6.12 |
| secp256k1 | standard | nodebound=1e6 | 9,316 | - | 7.59 | - | - | 12.49 | - |
| sha256 | large | baseline | 2,085,041 | 4 | 61.61 | 11.31 | 5.45x | 68.27 | 39.17 |
| sha256 | small | baseline | 265,041 | 2 | 14.49 | 4.24 | 3.42x | 19.19 | 10.59 |
| sha256 | small | compressed | 265,041 | 5 | 28.40 | 9.49 | 2.99x | 24.77 | 10.91 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.13 | 8.44 | 4.16x | 43.87 | 21.94 |
| sha256 | standard | permits=1 | 1,045,041 | 3 | 41.75 | 12.07 | 3.46x | 43.91 | 23.42 |
| sha256 | standard | permits=2 | 1,045,041 | 3 | 36.68 | 9.31 | 3.94x | 43.93 | 23.82 |
| sha256 | standard | permits=8 | 1,045,041 | 3 | 35.17 | 8.52 | 4.13x | 43.81 | 21.72 |
| sha256 | standard | shard=16MiB | 1,045,041 | 3 | 37.78 | 8.61 | 4.39x | 43.84 | 21.22 |
| sha256 | standard | shard=1MiB | 1,045,041 | 3 | 35.66 | 8.57 | 4.16x | 43.84 | 22.16 |
| sha256 | standard | shard=2MiB | 1,045,041 | 3 | 36.01 | 8.67 | 4.15x | 43.92 | 22.03 |
| sha256 | standard | shard=8MiB | 1,045,041 | 3 | 35.95 | 8.63 | 4.17x | 43.92 | 21.57 |
| sort | small | baseline | 2,198,832 | 2 | 12.99 | 3.77 | 3.44x | 18.48 | 8.41 |
| sort | standard | baseline | 21,603,702 | 4 | 64.95 | 11.77 | 5.52x | 67.36 | 28.42 |
| uint256 | large | baseline | 14,809 | 3 | 7.62 | 2.19 | 3.48x | 12.91 | 6.28 |
| uint256 | small | baseline | 7,309 | 3 | 7.41 | 2.17 | 3.40x | 12.73 | 6.18 |
| uint256 | standard | baseline | 9,809 | 3 | 7.48 | 2.16 | 3.46x | 12.67 | 6.26 |

speedup across 56 points: median 3.93x, range 2.16x - 5.97x
  best  bits/large 5.97x
  worst keccak/large 2.16x

6 FAILED runs (correctness or timeout):
  secp256k1/standard gpu: timeout after 900.1s
  bn254/standard gpu: timeout after 900.1s
