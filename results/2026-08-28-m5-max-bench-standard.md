# standard tier — Darwin 25.6.0 arm64 — git b6b89a7 — 2026-08-28T00:49:33
204 timed runs, 34 measured points, 0 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 82.51 | 13.96 | 5.91x | 92.98 | 46.46 |
| bits | small | baseline | 1,754,793 | 2 | 11.67 | 3.16 | 3.69x | 18.28 | 8.56 |
| bits | standard | baseline | 14,004,793 | 3 | 43.43 | 8.54 | 5.08x | 67.64 | 26.91 |
| bn254 | large | baseline | 13,800 | 3 | 7.79 | 2.21 | 3.52x | 12.68 | 6.21 |
| bn254 | small | baseline | 7,050 | 3 | 7.62 | 2.10 | 3.63x | 12.09 | 6.13 |
| bn254 | standard | baseline | 9,300 | 3 | 7.68 | 2.11 | 3.64x | 12.28 | 6.28 |
| branch | large | baseline | 35,781,952 | 6 | 103.56 | 18.24 | 5.68x | 96.20 | 49.03 |
| branch | small | baseline | 2,259,444 | 2 | 12.28 | 3.39 | 3.62x | 18.60 | 9.33 |
| branch | standard | baseline | 18,005,165 | 4 | 50.90 | 9.00 | 5.66x | 59.26 | 33.13 |
| divrem | large | baseline | 36,004,793 | 7 | 135.29 | 25.82 | 5.24x | 99.65 | 54.62 |
| divrem | small | baseline | 2,254,793 | 2 | 14.73 | 4.76 | 3.10x | 23.49 | 10.62 |
| divrem | standard | baseline | 17,999,993 | 4 | 64.13 | 12.80 | 5.01x | 76.08 | 40.00 |
| fib | large | baseline | 36,004,809 | 6 | 104.30 | 18.16 | 5.74x | 103.05 | 54.20 |
| fib | small | baseline | 2,254,809 | 2 | 13.15 | 3.49 | 3.77x | 18.20 | 9.87 |
| fib | small | compressed | 2,254,809 | 5 | 25.74 | 7.06 | 3.65x | 24.17 | 10.13 |
| fib | standard | baseline | 18,004,809 | 4 | 53.24 | 9.20 | 5.79x | 61.18 | 32.76 |
| keccak | large | baseline | 60,893 | 3 | 17.96 | 7.93 | 2.26x | 41.87 | 15.90 |
| keccak | small | baseline | 11,893 | 3 | 9.11 | 2.58 | 3.53x | 16.07 | 6.90 |
| keccak | standard | baseline | 32,893 | 3 | 12.63 | 4.84 | 2.61x | 27.26 | 9.76 |
| mem | large | baseline | 16,087,222 | 3 | 46.50 | 9.41 | 4.94x | 65.10 | 27.37 |
| mem | small | baseline | 1,087,222 | 2 | 9.43 | 2.86 | 3.30x | 14.04 | 6.71 |
| mem | standard | baseline | 8,087,222 | 2 | 27.79 | 7.30 | 3.81x | 42.05 | 16.40 |
| mulh | large | baseline | 36,004,812 | 6 | 110.60 | 23.07 | 4.79x | 107.80 | 61.51 |
| mulh | small | baseline | 2,254,812 | 2 | 12.78 | 3.62 | 3.53x | 19.34 | 11.35 |
| mulh | standard | baseline | 18,000,000 | 4 | 54.81 | 10.01 | 5.48x | 68.99 | 37.04 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.82 | 2.14 | 3.66x | 12.70 | 6.04 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.68 | 1.99 | 3.86x | 12.18 | 6.11 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.72 | 2.01 | 3.84x | 12.33 | 6.09 |
| sha256 | large | baseline | 2,085,041 | 4 | 53.72 | 10.38 | 5.17x | 69.20 | 37.66 |
| sha256 | small | baseline | 265,041 | 2 | 13.17 | 3.96 | 3.33x | 19.15 | 10.55 |
| sha256 | standard | baseline | 1,045,041 | 3 | 30.71 | 7.76 | 3.96x | 43.92 | 21.80 |
| uint256 | large | baseline | 14,809 | 3 | 7.83 | 2.12 | 3.70x | 13.00 | 6.24 |
| uint256 | small | baseline | 7,309 | 3 | 7.61 | 2.03 | 3.75x | 12.63 | 5.89 |
| uint256 | standard | baseline | 9,809 | 3 | 7.70 | 2.04 | 3.77x | 12.55 | 6.02 |

speedup across 34 points: median 3.77x, range 2.26x - 5.91x
  best  bits/large 5.91x
  worst keccak/large 2.26x

Every run verified its proof.
