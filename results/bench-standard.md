# standard tier — Darwin 25.6.0 arm64 — git 85d07d0 — 2026-08-27T01:18:53
204 timed runs, 34 measured points, 0 failures

| guest | size | config | cyc | shd | cpu_med | gpu_med | speedup | cpu_GB | gpu_GB |
|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 28,004,793 | 5 | 90.39 | 15.04 | 6.01x | 92.32 | 48.18 |
| bits | small | baseline | 1,754,793 | 2 | 11.91 | 3.39 | 3.52x | 18.37 | 8.61 |
| bits | standard | baseline | 14,004,793 | 3 | 49.71 | 9.47 | 5.25x | 67.13 | 26.56 |
| bn254 | large | baseline | 13,800 | 3 | 7.74 | 2.24 | 3.45x | 12.73 | 6.05 |
| bn254 | small | baseline | 7,050 | 3 | 7.50 | 2.14 | 3.51x | 12.47 | 6.29 |
| bn254 | standard | baseline | 9,300 | 3 | 7.56 | 2.14 | 3.54x | 12.28 | 6.06 |
| branch | large | baseline | 35,781,952 | 6 | 107.21 | 18.83 | 5.69x | 95.66 | 53.67 |
| branch | small | baseline | 2,259,444 | 2 | 13.27 | 3.64 | 3.65x | 18.57 | 9.31 |
| branch | standard | baseline | 18,005,165 | 4 | 57.32 | 9.87 | 5.81x | 60.12 | 33.41 |
| divrem | large | baseline | 36,004,793 | 7 | 141.12 | 28.18 | 5.01x | 100.05 | 56.84 |
| divrem | small | baseline | 2,254,793 | 2 | 16.31 | 6.61 | 2.47x | 23.49 | 10.83 |
| divrem | standard | baseline | 17,999,993 | 4 | 72.66 | 19.35 | 3.75x | 75.86 | 41.07 |
| fib | large | baseline | 36,004,809 | 6 | 111.86 | 19.17 | 5.84x | 103.20 | 53.92 |
| fib | small | baseline | 2,254,809 | 2 | 13.75 | 3.71 | 3.70x | 18.28 | 9.75 |
| fib | small | compressed | 2,254,809 | 5 | 27.97 | 8.98 | 3.12x | 22.84 | 10.15 |
| fib | standard | baseline | 18,004,809 | 4 | 59.69 | 10.49 | 5.69x | 61.45 | 33.10 |
| keccak | large | baseline | 60,893 | 3 | 19.83 | 10.22 | 1.94x | 41.92 | 15.90 |
| keccak | small | baseline | 11,893 | 3 | 9.02 | 3.74 | 2.41x | 16.01 | 6.58 |
| keccak | standard | baseline | 32,893 | 3 | 13.38 | 6.48 | 2.07x | 27.42 | 10.68 |
| mem | large | baseline | 16,087,222 | 3 | 52.49 | 10.51 | 4.99x | 60.34 | 27.11 |
| mem | small | baseline | 1,087,222 | 2 | 10.08 | 3.04 | 3.32x | 14.07 | 6.70 |
| mem | standard | baseline | 8,087,222 | 2 | 31.52 | 8.18 | 3.85x | 42.08 | 16.34 |
| mulh | large | baseline | 36,004,812 | 6 | 117.09 | 22.37 | 5.23x | 110.73 | 64.29 |
| mulh | small | baseline | 2,254,812 | 2 | 14.27 | 3.96 | 3.61x | 19.21 | 11.49 |
| mulh | standard | baseline | 18,000,000 | 4 | 62.07 | 11.37 | 5.46x | 69.13 | 41.15 |
| secp256k1 | large | baseline | 13,816 | 3 | 7.76 | 2.15 | 3.60x | 12.78 | 6.19 |
| secp256k1 | small | baseline | 7,066 | 3 | 7.53 | 2.09 | 3.60x | 12.42 | 6.15 |
| secp256k1 | standard | baseline | 9,316 | 3 | 7.60 | 2.17 | 3.51x | 12.28 | 6.11 |
| sha256 | large | baseline | 2,085,041 | 4 | 62.02 | 11.22 | 5.53x | 68.49 | 38.01 |
| sha256 | small | baseline | 265,041 | 2 | 14.27 | 4.17 | 3.42x | 19.19 | 10.62 |
| sha256 | standard | baseline | 1,045,041 | 3 | 35.35 | 8.47 | 4.17x | 43.85 | 22.08 |
| uint256 | large | baseline | 14,809 | 3 | 7.60 | 2.20 | 3.46x | 12.90 | 6.41 |
| uint256 | small | baseline | 7,309 | 3 | 7.41 | 2.10 | 3.53x | 12.73 | 6.20 |
| uint256 | standard | baseline | 9,809 | 3 | 7.49 | 2.14 | 3.50x | 12.72 | 6.27 |

speedup across 34 points: median 3.60x, range 1.94x - 6.01x
  best  bits/large 6.01x
  worst keccak/large 1.94x

Every run verified its proof.
