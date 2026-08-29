# Is the buffer hand-off worth keeping?

`SP1_METAL_REUSE` on against off, both arms the GPU path through one binary, git `163c3a2`, Darwin 25.6.0 arm64, 10 pairs per point, measured 2026-08-29.


No workload shows a wall-clock difference this measurement can tell from zero: every interval spans it, and every median difference is smaller than the scatter of the same arm across repeats.

The mechanism disagrees with itself. `blocked` is the only channel the hand-off can act through, and it favours keeping buffers on btc-headers, divrem while favouring dropping them on fib. A saving that changes sign between workloads is not a saving.

Against that, the codewords table left 94 of 1560 hand-offs unclaimed at the end of a commitment -- one in 17 -- each a pointer key on an allocation about to be freed. 0 were still there when that address came back.

The round arrays table left none of its 8040 hand-offs unclaimed.

## btc-headers/standard

| metric | reuse on | reuse off | median difference | 95% interval | noise floor | pairs favouring on |
|---|---|---|---|---|---|---|
| wall clock | 10.62 s | 10.73 s | +0.08 s (+0.8%) | -0.09 to +0.35 s | ±0.20 s | 7/10 |
| blocked (thread-s) | 4.38 s | 4.44 s | +0.01 s (+0.2%) | -0.07 to +0.15 s | ±0.16 s | 5/10 |
| gpu (thread-s) | 1.51 s | 1.52 s | -0.00 s (-0.2%) | -0.15 to +0.15 s | ±0.15 s | 4/10 |

| hand-off table | kept | taken | left unclaimed | on a dead entry's address |
|---|---|---|---|---|
| codewords | 370 | 350 | 20 | 0 |
| round arrays | 1800 | 600 | 0 | 0 |

Over 10 proofs with the hand-off on.

## divrem/large

| metric | reuse on | reuse off | median difference | 95% interval | noise floor | pairs favouring on |
|---|---|---|---|---|---|---|
| wall clock | 27.53 s | 27.55 s | -0.08 s (-0.3%) | -1.12 to +1.47 s | ±1.18 s | 5/10 |
| blocked (thread-s) | 16.47 s | 17.35 s | +0.78 s (+4.5%) | -0.42 to +1.16 s | ±0.47 s | 7/10 |
| gpu (thread-s) | 4.84 s | 4.78 s | +0.08 s (+1.6%) | -0.78 to +0.74 s | ±0.70 s | 5/10 |

| hand-off table | kept | taken | left unclaimed | on a dead entry's address |
|---|---|---|---|---|
| codewords | 650 | 611 | 39 | 0 |
| round arrays | 3390 | 1130 | 0 | 0 |

Over 10 proofs with the hand-off on.

## fib/large

| metric | reuse on | reuse off | median difference | 95% interval | noise floor | pairs favouring on |
|---|---|---|---|---|---|---|
| wall clock | 19.45 s | 19.72 s | +0.27 s (+1.4%) | -0.03 to +0.62 s | ±0.33 s | 7/10 |
| blocked (thread-s) | 14.21 s | 13.30 s | -0.90 s (-6.8%) | -1.47 to +0.57 s | ±0.59 s | 3/10 |
| gpu (thread-s) | 3.55 s | 3.63 s | -0.18 s (-4.9%) | -0.44 to +0.59 s | ±0.41 s | 4/10 |

| hand-off table | kept | taken | left unclaimed | on a dead entry's address |
|---|---|---|---|---|
| codewords | 540 | 505 | 35 | 0 |
| round arrays | 2850 | 950 | 0 | 0 |

Over 10 proofs with the hand-off on.
