// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to slop-dft 6.4.0 by the sp1-metal project; see CHANGES.md.
// ---------------------------------------------------------------------------
// Step 2: the Reed-Solomon encode inside a BaseFold commitment.
//
// slop's encode is not a coset LDE. `Dft::dft(src, log_blowup, BitReversed, 0)`
// zero-pads the trace to `rows << log_blowup` and takes a forward DFT of each
// column, then bit-reverses the rows. The core prover's blowup is four, not
// two: `CORE_LOG_BLOWUP` is 2. A decimation-in-frequency transform takes
// natural-order input to bit-reversed output directly, so matching slop's
// convention costs no permutation at all — which is why these kernels are
// Gentleman-Sande rather than the Cooley-Tukey shape `ntt_stage` uses.
//
// The trace is row-major, so a row is `cols` contiguous elements and one
// butterfly is elementwise across the row. Threads are laid out with the column
// fastest, which makes every access coalesced.
//
// The twiddle table holds w_n^j for j < n/2; stage m = 2^s reads it with stride
// 2^(log_n - 1 - s).
// ---------------------------------------------------------------------------

struct DifParams {
    uint log_n;
    uint log_m;
    uint cols;
    uint log_blowup;
};

// w_n^k for any k < n, from a table of only the first half.
//
// The second half of the table is the negation of the first, since w_n^(n/2) is
// -1, so the spread's twiddles cost no extra memory. Every other stage indexes
// below n/2 and reads the table directly.
inline uint dif_tw(device const uint* tw, uint k, uint log_n) {
    // Not `half`: that is a type in this language.
    uint mid = 1u << (log_n - 1u);
    uint t = tw[k & (mid - 1u)];
    return (k & mid) ? kb_neg(t) : t;
}

// The first `log_blowup` stages at once.
//
// Every one of them pairs a trace row with a row of the zero padding, so `u+v`
// is `u` and `(u-v)*w` is `u*w`: no butterfly survives, and what the stages
// amount to is that the codeword starts as `blowup` copies of the trace, copy
// `b` scaled row-wise by `w_n^(j * r)`. The exponent `r` is `b` with its
// `log_blowup` bits reversed, which is the decimation-in-frequency shape
// showing through: the copies come out in the same bit-reversed order the whole
// transform does.
//
// Running them together rather than as `log_blowup` passes means the trace is
// read once instead of once per stage, and the padding is never touched at all.
// Every stage below this pairs rows less than `n >> log_blowup` apart, so the
// copies are independent from here on and the rest of the transform can treat
// the codeword as one flat array.
kernel void dif_spread(device const uint* src [[buffer(0)]],
                       device uint* dst [[buffer(1)]],
                       device const uint* tw [[buffer(2)]],
                       constant DifParams& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint j = gid.y;
    uint u = src[j * p.cols + c];
    uint rows = 1u << (p.log_n - p.log_blowup);
    uint blocks = 1u << p.log_blowup;
    for (uint b = 0u; b < blocks; ++b) {
        // `j * r` is below n because j < n >> log_blowup and r < blowup, so the
        // mask is the modular reduction w_n is periodic under, not a truncation.
        uint r = reverse_bits(b) >> (32u - p.log_blowup);
        uint k = (j * r) & ((1u << p.log_n) - 1u);
        dst[(b * rows + j) * p.cols + c] = r == 0u ? u : kb_mul(u, dif_tw(tw, k, p.log_n));
    }
}

// One decimation-in-frequency stage over the whole codeword.
kernel void dif_stage(device uint* a [[buffer(0)]],
                      device const uint* tw [[buffer(1)]],
                      constant DifParams& p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint pair = gid.y;
    uint m = 1u << p.log_m;
    uint j = pair & (m - 1u);
    uint base = (pair >> p.log_m) << (p.log_m + 1u);
    uint i0 = (base + j) * p.cols + c;
    uint i1 = i0 + m * p.cols;
    uint u = a[i0];
    uint v = a[i1];
    a[i0] = kb_add(u, v);
    a[i1] = kb_mul(kb_sub(u, v), tw[j << (p.log_n - 1u - p.log_m)]);
}

// Two decimation-in-frequency stages in one pass over the codeword.
//
// The transform is memory-bound -- one multiply per element per stage, against
// a read and a write of the whole codeword -- so what it costs is passes, and
// a radix-4 butterfly does two stages for one pass at the same multiply count.
//
// Four points `q` apart, where `q` is half the upper stage's half-block. Write
// `W` for the n-th root and `t` for the upper stage's twiddle stride; then
// running stage `2q` and then stage `q` over them leaves
//
//     y0 = (x0+x2) + (x1+x3)
//     y1 = [(x0+x2) - (x1+x3)] * W^(2jt)
//     y2 = [(x0-x2) + i(x1-x3)] * W^(jt)
//     y3 = [(x0-x2) - i(x1-x3)] * W^(3jt)
//
// where `i` is `W^(n/4)`, which is where the second stage's twiddle on the odd
// half collapses to: that twiddle is `W^((j+q)t)` and `q*t` is exactly `n/4`
// whatever the stage. So the pair of stages costs four multiplies per four
// points, the same as the two radix-2 stages it replaces, for one pass instead
// of two.
//
// `i` is `tw[n/4]`, so it is read from the table rather than passed in.
kernel void dif_stage4(device uint* a [[buffer(0)]],
                       device const uint* tw [[buffer(1)]],
                       constant DifParams& p [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint quad = gid.y;
    uint q = 1u << (p.log_m - 1u);
    uint j = quad & (q - 1u);
    uint base = (quad >> (p.log_m - 1u)) << (p.log_m + 1u);
    uint step = q * p.cols;
    uint i0 = (base + j) * p.cols + c;
    uint i1 = i0 + step;
    uint i2 = i1 + step;
    uint i3 = i2 + step;
    uint x0 = a[i0];
    uint x1 = a[i1];
    uint x2 = a[i2];
    uint x3 = a[i3];
    // `j * t` is below n/4 and twice it is below n/2, so those two index the
    // table directly; only `3jt` can reach past its end.
    uint jt = j << (p.log_n - 1u - p.log_m);
    uint root4 = tw[1u << (p.log_n - 2u)];
    uint a02 = kb_add(x0, x2);
    uint s02 = kb_sub(x0, x2);
    uint a13 = kb_add(x1, x3);
    uint is13 = kb_mul(kb_sub(x1, x3), root4);
    a[i0] = kb_add(a02, a13);
    a[i1] = kb_mul(kb_sub(a02, a13), tw[2u * jt]);
    a[i2] = kb_mul(kb_add(s02, is13), tw[jt]);
    a[i3] = kb_mul(kb_sub(s02, is13), dif_tw(tw, 3u * jt, p.log_n));
}

// The last six stages, run while a 64-row by 64-column tile stays in
// threadgroup memory. Those stages pair rows less than 64 apart, so a
// contiguous 64-row window is self-contained and the six stages cost one
// memory pass rather than six.
constant uint DIF_TILE_ROWS = 64;
constant uint DIF_TILE_COLS = 64;
constant uint DIF_TILE_STAGES = 6;

kernel void dif_tail(device uint* a [[buffer(0)]],
                     device const uint* tw [[buffer(1)]],
                     constant DifParams& p [[buffer(2)]],
                     uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 tid2 [[thread_position_in_threadgroup]]) {
    threadgroup uint tile[DIF_TILE_ROWS * DIF_TILE_COLS];
    const uint cells = DIF_TILE_ROWS * DIF_TILE_COLS;
    uint tid = tid2.x;
    uint row0 = tgid.y * DIF_TILE_ROWS;
    uint col0 = tgid.x * DIF_TILE_COLS;
    for (uint k = tid; k < cells; k += 256u) {
        tile[k] = a[(row0 + (k / DIF_TILE_COLS)) * p.cols + col0 + (k % DIF_TILE_COLS)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint back = 0; back < DIF_TILE_STAGES; back++) {
        uint s = DIF_TILE_STAGES - 1u - back;
        uint m = 1u << s;
        uint stride = 1u << (p.log_n - 1u - s);
        for (uint k = tid; k < cells / 2u; k += 256u) {
            uint c = k % DIF_TILE_COLS;
            uint pair = k / DIF_TILE_COLS;
            uint j = pair & (m - 1u);
            uint base = (pair >> s) << (s + 1u);
            uint i0 = (base + j) * DIF_TILE_COLS + c;
            uint i1 = i0 + m * DIF_TILE_COLS;
            uint u = tile[i0];
            uint v = tile[i1];
            tile[i0] = kb_add(u, v);
            tile[i1] = kb_mul(kb_sub(u, v), tw[j * stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint k = tid; k < cells; k += 256u) {
        a[(row0 + (k / DIF_TILE_COLS)) * p.cols + col0 + (k % DIF_TILE_COLS)] = tile[k];
    }
}
