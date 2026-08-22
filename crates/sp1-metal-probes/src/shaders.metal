#include <metal_stdlib>
using namespace metal;

// KoalaBear base field, Montgomery form with R = 2^32, mirroring p3-koala-bear
// (p3-koala-bear-0.4.3-succinct/src/koala_bear.rs). All arithmetic stays in
// 32-bit integer lanes: Apple GPUs emulate 64-bit integer ops.
constant uint KB_P = 0x7f000001u;
// P^-1 mod 2^32 (p3's borrow-free convention: MU = P^-1, not -P^-1).
constant uint KB_MU = 0x81000001u;
// monty(3): the binomial-extension constant W for x^4 - 3.
constant uint KB_W_MONTY = 0x05fffffau;

// Montgomery reduction of the 64-bit value (hi, lo), valid for hi < P.
// Since t = lo * P^-1 mod 2^32, the low limb of t*P equals lo exactly, so the
// subtraction only involves the high limbs.
inline uint kb_redc(uint hi, uint lo) {
    uint t = lo * KB_MU;
    uint u = mulhi(t, KB_P);
    uint r = hi - u;
    return (hi < u) ? (r + KB_P) : r;
}

inline uint kb_mul(uint a, uint b) {
    return kb_redc(mulhi(a, b), a * b);
}

inline uint kb_add(uint a, uint b) {
    uint s = a + b;
    return (s >= KB_P) ? (s - KB_P) : s;
}

inline uint kb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + (KB_P - b));
}

inline uint kb_neg(uint a) {
    return (a == 0u) ? 0u : (KB_P - a);
}

inline uint kb_double(uint a) {
    return kb_add(a, a);
}

inline uint kb_cube(uint x) {
    return kb_mul(kb_mul(x, x), x);
}

// Degree-4 binomial extension over x^4 = 3; uint4 lanes are the coefficients
// (c0, c1, c2, c3) in Montgomery form, matching p3's BinomialExtensionField.
inline uint4 ext_add(uint4 a, uint4 b) {
    return uint4(kb_add(a.x, b.x), kb_add(a.y, b.y), kb_add(a.z, b.z), kb_add(a.w, b.w));
}

inline uint4 ext_sub(uint4 a, uint4 b) {
    return uint4(kb_sub(a.x, b.x), kb_sub(a.y, b.y), kb_sub(a.z, b.z), kb_sub(a.w, b.w));
}

inline uint4 ext_mul_base(uint4 a, uint b) {
    return uint4(kb_mul(a.x, b), kb_mul(a.y, b), kb_mul(a.z, b), kb_mul(a.w, b));
}

inline uint4 ext_mul(uint4 a, uint4 b) {
    uint c0 = kb_add(
        kb_mul(a.x, b.x),
        kb_mul(KB_W_MONTY, kb_add(kb_add(kb_mul(a.y, b.w), kb_mul(a.z, b.z)), kb_mul(a.w, b.y))));
    uint c1 = kb_add(
        kb_add(kb_mul(a.x, b.y), kb_mul(a.y, b.x)),
        kb_mul(KB_W_MONTY, kb_add(kb_mul(a.z, b.w), kb_mul(a.w, b.z))));
    uint c2 = kb_add(
        kb_add(kb_mul(a.x, b.z), kb_mul(a.y, b.y)),
        kb_add(kb_mul(a.z, b.x), kb_mul(KB_W_MONTY, kb_mul(a.w, b.w))));
    uint c3 = kb_add(
        kb_add(kb_mul(a.x, b.w), kb_mul(a.y, b.z)),
        kb_add(kb_mul(a.z, b.y), kb_mul(a.w, b.x)));
    return uint4(c0, c1, c2, c3);
}

// ---------------------------------------------------------------------------
// A cheaper extension multiply.
//
// Two facts about this field make the schoolbook version wasteful.
//
// W is monty(3), and Montgomery form is linear, so multiplying by W is
// tripling: three multiplies become two additions.
//
// Montgomery reduction is also linear, so REDC(ab) + REDC(cd) = REDC(ab + cd)
// as long as the 64-bit sum stays below P * 2^32. Two products of reduced
// inputs reach at most 2*(P-1)^2 = 9.079e18 against a bound of 9.152e18, so
// every pair of products can share a single reduction.
//
// Together these take the multiply from 19 kb_mul (76 integer multiplies) to
// 52 integer multiplies, with the same result bit for bit.
// ---------------------------------------------------------------------------

struct kb_wide {
    uint hi;
    uint lo;
};

inline kb_wide kb_wmul(uint a, uint b) {
    kb_wide r;
    r.lo = a * b;
    r.hi = mulhi(a, b);
    return r;
}

inline kb_wide kb_wadd(kb_wide a, kb_wide b) {
    kb_wide r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + ((r.lo < a.lo) ? 1u : 0u);
    return r;
}

/// a*b + c*d, with one Montgomery reduction instead of two.
inline uint kb_mul2(uint a, uint b, uint c, uint d) {
    kb_wide t = kb_wadd(kb_wmul(a, b), kb_wmul(c, d));
    return kb_redc(t.hi, t.lo);
}

/// Multiplication by W = monty(3).
inline uint kb_triple(uint x) {
    return kb_add(kb_add(x, x), x);
}

inline uint4 ext_mul_fast(uint4 a, uint4 b) {
    uint c0 = kb_add(
        kb_mul(a.x, b.x),
        kb_triple(kb_add(kb_mul2(a.y, b.w, a.z, b.z), kb_mul(a.w, b.y))));
    uint c1 = kb_add(
        kb_mul2(a.x, b.y, a.y, b.x),
        kb_triple(kb_mul2(a.z, b.w, a.w, b.z)));
    uint c2 = kb_add(
        kb_add(kb_mul2(a.x, b.z, a.y, b.y), kb_mul(a.z, b.x)),
        kb_triple(kb_mul(a.w, b.w)));
    uint c3 = kb_add(kb_mul2(a.x, b.w, a.y, b.z), kb_mul2(a.z, b.y, a.w, b.x));
    return uint4(c0, c1, c2, c3);
}

// ---------------------------------------------------------------------------
// Karatsuba over the same extension.
//
// Splitting a = A0 + A1 x^2 and b = B0 + B1 x^2 turns the quartic product into
// three products of linear factors, and each of those is itself a Karatsuba,
// so 16 base multiplies become 9. Reassembly uses x^4 = W:
//
//   a*b = P0 + M x^2 + P1 x^4,   M = P2 - P0 - P1,  P2 = (A0+A1)(B0+B1)
//
// which gives c0 = p00 + W(m2 + p10), c1 = p01 + W p11,
//             c2 = p02 + m0 + W p12,  c3 = m1.
// ---------------------------------------------------------------------------

struct kb_deg2 {
    uint r0;
    uint r1;
    uint r2;
};

/// (c0 + c1 x)(d0 + d1 x) as a quadratic, in three multiplies.
inline kb_deg2 kb_kmul2(uint c0, uint c1, uint d0, uint d1) {
    kb_deg2 r;
    r.r0 = kb_mul(c0, d0);
    r.r2 = kb_mul(c1, d1);
    r.r1 = kb_sub(kb_mul(kb_add(c0, c1), kb_add(d0, d1)), kb_add(r.r0, r.r2));
    return r;
}

inline uint4 ext_mul_kara(uint4 a, uint4 b) {
    kb_deg2 p0 = kb_kmul2(a.x, a.y, b.x, b.y);
    kb_deg2 p1 = kb_kmul2(a.z, a.w, b.z, b.w);
    kb_deg2 p2 = kb_kmul2(
        kb_add(a.x, a.z), kb_add(a.y, a.w), kb_add(b.x, b.z), kb_add(b.y, b.w));
    uint m0 = kb_sub(p2.r0, kb_add(p0.r0, p1.r0));
    uint m1 = kb_sub(p2.r1, kb_add(p0.r1, p1.r1));
    uint m2 = kb_sub(p2.r2, kb_add(p0.r2, p1.r2));
    return uint4(
        kb_add(p0.r0, kb_triple(kb_add(m2, p1.r0))),
        kb_add(p0.r1, kb_triple(p1.r1)),
        kb_add(kb_add(p0.r2, m0), kb_triple(p1.r2)),
        m1);
}

// ---- end of shared field arithmetic ----

// ---------------------------------------------------------------------------
// Probe: streaming bandwidth (STREAM triad, integer lanes).
// ---------------------------------------------------------------------------

kernel void triad(device const uint4* a [[buffer(0)]],
                  device const uint4* b [[buffer(1)]],
                  device uint4* c [[buffer(2)]],
                  constant uint& scale [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid] + scale * b[gid];
}

// ---------------------------------------------------------------------------
// Probes: register-resident integer op throughput. Eight independent
// dependency chains per thread hide instruction latency; the xor keeps every
// chain data-dependent so nothing folds at compile time. Each iteration does
// one op-of-interest plus one xor per chain.
// ---------------------------------------------------------------------------

constant uint TP_MULC[8] = {
    0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu,
    0x165667b1u, 0xd3a2646du, 0xfd7046c5u, 0xb55a4f09u,
};

kernel void tp_add(device uint* out [[buffer(0)]],
                   constant uint& iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    uint a[4];
    uint b[4];
    for (uint j = 0; j < 4; j++) {
        a[j] = gid * TP_MULC[j] + 1u;
        b[j] = gid ^ TP_MULC[j + 4];
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 4; j++) {
            a[j] += b[j];
            b[j] ^= a[j];
        }
    }
    out[gid] = a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3];
}

kernel void tp_mul_lo(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = gid * TP_MULC[j] + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = (x[j] * TP_MULC[j]) ^ gid;
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

kernel void tp_mul_hi(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = gid * TP_MULC[j] + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = mulhi(x[j], TP_MULC[j]) ^ gid;
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

kernel void tp_kb_mul(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = (gid * TP_MULC[j]) % (KB_P - 1u) + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = kb_mul(x[j], x[j] | 1u);
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// Probes: streamed elementwise field multiplication (bandwidth-bound shape).
// ---------------------------------------------------------------------------

kernel void kb_mul_elems(device const uint* a [[buffer(0)]],
                         device const uint* b [[buffer(1)]],
                         device uint* c [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    c[gid] = kb_mul(a[gid], b[gid]);
}

kernel void ext_mul_elems(device const uint4* a [[buffer(0)]],
                          device const uint4* b [[buffer(1)]],
                          device uint4* c [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    c[gid] = ext_mul(a[gid], b[gid]);
}

// ---------------------------------------------------------------------------
// Probes: BaseFold fold and sumcheck fix-last-variable, mirroring
// slop-multilinear's fold_mle (out[i] = in[2i] + beta * in[2i+1]) and
// mle_fix_last_variable (out[i] = x + alpha * (y - x) on interleaved pairs).
// ---------------------------------------------------------------------------

kernel void fold_base(device const uint* in [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      constant uint& beta [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = kb_add(in[2u * gid], kb_mul(beta, in[2u * gid + 1u]));
}

kernel void fold_ext(device const uint4* in [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint4& beta [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = ext_add(in[2u * gid], ext_mul(beta, in[2u * gid + 1u]));
}

kernel void flv_base(device const uint* in [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint4& alpha [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint x = in[2u * gid];
    uint y = in[2u * gid + 1u];
    uint4 r = ext_mul_base(alpha, kb_sub(y, x));
    r.x = kb_add(r.x, x);
    out[gid] = r;
}

kernel void flv_ext(device const uint4* in [[buffer(0)]],
                    device uint4* out [[buffer(1)]],
                    constant uint4& alpha [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint4 x = in[2u * gid];
    uint4 y = in[2u * gid + 1u];
    out[gid] = ext_add(ext_mul(alpha, ext_sub(y, x)), x);
}

// ---------------------------------------------------------------------------
// Probe: Poseidon2 width-16 over KoalaBear, D=3, 8 external + 20 internal
// rounds, mirroring p3-poseidon2 (external M4 light MDS + block sums) and
// p3-koala-bear's DiffusionMatrixKoalaBear (shift-based internal diagonal).
// Round constants arrive in Montgomery form via buffer(2): 8*16 external
// (first 4 rounds then last 4), then 20 internal.
// ---------------------------------------------------------------------------

constant uint P2_DIAG_SHIFTS[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};

inline void p2_mat4(thread uint* x) {
    uint t01 = kb_add(x[0], x[1]);
    uint t23 = kb_add(x[2], x[3]);
    uint t0123 = kb_add(t01, t23);
    uint t01123 = kb_add(t0123, x[1]);
    uint t01233 = kb_add(t0123, x[3]);
    x[3] = kb_add(t01233, kb_double(x[0]));
    x[1] = kb_add(t01123, kb_double(x[2]));
    x[0] = kb_add(t01123, t01);
    x[2] = kb_add(t01233, t23);
}

inline void p2_external(thread uint* s) {
    for (uint i = 0; i < 16; i += 4) {
        p2_mat4(s + i);
    }
    uint sums[4];
    for (uint j = 0; j < 4; j++) {
        sums[j] = kb_add(kb_add(s[j], s[4 + j]), kb_add(s[8 + j], s[12 + j]));
    }
    for (uint i = 0; i < 16; i++) {
        s[i] = kb_add(s[i], sums[i & 3u]);
    }
}

inline void p2_full_round(thread uint* s, constant uint* rc) {
    for (uint i = 0; i < 16; i++) {
        s[i] = kb_cube(kb_add(s[i], rc[i]));
    }
    p2_external(s);
}

// The internal diffusion mirrors DiffusionMatrixKoalaBear::permute_mut, which
// accumulates raw Montgomery values into a 64-bit sum before one reduction;
// here the 64-bit accumulator is two explicit 32-bit limbs (hi stays tiny, so
// kb_redc's hi < P precondition holds).
inline void p2_internal(thread uint* s, uint rc) {
    s[0] = kb_cube(kb_add(s[0], rc));
    uint plo = 0u;
    uint phi = 0u;
    for (uint i = 1; i < 16; i++) {
        plo += s[i];
        phi += (plo < s[i]) ? 1u : 0u;
    }
    uint flo = plo + s[0];
    uint fhi = phi + ((flo < s[0]) ? 1u : 0u);
    uint n0 = kb_neg(s[0]);
    uint slo = plo + n0;
    uint shi = phi + ((slo < n0) ? 1u : 0u);
    uint new0 = kb_redc(shi, slo);
    for (uint i = 1; i < 16; i++) {
        uint v = s[i];
        uint sh = P2_DIAG_SHIFTS[i - 1];
        uint addlo = v << sh;
        uint addhi = (sh == 0u) ? 0u : (v >> (32u - sh));
        uint l = flo + addlo;
        uint h = fhi + addhi + ((l < addlo) ? 1u : 0u);
        s[i] = kb_redc(h, l);
    }
    s[0] = new0;
}

inline void p2_permute(thread uint* s, constant uint* rc) {
    p2_external(s);
    for (uint r = 0; r < 4; r++) {
        p2_full_round(s, rc + r * 16u);
    }
    for (uint r = 0; r < 20; r++) {
        p2_internal(s, rc[128u + r]);
    }
    for (uint r = 4; r < 8; r++) {
        p2_full_round(s, rc + r * 16u);
    }
}

kernel void poseidon2_16(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint* rc [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    uint s[16];
    for (uint i = 0; i < 16; i++) {
        s[i] = in[gid * 16u + i];
    }
    p2_permute(s, rc);
    for (uint i = 0; i < 16; i++) {
        out[gid * 16u + i] = s[i];
    }
}

// ---------------------------------------------------------------------------
// Probe: synthetic zerocheck round. One thread per row of a column-major
// base-field trace evaluates T degree-3 monomials, combines them with powers
// of alpha (extension field), multiplies by eq(row), and threadgroup-reduces
// to one partial extension-field sum per group. This prices the
// constraint-evaluation shape without a real chip's constraint transpiler.
// ---------------------------------------------------------------------------

struct ZcParams {
    uint n;
    uint num_cols;
    uint num_terms;
};

kernel void zc_round(device const uint* trace [[buffer(0)]],
                     device const uint4* eq [[buffer(1)]],
                     device const uint4* alpha_pows [[buffer(2)]],
                     device const uint4* term_cols [[buffer(3)]],
                     constant ZcParams& p [[buffer(4)]],
                     device uint4* partials [[buffer(5)]],
                     uint gid [[thread_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 shared[256];
    uint4 acc = uint4(0u);
    if (gid < p.n) {
        for (uint t = 0; t < p.num_terms; t++) {
            uint4 cols = term_cols[t];
            uint m = kb_mul(
                kb_mul(trace[cols.x * p.n + gid], trace[cols.y * p.n + gid]),
                trace[cols.z * p.n + gid]);
            acc = ext_add(acc, ext_mul_base(alpha_pows[t], m));
        }
        acc = ext_mul(acc, eq[gid]);
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] = ext_add(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[tgid] = shared[0];
    }
}

// A kernel that does almost nothing: used to price the fixed cost of a
// dispatch, separately from the fixed cost of a command-buffer round trip.
kernel void noop(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        out[0] = out[0] + 1u;
    }
}

// One radix-2 decimation-in-time butterfly stage, the inner loop of an NTT and
// so of the Reed-Solomon encode inside a BaseFold commitment. Each thread owns
// one butterfly: the pair it touches is `m` apart, and `tw` holds the `m`
// twiddles for this stage.
kernel void ntt_stage(device uint* a [[buffer(0)]],
                      device const uint* tw [[buffer(1)]],
                      constant uint& log_m [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint m = 1u << log_m;
    uint j = gid & (m - 1u);
    uint block = gid >> log_m;
    uint i = (block << (log_m + 1u)) + j;
    uint u = a[i];
    uint t = kb_mul(a[i + m], tw[j]);
    a[i] = kb_add(u, t);
    a[i + m] = kb_sub(u, t);
}

// One sumcheck round of a logup GKR layer, in the shape sp1-hypercube's
// `sum_as_poly_in_last_t_variables` computes it: over each pair of consecutive
// entries, the round polynomial's evaluation at the first point uses the even
// entry, and its evaluation at the second uses the pair's sum. Five extension
// accumulators come out — numerator and denominator at each of the two points,
// plus the eq sum — and the caller combines them with lambda.
kernel void gkr_round(device const uint4* num_0 [[buffer(0)]],
                      device const uint4* num_1 [[buffer(1)]],
                      device const uint4* den_0 [[buffer(2)]],
                      device const uint4* den_1 [[buffer(3)]],
                      device const uint4* eq [[buffer(4)]],
                      constant uint& pairs [[buffer(5)]],
                      device uint4* partials [[buffer(6)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 shared[256];
    uint4 acc[5];
    for (uint k = 0u; k < 5u; k++) {
        acc[k] = uint4(0u);
    }
    if (gid < pairs) {
        uint i = 2u * gid;
        uint4 n0a = num_0[i], n0b = num_0[i + 1u];
        uint4 n1a = num_1[i], n1b = num_1[i + 1u];
        uint4 d0a = den_0[i], d0b = den_0[i + 1u];
        uint4 d1a = den_1[i], d1b = den_1[i + 1u];
        uint4 qa = eq[i], qb = eq[i + 1u];
        uint4 numer = ext_add(ext_mul(d0a, n1a), ext_mul(d1a, n0a));
        acc[0] = ext_mul(qa, numer);
        acc[1] = ext_mul(qa, ext_mul(d0a, d1a));
        uint4 n0h = ext_add(n0a, n0b);
        uint4 n1h = ext_add(n1a, n1b);
        uint4 d0h = ext_add(d0a, d0b);
        uint4 d1h = ext_add(d1a, d1b);
        uint4 qh = ext_add(qa, qb);
        uint4 numer_h = ext_add(ext_mul(d0h, n1h), ext_mul(d1h, n0h));
        acc[2] = ext_mul(qh, numer_h);
        acc[3] = ext_mul(qh, ext_mul(d0h, d1h));
        acc[4] = qh;
    }
    for (uint k = 0u; k < 5u; k++) {
        shared[tid] = acc[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shared[tid] = ext_add(shared[tid], shared[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            partials[tgid * 5u + k] = shared[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// Step 1: the GPU-side challenger.
//
// Every Fiat-Shamir challenge in SP1 comes from a duplex sponge over the same
// width-16 Poseidon2 permutation above: SP1GlobalContext is
// KoalaBearDegree4Duplex, whose Challenger is p3's
// DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>. Because each sumcheck round's
// challenge is derived from the previous round's output, a challenger that
// lives on the host forces one command-buffer round trip per round, and the
// launch probe measured that at 87 us against a 1.6 us marginal dispatch.
// Running the sponge on the device instead lets every round of a fold be
// encoded into a single submission.
//
// The state machine mirrors p3 exactly, including the details that matter for
// parity: observing clears the output buffer, a full input buffer duplexes
// immediately, sampling duplexes if there is pending input or no output left,
// and samples are *popped from the end* of the output buffer.
// ---------------------------------------------------------------------------

struct Challenger {
    uint sponge[16];
    uint input_buf[8];
    uint output_buf[8];
    uint input_len;
    uint output_len;
};

inline void chal_duplex(thread Challenger* c, constant uint* rc) {
    uint s[16];
    for (uint i = 0; i < 16; i++) {
        s[i] = c->sponge[i];
    }
    // Overwrite the first `input_len` elements with the buffered input.
    for (uint i = 0; i < c->input_len; i++) {
        s[i] = c->input_buf[i];
    }
    c->input_len = 0u;
    p2_permute(s, rc);
    for (uint i = 0; i < 16; i++) {
        c->sponge[i] = s[i];
    }
    for (uint i = 0; i < 8; i++) {
        c->output_buf[i] = s[i];
    }
    c->output_len = 8u;
}

inline void chal_observe(thread Challenger* c, constant uint* rc, uint v) {
    c->output_len = 0u;
    c->input_buf[c->input_len] = v;
    c->input_len += 1u;
    if (c->input_len == 8u) {
        chal_duplex(c, rc);
    }
}

inline uint chal_sample_base(thread Challenger* c, constant uint* rc) {
    if (c->input_len != 0u || c->output_len == 0u) {
        chal_duplex(c, rc);
    }
    c->output_len -= 1u;
    return c->output_buf[c->output_len];
}

inline void chal_observe_ext(thread Challenger* c, constant uint* rc, uint4 v) {
    chal_observe(c, rc, v.x);
    chal_observe(c, rc, v.y);
    chal_observe(c, rc, v.z);
    chal_observe(c, rc, v.w);
}

inline uint4 chal_sample_ext(thread Challenger* c, constant uint* rc) {
    uint c0 = chal_sample_base(c, rc);
    uint c1 = chal_sample_base(c, rc);
    uint c2 = chal_sample_base(c, rc);
    uint c3 = chal_sample_base(c, rc);
    return uint4(c0, c1, c2, c3);
}

kernel void chal_init(device Challenger* c [[buffer(0)]]) {
    for (uint i = 0; i < 16; i++) {
        c->sponge[i] = 0u;
    }
    c->input_len = 0u;
    c->output_len = 0u;
}

// Observe `n_obs` base elements, then sample `n_samp` extension elements.
// One thread: the sponge is inherently sequential.
kernel void chal_observe_sample(device Challenger* state [[buffer(0)]],
                                device const uint* vals [[buffer(1)]],
                                device uint4* out [[buffer(2)]],
                                constant uint& n_obs [[buffer(3)]],
                                constant uint& n_samp [[buffer(4)]],
                                constant uint* rc [[buffer(5)]]) {
    Challenger c = *state;
    for (uint i = 0; i < n_obs; i++) {
        chal_observe(&c, rc, vals[i]);
    }
    for (uint i = 0; i < n_samp; i++) {
        out[i] = chal_sample_ext(&c, rc);
    }
    *state = c;
}

// ---------------------------------------------------------------------------
// Step 1: a BaseFold round with Fiat-Shamir in the loop.
//
// One round of the fold slop's `fold_mle` performs, in the shape the prover
// runs it: sum the even and odd entries to form the round message, observe both
// into the challenger, sample the fold challenge, then fold. Splitting the
// reduction from the challenger keeps one kernel usable by both the host-driven
// and the device-driven paths, so the two differ only in where the challenge
// comes from.
// ---------------------------------------------------------------------------

// Per-threadgroup partial sums of the even and odd entries.
kernel void sc_partials(device const uint4* in [[buffer(0)]],
                        device uint4* partials [[buffer(1)]],
                        constant uint& n_pairs [[buffer(2)]],
                        uint gid [[thread_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 sh_even[256];
    threadgroup uint4 sh_odd[256];
    uint4 e = uint4(0u, 0u, 0u, 0u);
    uint4 o = uint4(0u, 0u, 0u, 0u);
    if (gid < n_pairs) {
        e = in[2u * gid];
        o = in[2u * gid + 1u];
    }
    sh_even[tid] = e;
    sh_odd[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh_even[tid] = ext_add(sh_even[tid], sh_even[tid + s]);
            sh_odd[tid] = ext_add(sh_odd[tid], sh_odd[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[2u * tgid] = sh_even[0];
        partials[2u * tgid + 1u] = sh_odd[0];
    }
}

// Reduce the per-threadgroup partials to the two-element round message.
// One threadgroup.
kernel void sc_reduce(device const uint4* partials [[buffer(0)]],
                      device uint4* msg [[buffer(1)]],
                      constant uint& n_partials [[buffer(2)]],
                      uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint4 sh_even[256];
    threadgroup uint4 sh_odd[256];
    uint4 e = uint4(0u, 0u, 0u, 0u);
    uint4 o = uint4(0u, 0u, 0u, 0u);
    for (uint i = tid; i < n_partials; i += 256u) {
        e = ext_add(e, partials[2u * i]);
        o = ext_add(o, partials[2u * i + 1u]);
    }
    sh_even[tid] = e;
    sh_odd[tid] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh_even[tid] = ext_add(sh_even[tid], sh_even[tid + s]);
            sh_odd[tid] = ext_add(sh_odd[tid], sh_odd[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        msg[0] = sh_even[0];
        msg[1] = sh_odd[0];
    }
}

// Observe the round message and sample the fold challenge, on the device.
// This is the dispatch that removes the host from the round loop.
kernel void sc_challenge(device const uint4* msg [[buffer(0)]],
                         device Challenger* state [[buffer(1)]],
                         device uint4* beta [[buffer(2)]],
                         constant uint* rc [[buffer(3)]]) {
    Challenger c = *state;
    chal_observe_ext(&c, rc, msg[0]);
    chal_observe_ext(&c, rc, msg[1]);
    beta[0] = chal_sample_ext(&c, rc);
    *state = c;
}

// `fold_ext` with the challenge read from device memory rather than baked into
// the command buffer, so the fold can be encoded before the challenge exists.
kernel void fold_ext_dev(device const uint4* in [[buffer(0)]],
                         device uint4* out [[buffer(1)]],
                         device const uint4* beta [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = ext_add(in[2u * gid], ext_mul(beta[0], in[2u * gid + 1u]));
}

// ---------------------------------------------------------------------------
// Step 2: the Reed-Solomon encode inside a BaseFold commitment.
//
// slop's encode is not a coset LDE. `Dft::dft(src, log_blowup, BitReversed, 0)`
// zero-pads the trace to `rows << log_blowup` and takes a forward DFT of each
// column, then bit-reverses the rows. A decimation-in-frequency transform takes
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
};

// The first stage, specialised for a blowup of 2. Its partner rows are the
// zero padding, so `u + v` is `u` and `(u - v) * w^j` is `u * w^j`: the stage
// reads the trace once and never touches the padding.
kernel void dif_spread(device const uint* src [[buffer(0)]],
                       device uint* dst [[buffer(1)]],
                       device const uint* tw [[buffer(2)]],
                       constant DifParams& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x;
    uint j = gid.y;
    uint u = src[j * p.cols + c];
    dst[j * p.cols + c] = u;
    dst[(j + (1u << (p.log_n - 1u))) * p.cols + c] = kb_mul(u, tw[j]);
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

// ---------------------------------------------------------------------------
// Step 6: the Merkle tree over a committed codeword.
//
// slop commits a tensor by hashing each row of the row-major matrix into a
// digest with `PaddingFreeSponge<KoalaPerm, 16, 8, 8>` and then compressing
// pairs of digests up the tree with `TruncatedPermutation<KoalaPerm, 2, 8, 16>`.
// Both are the same width-16 Poseidon2 permutation, so both are one kernel each
// and neither needs the host between layers: the whole tree is a single
// submission, which is the lesson the challenger spike ended on.
// ---------------------------------------------------------------------------

// One digest per row. Overwrite-mode sponge: absorb the row eight elements at a
// time into the first eight lanes, permuting after each chunk, including a
// short final chunk.
kernel void merkle_leaves(device const uint* rows [[buffer(0)]],
                          device uint* digests [[buffer(1)]],
                          constant uint& width [[buffer(2)]],
                          constant uint& n_rows [[buffer(3)]],
                          constant uint* rc [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= n_rows) {
        return;
    }
    uint s[16];
    for (uint i = 0; i < 16; i++) {
        s[i] = 0u;
    }
    device const uint* row = rows + gid * width;
    for (uint base = 0; base < width; base += 8u) {
        uint take = min(8u, width - base);
        for (uint i = 0; i < take; i++) {
            s[i] = row[base + i];
        }
        p2_permute(s, rc);
    }
    for (uint i = 0; i < 8; i++) {
        digests[gid * 8u + i] = s[i];
    }
}

// One layer of the tree: a parent is the first eight lanes of the permutation
// applied to its two children laid end to end.
kernel void merkle_compress(device const uint* children [[buffer(0)]],
                            device uint* parents [[buffer(1)]],
                            constant uint& n_parents [[buffer(2)]],
                            constant uint* rc [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n_parents) {
        return;
    }
    uint s[16];
    for (uint i = 0; i < 8; i++) {
        s[i] = children[(2u * gid) * 8u + i];
        s[8u + i] = children[(2u * gid + 1u) * 8u + i];
    }
    p2_permute(s, rc);
    for (uint i = 0; i < 8; i++) {
        parents[gid * 8u + i] = s[i];
    }
}

// The commitment is not the Merkle root. slop binds the shape into it:
// `compress([root, hash([log_height, total_width])])`, so a commitment carries
// the dimensions it was made over. A backend that returned the bare root would
// produce commitments the verifier rejects.
kernel void merkle_finalize(device const uint* root [[buffer(0)]],
                            device uint* out [[buffer(1)]],
                            constant uint& log_height [[buffer(2)]],
                            constant uint& total_width [[buffer(3)]],
                            constant uint* rc [[buffer(4)]]) {
    uint s[16];
    for (uint i = 0; i < 16; i++) {
        s[i] = 0u;
    }
    s[0] = log_height;
    s[1] = total_width;
    p2_permute(s, rc);
    uint h[8];
    for (uint i = 0; i < 8; i++) {
        h[i] = s[i];
    }
    for (uint i = 0; i < 8; i++) {
        s[i] = root[i];
        s[8u + i] = h[i];
    }
    p2_permute(s, rc);
    for (uint i = 0; i < 8; i++) {
        out[i] = s[i];
    }
}
