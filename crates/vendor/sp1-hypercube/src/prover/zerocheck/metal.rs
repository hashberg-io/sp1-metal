// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.
//! The zerocheck round reduction, generated per chip and run on Metal.
//!
//! The plan for this phase assumed the generator could not live here, because
//! enumerating chips needs `sp1-core-machine` and that depends on this crate.
//! It does not have to enumerate anything: the prover is handed the AIR, and
//! this crate already asks AIRs for their symbolic constraints elsewhere. So
//! the constraints are read from the AIR in hand, turned into a kernel, and
//! compiled the first time a chip is seen.
//!
//! It also assumed compiling would have to happen ahead of time. That is true
//! of the elliptic-curve precompiles, which run to 178,000 constraint nodes,
//! and false of everything else: the sixty-two core chips compile in 15.5
//! seconds between them, once per process.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use slop_multilinear::{Mle, PaddedMle};
use slop_uni_stark::{Entry, SymbolicExpression, SymbolicVariable};

use sp1_metal_probes::stats::{self, Input, Output};

use crate::logup_gkr::metal::EF;

const P: u32 = 0x7f00_0001;
const THREADS: u64 = 256;

/// Field arithmetic shared with the probe suite's kernels, so the generated
/// code computes in the representation everything else was checked in.
fn header() -> &'static str {
    include_str!("../../../../../sp1-metal-probes/src/shaders.metal")
        .split("// ---- end of shared field arithmetic ----")
        .next()
        .expect("shader header sentinel")
}

/// Turning a row-major trace into a column-major one.
///
/// The round kernel reads a whole row per thread, so consecutive threads start
/// a row apart; the probe that measures the same arithmetic on a column-major
/// trace runs **3.51 times faster** on identical data. A transpose is one
/// coalesced pass at close to the machine's bandwidth, which is cheap against a
/// reduction that is nowhere near it.
///
/// Tiled through threadgroup memory so that both the read and the write are
/// coalesced -- the padding column is what keeps the tile's banks from
/// colliding. Sixteen-wide tiles for extension elements, where a wider one
/// would take more threadgroup memory than a group is worth.
const TRANSPOSE_KERNELS: &str = r#"
kernel void zc_transpose_base(device const uint* src [[buffer(0)]],
                              device uint* dst [[buffer(1)]],
                              constant uint2& dims [[buffer(2)]],
                              uint2 tgid [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup uint tile[32][33];
    uint r0 = tgid.y * 32u, c0 = tgid.x * 32u;
    uint r = r0 + tid.y, c = c0 + tid.x;
    if (r < dims.x && c < dims.y) {
        tile[tid.y][tid.x] = src[r * dims.y + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rt = r0 + tid.x, ct = c0 + tid.y;
    if (rt < dims.x && ct < dims.y) {
        dst[ct * dims.x + rt] = tile[tid.x][tid.y];
    }
}

kernel void zc_transpose_ext(device const uint4* src [[buffer(0)]],
                             device uint4* dst [[buffer(1)]],
                             constant uint2& dims [[buffer(2)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup uint4 tile[16][17];
    uint r0 = tgid.y * 16u, c0 = tgid.x * 16u;
    uint r = r0 + tid.y, c = c0 + tid.x;
    if (r < dims.x && c < dims.y) {
        tile[tid.y][tid.x] = src[r * dims.y + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rt = r0 + tid.x, ct = c0 + tid.y;
    if (rt < dims.x && ct < dims.y) {
        dst[ct * dims.x + rt] = tile[tid.x][tid.y];
    }
}
"#;

/// Whether the round reads a column-major copy of the trace. **Off, measured.**
///
/// The probe says a column-major trace runs the same arithmetic 3.51 times
/// faster, and in the real kernel it is 21% *slower*: 3.24 GPU-seconds against
/// 2.68. Two reasons, and the first is the interesting one.
///
/// The probe's row-major variant reads three random columns of sixty-four per
/// term, so it touches a scattered handful of each row and wastes most of every
/// cache line it fetches. A real constraint DAG reads most of the columns of
/// the row it is given, so the line is used either way and there was never much
/// coalescing to win. **The probe measured a worse access pattern than the one
/// it was standing in for.**
///
/// And the transpose is not free: a read and a write of the whole trace, about
/// 36 GB a proof, against a round that was not bandwidth-bound to begin with.
///
/// Kept behind the switch because the row-major reads look like an obvious
/// problem and are not.
fn column_major() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_ZC_LAYOUT").map(|v| v == "1").unwrap_or(false))
}

/// Whether the three interpolation points' bodies are emitted interleaved.
///
/// **On.** The three compute independent things and name only their own
/// temporaries, so emitting them one after another leaves the machine one
/// dependency chain to follow where there could be three -- and the compiler
/// does not do this for itself. Interleaving them line by line takes the
/// zerocheck kernel from 2.85 GPU-seconds to 2.50, measured three runs each
/// with no overlap between the two sets, and costs no register pressure: every
/// pipeline still reports the hardware maximum of 1024 threads a group.
///
/// It is a textual operation on the emitted lines and changes nothing about
/// what is computed.
fn interleave() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_ZC_ILP").map(|v| v != "0").unwrap_or(true))
}

fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % P as u64) as u32
}

fn monty_of<Fx: PrimeField32>(x: Fx) -> u32 {
    to_monty(x.as_canonical_u32())
}

/// Whether a value lives in the base field or the extension.
///
/// A subexpression built only from constants stays in the base field however
/// deep it is; anything that touches a trace column does not, once the columns
/// themselves are extension elements. Tracking this per node rather than
/// promoting everything keeps the cheap arithmetic cheap: an `ext_mul` costs
/// nine base multiplies, an `ext_mul_base` costs four, and a `kb_mul` one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Base,
    Ext,
}

impl Ty {
    fn decl(self) -> &'static str {
        match self {
            Ty::Base => "uint",
            Ty::Ext => "uint4",
        }
    }
}

struct Codegen {
    body: String,
    memo: HashMap<usize, (String, Ty)>,
    next: usize,
    adds: usize,
    muls: usize,
    /// Multiplies counted in base-field multiplies, which is what the machine
    /// actually issues: one for base by base, four for an extension by a base,
    /// nine for Karatsuba on two extensions.
    base_muls: usize,
    consts: usize,
    loads: usize,
    /// Whether the trace columns hold extension elements, as they do in every
    /// zerocheck round after the first.
    ext_columns: bool,
    /// Set if any constraint reads a row other than the current one. The
    /// zerocheck folder hands the AIR a single row, so this must stay clear.
    reads_next_row: bool,
    /// Distinguishes the three copies of a body emitted for the interpolation
    /// points, so their temporaries do not collide.
    prefix: &'static str,
    /// Which interpolation point this copy reads its columns at.
    point: u8,
    /// Whether the trace this body reads has been transposed to column-major.
    column_major: bool,
    /// Whether this is a zerocheck round body, whose trace is row-major and
    /// read a row pair at a time, rather than the single-point kernel the
    /// spike measured, whose synthetic trace is column-major.
    round: bool,
}

impl Codegen {
    fn new(ext_columns: bool) -> Self {
        Self {
            body: String::new(),
            memo: HashMap::new(),
            next: 0,
            adds: 0,
            muls: 0,
            base_muls: 0,
            consts: 0,
            loads: 0,
            ext_columns,
            reads_next_row: false,
            prefix: "t",
            point: 0,
            column_major: false,
            round: false,
        }
    }

    /// A copy that reads its columns interpolated to `point`.
    fn at_point(ext_columns: bool, prefix: &'static str, point: u8) -> Self {
        Self { prefix, point, round: true, column_major: column_major(), ..Self::new(ext_columns) }
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("{}{}", self.prefix, self.next - 1)
    }

    fn bind(&mut self, ty: Ty, expr: &str) -> (String, Ty) {
        let name = self.fresh();
        writeln!(self.body, "    {} {name} = {expr};", ty.decl()).unwrap();
        (name, ty)
    }

    fn column(&mut self, table: &str, index: usize, offset: usize) -> (String, Ty) {
        self.loads += 1;
        if offset > 0 {
            self.reads_next_row = true;
        }
        let ty = if self.ext_columns { Ty::Ext } else { Ty::Base };
        // The interpolation helpers index `m[base + i]`, so a column-major
        // trace needs nothing new from them: the base becomes the row and the
        // "column index" becomes the column's offset into its own run.
        let idx = if self.column_major {
            format!("{index}u * p.rows")
        } else {
            format!("{index}u")
        };
        let expr = if self.point == 0 {
            format!("{table}[{table}_b0 + {idx}]")
        } else {
            let pt = self.point;
            let suffix = if self.ext_columns { "" } else { "_base" };
            // Each trace has its own row count, and the CPU decides whether a
            // row pair has a second row per trace as well, so the guard is not
            // shared: a preprocessed trace one row shorter than the main one
            // would otherwise interpolate against a row that is not there.
            let has1 = if table == "prep" { "has1p" } else { "has1" };
            format!("zc_at_{pt}{suffix}({table}, {table}_b0, {table}_b1, {idx}, {has1})")
        };
        self.bind(ty, &expr)
    }

    fn var<Fx: PrimeField32>(&mut self, v: &SymbolicVariable<Fx>) -> (String, Ty) {
        match v.entry {
            Entry::Main { offset } => self.column("main", v.index, offset),
            Entry::Permutation { offset } => self.column("main", v.index, offset),
            Entry::Preprocessed { offset } => self.column("prep", v.index, offset),
            // Public values and challenges are base-field however the columns look.
            Entry::Public | Entry::Challenge => {
                let expr = format!("pv[{}u]", v.index);
                self.bind(Ty::Base, &expr)
            }
        }
    }

    fn emit_rc<Fx: PrimeField32>(&mut self, r: &Rc<SymbolicExpression<Fx>>) -> (String, Ty) {
        let key = Rc::as_ptr(r) as usize;
        if let Some(hit) = self.memo.get(&key) {
            return hit.clone();
        }
        let out = self.emit(r);
        self.memo.insert(key, out.clone());
        out
    }

    /// `a + b`, in whichever field the two operands force.
    fn add(&mut self, a: (String, Ty), b: (String, Ty)) -> (String, Ty) {
        self.adds += 1;
        match (a.1, b.1) {
            (Ty::Base, Ty::Base) => {
                let e = format!("kb_add({}, {})", a.0, b.0);
                self.bind(Ty::Base, &e)
            }
            (Ty::Ext, Ty::Ext) => {
                let e = format!("ext_add({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Ext, Ty::Base) => {
                let e = format!("ext_add_base({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Base, Ty::Ext) => {
                let e = format!("ext_add_base({}, {})", b.0, a.0);
                self.bind(Ty::Ext, &e)
            }
        }
    }

    fn sub(&mut self, a: (String, Ty), b: (String, Ty)) -> (String, Ty) {
        self.adds += 1;
        match (a.1, b.1) {
            (Ty::Base, Ty::Base) => {
                let e = format!("kb_sub({}, {})", a.0, b.0);
                self.bind(Ty::Base, &e)
            }
            (Ty::Ext, Ty::Ext) => {
                let e = format!("ext_sub({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Ext, Ty::Base) => {
                let e = format!("ext_sub_base({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Base, Ty::Ext) => {
                let e = format!("base_sub_ext({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
        }
    }

    fn mul(&mut self, a: (String, Ty), b: (String, Ty)) -> (String, Ty) {
        self.muls += 1;
        self.base_muls += match (a.1, b.1) {
            (Ty::Base, Ty::Base) => 1,
            (Ty::Ext, Ty::Ext) => 9,
            _ => 4,
        };
        match (a.1, b.1) {
            (Ty::Base, Ty::Base) => {
                let e = format!("kb_mul({}, {})", a.0, b.0);
                self.bind(Ty::Base, &e)
            }
            (Ty::Ext, Ty::Base) => {
                let e = format!("ext_mul_base({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Base, Ty::Ext) => {
                let e = format!("ext_mul_base({}, {})", b.0, a.0);
                self.bind(Ty::Ext, &e)
            }
            (Ty::Ext, Ty::Ext) => {
                let e = format!("ext_mul_kara({}, {})", a.0, b.0);
                self.bind(Ty::Ext, &e)
            }
        }
    }

    fn emit<Fx: PrimeField32>(&mut self, e: &SymbolicExpression<Fx>) -> (String, Ty) {
        match e {
            SymbolicExpression::Variable(v) => self.var(v),
            // The zerocheck folder leaves the row selectors unimplemented, so a
            // constraint set that reached for one would not evaluate on the CPU
            // either. Kept as base-field zero so the shape stays honest.
            SymbolicExpression::IsFirstRow => self.bind(Ty::Base, "is_first"),
            SymbolicExpression::IsLastRow => self.bind(Ty::Base, "is_last"),
            SymbolicExpression::IsTransition => self.bind(Ty::Base, "is_transition"),
            SymbolicExpression::Constant(c) => {
                self.consts += 1;
                let e = format!("{}u", monty_of(*c));
                self.bind(Ty::Base, &e)
            }
            SymbolicExpression::Add { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                self.add(a, b)
            }
            SymbolicExpression::Sub { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                self.sub(a, b)
            }
            SymbolicExpression::Neg { x, .. } => {
                let a = self.emit_rc(x);
                self.adds += 1;
                match a.1 {
                    Ty::Base => {
                        let e = format!("kb_neg({})", a.0);
                        self.bind(Ty::Base, &e)
                    }
                    Ty::Ext => {
                        let e = format!("ext_neg({})", a.0);
                        self.bind(Ty::Ext, &e)
                    }
                }
            }
            SymbolicExpression::Mul { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                self.mul(a, b)
            }
        }
    }
}


/// One chip's zerocheck round reduction, as a Metal kernel.
///
/// The reduction is a sum over row pairs of the constraint polynomial evaluated
/// at three interpolation points, weighted by eq. Only the constraint
/// evaluation is chip-specific; the interpolation, the interaction adjustment
/// and the reduction are the same for every chip, so the body is generated
/// three times — once per point — and dropped into a fixed wrapper.
///
/// Columns are read interpolated rather than staged into a per-thread array: a
/// wide chip's row would be tens of kilobytes, which no thread has.
/// A chip's constraint DAG, counted by how each node varies with the
/// interpolation point.
///
/// The body below is emitted once per point, and what makes the three copies
/// differ is only how a column is read. A column's value is affine in the
/// point, so every node is a polynomial in it whose degree is that node's
/// multiplicative depth in columns — and the degree says how much of the three
/// copies is redundant:
///
/// - degree 0 touches no column, so one evaluation serves all three points;
/// - degree 1 is a straight line, so the third point is `2*v(2) - v(0)` — worth
///   taking only where the node is a multiply, since replacing an add with an
///   add and a double is no saving;
/// - degree 2 and above need all three.
///
/// Counted before anything is changed, because "the DAG is emitted three times"
/// does not say how much of it could be emitted once.
fn dag_shape<Fx: PrimeField32>(constraints: &[SymbolicExpression<Fx>]) -> [usize; 4] {
    // total, degree 0, degree-1 multiplies, everything else
    fn walk<Fx: PrimeField32>(
        e: &SymbolicExpression<Fx>,
        seen: &mut HashMap<usize, u8>,
        out: &mut [usize; 4],
    ) -> u8 {
        fn go<Fx: PrimeField32>(
            r: &Rc<SymbolicExpression<Fx>>,
            seen: &mut HashMap<usize, u8>,
            out: &mut [usize; 4],
        ) -> u8 {
            let key = Rc::as_ptr(r) as usize;
            if let Some(hit) = seen.get(&key) {
                return *hit;
            }
            let d = walk(r, seen, out);
            seen.insert(key, d);
            d
        }
        out[0] += 1;
        let (degree, is_mul) = match e {
            SymbolicExpression::Variable(v) => {
                (u8::from(!matches!(v.entry, Entry::Public | Entry::Challenge)), false)
            }
            SymbolicExpression::Constant(_)
            | SymbolicExpression::IsFirstRow
            | SymbolicExpression::IsLastRow
            | SymbolicExpression::IsTransition => (0, false),
            SymbolicExpression::Neg { x, .. } => (go(x, seen, out), false),
            SymbolicExpression::Add { x, y, .. } | SymbolicExpression::Sub { x, y, .. } => {
                (go(x, seen, out).max(go(y, seen, out)), false)
            }
            SymbolicExpression::Mul { x, y, .. } => {
                (go(x, seen, out).saturating_add(go(y, seen, out)), true)
            }
        };
        match (degree, is_mul) {
            (0, _) => out[1] += 1,
            (1, true) => out[2] += 1,
            _ => out[3] += 1,
        }
        degree
    }
    let (mut out, mut seen) = ([0usize; 4], HashMap::new());
    for c in constraints {
        walk(c, &mut seen, &mut out);
    }
    out
}

fn transpile_round<Fx: PrimeField32>(
    constraints: &[SymbolicExpression<Fx>],
    is_first_round: bool,
    ext_columns: bool,
) -> (String, usize, usize, bool) {
    let shape = dag_shape(constraints);
    sp1_metal_probes::tally::record("zerocheck: dag nodes", shape[0], 0);
    sp1_metal_probes::tally::record("zerocheck: dag degree0", shape[1], 0);
    sp1_metal_probes::tally::record("zerocheck: dag degree1 mul", shape[2], 0);
    sp1_metal_probes::tally::record("zerocheck: dag rest", shape[3], 0);
    let mut bodies = String::new();
    let mut nodes = 0usize;
    let mut base_muls = 0usize;
    let mut copies: Vec<String> = Vec::new();
    let mut reads_next_row = false;
    for (prefix, point, acc) in [("a", 0u8, "acc0"), ("b", 2, "acc2"), ("c", 4, "acc4")] {
        // The first round knows the polynomial vanishes at 0, so that copy of
        // the body would be dead weight.
        if point == 0 && is_first_round {
            writeln!(bodies, "        uint4 {acc} = uint4(0u);").unwrap();
            continue;
        }
        let mut copy = String::new();
        let mut cg = Codegen::at_point(ext_columns, prefix, point);
        let names: Vec<(String, Ty)> = constraints.iter().map(|c| cg.emit(c)).collect();
        nodes += cg.next;
        base_muls += cg.base_muls;
        reads_next_row |= cg.reads_next_row;
        write!(copy, "{}", cg.body).unwrap();
        writeln!(copy, "        uint4 {acc} = uint4(0u);").unwrap();
        for (k, (n, ty)) in names.iter().enumerate() {
            let term = match ty {
                Ty::Base => format!("ext_mul_base(ap[{k}u], {n})"),
                Ty::Ext => format!("ext_mul_kara(ap[{k}u], {n})"),
            };
            writeln!(copy, "        {acc} = ext_add({acc}, {term});").unwrap();
        }
        copies.push(copy);
    }
    // The three copies are independent -- each names only its own temporaries --
    // so emitting them one after another leaves the machine one dependency
    // chain to follow where there could be three. Interleaving them line by
    // line is a textual operation and changes nothing about what is computed.
    if interleave() {
        let mut lines: Vec<std::str::Lines> = copies.iter().map(|c| c.lines()).collect();
        let mut any = true;
        while any {
            any = false;
            for l in &mut lines {
                if let Some(line) = l.next() {
                    writeln!(bodies, "{line}").unwrap();
                    any = true;
                }
            }
        }
    } else {
        for c in &copies {
            write!(bodies, "{c}").unwrap();
        }
    }
    let col = if ext_columns { "uint4" } else { "uint" };
    let gkr_mul = if ext_columns { "ext_mul_kara" } else { "ext_mul_base" };
    let at2 = if ext_columns { "zc_at_2" } else { "zc_at_2_base" };
    // Column-major puts a row pair's two entries next to each other, so the
    // base is the row and the stride is folded into the column index.
    let bases = if column_major() {
        "        uint main_b0 = 2u * q;\n        uint main_b1 = 2u * q + 1u;\n        uint prep_b0 = main_b0;\n        uint prep_b1 = main_b1;"
    } else {
        "        uint main_b0 = (2u * q) * p.main_w;\n        uint main_b1 = (2u * q + 1u) * p.main_w;\n        uint prep_b0 = (2u * q) * p.prep_w;\n        uint prep_b1 = (2u * q + 1u) * p.prep_w;"
    };
    let kernel = format!(
        r#"
struct ZcParams {{
    uint pairs;
    uint rows;
    uint main_w;
    uint prep_w;
    uint prep_rows;
}};

kernel void zc_round(device const {col}* main [[buffer(0)]],
                     device const {col}* prep [[buffer(1)]],
                     device const uint* pv [[buffer(2)]],
                     device const uint4* eq [[buffer(3)]],
                     device const uint4* ap [[buffer(4)]],
                     device const uint4* gp [[buffer(5)]],
                     device uint4* partials [[buffer(6)]],
                     constant ZcParams& p [[buffer(7)]],
                     uint gid [[thread_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tgid [[threadgroup_position_in_grid]]) {{
    threadgroup uint4 s0[256];
    threadgroup uint4 s2[256];
    threadgroup uint4 s4[256];
    uint4 y0 = uint4(0u), y2 = uint4(0u), y4 = uint4(0u);
    if (gid < p.pairs) {{
        uint q = gid;
        bool has1 = (2u * q + 1u) < p.rows;
        bool has1p = (2u * q + 1u) < p.prep_rows;
{bases}
{bodies}
        // The interaction adjustment: every column, main then preprocessed,
        // against its batching power. The point-4 adjustment is inferred from
        // the other two rather than computed again, as the CPU does.
        uint4 g0 = uint4(0u), g2 = uint4(0u);
        for (uint i = 0u; i < p.main_w; i++) {{
            uint mi = i * {stride};
            g0 = ext_add(g0, {gkr_mul}(gp[i], main[main_b0 + mi]));
            g2 = ext_add(g2, {gkr_mul}(gp[i], {at2}(main, main_b0, main_b1, mi, has1)));
        }}
        for (uint i = 0u; i < p.prep_w; i++) {{
            uint pi = i * {stride};
            uint4 w = gp[p.main_w + i];
            g0 = ext_add(g0, {gkr_mul}(w, prep[prep_b0 + pi]));
            g2 = ext_add(g2, {gkr_mul}(w, {at2}(prep, prep_b0, prep_b1, pi, has1p)));
        }}
        uint4 g4 = ext_sub(ext_add(g2, g2), g0);
        uint4 e = eq[q];
        y0 = ext_mul_kara(ext_add(acc0, g0), e);
        y2 = ext_mul_kara(ext_add(acc2, g2), e);
        y4 = ext_mul_kara(ext_add(acc4, g4), e);
    }}
    s0[tid] = y0;
    s2[tid] = y2;
    s4[tid] = y4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            s0[tid] = ext_add(s0[tid], s0[tid + s]);
            s2[tid] = ext_add(s2[tid], s2[tid + s]);
            s4[tid] = ext_add(s4[tid], s4[tid + s]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
        partials[3u * tgid] = s0[0];
        partials[3u * tgid + 1u] = s2[0];
        partials[3u * tgid + 2u] = s4[0];
    }}
}}
"#,
        bodies = bodies,
        bases = bases,
        col = col,
        stride = if column_major() { "p.rows" } else { "1u" },
        gkr_mul = gkr_mul,
        at2 = at2,
    );
    let source = format!("{}\nconstant uint KB_ONE = {}u;\n{}", header(), to_monty(1), kernel);
    (source, nodes, base_muls, reads_next_row)
}


struct Context {
    device: Device,
    /// One pipeline per chip and round kind, built the first time it is asked
    /// for. `None` records a chip whose kernel would not build, so it is not
    /// attempted again.
    /// The pipeline for a chip and how many DAG nodes its body evaluates,
    /// which is what the reduction's cost is proportional to.
    pipelines: Mutex<HashMap<String, Option<(ComputePipelineState, usize, usize)>>>,
    /// Row-major to column-major, for the trace the round is about to read.
    transpose_base: Option<ComputePipelineState>,
    transpose_ext: Option<ComputePipelineState>,
}

impl Context {
    /// This thread's command queue; see [`sp1_metal_probes::nocopy::thread_queue`].
    fn queue(&self) -> CommandQueue {
        sp1_metal_probes::nocopy::thread_queue(&self.device)
    }
}

fn context() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device found");
        // Compiled once, unlike the per-chip bodies: nothing about a transpose
        // depends on the constraints.
        let source = format!("{}\n{TRANSPOSE_KERNELS}", header());
        let pso = |name: &str| -> Option<ComputePipelineState> {
            let _compiling = sp1_metal_probes::compile::lock();
            let library =
                device.new_library_with_source(&source, &metal::CompileOptions::new()).ok()?;
            let f = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        let transpose_base = pso("zc_transpose_base");
        let transpose_ext = pso("zc_transpose_ext");
        Context {
            transpose_base,
            transpose_ext,
            device,
            pipelines: Mutex::new(HashMap::new()),
        }
    })
}

/// Whether the zerocheck reduction goes to the GPU, read once.
fn zerocheck_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_ZEROCHECK").map(|v| v != "0").unwrap_or(true))
}

/// What the first use of a chip's kernel found when checked against the CPU.
///
/// A generated kernel is only as right as the transpiler, and the transpiler is
/// wrong for at least one recursion AIR in a way no static check here caught:
/// `BaseAlu`'s body reads the correct tables in the correct order and still
/// computes the wrong sum. Enumerating such hazards one at a time is a losing
/// game, so the backend proves each kernel correct instead of assuming it.
///
/// The first call for a chip runs both paths and compares. A chip that agrees
/// is trusted from then on; a chip that disagrees is declined for the life of
/// the process and its rounds go to the CPU. The cost is one extra CPU round
/// per chip -- not per call -- against a class of defects whose symptom is
/// otherwise a proof that fails to verify two phases later.
static VERDICTS: Mutex<Option<HashMap<String, bool>>> = Mutex::new(None);

/// Whether this chip's kernel has been checked, and what was found.
pub(crate) fn verdict(key: &str) -> Option<bool> {
    VERDICTS.lock().expect("verdicts poisoned").as_ref().and_then(|m| m.get(key).copied())
}

/// Records what checking this chip's kernel against the CPU found.
pub(crate) fn record_verdict(key: &str, agrees: bool) {
    let mut g = VERDICTS.lock().expect("verdicts poisoned");
    g.get_or_insert_with(HashMap::new).insert(key.to_string(), agrees);
    if !agrees {
        crate::prover::zerocheck::counters::note_no_pipeline(key, "disagreed with the CPU", 0);
    }
}

/// Whether every call is checked rather than only the first, read once.
///
/// The five other phases that moved to the GPU each have one of these, and each
/// of them is correct. This phase did not, and it is the one that turned out to
/// compute the wrong thing for the recursion AIRs -- silently, because a wrong
/// round polynomial does not fail until a proof is verified two phases later.
///
/// Slow by construction: it does the work twice and keeps the CPU arm's answer.
/// It is an instrument, not a mode to prove in.
pub(crate) fn checking() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_ZEROCHECK").is_ok_and(|v| v == "check"))
}

/// The largest constraint DAG this will generate a kernel for.
///
/// The elliptic-curve precompiles reach 178,000 nodes and megabytes of source,
/// which is minutes in the Metal compiler; the core chips are two orders of
/// magnitude smaller. Past this the chip stays on the CPU.
const MAX_NODES: usize = 16_384;

/// The node bound in force, which `SP1_METAL_ZEROCHECK_MAX_NODES` overrides.
///
/// The default was set from the precompiles, which are three orders of
/// magnitude past it, and nothing had ever chosen it against the core chips.
/// The counters say exactly one core chip is caught by it -- `Global`, at
/// 22,794 nodes -- and raising the bound to let that one through is measured at
/// **1,814 seconds of kernel compilation**, taking the acceptance proof from
/// 12.0 seconds to 463.6. The other sixty-one kernels compile in 0.04 seconds
/// between them, so the cost is not linear in the node count and the bound is
/// not conservative: it is the edge of a cliff.
///
/// What `Global` would need is the treatment the GKR first layer got -- the
/// constraints as a table one kernel walks, rather than a DAG unrolled into
/// source -- not a larger number here.
fn max_nodes() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("SP1_METAL_ZEROCHECK_MAX_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(MAX_NODES)
    })
}

/// The pipeline for a chip, generated and compiled on first sight.
///
/// **What that costs depends on a cache outside this process.** The figure this
/// project has always quoted -- sixty-one kernels in 0.04 thread-seconds -- is a
/// warm number: macOS keeps compiled shaders keyed by source, so a run that
/// changes none of it pays nothing. Changing a single character of the generated
/// source costs **271 thread-seconds** on the next run and nothing on the one
/// after, which is enough to make a proof of the acceptance workload take 108
/// seconds instead of 11.
///
/// Two things follow. A first run on a machine that has not seen this source is
/// four minutes slower than every run after it, which no measurement here has
/// ever reported. And a kernel timing taken on a run that is also compiling is
/// not a timing: the device is idle enough during the storm to make the kernel
/// look 29% faster than it is.
///
/// `constraints` is only called on a miss, because building the symbolic DAG is
/// not free either.
fn pipeline_for<Fx: PrimeField32>(
    key: &str,
    is_first_round: bool,
    ext_columns: bool,
    constraints: impl FnOnce() -> Vec<SymbolicExpression<Fx>>,
) -> Option<(ComputePipelineState, usize, usize)> {
    let ctx = context();
    if let Some(hit) = ctx.pipelines.lock().expect("pipelines poisoned").get(key) {
        return hit.clone();
    }
    // Creating a pipeline from two threads at once deadlocks in the driver; see
    // `sp1_metal_core::compile`. Taken here rather than around the lookup above
    // so that a thread wanting a kernel somebody else already compiled is never
    // put behind a cold compile.
    let _compiling = sp1_metal_probes::compile::lock();
    // Another thread may have compiled this very chip while this one queued.
    if let Some(hit) = ctx.pipelines.lock().expect("pipelines poisoned").get(key) {
        return hit.clone();
    }
    let cs = constraints();
    let compile_start = std::time::Instant::now();
    // Why a chip has no kernel is worth saying out loud: the three reasons are
    // very different problems, and a chip that falls back is silent otherwise.
    let built = if cs.is_empty() {
        crate::prover::zerocheck::counters::note_no_pipeline(key, "no constraints", 0);
        None
    } else {
        let (source, nodes, base_muls, reads_next_row) =
            transpile_round(&cs, is_first_round, ext_columns);
        // `SP1_METAL_ZC_DUMP=<chip key>` prints the generated kernel, which is
        // the only way to see what a chip's constraints actually became.
        if std::env::var("SP1_METAL_ZC_DUMP").is_ok_and(|v| v == key) {
            eprintln!("--- generated kernel for {key} ---\n{source}\n--- end ---");
        }
        if reads_next_row {
            // The zerocheck folder hands the AIR one row, so a constraint that
            // reaches for another cannot be served here -- and the emitted code
            // would silently read the current row in its place. The recursion
            // AIRs do reach: `BaseAlu` is one, and the wrong round polynomial
            // it produced did not fail until the proof was verified two phases
            // later. Declining is what the flag was always for; nothing read it
            // until a compressed proof refused to verify.
            crate::prover::zerocheck::counters::note_no_pipeline(key, "reads another row", nodes);
            None
        } else if nodes > max_nodes() {
            crate::prover::zerocheck::counters::note_no_pipeline(key, "over the node bound", nodes);
            None
        } else {
            let pso = ctx
                .device
                .new_library_with_source(&source, &metal::CompileOptions::new())
                .ok()
                .and_then(|library| library.get_function("zc_round", None).ok())
                .and_then(|f| ctx.device.new_compute_pipeline_state_with_function(&f).ok());
            if pso.is_none() {
                crate::prover::zerocheck::counters::note_no_pipeline(key, "did not compile", nodes);
            }
            // What the compiler will actually let this kernel run. A generated
            // body of hundreds of live extension values can need more registers
            // than a threadgroup of 256 has, and the driver says so here rather
            // than anywhere else: a pipeline that reports fewer than the 256
            // this dispatches is register-limited, and its occupancy with it.
            if let Some(p) = pso.as_ref() {
                crate::prover::zerocheck::counters::note_occupancy(
                    key,
                    p.max_total_threads_per_threadgroup(),
                    p.thread_execution_width(),
                    nodes,
                );
            }
            pso.map(|p| (p, base_muls, nodes))
        }
    };
    stats::record_compile(compile_start.elapsed().as_secs_f64());
    ctx.pipelines
        .lock()
        .expect("pipelines poisoned")
        .insert(key.to_string(), built.clone());
    built
}

/// Reinterprets field elements as the Montgomery words the kernels read.
fn as_words<T>(xs: &[T]) -> &[u8] {
    // SAFETY: KoalaBear is repr(transparent) over u32 and its degree-4
    // extension is repr(C) over four of them, so either is a plain array of
    // Montgomery words with no padding.
    unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs)) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ZcParams {
    pairs: u32,
    rows: u32,
    main_w: u32,
    prep_w: u32,
    prep_rows: u32,
}

/// The three points of the round polynomial, summed over row pairs.
///
/// Returns `None` whenever the chip is not one this can serve, leaving the
/// caller to do what it always did.
///
/// `max_pairs` stops the sum short of the trace. The reduction is a plain sum
/// over row pairs, so any prefix of it is a quantity the CPU can be asked for
/// too -- which is how a generated kernel is checked without paying for a whole
/// CPU round. Pass `usize::MAX` to reduce the trace.
pub(crate) fn sum_as_poly<Fx: PrimeField32, C: Field>(
    key: &str,
    is_first_round: bool,
    ext_columns: bool,
    constraints: impl FnOnce() -> Vec<SymbolicExpression<Fx>>,
    public_values: &[Fx],
    powers_of_alpha: &[EF],
    gkr_powers: &[EF],
    partial_lagrange: &Mle<EF>,
    preprocessed: Option<&PaddedMle<C>>,
    main: &PaddedMle<C>,
    max_pairs: usize,
) -> Option<(EF, EF, EF)> {
    // Every refusal below is tallied by reason and by the cells it turns away,
    // because the round reduction is most of zerocheck and an aggregate count
    // cannot say which refusal is carrying the work back to the CPU.
    let started = std::time::Instant::now();
    let cells = main.num_polynomials() << main.num_variables();
    let reject = |i: usize| -> Option<(EF, EF, EF)> {
        crate::prover::zerocheck::counters::REJECT[i]
            .record(cells, started.elapsed().as_nanos() as u64);
        None
    };
    if !zerocheck_on_gpu() {
        return reject(0);
    }
    let Some(main_inner) = main.inner().as_ref() else {
        return reject(1);
    };
    let main_w = main_inner.num_polynomials();
    let rows = main_inner.guts().sizes()[0];
    let pairs = main.num_real_entries().div_ceil(2).min(max_pairs);
    if main_w == 0 || rows == 0 || pairs == 0 {
        return reject(2);
    }
    let prep_inner = preprocessed.and_then(|p| p.inner().as_ref());
    let prep_w = preprocessed.map_or(0, PaddedMle::num_polynomials);
    let prep_rows = prep_inner.map_or(0, |p| p.guts().sizes()[0]);
    if prep_w > 0 && prep_inner.is_none() {
        return reject(3);
    }
    if gkr_powers.len() < main_w + prep_w {
        return reject(4);
    }
    let eq = partial_lagrange.guts().as_slice();
    if eq.len() < pairs {
        return reject(5);
    }
    let Some((pso, base_muls, body_nodes)) =
        pipeline_for(key, is_first_round, ext_columns, constraints)
    else {
        return reject(6);
    };
    // The base-field multiplies this dispatch will issue: every interpolation
    // point's body, for every row pair. Divided by the GPU seconds it takes,
    // this says whether the kernel is at the machine's arithmetic rate or a
    // long way under it -- which decides whether there is anything to win by
    // emitting fewer of them.
    sp1_metal_probes::tally::record("zerocheck: base muls", pairs * base_muls, 0);
    crate::prover::zerocheck::counters::ACCEPTED
        .record(cells, started.elapsed().as_nanos() as u64);

    let _phase = stats::enter(stats::Phase::Zerocheck);
    let ctx = context();
    // SAFETY: every buffer here is read by the one dispatch below, which is
    // awaited before this function returns, and nothing writes the slices in
    // the meantime.
    let upload =
        |bytes: &[u8]| unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, bytes) };
    let b_main = upload(as_words(main_inner.guts().as_slice()));
    let empty = [0u8; 16];
    let b_prep = prep_inner
        .map_or_else(|| upload(&empty), |p| upload(as_words(p.guts().as_slice())));
    let pv_words: Vec<u32> = public_values.iter().map(|x| monty_of(*x)).collect();
    let b_pv = upload(as_words(&pv_words));
    let b_eq = upload(as_words(&eq[..pairs]));
    let b_ap = upload(as_words(powers_of_alpha));
    let b_gp = upload(as_words(gkr_powers));

    let groups = (pairs as u64).div_ceil(THREADS);
    let partials =
        ctx.device.new_buffer(groups * 3 * 16, MTLResourceOptions::StorageModeShared);
    let params = ZcParams {
        pairs: pairs as u32,
        rows: rows as u32,
        main_w: main_w as u32,
        prep_w: prep_w as u32,
        prep_rows: prep_rows as u32,
    };
    // The column-major copies the round will read, and the pipeline that makes
    // them. Anything missing leaves this `None`, which puts the round back on
    // the row-major buffers it was reading before.
    let elem = if ext_columns { 16u64 } else { 4u64 };
    let tpso = if ext_columns { ctx.transpose_ext.as_ref() } else { ctx.transpose_base.as_ref() };
    let transposed = column_major().then(|| tpso).flatten().map(|t| {
        let m = ctx
            .device
            .new_buffer((rows * main_w) as u64 * elem, MTLResourceOptions::StorageModeShared);
        let pr = ctx
            .device
            .new_buffer((rows * prep_w.max(1)) as u64 * elem, MTLResourceOptions::StorageModeShared);
        (t.clone(), m, pr)
    });

    metal::objc::rc::autoreleasepool(|| {
        let queue = ctx.queue();
        let cb = queue.new_command_buffer();
        if let Some((t, m, pr)) = transposed.as_ref() {
            // Row-major in, column-major out, in the same command buffer as the
            // round that reads it -- so it costs a pass over the trace and no
            // round trip.
            let tile = if ext_columns { 16u64 } else { 32u64 };
            let mut tile_pass = |src: &metal::Buffer, dst: &metal::Buffer, w: usize| {
                if w == 0 {
                    return;
                }
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(t);
                enc.set_buffer(0, Some(src), 0);
                enc.set_buffer(1, Some(dst), 0);
                let dims = [rows as u32, w as u32];
                enc.set_bytes(2, 8, dims.as_ptr().cast());
                enc.dispatch_thread_groups(
                    MTLSize::new((w as u64).div_ceil(tile), (rows as u64).div_ceil(tile), 1),
                    MTLSize::new(tile, tile, 1),
                );
                enc.end_encoding();
            };
            tile_pass(&b_main, m, main_w);
            tile_pass(&b_prep, pr, prep_w);
        }
        let (r_main, r_prep) = match transposed.as_ref() {
            Some((_, m, pr)) => (m, pr),
            None => (&b_main, &b_prep),
        };
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in [r_main, r_prep, &b_pv, &b_eq, &b_ap, &b_gp].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(6, Some(&partials), 0);
        enc.set_bytes(
            7,
            std::mem::size_of::<ZcParams>() as u64,
            (&params as *const ZcParams).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new(groups, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        enc.end_encoding();
        let (_, gpu) = sp1_metal_probes::queue::submit(cb);
        crate::prover::zerocheck::counters::note_chip(
            key,
            (pairs * base_muls) as u64,
            gpu,
            body_nodes,
        );
    });

    stats::record_output(Output::Staged, groups as usize * 3 * 16, 0);
    let raw = partials.contents().cast::<EF>();
    let mut out = (EF::zero(), EF::zero(), EF::zero());
    for g in 0..groups as usize {
        // SAFETY: the kernel wrote three extension elements per threadgroup.
        unsafe {
            out.0 += *raw.add(3 * g);
            out.1 += *raw.add(3 * g + 1);
            out.2 += *raw.add(3 * g + 2);
        }
    }
    Some(out)
}
