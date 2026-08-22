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
            consts: 0,
            loads: 0,
            ext_columns,
            reads_next_row: false,
            prefix: "t",
            point: 0,
            round: false,
        }
    }

    /// A copy that reads its columns interpolated to `point`.
    fn at_point(ext_columns: bool, prefix: &'static str, point: u8) -> Self {
        Self { prefix, point, round: true, ..Self::new(ext_columns) }
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
        let expr = if self.point == 0 {
            format!("{table}[{table}_b0 + {index}u]")
        } else {
            let pt = self.point;
            format!("zc_at_{pt}({table}, {table}_b0, {table}_b1, {index}u, has1)")
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
fn transpile_round<Fx: PrimeField32>(
    constraints: &[SymbolicExpression<Fx>],
    is_first_round: bool,
) -> (String, usize) {
    let mut bodies = String::new();
    let mut nodes = 0usize;
    for (prefix, point, acc) in [("a", 0u8, "acc0"), ("b", 2, "acc2"), ("c", 4, "acc4")] {
        // The first round knows the polynomial vanishes at 0, so that copy of
        // the body would be dead weight.
        if point == 0 && is_first_round {
            writeln!(bodies, "        uint4 {acc} = uint4(0u);").unwrap();
            continue;
        }
        let mut cg = Codegen::at_point(true, prefix, point);
        let names: Vec<(String, Ty)> = constraints.iter().map(|c| cg.emit(c)).collect();
        nodes += cg.next;
        write!(bodies, "{}", cg.body).unwrap();
        writeln!(bodies, "        uint4 {acc} = uint4(0u);").unwrap();
        for (k, (n, ty)) in names.iter().enumerate() {
            let term = match ty {
                Ty::Base => format!("ext_mul_base(ap[{k}u], {n})"),
                Ty::Ext => format!("ext_mul_kara(ap[{k}u], {n})"),
            };
            writeln!(bodies, "        {acc} = ext_add({acc}, {term});").unwrap();
        }
    }
    let kernel = format!(
        r#"
struct ZcParams {{
    uint pairs;
    uint rows;
    uint main_w;
    uint prep_w;
}};

kernel void zc_round(device const uint4* main [[buffer(0)]],
                     device const uint4* prep [[buffer(1)]],
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
        uint main_b0 = (2u * q) * p.main_w;
        uint main_b1 = (2u * q + 1u) * p.main_w;
        uint prep_b0 = (2u * q) * p.prep_w;
        uint prep_b1 = (2u * q + 1u) * p.prep_w;
{bodies}
        // The interaction adjustment: every column, main then preprocessed,
        // against its batching power. The point-4 adjustment is inferred from
        // the other two rather than computed again, as the CPU does.
        uint4 g0 = uint4(0u), g2 = uint4(0u);
        for (uint i = 0u; i < p.main_w; i++) {{
            g0 = ext_add(g0, ext_mul_kara(gp[i], main[main_b0 + i]));
            g2 = ext_add(g2, ext_mul_kara(gp[i], zc_at_2(main, main_b0, main_b1, i, has1)));
        }}
        for (uint i = 0u; i < p.prep_w; i++) {{
            uint4 w = gp[p.main_w + i];
            g0 = ext_add(g0, ext_mul_kara(w, prep[prep_b0 + i]));
            g2 = ext_add(g2, ext_mul_kara(w, zc_at_2(prep, prep_b0, prep_b1, i, has1)));
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
    );
    let source = format!("{}\nconstant uint KB_ONE = {}u;\n{}", header(), to_monty(1), kernel);
    (source, nodes)
}


struct Context {
    device: Device,
    queue: CommandQueue,
    /// One pipeline per chip and round kind, built the first time it is asked
    /// for. `None` records a chip whose kernel would not build, so it is not
    /// attempted again.
    pipelines: Mutex<HashMap<String, Option<ComputePipelineState>>>,
}

fn context() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device found");
        Context {
            queue: device.new_command_queue(),
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

/// The largest constraint DAG this will generate a kernel for.
///
/// The elliptic-curve precompiles reach 178,000 nodes and megabytes of source,
/// which is minutes in the Metal compiler; the core chips are two orders of
/// magnitude smaller. Past this the chip stays on the CPU.
const MAX_NODES: usize = 16_384;

/// The pipeline for a chip, generated and compiled on first sight.
///
/// `constraints` is only called on a miss, because building the symbolic DAG is
/// not free either.
fn pipeline_for<Fx: PrimeField32>(
    key: &str,
    is_first_round: bool,
    constraints: impl FnOnce() -> Vec<SymbolicExpression<Fx>>,
) -> Option<ComputePipelineState> {
    let ctx = context();
    if let Some(hit) = ctx.pipelines.lock().expect("pipelines poisoned").get(key) {
        return hit.clone();
    }
    let cs = constraints();
    let built = (!cs.is_empty())
        .then(|| transpile_round(&cs, is_first_round))
        .filter(|(_, nodes)| *nodes <= MAX_NODES)
        .and_then(|(source, _)| {
            let library = ctx
                .device
                .new_library_with_source(&source, &metal::CompileOptions::new())
                .ok()?;
            let f = library.get_function("zc_round", None).ok()?;
            ctx.device.new_compute_pipeline_state_with_function(&f).ok()
        });
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
}

/// The three points of the round polynomial, summed over row pairs.
///
/// Returns `None` whenever the chip is not one this can serve, leaving the
/// caller to do what it always did.
pub(crate) fn sum_as_poly<Fx: PrimeField32>(
    key: &str,
    is_first_round: bool,
    constraints: impl FnOnce() -> Vec<SymbolicExpression<Fx>>,
    public_values: &[Fx],
    powers_of_alpha: &[EF],
    gkr_powers: &[EF],
    partial_lagrange: &Mle<EF>,
    preprocessed: Option<&PaddedMle<EF>>,
    main: &PaddedMle<EF>,
) -> Option<(EF, EF, EF)> {
    if !zerocheck_on_gpu() {
        return None;
    }
    let main_inner = main.inner().as_ref()?;
    let main_w = main_inner.num_polynomials();
    let rows = main_inner.guts().sizes()[0];
    let pairs = main.num_real_entries().div_ceil(2);
    if main_w == 0 || rows == 0 || pairs == 0 {
        return None;
    }
    let prep_inner = preprocessed.and_then(|p| p.inner().as_ref());
    let prep_w = preprocessed.map_or(0, PaddedMle::num_polynomials);
    if prep_w > 0 && prep_inner.is_none() {
        return None;
    }
    if gkr_powers.len() < main_w + prep_w {
        return None;
    }
    let eq = partial_lagrange.guts().as_slice();
    if eq.len() < pairs {
        return None;
    }
    let pso = pipeline_for(key, is_first_round, constraints)?;

    let ctx = context();
    let upload = |bytes: &[u8]| {
        ctx.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
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
    };
    metal::objc::rc::autoreleasepool(|| {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in [&b_main, &b_prep, &b_pv, &b_eq, &b_ap, &b_gp].into_iter().enumerate() {
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
        cb.commit();
        cb.wait_until_completed();
    });

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
