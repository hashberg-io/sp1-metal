//! Turning a real chip's symbolic AIR constraints into Metal.
//!
//! The probe suite priced the *shape* of constraint evaluation with a synthetic
//! zerocheck round — T degree-3 monomials per row over a column-major trace,
//! alpha-combined and eq-weighted — and deliberately left the per-chip
//! transpiler to the backend phase. This is that transpiler, at spike scale: it
//! reads the symbolic constraints SP1 already builds for every chip, emits a
//! Metal kernel per chip, compiles it, and measures what comes out.
//!
//! The question it exists to answer is not whether the arithmetic is fast — the
//! zerocheck probe answered that — but whether a real constraint set survives
//! being a straight-line program in a kernel: SP1's chips have hundreds to
//! thousands of constraints over DAGs with thousands of nodes, and a GPU thread
//! has a few hundred registers.
//!
//! `SymbolicExpression` is already a DAG: shared subexpressions are shared `Rc`
//! nodes, so common-subexpression elimination costs nothing more than
//! memoising on pointer identity.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use slop_algebra::{AbstractExtensionField, AbstractField, PrimeField32};
use slop_matrix::dense::RowMajorMatrixView;
use slop_uni_stark::{get_symbolic_constraints, Entry, SymbolicExpression, SymbolicVariable};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::air::MachineAir;
use sp1_hypercube::ConstraintSumcheckFolder;
use sp1_primitives::{SP1ExtensionField, SP1Field};

type F = SP1Field;
type EF = SP1ExtensionField;

const P: u32 = 0x7f00_0001;

/// Field arithmetic shared with the probe suite's kernels, taken verbatim so
/// the generated code computes in the same representation as everything
/// already measured.
fn header() -> &'static str {
    include_str!("../../sp1-metal-probes/src/shaders.metal")
        .split("// ---- end of shared field arithmetic ----")
        .next()
        .expect("shader header sentinel")
}

fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % P as u64) as u32
}

fn monty_of(x: F) -> u32 {
    to_monty(x.as_canonical_u32())
}

/// Emits one chip's constraints as a straight-line Metal program.
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
        }
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("t{}", self.next - 1)
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
        let expr = format!("{table}[{index}u * n + r{}]", offset.min(1));
        self.bind(ty, &expr)
    }

    fn var(&mut self, v: &SymbolicVariable<F>) -> (String, Ty) {
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

    fn emit_rc(&mut self, r: &Rc<SymbolicExpression<F>>) -> (String, Ty) {
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

    fn emit(&mut self, e: &SymbolicExpression<F>) -> (String, Ty) {
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

struct ChipReport {
    /// Set if any constraint reads a row other than the current one.
    reads_next_row: bool,
    name: String,
    constraints: usize,
    main_width: usize,
    prep_width: usize,
    nodes: usize,
    muls: usize,
    adds: usize,
    consts: usize,
    loads: usize,
    source: String,
}

/// Generates the zerocheck-round kernel for one chip: one thread per row, every
/// constraint evaluated, alpha-combined with precomputed powers, eq-weighted,
/// and reduced to one extension element per threadgroup. This is the shape the
/// `zerocheck` probe prices synthetically.
fn transpile(
    name: &str,
    constraints: &[SymbolicExpression<F>],
    base: usize,
    main_width: usize,
    prep_width: usize,
    ext_columns: bool,
) -> ChipReport {
    let mut cg = Codegen::new(ext_columns);
    let mut names = Vec::with_capacity(constraints.len());
    for c in constraints {
        names.push(cg.emit(c));
    }
    let mut combine = String::new();
    for (k, (n, ty)) in names.iter().enumerate() {
        let idx = base + k;
        // The folding challenge is an extension element whichever field the
        // constraint itself came out in.
        let term = match ty {
            Ty::Base => format!("ext_mul_base(ap[{idx}u], {n})"),
            Ty::Ext => format!("ext_mul_kara(ap[{idx}u], {n})"),
        };
        writeln!(combine, "        acc = ext_add(acc, {term});").unwrap();
    }
    let col = if ext_columns { "uint4" } else { "uint" };
    let kernel = format!(
        r#"
kernel void zc_chip(device const {col}* main [[buffer(0)]],
                    device const {col}* prep [[buffer(1)]],
                    device const uint* pv [[buffer(2)]],
                    device const uint4* eq [[buffer(3)]],
                    device const uint4* ap [[buffer(4)]],
                    device uint4* partials [[buffer(5)]],
                    constant uint& n [[buffer(6)]],
                    uint gid [[thread_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint tgid [[threadgroup_position_in_grid]]) {{
    threadgroup uint4 sh[256];
    uint4 acc = uint4(0u, 0u, 0u, 0u);
    if (gid < n) {{
        uint r0 = gid;
        uint r1 = (gid + 1u) & (n - 1u);
        uint is_first = (gid == 0u) ? KB_ONE : 0u;
        uint is_last = (gid == n - 1u) ? KB_ONE : 0u;
        uint is_transition = (gid == n - 1u) ? 0u : KB_ONE;
{body}
{combine}        acc = ext_mul_kara(acc, eq[gid]);
    }}
    sh[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {{
        if (tid < s) {{
            sh[tid] = ext_add(sh[tid], sh[tid + s]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
        partials[tgid] = sh[0];
    }}
}}
"#,
        body = cg.body,
        combine = combine,
    );
    let source = format!("{}\nconstant uint KB_ONE = {}u;\n{}", header(), to_monty(1), kernel);
    ChipReport {
        name: name.to_string(),
        constraints: constraints.len(),
        main_width,
        prep_width,
        nodes: cg.next,
        muls: cg.muls,
        adds: cg.adds,
        consts: cg.consts,
        loads: cg.loads,
        reads_next_row: cg.reads_next_row,
        source,
    }
}

/// Compiles a generated kernel and measures it on a synthetic trace of `n`
/// rows. The pipeline's maximum threadgroup size is the register-pressure
/// signal: a kernel the compiler cannot fit gets a lower occupancy limit.
fn measure(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    parts: &[ChipReport],
    n: usize,
    reps: usize,
) -> serde_json::Value {
    let r = &parts[0];
    let total_constraints: usize = parts.iter().map(|p| p.constraints).sum();
    let total_muls: usize = parts.iter().map(|p| p.muls).sum();
    let mut psos = Vec::with_capacity(parts.len());
    let mut compile_s = 0.0;
    let mut occupancy = u64::MAX;
    for part in parts {
        let t0 = std::time::Instant::now();
        let library =
            match device.new_library_with_source(&part.source, &metal::CompileOptions::new()) {
                Ok(l) => l,
                Err(e) => return serde_json::json!({ "compiled": false, "error": e.to_string() }),
            };
        compile_s += t0.elapsed().as_secs_f64();
        let function = library.get_function("zc_chip", None).expect("zc_chip");
        match device.new_compute_pipeline_state_with_function(&function) {
            Ok(p) => {
                occupancy = occupancy.min(p.max_total_threads_per_threadgroup());
                psos.push(p);
            }
            Err(e) => {
                return serde_json::json!({ "compiled": true, "pipeline": false, "error": e.to_string() })
            }
        }
    }
    let mut rng = StdRng::seed_from_u64(71);
    let mut rand_monty =
        |len: usize| -> Vec<u32> { (0..len).map(|_| monty_of(rng.gen::<F>())).collect() };
    let main = rand_monty(n * r.main_width.max(1));
    let prep = rand_monty(n * r.prep_width.max(1));
    let pv = rand_monty(1024);
    let eq = rand_monty(n * 4);
    let ap = rand_monty(total_constraints.max(1) * 4);
    let mk = |d: &[u32]| {
        device.new_buffer_with_data(
            d.as_ptr().cast(),
            (d.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };
    let (b_main, b_prep, b_pv, b_eq, b_ap) = (mk(&main), mk(&prep), mk(&pv), mk(&eq), mk(&ap));
    let groups = n.div_ceil(256);
    let b_part = device.new_buffer(
        (groups * 16 * psos.len()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let n32 = n as u32;
    let run = || {
        let cb = queue.new_command_buffer();
        for (i, pso) in psos.iter().enumerate() {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            enc.set_buffer(0, Some(&b_main), 0);
            enc.set_buffer(1, Some(&b_prep), 0);
            enc.set_buffer(2, Some(&b_pv), 0);
            enc.set_buffer(3, Some(&b_eq), 0);
            enc.set_buffer(4, Some(&b_ap), 0);
            enc.set_buffer(5, Some(&b_part), (i * groups * 16) as u64);
            enc.set_bytes(6, 4, (&n32 as *const u32).cast());
            enc.dispatch_thread_groups(
                metal::MTLSize::new(groups as u64, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        let t = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        t.elapsed().as_secs_f64()
    };
    run();
    let best = (0..reps).map(|_| run()).fold(f64::INFINITY, f64::min);
    serde_json::json!({
        "compiled": true,
        "pipeline": true,
        "kernels": psos.len(),
        "compile_seconds": compile_s,
        "source_bytes": parts.iter().map(|p| p.source.len()).sum::<usize>(),
        "max_total_threads_per_threadgroup": occupancy,
        "rows": n,
        "seconds_best": best,
        "mega_rows_per_s": n as f64 / best / 1e6,
        "giga_muls_per_s": (total_muls + 4 * total_constraints) as f64 * n as f64 / best / 1e9,
    })
}

/// A field element as the Montgomery word the kernels read.
fn monty4(x: EF) -> [u32; 4] {
    let b = x.as_base_slice();
    std::array::from_fn(|i| monty_of(b[i]))
}

fn from_monty4(w: [u32; 4]) -> EF {
    let inv = |m: u32| -> F {
        // Undo the Montgomery factor by reducing m * 1.
        let t = (m as u64).wrapping_mul(0x8100_0001u64) & 0xffff_ffff;
        let u = ((t * P as u64) >> 32) as u32;
        let r = 0u32.wrapping_sub(u);
        F::from_canonical_u32(if 0 < u { r.wrapping_add(P) } else { r })
    };
    EF::from_base_slice(&[inv(w[0]), inv(w[1]), inv(w[2]), inv(w[3])])
}

/// Checks a generated evaluator against the folder the prover actually uses.
///
/// The kernel computes, over `n` rows, the sum of `eq[r] * sum_k alpha^k * c_k(row r)`.
/// The CPU side asks the AIR for exactly that through `ConstraintSumcheckFolder`,
/// one row at a time. Columns are extension elements, which is what every
/// zerocheck round after the first hands the AIR, and the case the base-field
/// spike never covered.
#[allow(clippy::too_many_arguments)]
fn parity(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    air: &RiscvAir<F>,
    parts: &[ChipReport],
    num_constraints: usize,
    main_width: usize,
    prep_width: usize,
    n: usize,
) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let pvs = sp1_hypercube::PROOF_MAX_NUM_PVS;
    let public_values: Vec<F> = (0..pvs).map(|_| rng.gen()).collect();
    let alpha: Vec<EF> = (0..num_constraints.max(1)).map(|_| rng.gen()).collect();
    let eq: Vec<EF> = (0..n).map(|_| rng.gen()).collect();
    // Column-major, which is how the generated code indexes.
    let main: Vec<EF> = (0..n * main_width).map(|_| rng.gen()).collect();
    let prep: Vec<EF> = (0..n * prep_width.max(1)).map(|_| rng.gen()).collect();

    let words = |xs: &[EF]| -> Vec<u32> { xs.iter().flat_map(|x| monty4(*x)).collect() };
    let buf = |v: &[u32]| {
        device.new_buffer_with_data(
            v.as_ptr().cast(),
            std::mem::size_of_val(v).max(4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    };
    let b_main = buf(&words(&main));
    let b_prep = buf(&words(&prep));
    let b_pv = buf(&public_values.iter().map(|x| monty_of(*x)).collect::<Vec<_>>());
    let b_eq = buf(&words(&eq));
    let b_ap = buf(&words(&alpha));

    let threads = 256u64;
    let groups = (n as u64).div_ceil(threads);
    let mut got = EF::zero();
    for part in parts {
        let library = device
            .new_library_with_source(&part.source, &metal::CompileOptions::new())
            .map_err(|e| format!("compile: {e}"))?;
        let f = library.get_function("zc_chip", None).map_err(|e| format!("{e}"))?;
        let pso = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("{e}"))?;
        let partials = device.new_buffer(
            groups * 16,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in [&b_main, &b_prep, &b_pv, &b_eq, &b_ap].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(5, Some(&partials), 0);
        let n32 = n as u32;
        enc.set_bytes(6, 4, (&n32 as *const u32).cast());
        enc.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let raw = partials.contents().cast::<[u32; 4]>();
        for g in 0..groups as usize {
            got += from_monty4(unsafe { *raw.add(g) });
        }
    }

    let mut want = EF::zero();
    for r in 0..n {
        let main_row: Vec<EF> = (0..main_width).map(|i| main[i * n + r]).collect();
        let prep_row: Vec<EF> = (0..prep_width).map(|i| prep[i * n + r]).collect();
        let mut folder = ConstraintSumcheckFolder::<F, EF, EF> {
            preprocessed: RowMajorMatrixView::new_row(&prep_row),
            main: RowMajorMatrixView::new_row(&main_row),
            accumulator: EF::zero(),
            public_values: &public_values,
            constraint_index: 0,
            powers_of_alpha: &alpha,
        };
        slop_air::Air::eval(air, &mut folder);
        want += folder.accumulator * eq[r];
    }

    if got == want {
        Ok(())
    } else {
        Err(format!("got {got:?} want {want:?}"))
    }
}

fn main() {
    let mut only: Option<String> = None;
    let mut chunk = usize::MAX;
    let mut stats_only = false;
    let mut do_parity = false;
    let mut ext_columns = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = args.next().expect("--chunk needs a value").parse().unwrap(),
            "--stats" => stats_only = true,
            // Extension-valued columns, as every zerocheck round after the
            // first hands the AIR.
            "--ext" => ext_columns = true,
            "--parity" => {
                do_parity = true;
                ext_columns = true;
            }
            other => only = Some(other.to_string()),
        }
    }
    let n: usize = 1 << 16;
    let reps = 10;

    let machine = RiscvAir::<F>::machine();

    let mut reports = Vec::new();
    for chip in machine.chips() {
        let air = chip.air.as_ref();
        let name = air.name();
        if let Some(want) = only.as_deref() {
            if !name.contains(want) {
                continue;
            }
        }
        let prep_width = air.preprocessed_width();
        let main_width = slop_air::BaseAir::<F>::width(air);
        let constraints =
            get_symbolic_constraints(air, prep_width, sp1_hypercube::PROOF_MAX_NUM_PVS);
        if constraints.is_empty() {
            continue;
        }
        let size = chunk.min(constraints.len());
        let parts: Vec<ChipReport> = constraints
            .chunks(size)
            .enumerate()
            .map(|(i, c)| transpile(&name, c, i * size, main_width, prep_width, ext_columns))
            .collect();
        let r = &parts[0];
        if stats_only {
            let nodes: usize = parts.iter().map(|p| p.nodes).sum();
            let muls: usize = parts.iter().map(|p| p.muls).sum();
            let bytes: usize = parts.iter().map(|p| p.source.len()).sum();
            println!(
                "{:34} {:5} constr {:7} nodes {:7} mul {:5} wide {:5} prep {:6} kB",
                name, constraints.len(), nodes, muls, main_width, prep_width, bytes / 1024
            );
            reports.push(serde_json::json!({
                "chip": name, "constraints": constraints.len(), "main_width": main_width,
                "preprocessed_width": prep_width, "dag_nodes": nodes, "muls": muls,
                "source_bytes": bytes,
            }));
            continue;
        }
        let device = metal::Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        if do_parity {
            let rows = 256;
            let verdict = if parts.iter().any(|p| p.reads_next_row) {
                // The folder hands the AIR one row, so a constraint reaching for
                // the next one could not be evaluated on the CPU side either.
                Err("reads a row other than the current one".to_string())
            } else {
                parity(
                    &device,
                    &queue,
                    air,
                    &parts,
                    constraints.len(),
                    main_width,
                    prep_width,
                    rows,
                )
            };
            match verdict {
                Ok(()) => println!(
                    "{:34} {:5} constr {:6} nodes {:3} kernels  MATCHES the folder",
                    name,
                    constraints.len(),
                    parts.iter().map(|p| p.nodes).sum::<usize>(),
                    parts.len()
                ),
                Err(e) => println!(
                    "{:34} {:5} constr {:6} nodes {:3} kernels  DIFFERS: {e}",
                    name,
                    constraints.len(),
                    parts.iter().map(|p| p.nodes).sum::<usize>(),
                    parts.len()
                ),
            }
            continue;
        }
        let measured = measure(&device, &queue, &parts, n, reps);
        println!(
            "{:34} {:5} constr {:6} nodes {:6} mul {:3} kernels {:4} wide  {:>6} kB  {}",
            r.name,
            constraints.len(),
            parts.iter().map(|p| p.nodes).sum::<usize>(),
            parts.iter().map(|p| p.muls).sum::<usize>(),
            parts.len(),
            r.main_width,
            parts.iter().map(|p| p.source.len()).sum::<usize>() / 1024,
            if measured["compiled"].as_bool().unwrap_or(false) {
                format!(
                    "{:6.1} Mrow/s  occ {}",
                    measured["mega_rows_per_s"].as_f64().unwrap_or(0.0),
                    measured["max_total_threads_per_threadgroup"]
                )
            } else {
                "DID NOT COMPILE".to_string()
            }
        );
        reports.push(serde_json::json!({
            "chip": r.name,
            "constraints": constraints.len(),
            "chunk_size": size,
            "main_width": r.main_width,
            "preprocessed_width": r.prep_width,
            "dag_nodes": parts.iter().map(|p| p.nodes).sum::<usize>(),
            "muls": parts.iter().map(|p| p.muls).sum::<usize>(),
            "adds": parts.iter().map(|p| p.adds).sum::<usize>(),
            "constants": parts.iter().map(|p| p.consts).sum::<usize>(),
            "trace_loads": parts.iter().map(|p| p.loads).sum::<usize>(),
            "metal": measured,
        }));
    }
    std::fs::write(
        "transpile.json",
        serde_json::to_string_pretty(&serde_json::Value::Array(reports)).unwrap(),
    )
    .unwrap();
}
