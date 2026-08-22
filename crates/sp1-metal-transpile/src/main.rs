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
use slop_algebra::PrimeField32;
use slop_uni_stark::{get_symbolic_constraints, Entry, SymbolicExpression, SymbolicVariable};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::air::MachineAir;
use sp1_primitives::SP1Field;

type F = SP1Field;

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
struct Codegen {
    body: String,
    memo: HashMap<usize, String>,
    next: usize,
    adds: usize,
    muls: usize,
    consts: usize,
    loads: usize,
}

impl Codegen {
    fn new() -> Self {
        Self { body: String::new(), memo: HashMap::new(), next: 0, adds: 0, muls: 0, consts: 0, loads: 0 }
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("t{}", self.next - 1)
    }

    fn var(&mut self, v: &SymbolicVariable<F>) -> String {
        let name = self.fresh();
        let src = match v.entry {
            Entry::Main { offset } => {
                self.loads += 1;
                format!("main[{}u * n + r{}]", v.index, offset.min(1))
            }
            Entry::Preprocessed { offset } => {
                self.loads += 1;
                format!("prep[{}u * n + r{}]", v.index, offset.min(1))
            }
            Entry::Public => format!("pv[{}u]", v.index),
            Entry::Challenge => format!("pv[{}u]", v.index),
            Entry::Permutation { offset } => {
                self.loads += 1;
                format!("main[{}u * n + r{}]", v.index, offset.min(1))
            }
        };
        writeln!(self.body, "    uint {name} = {src};").unwrap();
        name
    }

    fn emit_rc(&mut self, r: &Rc<SymbolicExpression<F>>) -> String {
        let key = Rc::as_ptr(r) as usize;
        if let Some(name) = self.memo.get(&key) {
            return name.clone();
        }
        let name = self.emit(r);
        self.memo.insert(key, name.clone());
        name
    }

    fn emit(&mut self, e: &SymbolicExpression<F>) -> String {
        match e {
            SymbolicExpression::Variable(v) => self.var(v),
            SymbolicExpression::IsFirstRow => "is_first".to_string(),
            SymbolicExpression::IsLastRow => "is_last".to_string(),
            SymbolicExpression::IsTransition => "is_transition".to_string(),
            SymbolicExpression::Constant(c) => {
                let name = self.fresh();
                self.consts += 1;
                writeln!(self.body, "    uint {name} = {}u;", monty_of(*c)).unwrap();
                name
            }
            SymbolicExpression::Add { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                let name = self.fresh();
                self.adds += 1;
                writeln!(self.body, "    uint {name} = kb_add({a}, {b});").unwrap();
                name
            }
            SymbolicExpression::Sub { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                let name = self.fresh();
                self.adds += 1;
                writeln!(self.body, "    uint {name} = kb_sub({a}, {b});").unwrap();
                name
            }
            SymbolicExpression::Neg { x, .. } => {
                let a = self.emit_rc(x);
                let name = self.fresh();
                self.adds += 1;
                writeln!(self.body, "    uint {name} = kb_neg({a});").unwrap();
                name
            }
            SymbolicExpression::Mul { x, y, .. } => {
                let (a, b) = (self.emit_rc(x), self.emit_rc(y));
                let name = self.fresh();
                self.muls += 1;
                writeln!(self.body, "    uint {name} = kb_mul({a}, {b});").unwrap();
                name
            }
        }
    }
}

struct ChipReport {
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
) -> ChipReport {
    let mut cg = Codegen::new();
    let mut names = Vec::with_capacity(constraints.len());
    for c in constraints {
        names.push(cg.emit(c));
    }
    let mut combine = String::new();
    for (k, n) in names.iter().enumerate() {
        let idx = base + k;
        writeln!(combine, "        acc = ext_add(acc, ext_mul_base(ap[{idx}u], {n}));").unwrap();
    }
    let kernel = format!(
        r#"
kernel void zc_chip(device const uint* main [[buffer(0)]],
                    device const uint* prep [[buffer(1)]],
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
{combine}        acc = ext_mul(acc, eq[gid]);
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

fn main() {
    let mut only: Option<String> = None;
    let mut chunk = usize::MAX;
    let mut stats_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--chunk" => chunk = args.next().expect("--chunk needs a value").parse().unwrap(),
            "--stats" => stats_only = true,
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
            .map(|(i, c)| transpile(&name, c, i * size, main_width, prep_width))
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
