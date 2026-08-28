// SPDX-License-Identifier: MIT OR Apache-2.0
//! Proves a given RISC-V ELF with the SP1 CPU prover under span-timing
//! instrumentation, so tools/span_profile.py can split wall time into phases.
//!
//! The pair it wants is what `SP1_DUMP=1` writes: any sp1-sdk host run with
//! that variable set dumps `program.bin` and `stdin.bin` for the proof it was
//! about to generate and exits. Feeding those two files back in here reproduces
//! that exact workload under span timing, without touching the host that
//! produced them.
//!
//! Run with `RUST_LOGGER=flat RUST_LOG=debug`. The flat logger writes its span
//! lines to stdout, not stderr, so capture both:
//!
//! A guest that reads a single `u32` can be driven with `--stdin-u32 N` instead.
//!
//! `--repeat N` proves the same statement N times through one `ProverClient`,
//! which is the shape a consumer's driver has and the shape the intermittent
//! wrong proof of `knowledge/defect-wrong-proof.md` needs: five single-proof
//! runs were clean where a process making dozens of sequential calls failed
//! about one time in twenty-eight. `--timeout SECS` is what makes that
//! survivable unattended, because the failure does not raise: SP1 treats a
//! failed shard verification as retryable, retries, and parks with nothing
//! printed. A watchdog thread says so and leaves with status 2.
//!
//! ```sh
//! SP1_DUMP=1 path/to/real-host              # writes program.bin, stdin.bin
//! RUST_LOGGER=flat RUST_LOG=debug cargo run --release -- \
//!     --elf program.bin --stdin-bin stdin.bin --mode compressed > prove.log 2>&1
//! python3 ../../tools/span_profile.py prove.log --json profile.json
//! ```


// A general-purpose allocator instead of the system one, so that "does
// allocation churn cost wall time" can be answered by measurement rather than
// argued. A proof faults in about 54 GB of fresh pages; whether returning them
// to the kernel and taking them back is worth anything is the whole of the
// question, and this is the cheapest way to ask it.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use sp1_sdk::{Elf, ProveRequest, Prover, ProverClient, ProvingKey, SP1Stdin};

struct Args {
    elf: String,
    mode: String,
    stdin_file: Option<String>,
    stdin_bin: Option<String>,
    stdin_u32: Option<u32>,
    repeat: usize,
    timeout_s: u64,
}

fn parse_args() -> Result<Args> {
    let mut elf = None;
    let mut mode = "compressed".to_string();
    let mut stdin_file = None;
    let mut stdin_bin = None;
    let mut stdin_u32 = None;
    let mut repeat = 1usize;
    let mut timeout_s = 0u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--elf" => elf = Some(args.next().context("--elf needs a path")?),
            "--mode" => mode = args.next().context("--mode needs core|compressed")?,
            "--stdin-file" => stdin_file = Some(args.next().context("--stdin-file needs a path")?),
            "--stdin-bin" => stdin_bin = Some(args.next().context("--stdin-bin needs a path")?),
            "--execute" => mode = "execute".to_string(),
            "--repeat" => {
                repeat = args
                    .next()
                    .context("--repeat needs a count")?
                    .parse()
                    .context("parsing --repeat")?;
            }
            "--timeout" => {
                timeout_s = args
                    .next()
                    .context("--timeout needs seconds")?
                    .parse()
                    .context("parsing --timeout")?;
            }
            "--stdin-u32" => {
                stdin_u32 = Some(
                    args.next().context("--stdin-u32 needs a number")?.parse().context("parsing --stdin-u32")?,
                );
            }
            other => bail!("unknown argument `{other}`"),
        }
    }
    Ok(Args {
        elf: elf.context("--elf <path> is required")?,
        mode,
        stdin_file,
        stdin_bin,
        stdin_u32,
        repeat: repeat.max(1),
        timeout_s,
    })
}

/// Kills the process if a proof stops making progress.
///
/// A shard that fails verification is retried rather than raised, and when the
/// retries are exhausted every thread parks: no output, no exit, no Metal work.
/// Three separate diagnoses in this project started by mistaking that for
/// slowness. The watchdog runs on its own OS thread rather than as a task
/// because the runtime is exactly what has stopped turning.
struct Watchdog(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Watchdog {
    fn arm(seconds: u64, proof: usize) -> Self {
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        if seconds > 0 {
            let flag = std::sync::Arc::clone(&done);
            std::thread::spawn(move || {
                let deadline = Instant::now() + std::time::Duration::from_secs(seconds);
                while Instant::now() < deadline {
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                if flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                eprintln!(
                    "PROOF {proof} STUCK: no result after {seconds}s. \
                     A shard that fails verification is retried silently and then parks; \
                     re-run with RUST_LOG=error to see the cause."
                );
                std::process::exit(2);
            });
        }
        Self(done)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    sp1_sdk::utils::setup_logger();
    let elf_bytes = std::fs::read(&args.elf).with_context(|| format!("reading {}", args.elf))?;
    let elf = Elf::Dynamic(Arc::from(elf_bytes.into_boxed_slice()));
    let mut stdin = match &args.stdin_bin {
        Some(path) => {
            let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
            bincode::deserialize(&bytes).context("decoding the dumped SP1Stdin")?
        }
        None => SP1Stdin::new(),
    };
    if let Some(n) = args.stdin_u32 {
        stdin.write(&n);
    }
    if let Some(path) = &args.stdin_file {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        stdin.write_slice(&bytes);
    }
    let prover = ProverClient::builder().cpu().build().await;
    if args.mode == "execute" {
        // Cycles without proving: the cheap way to tell "more shards" from
        // "bigger shards" when reading a scaling study.
        let (_, report) =
            prover.execute(elf, stdin).await.map_err(|e| anyhow!("execution failed: {e:?}"))?;
        println!("cycles {}", report.total_instruction_count());
        return Ok(());
    }
    if !matches!(args.mode.as_str(), "core" | "compressed") {
        bail!("unknown mode `{}` (use core, compressed or --execute)", args.mode);
    }
    let pk = prover.setup(elf).await.map_err(|e| anyhow!("setup failed: {e:?}"))?;
    let started = Instant::now();
    let mut failures = 0usize;
    for proof_no in 1..=args.repeat {
        let _watchdog = Watchdog::arm(args.timeout_s, proof_no);
        let t0 = Instant::now();
        let proved = match args.mode.as_str() {
            "core" => prover.prove(&pk, stdin.clone()).core().await,
            _ => prover.prove(&pk, stdin.clone()).compressed().await,
        };
        let elapsed = t0.elapsed();
        // A failure does not stop the run. The rate is the measurement: three
        // events in eighty-four proofs is what the defect note has, and a
        // harness that gives up on the first one can never improve on that.
        let outcome = match proved {
            Err(e) => Err(anyhow!("proving failed: {e:?}")),
            Ok(proof) => prover
                .verify(&proof, pk.verifying_key(), None)
                .map_err(|e| anyhow!("verification failed: {e:?}")),
        };
        match outcome {
            Ok(()) => {
                if args.repeat == 1 {
                    eprintln!("proved and verified in {elapsed:?} (mode: {})", args.mode);
                } else {
                    eprintln!(
                        "proof {proof_no}/{} verified in {elapsed:?}",
                        args.repeat
                    );
                }
            }
            Err(e) => {
                failures += 1;
                // One proof reports through the error, as it always has. A run
                // of many reports each failure where it happened and carries on,
                // because the rate is the measurement.
                if args.repeat == 1 {
                    return Err(e);
                }
                eprintln!("PROOF {proof_no}/{} FAILED after {elapsed:?}: {e:#}", args.repeat);
            }
        }
    }
    if args.repeat > 1 {
        eprintln!(
            "{} of {} proofs verified, {failures} failed, in {:?}",
            args.repeat - failures,
            args.repeat,
            started.elapsed()
        );
    }
    report_metal(started.elapsed());
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// What the Metal path contributed, if it was built in.
///
/// The GPU figure comes from the hardware's own command-buffer timestamps, so
/// the share it takes of the proof is the honest answer to how much of this
/// machine's work has actually moved off the CPU.
#[cfg(feature = "metal")]
fn report_metal(elapsed: std::time::Duration) {
    let (_dispatches, gpu_s, _wall_s) = sp1_hypercube::metal::STATS.timings();
    let total = elapsed.as_secs_f64();
    // Everything but the GPU figure is thread-seconds: shards prove
    // concurrently, so those sums are over threads, not over the wall clock.
    eprintln!(
        "metal: {} | gpu is {:.1}% of the {:.1}s proof",
        sp1_hypercube::metal::STATS.report(),
        gpu_s / total * 100.0,
        total,
    );
    let (commits, on_gpu, commit_s, rows) = sp1_hypercube::prover::COMMIT_STATS.counts();
    eprintln!(
        "commit: {commits} tensor commitments ({on_gpu} on the GPU) over {rows} rows \
         in {commit_s:.3} thread-seconds",
    );
    // Buffers handed from the kernel that wrote an array to the one that reads
    // it. `collided` is the number that a *second* producer found still sitting
    // on its address -- so the number of times a pointer-keyed cache was asked
    // to describe memory that had been freed and handed out again. It is
    // reported because it is the only direct evidence that the hazard in
    // `sp1_metal_core::handoff` is live rather than merely arguable.
    for (what, (kept, taken, orphaned, collided)) in [
        ("codewords", sp1_metal_core::handoff::CODEWORDS.counts()),
        ("round arrays", sp1_metal_core::handoff::ROUND_ARRAYS.counts()),
    ] {
        if kept > 0 {
            eprintln!(
                "handoff {what}: {kept} kept, {taken} taken, {orphaned} dropped unclaimed, \
                 {collided} on an address a dead entry still held",
            );
        }
    }
    // Where the bytes went, split by phase. `resident` is the fraction a kernel
    // read with nothing done to it first, which is what the residency work is
    // trying to move.
    eprint!("\n{}", sp1_hypercube::metal_stats::report());
    let all = sp1_hypercube::metal_stats::grand_totals();
    eprintln!(
        "residency {:.1}% of {:.2} GB read | {:.2} GB written, {:.1}% of it copied back | \
         {:.3} thread-seconds copying",
        all.residency() * 100.0,
        all.in_bytes() as f64 / 1e9,
        all.out_bytes() as f64 / 1e9,
        if all.out_bytes() == 0 { 0.0 } else { all.out_staged as f64 / all.out_bytes() as f64 * 100.0 },
        all.in_copy_s + all.out_copy_s,
    );
    // What every fold in the proof cost, counted where they all funnel through
    // rather than at the call sites, which sampling could not separate.
    #[cfg(feature = "metal")]
    eprint!("{}", slop_multilinear::metal::report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_hypercube::prover::zerocheck::counters::report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_hypercube::prover::zerocheck::counters::reject_report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_hypercube::prover::zerocheck::counters::no_pipeline_report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_hypercube::prover::zerocheck::counters::occupancy_report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_hypercube::prover::zerocheck::counters::per_chip_report());
    eprint!("{}", sp1_hypercube::metal::site_report());
    eprint!("{}", sp1_hypercube::metal::gen_report());
    #[cfg(feature = "metal")]
    eprint!("{}", slop_jagged::metal::poly_report());
    #[cfg(feature = "metal")]
    eprint!("{}", sp1_metal_core::tally::report());
    let [rejected, interaction] = sp1_hypercube::metal::cpu_round_counts();
    eprintln!(
        "gkr rounds left on the CPU: {} rejected layers in {:.3}s, {} interaction layers in {:.3}s",
        rejected.0, rejected.1, interaction.0, interaction.1,
    );
    let (kernels, compile_s) = sp1_hypercube::metal_stats::compiles();
    if kernels > 0 {
        eprintln!(
            "compiled {kernels} kernels at run time in {compile_s:.3} thread-seconds; \
             Metal compiles in another process, so that wait is not CPU time here",
        );
    }
    // The union of the command buffers' own execution intervals: what fraction
    // of the proof the device was doing anything at all.
    let (busy, span) = sp1_hypercube::metal_stats::device_busy_s();
    eprintln!(
        "device busy {busy:.3}s of a {span:.3}s span, {:.1}% -- and {:.1}% of the whole proof",
        if span > 0.0 { busy / span * 100.0 } else { 0.0 },
        busy / elapsed.as_secs_f64() * 100.0,
    );
    let hist = sp1_hypercube::metal_stats::busy_histogram(40);
    if !hist.is_empty() {
        let bar = |v: f64| "#".repeat((v * 20.0).round() as usize);
        eprintln!("device busy over the proof, {:.2}s per column:", span / hist.len() as f64);
        for (i, v) in hist.iter().enumerate() {
            eprintln!("  {:5.1}s {:>5.0}% {}", i as f64 * span / hist.len() as f64, v * 100.0, bar(*v));
        }
    }
}

#[cfg(not(feature = "metal"))]
fn report_metal(_elapsed: std::time::Duration) {}
