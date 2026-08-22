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
//! ```sh
//! SP1_DUMP=1 path/to/real-host              # writes program.bin, stdin.bin
//! RUST_LOGGER=flat RUST_LOG=debug cargo run --release -- \
//!     --elf program.bin --stdin-bin stdin.bin --mode compressed > prove.log 2>&1
//! python3 ../../tools/span_profile.py prove.log --json profile.json
//! ```

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
}

fn parse_args() -> Result<Args> {
    let mut elf = None;
    let mut mode = "compressed".to_string();
    let mut stdin_file = None;
    let mut stdin_bin = None;
    let mut stdin_u32 = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--elf" => elf = Some(args.next().context("--elf needs a path")?),
            "--mode" => mode = args.next().context("--mode needs core|compressed")?,
            "--stdin-file" => stdin_file = Some(args.next().context("--stdin-file needs a path")?),
            "--stdin-bin" => stdin_bin = Some(args.next().context("--stdin-bin needs a path")?),
            "--execute" => mode = "execute".to_string(),
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
    })
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
    let pk = prover.setup(elf).await.map_err(|e| anyhow!("setup failed: {e:?}"))?;
    let t0 = Instant::now();
    let proof = match args.mode.as_str() {
        "core" => prover.prove(&pk, stdin).core().await,
        "compressed" => prover.prove(&pk, stdin).compressed().await,
        other => bail!("unknown mode `{other}` (use core, compressed or --execute)"),
    }
    .map_err(|e| anyhow!("proving failed: {e:?}"))?;
    let elapsed = t0.elapsed();
    prover
        .verify(&proof, pk.verifying_key(), None)
        .map_err(|e| anyhow!("verification failed: {e:?}"))?;
    eprintln!("proved and verified in {elapsed:?} (mode: {})", args.mode);
    report_metal(elapsed);
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
}

#[cfg(not(feature = "metal"))]
fn report_metal(_elapsed: std::time::Duration) {}
