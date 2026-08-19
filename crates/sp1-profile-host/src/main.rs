//! Proves a given RISC-V ELF with the SP1 CPU prover under span-timing
//! instrumentation, so tools/span_profile.py can split wall time into phases.
//!
//! Run with `RUST_LOGGER=flat RUST_LOG=debug` and capture stderr:
//!
//! ```sh
//! RUST_LOGGER=flat RUST_LOG=debug cargo run --release -- \
//!     --elf path/to/guest.elf --mode compressed 2> prove.log
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
}

fn parse_args() -> Result<Args> {
    let mut elf = None;
    let mut mode = "compressed".to_string();
    let mut stdin_file = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--elf" => elf = Some(args.next().context("--elf needs a path")?),
            "--mode" => mode = args.next().context("--mode needs core|compressed")?,
            "--stdin-file" => stdin_file = Some(args.next().context("--stdin-file needs a path")?),
            other => bail!("unknown argument `{other}`"),
        }
    }
    Ok(Args { elf: elf.context("--elf <path> is required")?, mode, stdin_file })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    sp1_sdk::utils::setup_logger();
    let elf_bytes = std::fs::read(&args.elf).with_context(|| format!("reading {}", args.elf))?;
    let elf = Elf::Dynamic(Arc::from(elf_bytes.into_boxed_slice()));
    let mut stdin = SP1Stdin::new();
    if let Some(path) = &args.stdin_file {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        stdin.write_slice(&bytes);
    }
    let prover = ProverClient::builder().cpu().build().await;
    let pk = prover.setup(elf).await.map_err(|e| anyhow!("setup failed: {e:?}"))?;
    let t0 = Instant::now();
    let proof = match args.mode.as_str() {
        "core" => prover.prove(&pk, stdin).core().await,
        "compressed" => prover.prove(&pk, stdin).compressed().await,
        other => bail!("unknown mode `{other}` (use core or compressed)"),
    }
    .map_err(|e| anyhow!("proving failed: {e:?}"))?;
    let elapsed = t0.elapsed();
    prover
        .verify(&proof, pk.verifying_key(), None)
        .map_err(|e| anyhow!("verification failed: {e:?}"))?;
    eprintln!("proved and verified in {elapsed:?} (mode: {})", args.mode);
    Ok(())
}
