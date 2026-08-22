use std::collections::HashMap;

use serde_json::{json, Value};
use sp1_metal_probes::gpu::Gpu;
use sp1_metal_probes::probes;

const USAGE: &str = "\
Usage: probes <command> [flags]

Commands:
  info                       Report Metal device information.
  check                      Run all GPU-vs-CPU parity checks (small inputs, fast).
  triad      [--n4 N] [--reps R] [--cpu]         Streaming bandwidth (uint4 triad).
  imul       [--grid N] [--iters I] [--reps R] [--cpu]  Integer op throughput (add/mul/mulhi/kb_mul).
  fieldops   [--n N] [--reps R] [--cpu]          Streamed base/ext field multiplication.
  fold       [--n-out N] [--reps R] [--cpu]      BaseFold fold + fix-last-variable.
  poseidon2  [--n N] [--reps R] [--cpu]          Poseidon2 width-16 permutations.
  zerocheck  [--n N] [--cols C] [--terms T] [--reps R] [--cpu]  Synthetic zerocheck round.
  gkr        [--n N] [--reps R] [--cpu]          One sumcheck round of a logup GKR layer.
  launch     [--n N] [--reps R]                  Dispatch, round-trip and fold-tail fixed costs.
  challenger [--n N] [--reps R]                  Duplex sponge on the GPU: a fold with Fiat-Shamir in the loop.
  backend    [--n N] [--reps R]                  slop Buffer on MetalBackend: what the device seam costs.
  encode     [--rows N] [--cols C] [--reps R]    Reed-Solomon encode inside a BaseFold commitment (CPU only).
  all        [--reps R] [--cpu]                  All throughput probes at default sizes.

Flags take `--key value` form. Default sizes are chosen for a ~1 GiB working
set per probe; all timings are host-clock around command-buffer execution.";

fn parse_flags(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut flags = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected argument `{}`", args[i]))?;
        if key == "cpu" {
            flags.insert("cpu".to_string(), "true".to_string());
            i += 1;
        } else {
            let value =
                args.get(i + 1).ok_or_else(|| format!("flag --{key} needs a value"))?;
            flags.insert(key.to_string(), value.clone());
            i += 2;
        }
    }
    Ok(flags)
}

fn flag_usize(flags: &HashMap<String, String>, key: &str, default: usize) -> Result<usize, String> {
    match flags.get(key) {
        None => Ok(default),
        Some(v) => v.parse().map_err(|_| format!("--{key} expects an integer, got `{v}`")),
    }
}

fn info(gpu: &Gpu) -> Value {
    json!({
        "device": gpu.device.name().to_string(),
        "unified_memory": gpu.device.has_unified_memory(),
        "recommended_max_working_set_bytes": gpu.device.recommended_max_working_set_size(),
        "max_threads_per_threadgroup": gpu.device.max_threads_per_threadgroup().width,
    })
}

fn run_checks(gpu: &Gpu) -> Result<(), String> {
    let checks: [(&str, fn(&Gpu) -> Result<(), String>); 11] = [
        ("triad", probes::triad::check),
        ("imul", probes::imul::check),
        ("fieldops", probes::fieldops::check),
        ("fold", probes::fold::check),
        ("poseidon2", probes::poseidon2::check),
        ("zerocheck", probes::zerocheck::check),
        ("launch", probes::launch::check),
        ("encode", probes::encode::check),
        ("gkr", probes::gkr::check),
        ("challenger", probes::challenger::check),
        ("backend", probes::backend::check),
    ];
    let mut failed = false;
    for (name, check) in checks {
        match check(gpu) {
            Ok(()) => println!("PASS {name}"),
            Err(e) => {
                failed = true;
                println!("FAIL {name}: {e}");
            }
        }
    }
    if failed {
        Err("one or more parity checks failed".to_string())
    } else {
        Ok(())
    }
}

fn run(command: &str, flags: &HashMap<String, String>, gpu: &Gpu) -> Result<Value, String> {
    let reps = flag_usize(flags, "reps", 10)?;
    let with_cpu = flags.contains_key("cpu");
    match command {
        "triad" => probes::triad::bench(gpu, flag_usize(flags, "n4", 1 << 26)?, reps, with_cpu),
        "imul" => probes::imul::bench(
            gpu,
            flag_usize(flags, "grid", 1 << 20)?,
            flag_usize(flags, "iters", 4096)? as u32,
            reps,
            with_cpu,
        ),
        "encode" => probes::encode::bench(
            gpu,
            flag_usize(flags, "rows", 1 << 20)?,
            flag_usize(flags, "cols", 64)?,
            reps,
        ),
        "launch" => probes::launch::bench(gpu, flag_usize(flags, "n", 1 << 24)?, reps),
        "challenger" => probes::challenger::bench(gpu, flag_usize(flags, "n", 1 << 24)?, reps),
        "backend" => probes::backend::bench(gpu, flag_usize(flags, "n", 1 << 26)?, reps),
        "fieldops" => probes::fieldops::bench(gpu, flag_usize(flags, "n", 1 << 27)?, reps, with_cpu),
        "gkr" => probes::gkr::bench(gpu, flag_usize(flags, "n", 1 << 23)?, reps, with_cpu),
        "fold" => probes::fold::bench(gpu, flag_usize(flags, "n-out", 1 << 25)?, reps, with_cpu),
        "poseidon2" => {
            probes::poseidon2::bench(gpu, flag_usize(flags, "n", 1 << 21)?, reps, with_cpu)
        }
        "zerocheck" => probes::zerocheck::bench(
            gpu,
            flag_usize(flags, "n", 1 << 22)?,
            flag_usize(flags, "cols", 64)?,
            flag_usize(flags, "terms", 128)?,
            reps,
            with_cpu,
        ),
        "all" => {
            let results = vec![
                probes::triad::bench(gpu, 1 << 26, reps, with_cpu)?,
                probes::imul::bench(gpu, 1 << 20, 4096, reps, with_cpu)?,
                probes::fieldops::bench(gpu, 1 << 27, reps, with_cpu)?,
                probes::fold::bench(gpu, 1 << 25, reps, with_cpu)?,
                probes::poseidon2::bench(gpu, 1 << 21, reps, with_cpu)?,
                probes::zerocheck::bench(gpu, 1 << 22, 64, 128, reps, with_cpu)?,
                probes::gkr::bench(gpu, 1 << 23, reps, with_cpu)?,
                probes::launch::bench(gpu, 1 << 24, reps)?,
                probes::challenger::bench(gpu, 1 << 24, reps)?,
                probes::backend::bench(gpu, 1 << 26, reps)?,
                probes::encode::bench(gpu, 1 << 20, 64, reps)?,
            ];
            Ok(Value::Array(results))
        }
        other => Err(format!("unknown command `{other}`\n\n{USAGE}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().map(String::as_str) else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let flags = match parse_flags(&args[1..]) {
        Ok(flags) => flags,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let outcome = match command {
        "info" => Ok(info(&gpu)),
        "check" => run_checks(&gpu).map(|()| json!({ "parity": "ok" })),
        other => run(other, &flags, &gpu),
    };
    match outcome {
        Ok(value) => {
            let output = json!({ "host": info(&gpu), "result": value });
            let rendered = serde_json::to_string_pretty(&output).expect("json rendering");
            println!("{rendered}");
            if let Some(path) = flags.get("out") {
                std::fs::write(path, &rendered).expect("write --out file");
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
