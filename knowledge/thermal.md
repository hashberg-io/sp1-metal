# Thermal management

How the benchmark suite keeps a measurement honest on a machine that throttles, what the operating system will and will not tell you about heat, and the ways this was got wrong first.

> **Written 2026-08-28**, prompted by an M5 Max that throttles under sustained GPU load where the M4 Max the rest of these notes were taken on did not visibly do so.
> The strategy is not specific to that machine; the pitfalls are the useful part.

## Why a benchmark has to care

Every wrong answer in this project's history came from a contaminated pair, and heat is a contaminant the existing gate could not see.
The suite already refused to run on a loaded machine, but a throttled machine is not a loaded one: it is idle by every measure the harness took, and the only symptom is that the silicon runs slower.
A pair timed in that state measures the temperature as much as the path under test.

The exposure is structural rather than incidental.
The matrix is hours of back-to-back proving with a cooldown nowhere in it, which is the definition of sustained load, and the arm that heats the machine most is the one whose advantage is being claimed.
Throttling therefore biases the result in the direction that flatters nothing: it makes the GPU arm look worse, and it does so most on exactly the large workloads where the advantage is largest.

## What the machine will tell you

Most of the obvious sources are useless on Apple Silicon, which is worth writing down so that nobody spends the afternoon rediscovering it.

- **The load average is a phantom.** It reads over 20 on an idle machine and has no relationship to anything. This was already known and is recorded in [benchmark-status.md](benchmark-status.md); it is repeated here because it is the first thing anyone reaches for.
- **`pmset -g therm` reports nothing.** It answers "No thermal warning level has been recorded" on a machine that has been proving for hours, and `pmset -g thermlog` prints no output at all.
- **There are no thermal `sysctl` keys.** The `machdep.xcpm` tree that carried them on Intel does not exist here.
- **`powermetrics` needs root.** It is the only route to actual die temperatures and GPU residency, and requiring a password makes it unusable from an unattended overnight run.
- **`ProcessInfo.thermalState` works, unprivileged, and is the right signal.** It reports one of `nominal`, `fair`, `serious` or `critical`, and it is the same signal the system uses to tell applications to back off. Reading it needs three lines of Swift, which is what [`tools/thermal_state.swift`](../tools/thermal_state.swift) is; the suite compiles it on demand rather than committing a binary.

Note what this signal is and is not.
It is the operating system's own judgement about whether software should reduce its demands, which is exactly the question a benchmark harness wants answered.
It is not a temperature, it is coarse, and it lags: by the time it reads `serious` the throttling has already happened.
That is why the suite does not rely on it alone.

## What the suite does

Three measures, in order of how much they matter.

- **A pause after every run, proportional to that run's wall clock** — a quarter of it, floored at five seconds and capped at two minutes. Proportionality is the whole idea: heat is roughly proportional to the work that produced it, so a fixed pause is either pointless after a three-second proof or ruinous after a three-hundred-second one. This is the protection that does not depend on being able to read anything.
- **The pause is extended while the system reports pressure**, up to ten further minutes, so a machine that is genuinely hot is waited out rather than merely counted to.
- **Every run records the conditions it was taken under** — thermal state before and after, reclaimable memory, seconds cooled — so that a figure which looks wrong in six months can be checked against the state of the machine instead of guessed at. This is worth more than the gating, because gating can only prevent the runs it anticipates while a record explains the ones it did not.

The idle gate also now waits on reclaimable memory, which is a related failure for the same reason: the largest guests peak above a hundred gigabytes resident, and a run started before its predecessor has handed its pages back will swap, which measures the swap.

## Pitfalls

**Gating that lives in one script protects only that script.**
This is the one that cost a machine.
The cooldown and the thermal gate were written into the suite, and then the shader cache was warmed by a *separate* script that ran twenty-two guests back to back with none of it — and the machine took a watchdog panic partway through the heaviest guest in the suite, with twelve of eighteen cores offline.
The protection was real, correct, and in the wrong file.
Anything that drives the prover in a loop needs the same treatment as the suite, and the warming pass needs it most, because compiling and proving together is the hottest thing this repository does.

**Warm-up runs are hotter than measured ones, and are easy to forget.**
The discarded warm-up before each point is the run that compiles, so it is the run that heats the machine, and it used to be the one run with no pause after it.

**A cold shader cache looks exactly like throttling.**
Both present as a Metal arm that is slower than it should be for no visible reason, and confusing them sends you to the wrong fix.
Tell them apart by the kernel count rather than the clock: the host reports how many rounds took the CPU while a kernel was still compiling, and a cold cache says hundreds where a warm one says none.
See [measurements.md](measurements.md) and the compile-budget material in [USING.md](../USING.md).

**Compiling is CPU load, not GPU load.**
A long warming pass pins one core in the Metal compiler service and barely touches the GPU, so a thermal strategy tuned only to GPU work will not anticipate it.

**Waiting on the wrong pattern never finishes.**
A `pgrep -f` waiter matches its own command line, so a shell waiting for a prover to exit can wait forever on itself; this was already recorded in [benchmark-status.md](benchmark-status.md) and it bites again as soon as cooldowns make people write wait loops.

**A panic clears the scratch directory but not the shader cache.**
Losing a machine mid-run costs the logs written under the system temporary directory and nothing else: the compiled kernels survive a reboot, so the expensive part of a warming pass does not have to be paid twice.
Check by proving one small workload and reading the dispatch count before assuming the work is gone.

## Checking a finished run

The point of recording conditions is to be able to ask, afterwards, whether they mattered.
The check that answers it compares runs of the *same* point taken under different thermal states, which controls for everything except the heat:

```sh
python3 - <<'PY'
import json, glob, statistics as st
from collections import defaultdict
by = defaultdict(lambda: defaultdict(list))
for f in glob.glob("results/bench-*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "meta" or r.get("warmup") or not r.get("verified"):
            continue
        k = (r["guest"], r["size"], r["config"], r["mode"], r["shard_size"], r["arm"])
        by[k][r.get("thermal_before")].append(r["wall_s"])
ratios = [st.median(d["fair"]) / st.median(d["nominal"])
          for d in by.values() if d.get("fair") and d.get("nominal")]
print(len(ratios), "points measured under both states")
if ratios:
    print(f"fair/nominal wall: median {st.median(ratios):.3f}, "
          f"range {min(ratios):.3f}-{max(ratios):.3f}")
PY
```

A median near 1.00 says the pauses held the machine below the point where heat changes the answer, and the run can be read as taken.
A median meaningfully above 1.00 says the cooldown was not enough and the figures are depressed by an unknown amount, which is a reason to re-run rather than to caveat.
