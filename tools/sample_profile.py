#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Turn a macOS `sample` call graph into a flat profile of where CPU time goes.

`sample` reports leaves and totals, neither of which answers "which operation
is this". A leaf like `BinomialExtensionField::mul` is a quarter of the
prover's CPU time and tells you nothing, because every phase multiplies.

So each sample's self time is attributed twice: to the *outermost* frame that
names a phase of proving, and to the *deepest* frame that names an operation.
Both are needed. Folding an MLE's last variable is a quarter of the prover's
CPU time, but it is worth knowing which sumcheck is asking -- and asking only
"which phase" would hide that one operation dominates three of them.

Usage: tools/sample_profile.py sample.txt [--by-leaf] [--anchor SUBSTRING]
                                [--callers SUBSTRING] [--variants SUBSTRING]

The sampling interval is read from the report's own header, so a report taken
every 5 ms is not silently reported as five times less work than it was.

`--anchor` adds an operation anchor for one run, so a block that comes out as
unattributed can be split without editing this file.
"""

import re
import sys
from collections import defaultdict

# Phases, matched root-first: the outermost frame that names one wins, so work
# a phase delegates still counts against the phase.
PHASES = [
    ("logup_gkr", "gkr"),
    ("LogupGkr", "gkr"),
    ("zerocheck", "zerocheck"),
    ("Zerocheck", "zerocheck"),
    ("jagged", "pcs"),
    ("Jagged", "pcs"),
    ("basefold", "pcs"),
    ("Basefold", "pcs"),
    ("commit_tensors", "commit"),
    ("commit_multilinears", "commit"),
    ("sp1_metal_probes::commit", "commit"),
    ("generate_dependencies", "tracegen"),
    ("generate_trace", "tracegen"),
    ("sp1_core_machine", "tracegen"),
    ("sp1_core_executor", "execute"),
    ("setup", "setup"),
    ("compile", "setup"),
    ("verify", "verify"),
]

# Operations, matched leaf-first: the most specific frame on the stack wins.
ANCHORS = [
    # --- logup GKR
    ("logup_gkr::execution::generate_interaction_vals", "gkr: interaction values"),
    ("logup_gkr::execution", "gkr: first layer / transition"),
    ("logup_gkr::cpu::", "gkr: circuit generation"),
    ("logup_poly::cpu_round_evals", "gkr: round evals on CPU"),
    ("logup_gkr::metal::run_and_record", "gpu: blocked on a dispatch"),
    ("waitUntilCompleted", "gpu: blocked on a dispatch"),
    ("logup_gkr::metal", "gpu: encoding a dispatch"),
    ("logup_gkr::prover", "gkr: round bookkeeping"),
    ("logup_gkr", "gkr: other"),
    # --- the Reed-Solomon encode
    ("radix_2_dit_parallel", "encode: DFT"),
    ("coset_dft", "encode: DFT"),
    ("reverse_matrix_index_bits", "encode: bit reversal"),
    ("transpose", "encode: transpose"),
    ("slop_dft", "encode: DFT"),
    ("p3_dft", "encode: DFT"),
    # --- commitment
    ("sp1_metal_probes::commit", "gpu: Merkle commitment"),
    ("poseidon2", "commit: Poseidon2 on CPU"),
    ("Poseidon2", "commit: Poseidon2 on CPU"),
    ("merkle", "commit: Merkle on CPU"),
    ("Merkle", "commit: Merkle on CPU"),
    ("commit_tensors", "commit: other"),
    # --- zerocheck
    ("zerocheck::metal", "gpu: zerocheck"),
    ("zerocheck", "zerocheck: on CPU"),
    ("ConstraintSumcheckFolder", "zerocheck: constraint folding"),
    ("SymbolicExpression", "zerocheck: symbolic constraints"),
    # --- jagged PCS
    ("slop_jagged::metal", "gpu: jagged"),
    ("jagged", "pcs: jagged on CPU"),
    ("basefold", "pcs: basefold"),
    ("Basefold", "pcs: basefold"),
    # --- generic multilinear work, whoever asked for it
    ("mle_fix_last_variable", "fold: fix_last_variable"),
    ("partial_lagrange", "eq: partial_lagrange"),
    ("partial_eq_with_basis", "eq: partial_lagrange"),
    ("slop_sumcheck", "sumcheck: driver"),
    ("dot_along_dim", "mle: dot products"),
    ("slop_multilinear", "mle: other"),
    ("slop_tensor", "tensor: other"),
    # --- trace generation and setup
    ("generate_dependencies", "tracegen: dependencies"),
    ("generate_trace", "tracegen: traces"),
    ("sp1_core_machine::riscv", "tracegen: riscv chips"),
    ("sp1_core_executor", "execute: the guest"),
    ("sp1_core_machine", "tracegen: other"),
    ("challenger", "challenger"),
    ("Challenger", "challenger"),
    ("compile", "setup: compiling"),
]

IDLE = (
    "__psynch_cvwait",
    "swtch_pri",
    "__workq_kernreturn",
    "__semwait_signal",
    "kevent",
    "wait_until_cold",
    "_pthread_cond_wait",
    "park",
)

LINE = re.compile(r"^([ +!:|]*)(\d+) (.*)$")


def frame_name(rest):
    """The symbol, without the `(in image) + offset [address]` tail."""
    cut = rest.find("  (in ")
    return rest[:cut] if cut >= 0 else rest.rstrip()


ABBREV = [
    ("p3_field::extension::binomial_extension::BinomialExtensionField<p3_koala_bear::koala_bear::KoalaBear, 4>", "EF"),
    ("p3_koala_bear::koala_bear::KoalaBear", "KB"),
    ("sp1_core_machine::riscv::RiscvAir<KB>", "RiscvAir"),
    ("sp1_hypercube::", ""),
    ("slop_multilinear::", ""),
    ("slop_sumcheck::", ""),
    ("slop_jagged::", ""),
    ("slop_tensor::", ""),
    ("core::iter::adapters::", ""),
    ("alloc::vec::", ""),
]


def short(name, keep_generics=False):
    """A frame name a person can read.

    Monomorphised Rust names run to several hundred characters of nested
    generics, and in a profile the interesting part is almost always the path
    plus, occasionally, whether the field was the base or the extension.
    """
    for long, brief in ABBREV:
        name = name.replace(long, brief)
    if not keep_generics:
        out, depth = [], 0
        for ch in name:
            if ch == "<":
                depth += 1
                if depth == 1:
                    continue
            elif ch == ">":
                depth -= 1
                continue
            if depth == 0:
                out.append(ch)
        name = "".join(out)
    return name.strip()


def classify(stack, anchors):
    """`(phase, operation)` for one stack: outermost phase, deepest operation."""
    phase = None
    for name in stack:
        hit = next((label for pat, label in PHASES if pat in name), None)
        if hit:
            phase = hit
            break
    op = None
    for name in reversed(stack):
        hit = next((label for pat, label in anchors if pat in name), None)
        if hit:
            op = hit
            break
    if op is None:
        leaf = stack[-1] if stack else ""
        if any(w in leaf for w in IDLE):
            return ("(idle)", "(idle: no work on this thread)")
        return (phase or "(none)", "(unattributed)")
    return (phase or "(none)", op)


INTERESTING = ("sp1_hypercube", "sp1_core_machine", "slop_", "p3_dft", "sp1_metal_probes")


def callers_of(nodes, self_counts, pattern, depth_wanted=6):
    """Where a given operation is called from, by inclusive sample count.

    The chain is filtered to project and slop frames, because the twenty rayon
    and core::iter frames between a caller and its callee say nothing about
    which part of the prover asked for the work.
    """
    chains = defaultdict(int)
    stack = []
    for i, (depth, count, name) in enumerate(nodes):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, i))
        if pattern not in name:
            continue
        if any(pattern in nodes[j][2] for _, j in stack[:-1]):
            continue  # a nested repeat of the same frame
        chain = [
            short(nodes[j][2])
            for _, j in stack[:-1]
            if any(k in nodes[j][2] for k in INTERESTING)
        ]
        seen, dedup = set(), []
        for c in reversed(chain[-depth_wanted * 3:]):
            if c not in seen:
                seen.add(c)
                dedup.append(c)
        chains[" < ".join(dedup[:depth_wanted])] += count
    return chains


def main():
    path = sys.argv[1]
    by_leaf = "--by-leaf" in sys.argv
    anchors = list(ANCHORS)
    for i, a in enumerate(sys.argv):
        if a == "--anchor":
            anchors.insert(0, (sys.argv[i + 1], f"custom: {sys.argv[i + 1]}"))

    lines = open(path, errors="replace").read().splitlines()
    # "Analysis of sampling <name> (pid N) every M milliseconds"
    ms = 1.0
    m = re.search(r"every (\d+(?:\.\d+)?) millisecond", lines[0] if lines else "")
    if m:
        ms = float(m.group(1))
    try:
        start = lines.index("Call graph:") + 1
    except ValueError:
        sys.exit("no call graph in that file")
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("Total number in stack")),
        len(lines),
    )

    # Parse into nodes, then self = count - sum(direct children).
    nodes = []  # (depth, count, name)
    for raw in lines[start:end]:
        m = LINE.match(raw)
        if not m:
            continue
        nodes.append((len(m.group(1)), int(m.group(2)), frame_name(m.group(3))))

    self_counts = [c for _, c, _ in nodes]
    stack = []  # (depth, index)
    for i, (depth, count, _) in enumerate(nodes):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            self_counts[stack[-1][1]] -= count
        stack.append((depth, i))

    # Attribute each node's self time to the deepest anchor above it.
    totals = defaultdict(int)
    leaves = defaultdict(int)
    stack = []
    for i, (depth, count, name) in enumerate(nodes):
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, i))
        if self_counts[i] <= 0:
            continue
        names = [nodes[j][2] for _, j in stack]
        totals[classify(names, anchors)] += self_counts[i]
        leaves[name] += self_counts[i]

    for i, a in enumerate(sys.argv):
        if a == "--variants":
            pattern = sys.argv[i + 1]
            seen = defaultdict(int)
            stack = []
            for j, (depth, count, name) in enumerate(nodes):
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                stack.append((depth, j))
                if pattern not in name:
                    continue
                if any(pattern in nodes[k][2] for _, k in stack[:-1]):
                    continue
                seen[short(name, keep_generics=True)] += count
            total = sum(seen.values())
            print(f"`{pattern}`: {total * ms / 1000:.2f} thread-seconds, by instantiation\n")
            for name, count in sorted(seen.items(), key=lambda kv: -kv[1])[:15]:
                print(f"{count * ms / 1000:>8.2f}  {name[:180]}")
            return
        if a == "--callers":
            pattern = sys.argv[i + 1]
            chains = callers_of(nodes, self_counts, pattern)
            total = sum(chains.values())
            print(f"`{pattern}` is entered {total * ms / 1000:.2f} thread-seconds' worth, from:\n")
            for chain, count in sorted(chains.items(), key=lambda kv: -kv[1])[:12]:
                print(f"{count * ms / 1000:>8.2f}  {chain or '(no project frame above it)'}")
            return

    grand = sum(totals.values())
    idle = sum(v for (_, op), v in totals.items() if op.startswith("(idle"))
    work = grand - idle
    print(f"{grand * ms / 1000:.1f} thread-seconds sampled, of which {work * ms / 1000:.1f} doing work\n")

    by_phase = defaultdict(int)
    for (phase, op), count in totals.items():
        if not op.startswith("(idle"):
            by_phase[phase] += count
    print(f"{'phase':<12} {'thread-s':>9} {'% of work':>10}")
    for phase, count in sorted(by_phase.items(), key=lambda kv: -kv[1]):
        print(f"{phase:<12} {count * ms / 1000:>9.2f} {count / work * 100:>9.1f}%")

    print(f"\n{'phase':<12} {'operation':<34} {'thread-s':>9} {'% of work':>10}")
    for (phase, op), count in sorted(totals.items(), key=lambda kv: -kv[1]):
        if op.startswith("(idle"):
            continue
        if count / work < 0.002:
            continue
        print(f"{phase:<12} {op:<34} {count * ms / 1000:>9.2f} {count / work * 100:>9.1f}%")
    print(f"\nidle: {idle * ms / 1000:.2f} thread-seconds on threads with no work on the stack")
    if by_leaf:
        print("\ntop leaves by self time:")
        for name, count in sorted(leaves.items(), key=lambda kv: -kv[1])[:25]:
            print(f"{count * ms / 1000:>9.2f}  {name[:150]}")


main()
