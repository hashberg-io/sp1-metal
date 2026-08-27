#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check what the documents claim against what the repository contains.

Reading the documents for errors finds almost nothing: a wrong count reads
exactly like a right one, and the eye confirms what it expects. Every check
here instead compares a claim against the artefact it describes -- names in a
document against names in the binary, a documented default against the
`unwrap_or` beside it, a published hash against the file. Run over documents
that had already been reviewed several times by hand, this found nine errors,
including three build artifacts committed by accident and three probes that
existed but were written down nowhere.

Set comparisons print both directions. "Documented but absent" catches a
deletion; "present but undocumented" catches an addition; either alone catches
half of what goes wrong.

Exits non-zero if anything fails, so it can gate a release.

Usage: tools/check_consistency.py [-v]
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED: list[str] = []
VERBOSE = "-v" in sys.argv[1:]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if detail and (not ok or VERBOSE) else ""))
    if not ok:
        FAILED.append(name)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def tracked() -> list[str]:
    return sh("git ls-files").split("\n")


def diff(a: set, b: set) -> str:
    return f"only in the first: {sorted(a - b)}; only in the second: {sorted(b - a)}"


def check_hygiene() -> None:
    """Nothing machine-specific, and nothing that was never meant to ship."""
    print("machine independence and hygiene")
    # Excluding this file: the patterns are string literals in it, so without
    # that the checker is the one thing it always reports. That is the same
    # self-match that made a `pgrep -f` waiter here wait on itself for hours.
    me = ":(exclude)tools/check_consistency.py"
    home = sh(f"git grep -lI '/Users/' -- . '{me}'")
    check("no absolute home-directory paths are tracked", home == "", home)
    scratch = sh(f"git grep -lI 'CLAUDE_JOB' -- . '{me}'")
    check("no scratch-directory paths are tracked", scratch == "", scratch)
    files = set(tracked())
    strays = {f for f in files if re.fullmatch(r"[^/]+\.(log|json)", f) and f != "Cargo.lock"}
    check("no stray logs or dumps at the repository root", not strays, str(sorted(strays)))
    for path in ("crates/vendor/p3-uni-stark/target", "transpile.json"):
        check(f"{path} is ignored", sh(f"git check-ignore {path}") != "")
    check("the span-profile fixture is not shadowed by an ignore rule",
          sh("git check-ignore tools/fixtures/span_profile_sample.log") == "")


def check_names() -> None:
    """Every list of names in a document, against the names themselves."""
    print("\nnames, counted where they are defined")
    listed = set(re.findall(r"\[`crates/([a-z0-9-]+)`\]", read("README.md"))) - {"vendor"}
    present = {d for d in os.listdir(os.path.join(ROOT, "crates")) if d != "vendor"}
    check("the README lists every crate", listed == present, diff(listed, present))

    doc = set(re.findall(r"^\d+\. `probes ([a-z0-9]+)`", read("knowledge/probes.md"), re.M))
    binary = set(re.findall(r'\("([a-z0-9]+)", probes::', read("crates/sp1-metal-probes/src/main.rs")))
    check("every probe the binary registers is documented", doc == binary, diff(doc, binary))

    code = set(re.findall(r"SP1_METAL_[A-Z_]+", sh("grep -rhoE 'SP1_METAL_[A-Z_]+' --include=*.rs crates/")))
    docd = set(re.findall(r"SP1_METAL_[A-Z_]+", read("USING.md")))
    check("every environment switch is documented, and no others", code == docd, diff(code, docd))

    idx = read("knowledge/README.md")
    absent = [d for d in sorted(os.listdir(os.path.join(ROOT, "knowledge")))
              if d.endswith(".md") and d != "README.md" and d not in idx]
    check("the notes index lists every note", not absent, str(absent))


def check_values() -> None:
    """Documented defaults and versions, against the code and the manifests."""
    print("\nvalues, against their definitions")
    src = read("crates/vendor/sp1-hypercube/src/prover/zerocheck/metal.rs")
    bound = re.search(r"const MAX_NODES: usize = ([0-9_]+)", src).group(1).replace("_", "")
    check("the documented node bound is the one in the code", f"| {bound} |" in read("USING.md"), bound)

    using = read("USING.md")
    stale = []
    vendor = os.path.join(ROOT, "crates/vendor")
    for d in sorted(os.listdir(vendor)):
        manifest = os.path.join(vendor, d, "Cargo.toml")
        if not os.path.exists(manifest):
            continue
        with open(manifest, encoding="utf-8") as fh:
            version = re.search(r'^version = "([^"]+)"', fh.read(), re.M).group(1)
        if not re.search(rf'^{re.escape(d)}\s*= "={re.escape(version)}"', using, re.M):
            stale.append(f"{d} {version}")
    check("every vendored version matches the pin consumers are told to use", not stale, str(stale))

    wrong = []
    for digest, path in re.findall(r"([0-9a-f]{64})\s+(\S+)", read("guests/README.md")):
        full = os.path.join(ROOT, "guests", path)
        if not os.path.exists(full):
            wrong.append(f"{path} (missing)")
            continue
        with open(full, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != digest:
                wrong.append(path)
    check("every published guest ELF hash matches its file", not wrong, str(wrong))


def check_references() -> None:
    """Every path a document points at, resolved."""
    print("\nreferences")
    md = [f for f in tracked() if f.endswith(".md")]
    broken = []
    for f in md:
        base = os.path.dirname(f)
        for text, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", read(f)):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.split("#")[0]
            if path and not os.path.exists(os.path.join(ROOT, os.path.normpath(os.path.join(base, path)))):
                broken.append(f"{f}: [{text[:30]}]({target})")
    check("every relative link resolves", not broken, "\n          ".join(broken[:5]))

    dangling = set()
    for f in md:
        for m in re.findall(r"`((?:results|tools|guests|crates|knowledge)/[A-Za-z0-9_./-]+)`", read(f)):
            if not os.path.exists(os.path.join(ROOT, m)):
                dangling.add(f"{f}: {m}")
    check("every path named in backticks exists", not dangling, "\n          ".join(sorted(dangling)[:5]))


def check_change_logs() -> None:
    """Each vendored crate claims its CHANGES.md lists every file that differs."""
    print("\nvendored change logs")
    missing = []
    vendor = os.path.join(ROOT, "crates/vendor")
    for d in sorted(os.listdir(vendor)):
        root = os.path.join(vendor, d)
        if not os.path.isdir(root):
            continue
        log = os.path.join(root, "CHANGES.md")
        if not os.path.exists(log):
            missing.append(f"{d}: no CHANGES.md")
            continue
        with open(log, encoding="utf-8") as fh:
            text = fh.read()
        for dirpath, _, names in os.walk(root):
            if f"{os.sep}target" in dirpath:
                continue
            for name in names:
                if not name.endswith((".rs", ".toml", ".metal")):
                    continue
                try:
                    with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                        body = fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
                if "sp1-metal project" in body and name not in text:
                    missing.append(os.path.relpath(os.path.join(dirpath, name), ROOT))
    check("each change log names every file it annotated", not missing, str(missing))


def main() -> int:
    check_hygiene()
    check_names()
    check_values()
    check_references()
    check_change_logs()
    print()
    if FAILED:
        print(f"{len(FAILED)} failing check(s): " + ", ".join(FAILED))
        return 1
    print("everything consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
