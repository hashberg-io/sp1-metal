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


def check_naming() -> None:
    """Both records folders name their files by when they were made."""
    print("\nfile naming")
    notes = [f for f in os.listdir(os.path.join(ROOT, "knowledge"))
             if f.endswith(".md") and f != "README.md"]
    bad = [f for f in sorted(notes) if not re.fullmatch(r"20\d\d-\d\d-\d\d-[a-z0-9-]+\.md", f)]
    check("every note is dated yyyy-mm-dd-<slug>.md", not bad, str(bad))

    res = [f for f in os.listdir(os.path.join(ROOT, "results")) if f != "README.md"]
    pat = r"20\d\d-\d\d-\d\d-m[45]-max-[a-z0-9.-]+\.[a-z]+"
    bad = [f for f in sorted(res) if not re.fullmatch(pat, f)]
    check("every result is dated yyyy-mm-dd-<machine>-<slug>.<ext>", not bad, str(bad))


def check_names() -> None:
    """Every list of names in a document, against the names themselves."""
    print("\nnames, counted where they are defined")
    listed = set(re.findall(r"\[`crates/([a-z0-9-]+)`\]", read("README.md"))) - {"vendor"}
    present = {d for d in os.listdir(os.path.join(ROOT, "crates")) if d != "vendor"}
    check("the README lists every crate", listed == present, diff(listed, present))

    doc = set(re.findall(r"^\d+\. `probes ([a-z0-9]+)`", read("knowledge/2026-08-19-probes.md"), re.M))
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


def check_defaults() -> None:
    """Every default the switch table states, against the code that supplies it.

    This is the check the docstring above promised and did not have. The table
    in `USING.md` is the only place a consumer learns what a switch does when
    they leave it alone, and it is a column of prose beside a value in a
    `unwrap_or` that nothing forced to agree with it -- the shape of drift this
    file exists to catch. It is not hypothetical: `SP1_METAL_REUSE` was flipped
    off after measurement, and the table and the code had to be edited
    separately to say so.

    A switch may be read at several sites. The `=check` modes are read as their
    own site and default off, so a switch counts as on by default if *any* site
    turns it on, which is what the table's column means.

    Anything this cannot classify fails rather than being skipped. A checker
    that quietly ignores a shape it does not know protects nothing, and the new
    shape is exactly where an unchecked default would hide.
    """
    print("\nswitch defaults, against the code that supplies them")
    src = {f: read(f) for f in tracked() if f.endswith(".rs")}

    # Constants that a default is spelled with, so `unwrap_or(MAX_NODES)` can
    # be compared against a number in a table.
    consts: dict[str, int] = {}
    for text in src.values():
        for name, value in re.findall(r"const ([A-Z][A-Z0-9_]*): usize = ([^;]+);", text):
            try:
                consts[name] = int(eval(value.replace("_", ""), {"__builtins__": {}}))
            except Exception:
                pass

    def number(expr: str):
        expr = expr.strip()
        if re.fullmatch(r"[0-9_]+\.[0-9]+", expr):
            return float(expr)
        if re.fullmatch(r"[0-9_]+", expr):
            return int(expr.replace("_", ""))
        if expr in consts:
            return consts[expr]
        if re.fullmatch(r"[0-9_]+\s*<<\s*[0-9]+", expr):
            return int(eval(expr.replace("_", ""), {"__builtins__": {}}))
        return None

    # Which switch a helper function's own default defers to, so the table may
    # name a switch rather than a value.
    defers: dict[str, str] = {}
    for text in src.values():
        for name, body in re.findall(r"\nfn ([a-z_]+)\(\)[^{]*\{(.{0,400})", text, re.S):
            m = re.search(r'env::var\("(SP1_METAL_[A-Z_]+)"\)', body)
            if m:
                defers[name] = m.group(1)

    def classify(window: str):
        """What one read site says the default is."""
        m = re.search(r"unwrap_or_else\(\s*([a-z_]+)\s*\)", window)
        if m:
            return ("defers", defers.get(m.group(1), m.group(1)))
        m = re.search(r"unwrap_or\(([^()]*(?:\([^()]*\))?[^()]*)\)", window)
        if m:
            arg = m.group(1).strip()
            if arg in ("true", "false"):
                return ("bool", arg == "true")
            n = number(arg)
            if n is not None:
                return ("number", n)
            if re.fullmatch(r"[a-z_][a-z0-9_]*", arg):
                return ("caller", arg)
            return None
        if re.search(r"_ => Mode::(Gpu|Cpu)", window):
            return ("bool", "Mode::Gpu" in re.search(r"_ => (Mode::\w+)", window).group(1))
        if "is_ok_and" in window:
            return ("bool", False)
        return None

    documented = {}
    for name, stated in re.findall(r"^\| `(SP1_METAL_[A-Z_]+)` \| [^|]* \| ([^|]+)\|", read("USING.md"), re.M):
        documented[name] = stated.strip()

    wrong, unknown = [], []
    for name, stated in sorted(documented.items()):
        sites = []
        for text in src.values():
            for m in re.finditer(rf'env::var\("{name}"\)', text):
                got = classify(text[m.start():m.start() + 400])
                if got:
                    sites.append(got)
        if not sites:
            unknown.append(f"{name}: no read site whose default this can read")
            continue
        # A switch is on by default if any site turns it on; the `=check` modes
        # are sites of their own and every one of them defaults off.
        bools = [v for kind, v in sites if kind == "bool"]
        other = [(kind, v) for kind, v in sites if kind != "bool"]
        # What the table says, with the reason it gives stripped off.
        plain = re.split(r"\s+[—-]{1,2}\s+", stated.replace("**", ""))[0].strip()
        if other:
            kind, value = other[0]
            if kind == "number":
                want = number(plain)
                ok = want is not None and float(want) == float(value)
            elif kind == "defers":
                ok = plain.strip("`") == value
            else:
                ok = plain == "as the caller set it"
            if not ok:
                wrong.append(f"{name}: code says {value}, the table says {plain!r}")
        elif bools:
            on = any(bools)
            if plain not in ("on", "off"):
                unknown.append(f"{name}: the table says {plain!r}, which is not on or off")
            elif (plain == "on") != on:
                wrong.append(f"{name}: code defaults {'on' if on else 'off'}, the table says {plain!r}")

    check("every documented default is the one the code supplies", not wrong,
          "\n          ".join(wrong))
    check("every switch's default is one this can read", not unknown,
          "\n          ".join(unknown))


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

    # Rust doc comments cite these files the same way documents do, and a
    # rename that misses one leaves a reader chasing a path that is gone.
    dangling = set()
    for f in md + [f for f in tracked() if f.endswith(".rs")]:
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
    check_naming()
    check_names()
    check_values()
    check_defaults()
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
