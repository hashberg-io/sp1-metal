# Release plan

> **Status: executed.** Items 1 to 6 are done and v0.1.0 is tagged. Pushing is the only step outstanding.

This plans a public release of sp1-metal from one repository at a tag, so that people can clone it, verify what it claims, and use it in their own SP1 host programs.
It does not plan a crates.io publication or an upstream contribution; both are ruled out below, the first by construction and the second by choice.

The release is deliberately artisanal.
The bar is that setting it up should be as streamlined as the mechanism allows, not that it should look like a product.

## What the release is

**One repository, one tag, two audiences.**

An *integrator* wants GPU proving in their own SP1 host program.
They need no clone: twenty-two lines of TOML in their own manifest, pointing at this repository's git URL.

An *evaluator* wants to see whether the claim is true on their machine.
They clone, run one command, and get a CPU/GPU pair with the speedup printed.

Everything below serves one of those two.

### Why not crates.io

`[patch.crates-io]` matches by package *name*.
Renaming the five forks breaks the patch; keeping the names is impossible, because upstream holds them.
So git-tag distribution is not a stage this project would outgrow — it is the only mechanism that exists for an overlay of this shape, and the release is designed around it rather than apologising for it.

`sp1-metal-core` and `sp1-metal-probes` could be published, but neither is useful on its own, so all six of this project's crates stay `publish = false`.

### Why not upstream, for now

The four CPU fixes contain no Metal and would help every SP1 user on any platform, and the `LogupRoundOps` seam leaves the CPU path bit-identical.
They remain offerable — that is what the per-crate `CHANGES.md` discipline is for — but this is experimental software, and offering it upstream is a different undertaking from releasing it.
Revisit when the fork has been used by someone other than its author.

## What was established before planning this

Each of these was measured or tested, not assumed, and each shaped an item below.

**A consumer needs no clone.**
Cargo searches a git repository's whole tree for `[patch]` sources, so it finds `crates/vendor/*` even though they are not workspace members.
Tested from an outside project against a `file://` URL of this repository: all seven crates resolved to the git source — the five forks plus `sp1-metal-core` and `sp1-metal-probes`, which are path dependencies inside the checkout — and all five forks resolved with the `metal` feature on.
`slop-jagged` picks the feature up transitively through `sp1-hypercube/metal`, so a consumer's feature block names four crates, not five.

**The runtime gates default to on.**
Every boolean `SP1_METAL_*` switch reads `unwrap_or(true)` — or, for the encode, falls through to `Mode::Gpu` — except the two that are deliberately off, `SP1_METAL_ZC_LAYOUT` and `SP1_METAL_DEFER`.
The remaining five are numeric thresholds rather than switches: `MIN_ENTRIES`, `MIN_LAYER`, `PERMITS`, `ZEROCHECK_FOLD` and `ZEROCHECK_MAX_NODES`.
Enabling the cargo feature is therefore the whole of the setup.
The eleven variables the acceptance script sets exist to force an honest *CPU* arm, not to switch the GPU on, and no consumer needs them.

**The reproduction harness is not in the repository.**
The three guest programs behind every number this project has quoted live only in a session scratchpad under `/private/tmp`.
They are three crates of about twenty lines each with one dependency, and their built ELFs are 150 KB apiece.
`tools/scaling_study.sh` takes their location as an argument because there was nowhere in the repository to put them.

**The guests cannot be rebuilt on this machine.**
`cargo prove` and `sp1up` are not installed.
The committed ELFs will be the only copies, so they must be committed exactly as they are, and the crates must not be renamed.

**The evidence is not in the repository either.**
`results/` is 29 files and 156 KB, entirely untracked because `.gitignore` holds `/results`, and three knowledge documents already cite paths that do not ship.

**The trunk contradicts the branch.**
`main` is 115 commits behind and has never seen this work.
`knowledge/verdict.md`, which the README calls "Start here", says the project is not worth building and predicts 3.3x; the branch measured 5.56x.

## The work, in order

### 1. Rescue the guests

Move the three guest crates into `guests/fib`, `guests/mem` and `guests/bits`, with sources *and* prebuilt ELFs.

The directory names are the workloads; the crate and ELF names stay `acceptance-guest`, `acceptance-mem` and `acceptance-bits`, because renaming would need a rebuild this machine cannot do.
`tools/scaling_study.sh` already takes the directory and the ELF suffix as separate arguments, so it needs one line changed per row and a default for its guest-root argument.

Do this first and independently of everything else: the scratchpad is temporary, and until this is done the project can lose the ability to reproduce its own headline claim to a routine cleanup.

*Verified by:* running the ladder against the in-repo guests and getting a row that matches `results/scaling-quiet.tsv`.

### 2. Commit the measurements

Drop `/results` from `.gitignore` and commit the 29 files.

The central claim of this project is a number, and the measurements that produce it should travel with it — particularly since three knowledge documents already link to them.

*Verified by:* `git ls-files results | wc -l` reporting 29, and the links in `next-levers.md`, `cpu-plan.md` and `spikes.md` resolving.

### 3. Merge to main, and rewrite the verdict as part of the merge

Merge the branch, then replace `knowledge/verdict.md`'s headline and update the top-level `README.md`.

The old verdict was not wrong.
It measured a *predicted* port against a *packed* CPU baseline and concluded 3.3x against a bar of 5x, which was the right answer to the question it asked.
What happened next is that the thing got built and beat the prediction, for reasons the old document could not have known — chiefly that the prover as it ships is scalar, and that four phases moved rather than the two the model priced.
Keep that reasoning as history, because the gap between the prediction and the outcome is the most instructive thing in the repository.
Replace only the headline and the recommendation.

The README update belongs to the merge rather than to the branch, which is the standing rule for an index that sits above the work.

*Verified by:* reading the front page as a stranger would and finding it agrees with `results/scaling-quiet.tsv`.

### 4. Write the integration guide

`USING.md` at the root, holding:

- the twenty-two line TOML block, pinned to the tag, with the reason a consumer must name the crates in their own manifest (cargo unifies features, so the names must appear locally to reach the SDK's copies);
- the note that the gates default to on and no environment variable is needed;
- the table of `SP1_METAL_*` switches, for turning parts off and for the `=check` parity mode;
- the compatibility statement: this tracks SP1 6.4.x exactly, and a new SP1 minor version means re-vendoring, which the per-crate `CHANGES.md` files are what make tractable;
- `crates/sp1-profile-host` named as the worked example, since it already is one.

*Verified by:* building a scratch consumer from the block as written, from the tag rather than a branch, and confirming the five forks resolve with `metal` on.

### 5. One command to see the speedup

`tools/quickstart.sh`: build `sp1-profile-host`, run one CPU/GPU pair against `guests/fib`, print both times and the ratio.

Honest about cost in its own output: the first build is about eight minutes, the pair about two on an M4 Max.
It should refuse to run, rather than produce a misleading number, if the machine is busy — the project has already recorded one corrupted measurement taken during a competing build.

*Verified by:* running it on a quiet machine and getting a ratio inside the 3.71x-5.63x envelope.

### 6. Release hygiene

- **Prerequisites**, stated plainly: Apple Silicon, Rust 1.88 or later, macOS with Metal 3.
  Whether runtime shader compilation needs the Xcode Command Line Tools is *not yet known* and must be checked rather than guessed; see the risks below.
- **An affiliation line.** This repository carries Succinct's project name and forks five of their crates while saying nothing about being independent. One sentence: an independent fork, not affiliated with or endorsed by Succinct Labs.
- **Workspace hygiene.** Add `crates/vendor/slop-*` to the root workspace's `exclude` list, with matching `.gitignore` entries, so `cargo test` works inside them; today it fails with "believes it's in a workspace when it's not".

No CI.
The only tests worth running are the GPU parity suites, and GitHub's macOS runners cannot be relied on for Metal compute; a green badge covering the CPU paths alone would claim more than it checks.
Say in the README that the suites are run by hand, and say which ones.

### 7. Tag and push

Tag `v0.1.0` on `main`.
Push, once, with explicit authorisation.

This repository has never been pushed: `origin/main` holds a single seed commit and 115 commits of work are local to one machine.
That is itself a risk, and it ends here.

## Out of scope

- crates.io, for the reason in the first section.
- Upstreaming, deliberately deferred.
- A `sp1-metal` facade crate whose one feature turns on all five: it would shorten the feature block but cannot touch the patch block, which is not worth a new crate to maintain.
- Any further backend work. The release ships what is measured, not what is next.

## Risks and open questions

**Does the runtime shader path need the Xcode Command Line Tools?**
Every measurement here was taken on a machine that has them, so their absence has never been tested.
If they are required, that is a prerequisite a cloner must be told about, and it belongs in item 6.
Check before the tag, on a machine without them if one can be found, or by reading what `new_library_with_source` links against if not.

**The guest ELFs cannot be regenerated here.**
Committing them makes the repository self-contained, but it also means a binary artefact whose provenance rests on this commit.
Commit the sources beside them so a reader with `sp1up` can rebuild and compare, and say in `guests/README.md` which SP1 version produced them.

**The SP1 version pin is exact.**
`=6.4.0` throughout. A consumer on any other 6.x will not resolve.
This is stated rather than solved; solving it means re-vendoring against each release.
