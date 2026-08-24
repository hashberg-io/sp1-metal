#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Finish a benchmark run: clear any dead provers, render the reports, commit.
# Safe to run while the suite is still going -- the JSON Lines file is valid at
# every moment, so the reports simply cover what has been measured so far.
set -u
cd "$(dirname "$0")/.."

echo "== dead provers (0% CPU, older than 30 min) =="
killed=0
for pid in $(ps -Ao pid,etime,%cpu,command | grep '[s]p1-profile-host --elf' \
    | awk '$2 ~ /:/ && $3+0 < 1.0 {n=split($2,a,":"); m=(n==3 ? a[1]*60+a[2] : a[1]); if (m>=30) print $1}'); do
  echo "   killing $pid"; kill -9 "$pid" 2>/dev/null; killed=$((killed+1))
done
[ "$killed" = 0 ] && echo "   none"

echo "== suite =="
if pgrep -f bench_suite.py > /dev/null; then
  echo "   still running: $(wc -l < results/bench-full.jsonl 2>/dev/null) rows so far"
else
  echo "   finished"
fi

for tier in standard full; do
  f="results/bench-$tier.jsonl"
  [ -f "$f" ] || continue
  echo "== $tier =="
  python3 tools/bench_report.py "$f" | tail -14
  python3 tools/bench_report.py "$f" --md > "results/bench-$tier.md"
done

echo "== committing =="
git add results/bench-*.jsonl results/bench-*.md 2>/dev/null
if git diff --cached --quiet; then
  echo "   nothing new"
else
  git commit -q -m "Record benchmark results

Rendered by tools/bench_finish.sh from the JSON Lines the suite wrote.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
  echo "   committed $(git log --oneline -1 | cut -c1-60)"
fi
