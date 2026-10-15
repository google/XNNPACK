#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize paired intervals without treating them as independent processes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def pair_run(rows, passes):
  """Return medians of within-pass on/off ratios; reject incomplete pairings."""
  if len(rows) != 2 * passes:
    raise ValueError("incomplete paired run")
  pairs = defaultdict(dict)
  for row in rows:
    p, rewrite = int(row["pass"]), int(row["rewrite"])
    if not 0 <= p < passes or rewrite not in (0, 1) or rewrite in pairs[p]:
      raise ValueError("duplicate/invalid pair")
    for key in ["run_us", "invoke_us"]:
      if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
        raise ValueError("nonpositive/nonfinite timing")
    pairs[p][rewrite] = row
  if len(pairs) != passes or any(set(p) != {0, 1} for p in pairs.values()):
    raise ValueError("unmatched pair")
  result = {}
  for metric in ["run_us", "invoke_us"]:
    result[metric + "_ratio"] = statistics.median(
        float(pair[1][metric]) / float(pair[0][metric])
        for pair in pairs.values()
    )
    for rewrite, label in [(1, "on"), (0, "off")]:
      result[metric + "_" + label] = statistics.median(
          float(p[rewrite][metric]) for p in pairs.values()
      )
  return result


def classify(ratios, margin):
  # A conservative screening rule, not a statistical confidence interval.
  if len(ratios) < 3:
    return "insufficient_repetitions"
  if min(ratios) > 1 + margin:
    return "prefer_off"
  if max(ratios) < 1 / (1 + margin):
    return "prefer_on"
  return "uncertain"


def symbols(root):
  loads = [
      int(line.split()[2], 16)
      for line in (root / "elf.txt").read_text().splitlines()
      if line.strip().startswith("LOAD ")
  ]
  if not loads:
    return {}
  base = min(loads)
  result = {}
  for line in (root / "symbols.txt").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 3:
      try:
        result[int(parts[0], 16) - base] = parts[-1]
      except ValueError:
        pass
  return result


def resolve(text, names):
  return ";".join(
      slot + ":" + names.get(int(offset), "UNRESOLVED_" + offset)
      for slot, offset in (pair.split(":") for pair in text.split(";") if pair)
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("roots", nargs="+", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--margin", type=float, default=0.05)
  args = parser.parse_args()
  if not 0 <= args.margin < 1:
    parser.error("margin must be in [0, 1)")
  if len({p.resolve() for p in args.roots}) != len(args.roots):
    parser.error("do not count the same collection twice")
  groups = defaultdict(list)
  details = []
  coverage = []
  collections = set()
  for root in args.roots:
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    collection_id = manifest.get(
        "collection_id", hashlib.sha256(manifest_bytes).hexdigest()
    )
    if collection_id in collections:
      parser.error(
          "duplicate collection; copied results are not independent repetitions"
      )
    collections.add(collection_id)
    names = symbols(root)
    counts = Counter(r["status"] for r in manifest["runs"])
    coverage.append(
        dict(
            collection=root.name,
            complete=manifest["complete"],
            counts=dict(counts),
        )
    )
    for run in manifest["runs"]:
      if run["status"] != "ok":
        continue
      rows = list(csv.DictReader((root / run["directory"] / "rows.csv").open()))
      paired = pair_run(rows, manifest["settings"]["passes"])
      on = next(r for r in rows if r["rewrite"] == "1")
      off = next(r for r in rows if r["rewrite"] == "0")
      fields = [
          "m",
          "k",
          "n",
          "batch_a",
          "batch_b",
          "transpose_b",
          "positive_a",
          "threads",
          "scale",
          "churn_bytes",
      ]
      identity = {k: on[k] for k in fields}
      if any(any(r[k] != identity[k] for k in fields) for r in rows):
        raise ValueError("a run mixes input configurations")
      base = dict(
          label=manifest["label"],
          binary_sha256=manifest["binary_sha256"],
          cpus=",".join(map(str, manifest["cpus"])),
          case=run["case"]["id"],
          **identity,
          kernel_on=resolve(on["kernel_offsets"], names),
          kernel_off=resolve(off["kernel_offsets"], names),
          bmm_type_on=on["bmm_type"],
          bmm_type_off=off["bmm_type"],
          mr_on=on["mr_config"],
          mr_off=off["mr_config"],
          nr_on=on["nr"],
          nr_off=off["nr"],
          workspace_on=int(on["workspace_bytes"]),
          workspace_off=int(off["workspace_bytes"]),
          scale_bytes_on=int(on["scale_bytes"]),
          scale_bytes_off=int(off["scale_bytes"]),
          external_bytes=int(on["external_bytes"]),
          rhs_bytes=int(on["rhs_bytes"])
      )
      detail = dict(
          base,
          collection=root.name,
          directory=run["directory"],
          repetition=run["repetition"],
          **paired
      )
      details.append(detail)
      groups[tuple(base.items())].append(detail)
  output = []
  for identity, runs in groups.items():
    row = dict(identity)
    ratios = [r["run_us_ratio"] for r in runs]
    ratio = statistics.median(ratios)
    row.update(
        processes=len(runs),
        ratio_median=ratio,
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        on_ms=statistics.median(r["run_us_on"] for r in runs) / 1000,
        off_ms=statistics.median(r["run_us_off"] for r in runs) / 1000,
        invoke_ratio_median=statistics.median(
            r["invoke_us_ratio"] for r in runs
        ),
        off_latency_reduction_percent=100 * (1 - 1 / ratio),
        off_throughput_gain_percent=100 * (ratio - 1),
        preference=classify(ratios, args.margin),
    )
    batch_out = max(int(row["batch_a"]), int(row["batch_b"]))
    flops = 2 * batch_out * int(row["m"]) * int(row["k"]) * int(row["n"])
    row.update(
        work_flops=flops,
        rhs_fp32_bytes=4 * row["rhs_bytes"],
        row_tiles_per_rhs_estimate=(batch_out // int(row["batch_b"]))
        * math.ceil(int(row["m"]) / int(row["mr_on"])),
        effective_gflops_on=flops / (row["on_ms"] * 1e6),
        effective_gflops_off=flops / (row["off_ms"] * 1e6),
    )
    output.append(row)
  args.out.mkdir(parents=True, exist_ok=False)
  for name, rows in [("summary.csv", output), ("processes.csv", details)]:
    with (args.out / name).open("w") as f:
      if rows:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
  (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
  print("Configurations:", len(output), "valid processes:", len(details))
  print("Preferences:", dict(Counter(row["preference"] for row in output)))
  for row in coverage:
    print("Coverage:", row)
  if any(not r["complete"] or r["counts"].get("failed", 0) for r in coverage):
    print(
        "WARNING: incomplete or failed collections; inspect raw records before"
        " fitting a heuristic"
    )


if __name__ == "__main__":
  main()
