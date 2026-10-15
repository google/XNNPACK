#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect matched BMM measurements on Linux or an adb-connected Android device."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def save(path, value):
  path.write_text(json.dumps(value, indent=2) + "\n")


def shapes(profile):
  result = []

  def attention(name, tokens, history, dim, broadcast=False):
    m, ba = (tokens, 8) if broadcast else (8 * tokens, 1)
    for op, transpose in [("qk", 1), ("pv", 0), ("pv_transposed", 1)]:
      k, n = (dim, history) if op == "qk" else (history, dim)
      result.append(
          dict(
              id=f"{name}_{op}",
              m=m,
              k=k,
              n=n,
              batch_a=ba,
              batch_b=1,
              transpose_b=transpose,
              positive_a=int(op != "qk"),
              tokens=tokens,
              history=history,
              head_dim=dim,
              query_heads=8,
              representation="broadcast" if broadcast else "folded",
          )
      )

  if profile == "smoke":
    attention("prefill", 128, 1024, 512)
    attention("decode8k", 1, 8192, 512)
  elif profile == "core":
    for name, t, s, d, broadcast in [
        ("decode_local", 1, 544, 256, False),
        ("decode_global", 1, 1088, 512, False),
        ("decode8k", 1, 8192, 512, False),
        ("chunk16_local", 16, 544, 256, False),
        ("chunk128_local", 128, 640, 256, False),
        ("chunk128_global", 128, 1024, 512, False),
        ("chunk128_global8k", 128, 8192, 512, False),
        ("broadcast_local", 128, 640, 256, True),
        ("broadcast_global", 128, 1024, 512, True),
        ("published_local", 1024, 1535, 256, False),
        ("published_global", 1024, 2048, 512, False),
    ]:
      attention(name, t, s, d, broadcast)
  elif profile == "boundary":
    for m in [1, 2, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128]:
      for k, n, trans in [
          (512, 1024, 1),
          (512, 8192, 1),
          (8192, 512, 0),
          (8192, 512, 1),
      ]:
        result.append(
            dict(
                id=f"m{m}_k{k}_n{n}_t{trans}",
                m=m,
                k=k,
                n=n,
                batch_a=1,
                batch_b=1,
                transpose_b=trans,
                positive_a=0,
            )
        )
  elif profile == "batch":
    for m in [1, 8, 128]:
      for ba, bb in [(1, 1), (8, 1), (8, 8), (1, 8)]:
        for trans in [0, 1]:
          result.append(
              dict(
                  id=f"m{m}_a{ba}_b{bb}_t{trans}",
                  m=m,
                  k=512,
                  n=1024,
                  batch_a=ba,
                  batch_b=bb,
                  transpose_b=trans,
                  positive_a=0,
              )
          )
  elif profile == "history":
    for t in [1, 16, 128]:
      for s in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        for d in [256, 512]:
          attention(f"t{t}_s{s}_d{d}", t, s, d)
  return result


def read_command(argv):
  try:
    p = subprocess.run(argv, text=True, capture_output=True, timeout=30)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
  except (OSError, subprocess.TimeoutExpired) as exc:
    return {"error": str(exc)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--binary", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--label", default="unnamed")
  parser.add_argument(
      "--serial", help="Android adb serial; omit for local Linux"
  )
  parser.add_argument(
      "--cpus", help="Explicit comma-separated CPU IDs, e.g. 0,2,4,6"
  )
  parser.add_argument("--threads", nargs="+", type=int, default=[1])
  parser.add_argument(
      "--profile",
      choices=["smoke", "core", "boundary", "batch", "history"],
      default="smoke",
  )
  parser.add_argument(
      "--cases", type=Path, help="JSON list of custom cases; replaces --profile"
  )
  parser.add_argument("--repetitions", type=int, default=3)
  parser.add_argument("--passes", type=int, default=4)
  parser.add_argument("--ms", type=float, default=60)
  parser.add_argument("--scales", nargs="+", type=float, default=[0.03125])
  parser.add_argument("--churn-mib", type=int, default=0)
  parser.add_argument("--max-pair-mib", type=float, default=512)
  parser.add_argument("--cooldown-seconds", type=float, default=3)
  parser.add_argument("--timeout-seconds", type=float, default=180)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--nm", default="nm")
  parser.add_argument("--readelf", default="readelf")
  parser.add_argument("--build-info", action="append", type=Path, default=[])
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Print the matrix without running or creating files",
  )
  args = parser.parse_args()
  cases = (
      json.loads(args.cases.read_text()) if args.cases else shapes(args.profile)
  )
  if not cases or len({c["id"] for c in cases}) != len(cases):
    parser.error("cases must be nonempty and have unique IDs")
  fields = ["m", "k", "n", "batch_a", "batch_b", "transpose_b", "positive_a"]
  for c in cases:
    for key in fields:
      if type(c.get(key)) is not int or c[key] < (
          0 if key in fields[-2:] else 1
      ):
        parser.error(f"invalid {key} in {c['id']}")
    if c["transpose_b"] not in (0, 1) or c["positive_a"] not in (0, 1):
      parser.error("transpose_b and positive_a must be 0 or 1")
    if c["batch_a"] != c["batch_b"] and min(c["batch_a"], c["batch_b"]) != 1:
      parser.error("batch sizes must match or broadcast")
  if (
      args.repetitions < 1
      or not 2 <= args.passes <= 100
      or args.passes % 2
      or any(not 1 <= t <= 256 for t in args.threads)
      or len(set(args.threads)) != len(args.threads)
      or not 0 <= args.seed <= 2**32 - 1
      or args.churn_mib < 0
      or not math.isfinite(args.cooldown_seconds)
      or args.cooldown_seconds < 0
      or not math.isfinite(args.timeout_seconds)
      or args.timeout_seconds <= 0
      or not math.isfinite(args.ms)
      or not 10 <= args.ms <= 10000
      or not math.isfinite(args.max_pair_mib)
      or args.max_pair_mib <= 0
      or any(not math.isfinite(s) or s <= 0 for s in args.scales)
  ):
    parser.error(
        "invalid repetition/pass/thread/seed/timing/scale/budget settings"
    )
  schedule = [
      (c, t, s) for c in cases for t in args.threads for s in args.scales
  ]
  if args.dry_run:
    print(
        json.dumps(
            {
                "cases": cases,
                "processes": len(schedule) * args.repetitions,
                "paired_intervals": (
                    len(schedule) * args.repetitions * args.passes
                ),
            },
            indent=2,
        )
    )
    return
  if not args.binary or not args.out or not args.cpus:
    parser.error("--binary, --out and --cpus are required")
  cpus = [int(s) for s in args.cpus.split(",")]
  if (
      len(set(cpus)) != len(cpus)
      or min(cpus) < 0
      or max(args.threads) > len(cpus)
  ):
    parser.error(
        "CPU IDs must be unique/nonnegative; threads must not exceed CPU count"
    )
  if not args.serial and not set(cpus).issubset(os.sched_getaffinity(0)):
    parser.error("requested CPU is outside this process's allowed affinity")
  binary = args.binary.resolve(strict=True)
  args.out.mkdir(parents=True, exist_ok=False)
  root = args.out.resolve()
  save(root / "cases.json", cases)
  for i, path in enumerate(args.build_info):
    (root / f"build-info-{i}-{path.name}").write_bytes(path.read_bytes())
  digest = hashlib.sha256(binary.read_bytes()).hexdigest()
  manifest = {
      "schema": 1,
      "collection_id": str(uuid.uuid4()),
      "complete": False,
      "label": args.label,
      "platform": "android" if args.serial else "linux",
      "cpus": cpus,
      "binary_name": binary.name,
      "binary_sha256": digest,
      "source": read_command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
      "settings": {
          k: v
          for k, v in vars(args).items()
          if k not in ["binary", "out", "cases", "build_info", "serial"]
      },
      "runs": [],
  }
  manifest["harness_sha256"] = {
      name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
      for name in ["probe.cc", "run.py", "summarize.py"]
  }
  diff = read_command(["git", "-C", str(REPO), "diff", "HEAD"])
  (root / "source.patch").write_text(diff.get("stdout", ""))
  save(root / "manifest.json", manifest)
  symbols = read_command([args.nm, "-n", "--defined-only", str(binary)])
  (root / "symbols.txt").write_text(symbols.get("stdout", ""))
  elf = read_command([args.readelf, "-lW", str(binary)])
  (root / "elf.txt").write_text(elf.get("stdout", ""))
  if (
      symbols.get("returncode")
      or elf.get("returncode")
      or "error" in symbols
      or "error" in elf
  ):
    print(
        "WARNING: symbol collection failed; preserve the unstripped binary for"
        " later resolution",
        flush=True,
    )

  def shell(command):
    return (
        read_command(["adb", "-s", args.serial, "shell", command])
        if args.serial
        else read_command(["sh", "-c", command])
    )

  metadata = {
      "uname": shell("uname -a"),
      "cpuinfo": shell("cat /proc/cpuinfo"),
      "topology": shell(
          "for c in /sys/devices/system/cpu/cpu[0-9]*; do echo $c; cat"
          " $c/topology/core_id $c/topology/physical_package_id"
          " $c/topology/thread_siblings_list $c/cpu_capacity 2>/dev/null; done"
      ),
      "cache": shell(
          "for c in /sys/devices/system/cpu/cpu[0-9]*/cache/index*; do echo $c;"
          " cat $c/level $c/type $c/size $c/shared_cpu_list 2>/dev/null; done"
      ),
      "policies": shell(
          "for p in /sys/devices/system/cpu/cpufreq/policy*; do echo $p; cat"
          " $p/related_cpus $p/cpuinfo_max_freq $p/scaling_governor"
          " 2>/dev/null; done"
      ),
  }
  if args.serial:
    metadata.update({
        "device": shell(
            "getprop ro.product.model; getprop ro.soc.model; getprop"
            " ro.build.fingerprint"
        ),
        "power": shell("dumpsys power"),
        "battery": shell("dumpsys battery"),
    })
    remote = "/data/local/tmp/xnnpack-bmm-" + digest[:16]
    subprocess.run(
        ["adb", "-s", args.serial, "push", str(binary), remote], check=True
    )
    subprocess.run(
        ["adb", "-s", args.serial, "shell", "chmod", "755", remote], check=True
    )
    remote_hash = shell("sha256sum " + shlex.quote(remote))
    if (
        remote_hash.get("returncode") != 0
        or remote_hash["stdout"].split()[0] != digest
    ):
      raise RuntimeError("uploaded binary hash mismatch")
    mask = format(sum(1 << c for c in cpus), "x")
    base = ["taskset", mask, remote]
  else:
    metadata["lscpu"] = read_command(["lscpu", "-J"])
    base = ["taskset", "-c", args.cpus, str(binary)]
  save(root / "device.json", metadata)

  def telemetry():
    record = {
        "time": time.time(),
        "frequency": shell(
            "cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq"
            " 2>/dev/null"
        ),
    }
    record["thermal"] = shell(
        "dumpsys thermalservice"
        if args.serial
        else (
            "for t in /sys/class/thermal/thermal_zone*; do echo $t; cat $t/type"
            " $t/temp 2>/dev/null; done"
        )
    )
    return record

  failures = 0
  for rep in range(args.repetitions):
    order = list(schedule)
    random.Random(args.seed + rep).shuffle(order)
    for case, threads, scale in order:
      run_id = f"run-{len(manifest['runs']):05d}"
      folder = root / run_id
      folder.mkdir()
      argv = base + [f"--{k.replace('_', '-')}={case[k]}" for k in fields]
      argv += [
          f"--threads={threads}",
          f"--scale={scale}",
          f"--seed={args.seed}",
          f"--passes={args.passes}",
          f"--ms={args.ms}",
          f"--off-first={rep % 2}",
          f"--churn-mib={args.churn_mib}",
          f"--max-pair-mib={args.max_pair_mib}",
      ]
      command = (
          ["adb", "-s", args.serial, "shell", shlex.join(argv)]
          if args.serial
          else argv
      )
      record = {
          "directory": run_id,
          "case": case,
          "repetition": rep,
          "threads": threads,
          "scale": scale,
          "command": command,
      }
      time.sleep(args.cooldown_seconds)
      samples = [telemetry()]
      print(
          run_id, case["id"], "threads", threads, "repetition", rep, flush=True
      )
      start = time.monotonic()
      timed_out = False
      with (folder / "rows.csv").open("w") as output, (
          folder / "stderr.txt"
      ).open("w") as err:
        process = subprocess.Popen(command, stdout=output, stderr=err)
        while process.poll() is None:
          try:
            process.wait(timeout=5)
          except subprocess.TimeoutExpired:
            samples.append(telemetry())
          if time.monotonic() - start > args.timeout_seconds:
            process.terminate()
            process.wait(timeout=10)
            timed_out = True
            break
      samples.append(telemetry())
      record.update(
          returncode=process.returncode,
          timed_out=timed_out,
          wall_seconds=time.monotonic() - start,
      )
      record["status"] = (
          "ok"
          if process.returncode == 0 and not timed_out
          else "skipped"
          if process.returncode == 77 and not timed_out
          else "failed"
      )
      if record["status"] == "ok":
        rows = list(
            csv.DictReader(io.StringIO((folder / "rows.csv").read_text()))
        )
        try:
          from summarize import pair_run

          pair_run(rows, args.passes)
          for row in rows:
            for key in ["mr_config", "mr_selected", "nr"]:
              if int(row[key]) < 1:
                raise ValueError("invalid tile size")
        except (ValueError, KeyError) as exc:
          record.update(status="failed", error="invalid CSV: " + str(exc))
      if record["status"] == "failed":
        failures += 1
      save(folder / "telemetry.json", samples)
      save(folder / "run.json", record)
      manifest["runs"].append(record)
      save(root / "manifest.json", manifest)
      if timed_out and args.serial:
        # Closing adb does not reliably kill a remote child. Stop the
        # sweep so a leftover process cannot contaminate later runs.
        raise RuntimeError(
            "remote timeout; stop the probe on the device before continuing:"
            f" {remote}"
        )
  manifest["complete"] = True
  save(root / "manifest.json", manifest)
  print("Complete:", root, "failures:", failures)
  raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
  main()
