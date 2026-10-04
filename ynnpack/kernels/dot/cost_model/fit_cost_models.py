#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fits dot kernel cost model coefficients and generates per-CPU header files.

Scans benchmark JSON files under `cost_model/data/**/<cpu>/*.json`, fits the 3
parameters of each `dot_cost_model`:
  overhead +
  num_blocks * (
    load_a_cost * (block_m * k) +
    load_b_cost * (block_n * k) +
    output_cost * (block_m * block_n)
  )
and writes `<cpu>.h` for each CPU in `cost_model/`.
"""

import argparse
import csv
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import nnls


def ceil_div(a: int, b: int) -> int:
  return (a + b - 1) // b


def align_up(a: int, b: int) -> int:
  return ceil_div(a, b) * b


def load_benchmark_data(input_sources: List[str]) -> List[Dict[str, Any]]:
  """Loads benchmark entries from CSV files or stdin."""
  benchmarks = []

  def parse_csv_text(text: str):
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
      if line.startswith("name,iterations"):
        header_idx = i
        break
    if header_idx is None:
      return
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
      name = row.get("name", "")
      if not name:
        continue
      real_time = float(row["real_time"]) if row.get("real_time") else 0.0
      cpu_time = float(row["cpu_time"]) if row.get("cpu_time") else 0.0
      entry = {
          "name": name,
          "real_time": real_time,
          "cpu_time": cpu_time,
          "time_unit": row.get("time_unit", "ns"),
      }
      # The benchmark emits the problem shape and the kernel block/tile sizes as
      # CSV counter columns (see bench.cc), so we read them directly instead of
      # parsing them out of the benchmark name.
      for col in ("m", "n", "k", "block_m", "block_n", "block_k", "tile_m",
                  "tile_n", "tile_k"):
        entry[col] = _int_counter(row.get(col))
      if (row.get("error_occurred") or "").lower() in ("true", "1"):
        entry["error_occurred"] = True
      if row.get("error_message"):
        entry["error_message"] = row["error_message"]
      benchmarks.append(entry)

  if not input_sources or (len(input_sources) == 1 and input_sources[0] == "-"):
    parse_csv_text(sys.stdin.read())
  else:
    for src in input_sources:
      if src == "-":
        parse_csv_text(sys.stdin.read())
      elif "\n" in src or src.startswith("name,iterations"):
        parse_csv_text(src)
      elif os.path.exists(src):
        with open(src, "r") as fp:
          parse_csv_text(fp.read())
      else:
        print(f"Warning: input file '{src}' not found.", file=sys.stderr)

  return benchmarks


def _int_counter(value: Any) -> Optional[int]:
  """Parses a CSV counter cell (a float-formatted integer) into an int."""
  if value is None or value == "":
    return None
  try:
    return int(round(float(value)))
  except (TypeError, ValueError):
    return None


def kernel_dims(
    entry: Dict[str, Any],
) -> Optional[Tuple[int, int, int, int, int, int]]:
  """Reads (block_m, block_n, block_k, tile_m, tile_n, tile_k) from CSV columns."""
  dims = tuple(
      entry.get(col)
      for col in ("block_m", "block_n", "block_k", "tile_m", "tile_n", "tile_k")
  )
  if any(d is None for d in dims):
    return None
  return dims


def kernel_shape(entry: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
  """Reads (m, n, k) from CSV columns."""
  shape = tuple(entry.get(col) for col in ("m", "n", "k"))
  if any(s is None for s in shape):
    return None
  return shape


# TODO: b/549795065 - The following two functions are big and messy, and exist
# because the mapping between target names and dot kernels is sloppy. If we
# clean that up, we can just directly map part of the kernel name to a dot cost
# model.
def parse_dot_costs_header(
    header_path: str,
) -> List[Tuple[Tuple[str, ...], str]]:
  """Parses `struct dot_cost_models` in `dot_costs.h`.

  Args:
    header_path: Path to the cost_model.h header file to parse.

  Returns:
    Ordered list of (guard_stack, field_name) pairs.
  """
  with open(header_path, "r") as fp:
    lines = fp.readlines()

  in_struct = False
  guard_stack: List[str] = []
  fields: List[Tuple[Tuple[str, ...], str]] = []

  for line in lines:
    stripped = line.strip()
    if stripped.startswith("struct dot_cost_models"):
      in_struct = True
      continue
    if not in_struct:
      continue
    if stripped.startswith("};"):
      break

    if stripped.startswith("#ifdef ") or stripped.startswith("#ifndef "):
      guard_stack.append(stripped)
    elif stripped.startswith("#endif"):
      if guard_stack:
        guard_stack.pop()
    else:
      m = re.match(r"dot_cost_model\s+(\w+);", stripped)
      if m:
        fields.append((tuple(guard_stack), m.group(1)))

  return fields


_KERNEL_PAT = re.compile(
    r"dot_(.+?)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_([a-z0-9_]+)"
)
_SME_PAT = re.compile(r"dot_(.+?)_(sme2?)(?:/|$)")


def infer_kernel_model_name(
    benchmark_name: str, valid_models: Set[str]
) -> Optional[str]:
  """Maps a single benchmark kernel name to its `dot_cost_models` field name."""
  m_sme = _SME_PAT.search(benchmark_name)
  if m_sme:
    type_str, arch = m_sme.group(1), m_sme.group(2)
    parts = type_str.split("_")
    short_type = (
        parts[1] if len(parts) >= 2 and parts[0] == "int8" else parts[0]
    )
    cand = f"arm64_{arch}_{short_type}"
    return cand if cand in valid_models else None

  m = _KERNEL_PAT.search(benchmark_name)
  if not m:
    return None

  type_str = m.group(1)
  bk = int(m.group(4))
  tn = int(m.group(6))
  tk = int(m.group(7))
  arch = m.group(8)

  arch_variants = [arch]
  if arch.startswith("amx"):
    arch_variants.append("amx")
  if arch == "fma3":
    arch_variants.append("f16c_fma3")

  type_variants = [type_str]
  first_part = type_str.split("_")[0]
  if first_part not in type_variants:
    type_variants.append(first_part)

  for a_var in arch_variants:
    for t_var in type_variants:
      for k_suf in [f"_k{tk}", f"_k{bk}", f"_{tn}x{tk}", ""]:
        for prefix in ["x86_", "arm64_", "arm_", "wasm_", "hexagon_", ""]:
          cand = f"{prefix}{a_var}_{t_var}{k_suf}"
          if cand in valid_models:
            return cand
  return None


def solve_nnls(
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    fallback_x: np.ndarray,
    fallback_y: np.ndarray,
) -> np.ndarray:
  """Non-negative least squares with fallbacks for hard-to-fit data.

  scipy's `nnls` defaults to a low iteration cap (3 * num_columns) and raises if
  it is reached, which happens for some ill-conditioned data (e.g. the slow
  efficiency-core SME runs). We give it more room, and fall back to an ordinary
  least-squares solve (clamped to non-negative) if it still does not converge or
  returns all zeros.
  """
  try:
    c, _ = nnls(a_mat, b_vec, maxiter=100 * a_mat.shape[1])
  except RuntimeError:
    c = None
  if c is None or np.all(c == 0):
    c = np.linalg.lstsq(fallback_x, fallback_y, rcond=None)[0]
    c = np.maximum(c, 1e-12)
  return c


def fit_model(
    benchmarks: List[Dict[str, Any]],
    shared_load: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Fits a single cost model to all valid benchmarks.

  If `shared_load` is set, load_a_cost and load_b_cost are constrained to a
  single shared coefficient (see the AMX/SME note below).
  """
  entries = {}

  # Only exclude m = 1 if the model has kernels with m > 1, so models that are
  # legitimately m=1 only (like x86_avx512_int8_int8_int32_symmetric_b) can
  # still be fit.
  has_large_m = False
  for b in benchmarks:
    dims = kernel_dims(b)
    if dims and dims[0] > 1:
      has_large_m = True
      break

  for b in benchmarks:
    if b.get("skipped") or b.get("error_occurred") or b.get("error_message"):
      continue

    name = b.get("name", "")
    dims = kernel_dims(b)
    if not dims:
      continue
    block_m, block_n, _, tile_m, tile_n, _ = dims
    # We avoid m = 1 and kernels with few tiles, because these kernels tend to
    # distort the fits for the larger cases.
    if has_large_m and block_m == 1:
      continue
    if (block_m // tile_m) * (block_n // tile_n) < 4:
      continue

    shape = kernel_shape(b)
    if not shape:
      continue
    m, n, k = shape

    real_time_ns = b.get("real_time") or b.get("cpu_time")
    if not real_time_ns or real_time_ns <= 0:
      continue
    real_time_sec = float(real_time_ns) * 1e-9

    base_name = re.sub(r"/(\d+/)*(mn|nm|real_time)$", "", name)
    key = (base_name, (m, n, k), dims)
    if key not in entries:
      entries[key] = []
    entries[key].append(real_time_sec)

  x_list = []
  y_list = []

  for (_, (m, n, k), dims), times in entries.items():
    block_m, block_n, block_k = dims[:3]
    t_sec = min(times)

    blocks_m = ceil_div(m, block_m)
    blocks_n = ceil_div(n, block_n)
    k_aligned = align_up(k, block_k)

    num_blocks = blocks_m * blocks_n

    # The benchmark includes a fixed invocation overhead (x0 = 1.0) for the
    # harness and function call, while per-block overhead (x1 = num_blocks) and
    # per-element memory/compute operations scale with the grid.
    x0 = 1.0
    x1 = float(num_blocks)
    x2 = num_blocks * (block_m * k_aligned)
    x3 = num_blocks * (block_n * k_aligned)
    x4 = num_blocks * (block_m * block_n)

    x_list.append([x0, x1, x2, x3, x4])
    y_list.append(t_sec)

  if not x_list:
    raise ValueError("No valid benchmark data points found to fit.")

  x = np.array(x_list, dtype=np.float64)
  y = np.array(y_list, dtype=np.float64)

  if shared_load:
    # Hack: AMX and SME kernels load A and B symmetrically into tiles, so their
    # load costs should be equal. The benchmark data does not constrain the two
    # separately (e.g. AMX only has single-M iterations where 1x4 and 2x2 take
    # the same time; SME sweeps leave load_a and load_b collinear), so an
    # unconstrained regression splits the load cost arbitrarily between them
    # (typically driving one to zero). Constrain load_a_cost == load_b_cost by
    # fitting a single shared load coefficient.
    x_reduced = np.column_stack([x[:, 0], x[:, 1], x[:, 2] + x[:, 3], x[:, 4]])
    a_mat = x_reduced / y[:, np.newaxis]
    b_vec = np.ones(len(y))

    c_reduced = solve_nnls(a_mat, b_vec, x_reduced, y)

    c = np.array(
        [c_reduced[0], c_reduced[1], c_reduced[2], c_reduced[2], c_reduced[3]]
    )
  else:
    # Weighted NNLS minimizing sum( ((x*c - y) / y)^2 ) = sum( (x/y * c - 1)^2 )
    a_mat = x / y[:, np.newaxis]
    b_vec = np.ones(len(y))

    c = solve_nnls(a_mat, b_vec, x, y)

  # Scale so mean ratio is 1.000
  pred = x @ c
  ratios = pred / y
  mean_ratio = np.mean(ratios)
  if mean_ratio > 0:
    c = c / mean_ratio
    ratios = (x @ c) / y

  # Return the 4 dot_cost_model parameters (block_overhead, load_a, load_b,
  # output_c), discarding the one-time fixed invocation overhead c[0].
  return c[1:], ratios, y


def generate_cpu_header(
    cpu: str,
    header_fields: List[Tuple[Tuple[str, ...], str]],
    fitted_models: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> str:
  """Generates the C++ contents of `<cpu>.h`."""
  cpu_ident = re.sub(r"[^a-zA-Z0-9_]", "_", cpu)
  guard_macro = (
      "XNNPACK_YNNPACK_KERNELS_DOT_COST_MODEL_"
      f"{cpu_ident.upper()}_H_"
  )

  lines = f"""// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Generated by fit_cost_models.py.

// clang-format off

#ifndef {guard_macro}
#define {guard_macro}

#include "ynnpack/kernels/dot/cost_model/cost_model.h"

namespace ynn {{

static constexpr dot_cost_models {cpu_ident} = {{""".split("\n")

  active_guards: List[str] = []

  def close_guard(guard: str) -> str:
    macro = guard.split(None, 1)[1]
    return f"#endif  // {macro}"

  for guards, field_name in header_fields:
    if field_name not in fitted_models:
      continue
    target_guards = list(guards)
    common_len = 0
    while (
        common_len < len(active_guards)
        and common_len < len(target_guards)
        and active_guards[common_len] == target_guards[common_len]
    ):
      common_len += 1

    while len(active_guards) > common_len:
      lines.append(close_guard(active_guards.pop()))

    while len(active_guards) < len(target_guards):
      next_guard = target_guards[len(active_guards)]
      lines.append(next_guard)
      active_guards.append(next_guard)

    coeffs, ratios, y = fitted_models[field_name]
    lines.append(
        f"    // Ratio (predicted / actual): std={np.std(ratios):.3f},"
        f" min={np.min(ratios):.3f}, max={np.max(ratios):.3f} (n={len(y)})"
    )
    lines.append(f"""    .{field_name} = {{
        /*block_overhead=*/{coeffs[0]:.2e}f,
        /*load_a_cost=*/{coeffs[1]:.2e}f,
        /*load_b_cost=*/{coeffs[2]:.2e}f,
        /*output_c=*/{coeffs[3]:.2e}f,
    }},""")

  while active_guards:
    lines.append(close_guard(active_guards.pop()))

  lines.extend([
      "};",
      "",
      "}  // namespace ynn",
      "",
      f"#endif  // {guard_macro}",
      "",
  ])
  return "\n".join(lines)


def find_cpu_data_dirs(data_dir: str) -> Dict[str, List[str]]:
  """Finds per-CPU `.csv` files under `data_dir`."""
  cpu_files: Dict[str, List[str]] = {}
  if not os.path.isdir(data_dir):
    return cpu_files
  for entry in sorted(os.listdir(data_dir)):
    full_path = os.path.join(data_dir, entry)
    if os.path.isfile(full_path) and entry.endswith(".csv"):
      cpu = os.path.splitext(entry)[0]
      cpu_files.setdefault(cpu, []).append(full_path)
    elif os.path.isdir(full_path):
      data_files = sorted(
          os.path.join(full_path, f)
          for f in os.listdir(full_path)
          if f.endswith(".csv")
      )
      if data_files:
        cpu_files.setdefault(entry, []).extend(data_files)
  return dict(sorted(cpu_files.items()))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--data_dir",
      default=None,
      help=(
          "Root directory containing per-CPU benchmark CSV files (default:"
          " cost_model/data)."
      ),
  )
  parser.add_argument(
      "--output_dir",
      default=None,
      help=(
          "Output directory for generated <cpu>.h files (default: cost_model/)."
      ),
  )
  parser.add_argument(
      "--cpus",
      nargs="*",
      default=None,
      help="Specific CPUs to generate cost model headers for (default: all).",
  )
  args = parser.parse_args()

  script_dir = os.path.dirname(os.path.abspath(__file__))
  header_path = os.path.join(script_dir, "cost_model.h")
  if not os.path.exists(header_path):
    raise FileNotFoundError(f"Could not find cost_model.h at {header_path}")

  header_fields = parse_dot_costs_header(header_path)
  valid_models = {field for _, field in header_fields}

  data_dir = args.data_dir or os.path.join(script_dir, "data")
  output_dir = args.output_dir or script_dir

  cpu_data = find_cpu_data_dirs(data_dir)
  if args.cpus:
    requested = set(args.cpus)
    cpu_data = {c: files for c, files in cpu_data.items() if c in requested}

  if not cpu_data:
    print(f"No CPU benchmark CSV files found under {data_dir}", file=sys.stderr)
    sys.exit(1)

  os.makedirs(output_dir, exist_ok=True)

  for cpu, csv_files in cpu_data.items():
    benchmarks = load_benchmark_data(csv_files)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for b in benchmarks:
      if b.get("skipped") or b.get("error_occurred") or b.get("error_message"):
        continue
      if (b.get("real_time") or b.get("cpu_time") or 0) <= 0:
        continue
      name = b.get("name", "")
      model_name = infer_kernel_model_name(name, valid_models)
      if model_name:
        by_model.setdefault(model_name, []).append(b)

    # Fit int8 and uint8 AMX models to the combined data so they share the exact
    # same cost model parameters.
    amx_int8_combined = by_model.get("x86_amx_int8", []) + by_model.get(
        "x86_amx_uint8", []
    )
    if amx_int8_combined:
      if "x86_amx_int8" in valid_models:
        by_model["x86_amx_int8"] = amx_int8_combined
      if "x86_amx_uint8" in valid_models:
        by_model["x86_amx_uint8"] = amx_int8_combined

    fitted_models: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, field_name in header_fields:
      if field_name not in by_model:
        continue
      try:
        coeffs, ratios, y = fit_model(
            by_model[field_name],
            shared_load="amx" in field_name or "sme" in field_name,
        )
      except ValueError:
        continue
      fitted_models[field_name] = (coeffs, ratios, y)
      print(f"[{cpu}] {field_name}")

    out_path = os.path.join(output_dir, f"{cpu}.h")
    header_content = generate_cpu_header(cpu, header_fields, fitted_models)
    with open(out_path, "w") as fp:
      fp.write(header_content)
    print(f"[{cpu}] Wrote {out_path} ({len(fitted_models)} models)\n")


if __name__ == "__main__":
  main()
