#!/usr/bin/env python
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='GAvgPool microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument(
    "-b",
    "--output-bench",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f16_f32acc|f32|qs8|u8)_(rminmax|rmax|rmin|rsum|rdsum)(_minmax_(fp32))?_ukernel_(.*)_(u|c)(\d+)(v)?(_acc\d+)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  dtype = match.group(1)
  op = match.group(2)
  target_name = match.group(5)
  arch, isa, _ = xnncommon.parse_target_name(target_name=target_name)
  return dtype, arch, isa, op


BENCHMARK_TEMPLATE = """\
BENCHMARK_CAPTURE(${OP_NAME}, ${BENCHMARK_NAME},
                  ${KERNEL},
                  $if CHECK_ISA:
                    ${INIT_PARAMS},
                    ${CHECK_ISA})
                  $else:
                    ${INIT_PARAMS})
  ->Apply(Benchmark${OP})
  ->UseRealTime();
"""

def generate_benchmark_cases(
    ukernel: str,
    dtype: str,
    op: str,
    isa: str,
    init_fn: str | None = None,
):
  """Generates all tests cases for a RSUM Discontig micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    dtype: input datatype.
    op: reduction operator.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.
    init_fn: C name of the function to initialize microkernel parameters.

  Returns:
    Code for the test case.
  """
  datatype = ukernel.split("_", 2)[1]
  check_isa = xnncommon.generate_isa_utilcheck_macro(isa)
  if check_isa:
    check_isa = f"benchmark::utils::{check_isa}"
  return xngen.preprocess(
      BENCHMARK_TEMPLATE,
      {
          "OP": op.upper(),
          "OP_NAME": dtype + "_" + op,
          "BENCHMARK_NAME": ukernel.split("__", 1)[1],
          "KERNEL": ukernel,
          "INIT_PARAMS": init_fn or "/*init_params=*/nullptr",
          "CHECK_ISA": check_isa,
          "DATATYPE": datatype,
      },
  )

def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    benches = """\
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      dtype, arch, isa, op = split_ukernel_name(name)

      benchmark_case = generate_benchmark_cases(name, dtype, op, isa, init_fn)
      benches += "\n\n" + xnncommon.postprocess_test_case(benchmark_case, arch, isa)

    # Footer with `main` function.
    benches += "\n\n" + """\
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

    xnncommon.overwrite_if_changed(options.output_bench, benches)


if __name__ == "__main__":
  main(sys.argv[1:])
