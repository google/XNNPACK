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
  match = re.match(r"xnn_(qs8|qu8|f16_f32acc|f32)_(rdsum)(_(minmax))?(_(fp32|rndnu))?_ukernel_((\d+)p)?(\d+)x__(.+)_c(\d+)(_acc(\d+))?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  dtype = match.group(1)
  arch, isa, _ = xnncommon.parse_target_name(target_name=match.group(10))
  return dtype, arch, isa


BENCHMARK_TEMPLATE = """\
BENCHMARK_CAPTURE(${OP_NAME}, ${BENCHMARK_NAME},
                  ${KERNEL},
                  $if CHECK_ISA:
                    ${INIT_PARAMS},
                    ${CHECK_ISA})
                  $else:
                    ${INIT_PARAMS})
  ->Apply(BenchmarkBatch)
  ->UseRealTime();
"""

def generate_benchmark_cases(
    ukernel: str,
    dtype: str,
    isa: str,
    init_fn: str | None = None,
):
  """Generates all tests cases for a RSUM Discontig micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    dtype: input datatype.
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
          "OP_NAME": dtype + "_rsum_discontig",
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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      dtype, arch, isa = split_ukernel_name(name)

      benchmark_case = generate_benchmark_cases(name, dtype, isa, init_fn)
      # Only generate benchmarks for generic and multipass kernels. The unipass
      # kernels will be removed.
      regexp = re.compile(r'\dp\dx')
      if regexp.search(name):
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
