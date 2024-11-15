.#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import codecs
import collections
import os
import sys
import yaml
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument(
    "-o",
    "--output-test",
    action="append",
    metavar="FILE",
    required=True,
    help="Test output (C++ source) file(s)")
parser.add_argument( "-b", "--output-bench",
                    metavar="FILE", required=False,
                    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]
  mr, nr = map(int, param_spec.split("x"))
  arch, isa, assembly = xnncommon.parse_target_name(target_name)
  return mr, nr, arch, isa

SPMM_BENCH_CODE = """\
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  f32_spmm(state, ${SPMM}, ${MR}, ${NR},
    /*sparsity=*/0.8f, ${INIT_PARAMS},
  $if ISA_CHECK:
    benchmark::utils::${ISA_CHECK}
  $else:
    /*isa_check=*/nullptr
  );
}\n
BENCHMARK_SPMM(${UKERNEL_NAME})
"""

TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, datatype, params_type, init_params) \
XNN_TEST_SPMM_K_EQ(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_K_LT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_K_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_SPMM_K_DIV(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_N_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});
XNN_TEST_SPMM_N_DIV(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_M_LT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_M_DIV(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_M_GT(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_OUTPUT_STRIDE(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_QMIN(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_QMAX(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_HALF_SPARSE(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
XNN_TEST_SPMM_ZERO_WEIGHTS(ukernel,arch_flags, ${", ".join(TEST_ARGS)});  
"""


def main(args):
  options = parser.parse_args(args)
  num_output_files = len(options.output_test)

  tester = options.tester
  tester_header = {
  "SpMMMicrokernelTester": "spmm-microkernel-tester.h",
  }[tester]
  ukernel = options.ukernel

  tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"
#include "spmm-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    benches = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <benchmark/benchmark.h>
#include "spmm-benchmark.h"
#include "utils.h"
#include "xnnpack/gemm.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
""".format(specification=options.spec, generator=sys.argv[0])

    test_outputs = collections.defaultdict(lambda: tests)
    bench_outputs = benches
    sorted_spec_yaml = collections.defaultdict(list)
    isa_hierarchy = xnncommon.isa_hierarchy_map()

    benches = [""] * len(isa_hierarchy)
    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec["init"]
      k_block = int(ukernel_spec["k-block"])
      pipelined = bool(ukernel_spec.get("pipelined", False))
      mr, nr, arch, isa = split_ukernel_name(name)

      test_case, bench_case = generate_test_cases(name, init_fn, mr, nr, k_block,
                                      pipelined, isa)
      # Hash the name of each microkernel and figure out which output file to
      # write it to.
      output_index = zlib.crc32(bytes(name, "utf-8")) % num_output_files
      test_outputs[options.output_test[output_index]] += "\n\n" + xnncommon.postprocess_test_case(
          test_case, arch, isa)
      benches[isa_hierarchy.get(isa, 0)] +=  "\n\n" + xnncommon.postprocess_test_case(bench_case, arch, isa)

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_outputs += benches[arch_idx]

    bench_outputs += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""
    for output_name in options.output_test:
      xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])

    if options.output_bench:
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_outputs)


if __name__ == "__main__":
  main(sys.argv[1:])
