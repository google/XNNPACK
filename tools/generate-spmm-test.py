#!/usr/bin/env python
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
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["SpMMMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-k", "--ukernel", metavar="FILE", required=True,
                    help="Microkernel type")
parser.add_argument("-o", "--output", metavar="FILE", required=True, 
                    help="Test output (C++ source) file(s)")
parser.add_argument( "-b", "--output-bench",
                    metavar="FILE", required=False,
                    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())

SPMM_BENCH_CODE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params)
static void ukernel(benchmark::State& state, const char* net) {
  f32_spmm(state, ukernel, mr, nr,
    /*sparsity=*/0.8f, init_params);
}\n 
BENCHMARK_SPMM(ukernel)
"""

TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params) \
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
//   Microkernel: {specification}
//   Generator: {generator}

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"
#include "spmm-microkernel-tester.h"
""".format(specification=options.ukernel, generator=sys.argv[0])

  if options.output_bench:
    benches = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {specification}
//   Generator: {generator}

#include <benchmark/benchmark.h>
#include "spmm-benchmark.h"
#include "utils.h"
#include "xnnpack/gemm.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
""".format(specification=options.ukernel, generator=sys.argv[0])
  
  ukernel_parts = options.ukernel.split("-")
  datatype = ukernel_parts[0]
  op = ukernel_parts[1]
  test_args = ["mr"]
  test_args.append("nr")
  test_args.append("pipelined")
  test_args.append("kblock")
  test_args.append("init_params")
  tests += xnncommon.make_multiline_macro(xngen.preprocess(
    TEST_TEMPLATE,
    {
      "TEST_ARGS": test_args,
      "TESTER": tester,
      "DATATYPE": datatype,
    },
  ))

  if options.output_bench:
    benches += xnncommon.make_multiline_macro(xngen.preprocess(
      SPMM_BENCH_CODE,
      {
        "TEST_ARGS": test_args,
        "TESTER": tester,
        "DATATYPE": datatype,
      },
    ))

  folder = datatype + "-" + ("spmm" if datatype.startswith("f") else op)
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"
  if options.output_bench:
    benches += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
    benches += "#undef XNN_UKERNEL_WITH_PARAMS\n"
    benches += "\n\n" + """\
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

  # xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])
  xnncommon.overwrite_if_changed(options.output, tests)

  if options.output_bench:
    xnncommon.overwrite_if_changed(options.output_bench, benches)

if __name__ == "__main__":
  main(sys.argv[1:])
