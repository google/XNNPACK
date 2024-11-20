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
//   Specification: {specification}
//   Generator: {generator}

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"
#include "spmm-microkernel-tester.h"
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
  folder = datatype + "-" + ("spmm" if datatype.startswith("f") else op)
  tests += f'#include "{xnncommon.xnnpack_src()}{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  # xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])
  xnncommon.overwrite_if_changed(options.output, tests)

if __name__ == "__main__":
  main(sys.argv[1:])
