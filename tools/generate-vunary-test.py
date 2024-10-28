#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xngen
import xnncommon


parser = argparse.ArgumentParser(
    description="Vector unary operation microkernel test generator"
)
parser.add_argument(
    "-k",
    "--ukernel",
    required=True,
    help="Microkernel to generate",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())

OP_TYPES = {
    "vabs": "Abs",
    "vclamp": "Clamp",
    "velu": "ELU",
    "vexp": "Exp",
    "vgelu": "GELU",
    "vhswish": "HardSwish",
    "vlog": "Log",
    "vlrelu": "LeakyReLU",
    "vneg": "Negate",
    "vrelu": "ReLU",
    "vrndd": "RoundDown",
    "vrndne": "RoundToNearestEven",
    "vrndz": "RoundTowardsZero",
    "vrndu": "RoundUp",
    "vrsqrt": "ReciprocalSquareRoot",
    "vsigmoid": "Sigmoid",
    "vsqr": "Square",
    "vsqrt": "SquareRoot",
    "vtanh": "TanH",
}

PARAMS_TYPES = ["Clamp", "ELU", "LeakyReLU"]

SPECIAL_VALUES_F32 = {
    "SquareRoot": (
        4,  # Number of elements.
        "{0.0f, -0.0f, 1.0f, -1.0f}",  # Inputs.
        "{0.0f, -0.0f, 1.0f, NAN}",  # Expected outputs.
        1,  # Error margin in ULP.
    ),
    "TanH": (
        7,  # Number of elements.
        "{0.0f, -0.0f, 10.0f, -10.0f, INFINITY, -INFINITY, NAN}",
        "{0.0f, -0.0f, 1.0f, -1.0f, 1.0f, -1.0f, NAN}",
        # TODO: b/338934971 - This should be `1` ulp, but this fails on
        # `cmake-linux-riscv64-rvv` (but not on `cmake-linux-riscv64`).
        3,
    ),
    "Log": (
        4,  # Number of elements.
        "{1.0f, -1.0f, 0.0f, -0.0f}",  # Inputs.
        "{0.0f, NAN, -INFINITY, -INFINITY}",  # Expected outputs.
        1,  # Error margin in ULP.
    ),
    "GELU": (
        3,  # Number of elements.
        "{-6.0f, 6.0f, 0.0f}",  # Inputs.
        "{0.0f, 6.0f, 0.0f}",  # Expected outputs.
        1,  # Error margin in ULP.
    ),
    "Exp": (
        3,  # Number of elements.
        "{0.0f, -1e3f, 1e3f}",  # Inputs.
        "{1.0f, 0.0f, INFINITY}",  # Expected outputs.
        1,  # Error margin in ULP.
    ),
}

TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)
  TEST(ukernel, batch_eq) { TestBatchEq<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, batch_div) { TestBatchDiv<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, batch_lt) { TestBatchLT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, batch_gt) { TestBatchGT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, inplace) { TestInPlace<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
$if OP_TYPE == "Clamp":
  TEST(ukernel, clamp_min) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_size = batch_tile * batch_scale;
    for (int16_t min : {-128, -20, -1, 0, 1, 30, 127, 255}) {
      xnn_unary_params params;
      params.clamp.min = min;
      params.clamp.max = 255;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .Test<TestInfo>(ukernel, init_params, params);
    }
  }

  TEST(ukernel, clamp_max) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_size = batch_tile * batch_scale;
    for (int16_t max : {-127, -11, 0, 40, 127, 255}) {
      xnn_unary_params params;
      params.clamp.min = -128;
      params.clamp.max = max;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .Test<TestInfo>(ukernel, init_params, params);
    }
  }
$if OP_TYPE == "ELU":
  TEST(ukernel, alpha) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_size = batch_tile * batch_scale;
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      xnn_unary_params params;
      params.elu.alpha = alpha;
      ${TESTER}()
        .batch_size(batch_size)
        .Test<TestInfo>(ukernel, init_params, params);
    }
  }
$if OP_TYPE == "LeakyReLU":
  TEST(ukernel, negative_slope) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_size = batch_tile * batch_scale;
    for (float negative_slope : {0.01f, 0.3f, 1.3f}) {
      xnn_unary_params params;
      params.leaky_relu.negative_slope = negative_slope;
      ${TESTER}()
        .batch_size(batch_size)
        .Test<TestInfo>(ukernel, init_params, params);
    }
  }
$if "q" in DATATYPE:
  TEST(ukernel, input_scale) { TestInputScale<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, output_scale) { TestOutputScale<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, input_zero_point) { TestInputZeroPoint<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
  TEST(ukernel, output_zero_point) { TestOutputZeroPoint<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }
$if DATATYPE == "f32" and OP_TYPE in SPECIAL_VALUES_F32:
  TEST(ukernel, special_values) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    VUnaryMicrokernelTester().Test<TestInfo>(ukernel, init_params,
      /*inputs=*/${SPECIAL_VALUES_F32[OP_TYPE][1]},
      /*outputs=*/${SPECIAL_VALUES_F32[OP_TYPE][2]},
      /*tolerance_ulp=*/${SPECIAL_VALUES_F32[OP_TYPE][3]});
  }
"""

def main(args):
  options = parser.parse_args(args)

  parts = options.ukernel.split("-")
  datatype = parts[-2]
  op = parts[-1]
  op_type = OP_TYPES[op]

  tester = "VUnaryMicrokernelTester"
  tester_header = "vunary-microkernel-tester.h"
  op_header = "vunary.h"
  tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: {microkernel}
//   Generator: {generator}


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/{op_header}"
#include "next_prime.h"
#include "{tester_header}"

""".format(
      microkernel=options.ukernel,
      generator=sys.argv[0],
      op_header=op_header,
      tester_header=tester_header,
  )

  test_args = ["ukernel", "init_params"]

  tests += """\
using TestInfo = {op_type};

""".format(op_type=op_type)

  tests += xnncommon.make_multiline_macro(xngen.preprocess(
      TEST_TEMPLATE,
      {
          "TESTER": tester,
          "TEST_ARGS": test_args,
          "DATATYPE": datatype,
          "OP_TYPE": op_type,
          "OP_NAME": op,
          "SPECIAL_VALUES_F32": SPECIAL_VALUES_F32,
      },
  ))

  folder = options.ukernel
  if "rnd" in folder:
    folder = folder[0:8]

  tests += f'#include "{xnncommon.xnnpack_src()}/{folder}/{options.ukernel}.h"\n'
  tests += "#undef XNN_UKERNEL_WITH_PARAMS\n"

  xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
