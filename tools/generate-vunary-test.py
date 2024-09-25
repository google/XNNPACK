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
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=[
                    "VLReLUMicrokernelTester",
                    "VUnaryMicrokernelTester"],
                    help="Tester class to be used in the generated test")
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

SPECIAL_VALUES_F32 = {
    "SquareRoot": (
        4,  # Number of elements.
        "{0.0f, -0.0f, 1.0f, -1.0f}",  # Inputs.
        "{0.0f, -0.0f, 1.0f, NAN}",  # Expected outputs.
        "struct xnn_f32_sqrt_params",  # Params name.
        1,  # Error margin in ULP.
    ),
    "TanH": (
        7,  # Number of elements.
        "{0.0f, -0.0f, 10.0f, -10.0f, INFINITY, -INFINITY, NAN}",
        "{0.0f, -0.0f, 1.0f, -1.0f, 1.0f, -1.0f, NAN}",
        "union xnn_f32_tanh_params",
        # TODO: b/338934971 - This should be `1` ulp, but this fails on
        # `cmake-linux-riscv64-rvv` (but not on `cmake-linux-riscv64`).
        3,
    ),
    "Log": (
        4,  # Number of elements.
        "{1.0f, -1.0f, 0.0f, -0.0f}",  # Inputs.
        "{0.0f, NAN, -INFINITY, -INFINITY}",  # Expected outputs.
        "struct xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
    "GELU": (
        3,  # Number of elements.
        "{-6.0f, 6.0f, 0.0f}",  # Inputs.
        "{0.0f, 6.0f, 0.0f}",  # Expected outputs.
        "struct xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
    "Exp": (
        3,  # Number of elements.
        "{0.0f, -1e3f, 1e3f}",  # Inputs.
        "{1.0f, 0.0f, INFINITY}",  # Expected outputs.
        "struct xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
}

TEST_TEMPLATE = """\
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)

XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});

$if OP_TYPE != "SquareRootShift":
  XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
$if OP_TYPE == "Clamp":
  XNN_TEST_UNARY_QMIN(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
  XNN_TEST_UNARY_QMAX(ukernel, arch_flags, batch_tile, datatype, ${", ".join(TEST_ARGS)});
$if OP_TYPE == "ELU":
  TEST(ukernel, prescale) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_end = batch_tile * batch_scale;
    const size_t batch_step = std::max(1, batch_tile - 1);
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {
        ${TESTER}()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(ukernel, alpha) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_end = batch_tile * batch_scale;
    const size_t batch_step = std::max(1, batch_tile - 1);
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {
        ${TESTER}()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(ukernel, beta) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_end = batch_tile * batch_scale;
    const size_t batch_step = std::max(1, batch_tile - 1);
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {
        ${TESTER}()
          .batch_size(batch_size)
          .beta(beta)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
$if OP_TYPE == "LeakyReLU":
  $if "f" in DATATYPE:
    TEST(ukernel, slope) {
      TEST_REQUIRES_ARCH_FLAGS(arch_flags);
      const size_t batch_scale = get_batch_scale<datatype>();
      const size_t batch_end = batch_tile * batch_scale;
      const size_t batch_step = std::max(1, batch_tile - 1);
      for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
        for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {
          ${TESTER}()
            .batch_size(batch_size)
            .slope(slope)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  $else:
    TEST(ukernel, positive_scale) {
      TEST_REQUIRES_ARCH_FLAGS(arch_flags);
      for (size_t batch_size = 1; batch_size <= batch_tile * 5; batch_size += std::max(1, batch_tile - 1)) {
        for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
          ${TESTER}()
            .batch_size(batch_size)
            .positive_scale(positive_scale)
            $if DATATYPE == "QU8":
              .input_zero_point(150)
              .output_zero_point(100)
            .Test(${", ".join(TEST_ARGS)});
          }
      }
    }

    TEST(ukernel, negative_scale) {
      TEST_REQUIRES_ARCH_FLAGS(arch_flags);
      for (size_t batch_size = 1; batch_size <= batch_tile * 5; batch_size += std::max(1, batch_tile - 1)) {
        for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
          ${TESTER}()
            .batch_size(batch_size)
            .negative_scale(negative_scale)
            $if DATATYPE == "QU8":
              .input_zero_point(150)
              .output_zero_point(100)
            .Test(${", ".join(TEST_ARGS)});
          }
      }
    }
$if OP_TYPE == "SquareRootShift":
  TEST(ukernel, shift) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    const size_t batch_scale = get_batch_scale<datatype>();
    const size_t batch_end = batch_tile * batch_scale;
    const size_t batch_step = std::max(1, batch_tile - 1);
    for (uint32_t shift = 0; shift < 32; shift++) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {
        ${TESTER}()
          .batch_size(batch_size)
          .shift(shift)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
$if DATATYPE == "f32" and OP_TYPE in SPECIAL_VALUES_F32:
  TEST(ukernel, special_values) {
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);
    constexpr size_t num_elements = ${SPECIAL_VALUES_F32[OP_TYPE][0]};
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        ${SPECIAL_VALUES_F32[OP_TYPE][1]};
    std::array<float, num_elements> expected =
        ${SPECIAL_VALUES_F32[OP_TYPE][2]};
    std::array<float, buffered_size> outputs;
    ${SPECIAL_VALUES_F32[OP_TYPE][3]} params;
    if (${TEST_ARGS[1]}) {
      ${TEST_ARGS[1]}(&params);
    }
    ${TEST_ARGS[0]}(
        num_elements * sizeof(float), inputs.data(), outputs.data(), &params);
    for (int i = 0; i < num_elements; i++) {
      if (std::isfinite(expected[i])) {
        EXPECT_NEAR(
            expected[i], outputs[i],
            ${SPECIAL_VALUES_F32[OP_TYPE][4]} * std::abs(expected[i]) * std::numeric_limits<float>::epsilon())
            << "for input " << inputs[i];
      } else {
        EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))
            << "for input " << inputs[i] << " and output " << outputs[i]
            << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN
            << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL
            << ", FP_ZERO=" << FP_ZERO << ")";
      }
    }
  }
"""

def main(args):
  options = parser.parse_args(args)

  parts = options.ukernel.split("-")
  datatype = parts[-2]
  op = parts[-1]
  op_type = OP_TYPES[op]

  tester = options.tester
  tester_header = {
      "VLReLUMicrokernelTester": "vlrelu-microkernel-tester.h",
      "VUnaryMicrokernelTester": "vunary-microkernel-tester.h",
  }[tester]

  op_header = {
      "VLReLUMicrokernelTester": "vlrelu.h",
      "VUnaryMicrokernelTester": "vunary.h",
  }[tester]
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

  test_args = ["ukernel"]
  if op_type.startswith("Round"):
    test_args.append(tester + "::OpType::" + op_type)
  test_args.append("init_params")

  disambiguate = {
      "Abs": "Abs",
      "GELU": "Gelu",
      "Exp": "Exp",
      "Log": "Log",
      "Negate": "Neg",
      "Square": "Sqr",
  }.get(op_type, None)
  if disambiguate:
    test_args.append(disambiguate + "()")

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
