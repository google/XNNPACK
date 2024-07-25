#!/usr/bin/env python
# Copyright 2019 Google LLC
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
import xngen
import xnncommon


parser = argparse.ArgumentParser(
    description="Vector unary operation microkernel test generator"
)
parser.add_argument(
    "-s",
    "--spec",
    metavar="FILE",
    required=True,
    help="Specification (YAML) file",
)
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(
      r"(?:xnn_|xnn_generate_)(s8|u8|bf16|f16|f32|u32|u64)(_(s8|u8|bf16|f16|f32|u32|u64))*_v(abs|clamp|elu|exp|gelu|hswish|log|lrelu|neg|relu|rndd|rndne|rndu|rndz|rsqrt|sigmoid|sqr|sqrt|sqrtshift|tanh)_(fact_)?ukernel__(.+)_u(\d+)(v)?",
      name,
  )
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  op_type = {
      "abs": "Abs",
      "clamp": "Clamp",
      "elu": "ELU",
      "exp": "Exp",
      "gelu": "GELU",
      "hswish": "HardSwish",
      "log": "Log",
      "lrelu": "LeakyReLU",
      "neg": "Negate",
      "relu": "ReLU",
      "rndd": "RoundDown",
      "rndne": "RoundToNearestEven",
      "rndz": "RoundTowardsZero",
      "rndu": "RoundUp",
      "rsqrt": "ReciprocalSquareRoot",
      "sigmoid": "Sigmoid",
      "sqr": "Square",
      "sqrt": "SquareRoot",
      "sqrtshift": "SquareRootShift",
      "tanh": "TanH",
  }[match.group(4)]
  batch_tile = int(match.group(7))
  vector_tile = bool(match.group(8))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(6))
  return op_type, batch_tile, vector_tile, arch, isa


SPECIAL_VALUES_F32 = {
    "SquareRoot": (
        4,  # Number of elements.
        "{0.0f, -0.0f, 1.0f, -1.0f}",  # Inputs.
        "{0.0f, -0.0f, 1.0f, NAN}",  # Expected outputs.
        "xnn_f32_sqrt_params",  # Params name.
        1,  # Error margin in ULP.
    ),
    "TanH": (
        7,  # Number of elements.
        "{0.0f, -0.0f, 10.0f, -10.0f, INFINITY, -INFINITY, NAN}",
        "{0.0f, -0.0f, 1.0f, -1.0f, 1.0f, -1.0f, NAN}",
        "xnn_f32_tanh_params",
        # TODO: b/338934971 - This should be `1` ulp, but this fails on
        # `cmake-linux-riscv64-rvv` (but not on `cmake-linux-riscv64`).
        3,
    ),
    "Log": (
        4,  # Number of elements.
        "{1.0f, -1.0f, 0.0f, -0.0f}",  # Inputs.
        "{0.0f, NAN, -INFINITY, -INFINITY}",  # Expected outputs.
        "xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
    "GELU": (
        3,  # Number of elements.
        "{-6.0f, 6.0f, 0.0f}",  # Inputs.
        "{0.0f, 6.0f, 0.0f}",  # Expected outputs.
        "xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
    "Exp": (
        3,  # Number of elements.
        "{0.0f, -1e3f, 1e3f}",  # Inputs.
        "{1.0f, 0.0f, INFINITY}",  # Expected outputs.
        "xnn_f32_default_params",
        1,  # Error margin in ULP.
    ),
}

TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VUnaryMicrokernelTester()
    $if BATCH_SCALE:
      .batch_size(${BATCH_TILE} * ${BATCH_SCALE})
    $else:
      .batch_size(${BATCH_TILE})
    .${TEST_FN}(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1 or BATCH_SCALE != "":
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .${TEST_FN}(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .${TEST_FN}(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if BATCH_SCALE:
    const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
  $else:
    const size_t batch_step = ${BATCH_TILE};
  for (size_t batch_size = batch_step + 1; batch_size < ${10 if BATCH_TILE == 1 else "2 * batch_step"}; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .${TEST_FN}(${", ".join(TEST_ARGS)});
  }
}

$if OP_TYPE != "SquareRootShift":
  TEST(${TEST_NAME}, inplace) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .${TEST_FN}(${", ".join(TEST_ARGS)});
    }
  }

$if OP_TYPE == "Clamp":
  TEST(${TEST_NAME}, qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1) if BATCH_SCALE == "" else "batch_step - 1"}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1) if BATCH_SCALE == "" else "batch_step - 1"}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "ELU":
  TEST(${TEST_NAME}, prescale) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, alpha) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, beta) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "LeakyReLU":
  TEST(${TEST_NAME}, slope) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "SquareRootShift":
  TEST(${TEST_NAME}, shift) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE:
      const size_t batch_step = ${BATCH_TILE} * ${BATCH_SCALE};
    $else:
      const size_t batch_step = ${BATCH_TILE};
    for (uint32_t shift = 0; shift < 32; shift++) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .shift(shift)
          .${TEST_FN}(${", ".join(TEST_ARGS)});
      }
    }
  }
$if DATATYPE == "f32" and OP_TYPE in SPECIAL_VALUES_F32:
  TEST(${TEST_NAME}, special_values) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    constexpr size_t num_elements = ${SPECIAL_VALUES_F32[OP_TYPE][0]};
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        ${SPECIAL_VALUES_F32[OP_TYPE][1]};
    std::array<float, num_elements> expected =
        ${SPECIAL_VALUES_F32[OP_TYPE][2]};
    std::array<float, buffered_size> outputs;
    $if len(TEST_ARGS) == 1:
      ${TEST_ARGS[0]}(
          num_elements * sizeof(float), inputs.data(), outputs.data(), nullptr);
    $else:
      union ${SPECIAL_VALUES_F32[OP_TYPE][3]} params;
      ${TEST_ARGS[1]}(&params);
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


def generate_test_cases(
    ukernel, op_type, init_fn, batch_tile, vector_tile, isa
):
  """Generates all tests cases for a Vector Unary Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    op_type: Operation type.
    init_fn: C name of the function to initialize microkernel parameters.
    batch_tile: Number of batch elements processed per one iteration of the
      inner loop of the micro-kernel.
    vector_tile: Indicates if batch tile is specified in vectors rather than
      elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  test_args = [ukernel]
  if op_type.startswith("Round"):
    test_args.append("VUnaryMicrokernelTester::OpType::" + op_type)
  if init_fn is not None:
    test_args.append(init_fn)
  batch_scale = ""
  if vector_tile:
    ctype = {"f16": "uint16_t", "f32": "float"}[datatype]
    batch_scale = {
        "rvv": "xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
        "rvvfp16arith": (
            "xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype
        ),
    }[isa]
  return xngen.preprocess(
      TEST_TEMPLATE,
      {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": test_args,
          "DATATYPE": datatype,
          "BATCH_TILE": batch_tile,
          "BATCH_SCALE": batch_scale,
          "BATCH_SUFFIX": "v" if vector_tile else "",
          "OP_TYPE": op_type,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "SPECIAL_VALUES_F32": SPECIAL_VALUES_F32,
          "TEST_FN": {
              "Abs": "TestAbs",
              "GELU": "TestGelu",
              "Exp": "TestExp",
              "Log": "TestLog",
              "Negate": "TestNeg",
              "Square": "TestSqr",
          }.get(op_type, "Test"),
      },
  )


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
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
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      op_type, batch_tile, vector_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(
          name, op_type, init_fn, batch_tile, vector_tile, isa
      )
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
