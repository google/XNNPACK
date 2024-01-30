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
  description='Vector unary operation microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"(?:xnn_|xnn_generate_)(s8|u8|f16|f32|u32|u64)(_(s8|u8|f16|f32|u32|u64))*_v(abs|clamp|elu|hswish|lrelu|neg|relu|rndd|rndne|rndu|rndz|rsqrt|sigmoid|sqr|sqrt|sqrtshift|tanh)_(fact_)?ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  op_type = {
    "abs": "Abs",
    "clamp": "Clamp",
    "elu": "ELU",
    "hswish": "HardSwish",
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


TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VUnaryMicrokernelTester()
    .batch_size(${BATCH_TILE}${BATCH_SCALE})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = ${BATCH_TILE*2}${BATCH_SCALE}; batch_size < ${BATCH_TILE*10}${BATCH_SCALE}; batch_size += ${BATCH_TILE}${BATCH_SCALE}) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size < ${BATCH_TILE}${BATCH_SCALE}; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = ${BATCH_TILE}${BATCH_SCALE} + 1; batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}${BATCH_SCALE}; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if OP_TYPE != "SquareRootShift":
  TEST(${TEST_NAME}, inplace) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if OP_TYPE == "Clamp":
  TEST(${TEST_NAME}, qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1 if BATCH_SCALE == "" else (BATCH_TILE * 10 - 1))}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1 if BATCH_SCALE == "" else (BATCH_TILE * 10 - 1))}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "ELU":
  TEST(${TEST_NAME}, prescale) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, alpha) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, beta) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "LeakyReLU":
  TEST(${TEST_NAME}, slope) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (float slope : std::vector<float>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if OP_TYPE == "SquareRootShift":
  TEST(${TEST_NAME}, shift) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t shift = 0; shift < 32; shift++) {
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}${BATCH_SCALE}; batch_size += ${max(1, BATCH_TILE-1)}) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .shift(shift)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
"""


def generate_test_cases(ukernel, op_type, init_fn, batch_tile, vector_tile, isa):
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
      "rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
      "rvvfp16arith": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
    }[isa]
  return xngen.preprocess(TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "BATCH_TILE": batch_tile,
      "BATCH_SCALE": batch_scale,
      "BATCH_SUFFIX": "v" if vector_tile else "",
      "OP_TYPE": op_type,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


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


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      op_type, batch_tile, vector_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(
        name, op_type, init_fn, batch_tile, vector_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
