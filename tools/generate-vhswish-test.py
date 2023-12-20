#!/usr/bin/env python
# Copyright 2023 Google LLC
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
  description='Vector Hardswish operation microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(qs8|qu8)_vhswish_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  datatype = match.group(1)
  batch_tile = int(match.group(3))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return datatype, batch_tile, arch, isa


HSWISH_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VHSwishMicrokernelTester()
    .batch_size(${BATCH_TILE})
    $if DATATYPE == "QU8":
      .input_zero_point(150)
      .output_zero_point(100)
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = ${BATCH_TILE*2}; batch_size < ${BATCH_TILE*10}; batch_size += ${BATCH_TILE}) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        $if DATATYPE == "QU8":
          .input_zero_point(150)
          .output_zero_point(100)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size < ${BATCH_TILE}; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        $if DATATYPE == "QU8":
          .input_zero_point(150)
          .output_zero_point(100)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = ${BATCH_TILE+1}; batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      $if DATATYPE == "QU8":
        .input_zero_point(150)
        .output_zero_point(100)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, input_scale) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
    for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_scale(input_scale)
        $if DATATYPE == "QU8":
          .input_zero_point(150)
          .output_zero_point(100)
        .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, output_scale) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
    for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .output_scale(output_scale)
        $if DATATYPE == "QU8":
          .input_zero_point(150)
          .output_zero_point(100)
        .Test(${", ".join(TEST_ARGS)});
      }
  }
}

TEST(${TEST_NAME}, input_zero_point) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        $if DATATYPE == "QU8":
          .output_zero_point(100)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, output_zero_point) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        $if DATATYPE == "QU8":
          .input_zero_point(150)
        .output_zero_point(output_zero_point)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def generate_test_cases(ukernel, init_fn, datatype, batch_tile, isa):
  """Generates all tests cases for a Vector Hardswish micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    datatype: data type.
    batch_tile: Number of batch elements processed per one iteration of the
                inner loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)
  return xngen.preprocess(HSWISH_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "BATCH_TILE": batch_tile,
      "DATATYPE": datatype.upper(),
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <vector>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vhswish.h>

#include "vhswish-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      datatype, batch_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(
        name, init_fn, datatype, batch_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
