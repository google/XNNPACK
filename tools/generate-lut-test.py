#!/usr/bin/env python
# Copyright 2021 Google LLC
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
  description='LUT microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_x8_lut_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  batch_tile = int(match.group(2))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(1))
  return batch_tile, arch, isa


LUT_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  LUTMicrokernelTester()
    .batch_size(${BATCH_TILE})
    .Test(${UKERNEL_NAME});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = ${BATCH_TILE*2}; batch_size < ${BATCH_TILE*10}; batch_size += ${BATCH_TILE}) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size < ${BATCH_TILE}; batch_size++) {
      LUTMicrokernelTester()
        .batch_size(batch_size)
        .Test(${UKERNEL_NAME});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = ${BATCH_TILE+1}; batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch_size++) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
    LUTMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(${UKERNEL_NAME});
  }
}
"""


def generate_test_cases(ukernel, batch_tile, isa):
  """Generates all tests cases for a LUT micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    batch_tile: Number of batch elements processed per one iteration of the
                inner loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  return xngen.preprocess(LUT_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "BATCH_TILE": batch_tile,
      "UKERNEL_NAME": ukernel,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/lut.h>
#include "lut-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      batch_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, batch_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
