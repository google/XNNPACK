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
  description='LUT Norm microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_u8_lut32norm_ukernel__(.+)", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(1))
  return arch, isa


LUT_NORM_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, n_eq_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  LUTNormMicrokernelTester()
    .n(1)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, small_n) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester()
      .n(n)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, large_n) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = 16; n <= 128; n+=2) {
    LUTNormMicrokernelTester()
      .n(n)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, n_eq_1_inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  LUTNormMicrokernelTester()
    .n(1)
    .inplace(true)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, small_n_inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = 2; n <= 16; n++) {
    LUTNormMicrokernelTester()
      .n(n)
      .inplace(true)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, large_n_inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t n = 16; n <= 128; n+=2) {
    LUTNormMicrokernelTester()
      .n(n)
      .inplace(true)
      .Test(${UKERNEL_NAME});
  }
}
"""


def generate_test_cases(ukernel, isa):
  """Generates all tests cases for a LUT Norm micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  return xngen.preprocess(LUT_NORM_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
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
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/lut.h>
#include "lut-norm-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      # We only have 1 scalar microkernel, so batch tile is always 0, and we
      # don't need ISA checks, so not including common.h and isa-checks.h.
      arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
