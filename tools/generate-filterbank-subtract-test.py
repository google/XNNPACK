#!/usr/bin/env python
# Copyright 2022 Google LLC
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
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='Filterbank Subtract microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_u32_filterbank_subtract_ukernel__(.+)_x(\d+)", name)
  assert match is not None
  batch_tile = int(match.group(2))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(1))
  return batch_tile, arch, isa


FILTERBANK_SUBTRACT_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  FilterbankSubtractMicrokernelTester()
    .batch(${BATCH_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch = ${BATCH_TILE*2}; batch < ${BATCH_TILE*10}; batch += ${BATCH_TILE}) {
      FilterbankSubtractMicrokernelTester()
        .batch(batch)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch = 2; batch < ${BATCH_TILE}; batch += 2) {
      FilterbankSubtractMicrokernelTester()
        .batch(batch)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch = ${BATCH_TILE+2}; batch < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch = ${BATCH_TILE+2}; batch < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(${", ".join(TEST_ARGS)});
  }
}
"""


def generate_test_cases(ukernel, batch_tile, isa):
  """Generates all tests cases for a Filterbank Subtract micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    batch_tile: Number of batch processed per one iteration of the inner
                  loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(FILTERBANK_SUBTRACT_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "DATATYPE": datatype,
      "BATCH_TILE": batch_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2022 Google LLC
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

#include <xnnpack/filterbank.h>
#include "filterbank-subtract-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      batch_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, batch_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
