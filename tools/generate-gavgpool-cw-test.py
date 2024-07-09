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
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='GAvgPoolCW microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_gavgpool_cw_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)

  element_tile = int(match.group(3))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return element_tile, arch, isa


AVGPOOL_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, elements_eq_${ELEMENT_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  GAvgPoolCWMicrokernelTester()
    .elements(${ELEMENT_TILE})
    .channels(${CHANNEL_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, elements_gt_${ELEMENT_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t elements = ${ELEMENT_TILE + 1}; elements < ${ELEMENT_TILE * 2}; elements++) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ELEMENT_TILE > 1:
  TEST(${TEST_NAME}, elements_lt_${ELEMENT_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t elements = 1; elements < ${ELEMENT_TILE}; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(${CHANNEL_TILE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, elements_div_${ELEMENT_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t elements = ${ELEMENT_TILE * 2}; elements < ${ELEMENT_TILE * 5}; elements += ${ELEMENT_TILE}) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(${CHANNEL_TILE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE + 1}; channels < ${CHANNEL_TILE * 4}; channels++) {
    GAvgPoolCWMicrokernelTester()
      .elements(${ELEMENT_TILE})
      .channels(channels)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t elements = 1; elements < ${ELEMENT_TILE * 2}; elements += ${1 if ELEMENT_TILE < 4 else 3}) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(${CHANNEL_TILE * 4})
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t elements = 1; elements < ${ELEMENT_TILE * 2}; elements += ${1 if ELEMENT_TILE < 4 else 3}) {
    GAvgPoolCWMicrokernelTester()
      .elements(elements)
      .channels(${CHANNEL_TILE * 4})
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }
}
"""


def generate_test_cases(ukernel, init_fn, element_tile, isa):
  """Generates all tests cases for a GAVGPOOL micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    element_tile: Number of elements/pixels processed per one iteration of the inner
                  loops of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel, init_fn]
  return xngen.preprocess(AVGPOOL_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "CHANNEL_TILE": 1,  # All microkernels process one channel at a time.
      "ELEMENT_TILE": element_tile,
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
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "gavgpool-cw-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      element_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, init_fn, element_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
