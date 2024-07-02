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
  description='Fill microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_xx_fill_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  channel_tile = int(match.group(2))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(1))
  return channel_tile, arch, isa


FILL_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  FillMicrokernelTester()
    .channels(${CHANNEL_TILE})
    .Test(${UKERNEL_NAME});
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE*2}; channels <= ${CHANNEL_TILE*3}; channels += ${CHANNEL_TILE}) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      FillMicrokernelTester()
        .channels(channels)
        .Test(${UKERNEL_NAME});
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    FillMicrokernelTester()
      .channels(channels)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, multiple_rows) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*3}; channels += ${1 if CHANNEL_TILE == 1 else (CHANNEL_TILE // 4 - 1)}) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .Test(${UKERNEL_NAME});
    }
  }
}

TEST(${TEST_NAME}, multiple_rows_with_output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 2; rows < 5; rows++) {
    for (size_t channels = 1; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*3}; channels += ${1 if CHANNEL_TILE == 1 else (CHANNEL_TILE // 4 - 1)}) {
      FillMicrokernelTester()
        .channels(channels)
        .rows(rows)
        .output_stride(${CHANNEL_TILE*3+1})
        .Test(${UKERNEL_NAME});
    }
  }
}
"""


def generate_test_cases(ukernel, channel_tile, isa):
  """Generates all tests cases for a Fill micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: Number of batch elements processed per one iteration of the
                inner loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  return xngen.preprocess(FILL_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "CHANNEL_TILE": channel_tile,
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
#include "xnnpack/common.h"
#include "xnnpack/fill.h"
#include "xnnpack/isa-checks.h"
#include "fill-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      channel_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, channel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
