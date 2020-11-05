#!/usr/bin/env python
# Copyright 2020 Google LLC
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
  description='Depth-to-Space CHW-to-HWC microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"^xnn_x(8|16|32)_depth_to_space_chw2hwc_ukernel__(.+)_c(\d+)(_ib(\d+))?$", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  channel_tile = int(match.group(3))
  try:
    inner_block_size_tile = int(match.group(5))
  except TypeError:
    inner_block_size_tile = 1

  arch, isa = xnncommon.parse_target_name(target_name=match.group(2))
  return channel_tile, inner_block_size_tile, arch, isa


DEPTH_TO_SPACE_CHW2HWC_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, smoke) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(${TEST_FUNC});
}

$if INNER_BLOCK_SIZE_TILE == 1:
  TEST(${TEST_NAME}, block_size_gt_2) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t block_size = 2; block_size <= 8; block_size++) {
      DepthToSpaceMicrokernelTester()
        .output_channels(1)
        .input_height(1)
        .input_width(1)
        .block_size(block_size)
        .Test(${TEST_FUNC});
    }
  }

TEST(${TEST_NAME}, channels_gt_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(${TEST_FUNC});
  }
}

TEST(${TEST_NAME}, non_unit_size) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(${TEST_FUNC});
      }
    }
  }
}

TEST(${TEST_NAME}, non_unit_size_block_size_3) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(${TEST_FUNC});
      }
    }
  }
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .output_channels(channels)
        .Test(${TEST_FUNC});
    }
  }

  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE}; channels <= 4 * ${CHANNEL_TILE}; channels += ${CHANNEL_TILE}) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .output_channels(channels)
        .Test(${TEST_FUNC});
    }
  }

  TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE} + 1; channels < 2 * ${CHANNEL_TILE}; channels++) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .output_channels(channels)
        .Test(${TEST_FUNC});
    }
  }

$if INNER_BLOCK_SIZE_TILE > 1:
  TEST(${TEST_NAME}, block_size_lt_${INNER_BLOCK_SIZE_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t block_size = 2; block_size < ${INNER_BLOCK_SIZE_TILE}; block_size++) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .block_size(block_size)
        .Test(${TEST_FUNC});
    }
  }

  TEST(${TEST_NAME}, block_size_div_${INNER_BLOCK_SIZE_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t block_size = ${INNER_BLOCK_SIZE_TILE}; block_size <= 3 * ${INNER_BLOCK_SIZE_TILE}; block_size += ${INNER_BLOCK_SIZE_TILE}) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .block_size(block_size)
        .Test(${TEST_FUNC});
    }
  }

  TEST(${TEST_NAME}, block_size_gt_${INNER_BLOCK_SIZE_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t block_size = ${INNER_BLOCK_SIZE_TILE} + 1; block_size < 2 * ${INNER_BLOCK_SIZE_TILE}; block_size++) {
      DepthToSpaceMicrokernelTester()
        .input_height(3)
        .input_width(3)
        .block_size(block_size)
        .Test(${TEST_FUNC});
    }
  }
"""


def generate_test_cases(ukernel, channel_tile, inner_block_size_tile, isa):
  """Generates all tests cases for a Clamp micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: Number of output channels processed per one iteration of the
                  innermost loop of the micro-kernel.
    inner_block_size_tile: Number of pixels along the `block_size` dimension
                           of the output width processed per one iteration of
                           the next-to-innermost loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  test_args = [ukernel]
  return xngen.preprocess(DEPTH_TO_SPACE_CHW2HWC_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_FUNC": ukernel,
      "DATATYPE": datatype,
      "CHANNEL_TILE": channel_tile,
      "INNER_BLOCK_SIZE_TILE": inner_block_size_tile,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2020 Google LLC
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

#include <xnnpack/depthtospace.h>
#include "depth-to-space-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      channel_tile, inner_block_size_tile, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, channel_tile, inner_block_size_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
