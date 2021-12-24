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
    description="Matrix transpose microkernel test generator")
parser.add_argument(
    "-s",
    "--spec",
    metavar="FILE",
    required=True,
    help="Specification (YAML) file")
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file")
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"^xnn_(x\d+)_transpose_ukernel__(\d+)x(\d+)_(.+)$", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  tile_height = int(match.group(2))
  tile_width = int(match.group(3))

  arch, isa = xnncommon.parse_target_name(target_name=match.group(4))
  return tile_height, tile_width, arch, isa


TRANSPOSE_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH})
    .output_stride(${TILE_HEIGHT})
    .block_width(${TILE_WIDTH})
    .block_height(${TILE_HEIGHT})
    .iterations(1)
    .Test(${KERNEL});
}

TEST(${TEST_NAME}, bh_1_${TILE_HEIGHT * 2}_bw_1_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = 1; i <= ${TILE_HEIGHT * 2}; ++i){
    for(size_t j = 1; j <= ${TILE_WIDTH * 2}; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH * 2})
    .output_stride(${TILE_HEIGHT})
    .block_width(${TILE_WIDTH * 2})
    .block_height(${TILE_HEIGHT})
    .iterations(1)
    .Test(${KERNEL});
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH + 1}_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${TILE_WIDTH + 1}; i < ${TILE_WIDTH * 2}; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(${TILE_HEIGHT})
      .block_width(i)
      .block_height(${TILE_HEIGHT})
      .iterations(1)
      .Test(${KERNEL});
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT * 2}_bw_${TILE_WIDTH + 1}_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${TILE_WIDTH + 1}; i < ${TILE_WIDTH * 2}; ++i){
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(${TILE_HEIGHT * 2})
      .block_width(i)
      .block_height(${TILE_HEIGHT * 2})
      .iterations(1)
      .Test(${KERNEL});
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT * 2}_bw_${TILE_WIDTH}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH})
    .output_stride(${TILE_HEIGHT * 2})
    .block_width(${TILE_WIDTH})
    .block_height(${TILE_HEIGHT * 2})
    .iterations(1)
    .Test(${KERNEL});
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT + 1}_${TILE_HEIGHT * 2}_bw_${TILE_WIDTH}){
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${TILE_HEIGHT + 1}; i < ${TILE_HEIGHT * 2}; ++i){
    TransposeMicrokernelTester()
      .input_stride(${TILE_WIDTH})
      .output_stride(i)
      .block_width(${TILE_WIDTH})
      .block_height(i)
      .iterations(1)
      .Test(${KERNEL});
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT + 1}_${TILE_HEIGHT * 2}_bw_${TILE_WIDTH * 2}){
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${TILE_HEIGHT + 1}; i < ${TILE_HEIGHT * 2}; ++i){
    TransposeMicrokernelTester()
      .input_stride(${TILE_WIDTH * 2})
      .output_stride(i)
      .block_width(${TILE_WIDTH * 2})
      .block_height(i)
      .iterations(1)
      .Test(${KERNEL});
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT + 1}_${TILE_HEIGHT * 2}_bw_${TILE_WIDTH + 1}_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${TILE_HEIGHT + 1}; i < ${TILE_HEIGHT * 2}; ++i){
    for(size_t j = ${TILE_WIDTH + 1}; j < ${TILE_WIDTH * 2}; ++j){
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH}_is_${TILE_WIDTH * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH * 2})
    .output_stride(${TILE_HEIGHT})
    .block_width(${TILE_WIDTH})
    .block_height(${TILE_HEIGHT})
    .iterations(1)
    .Test(${KERNEL});
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH}_os_${TILE_HEIGHT * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH})
    .output_stride(${TILE_HEIGHT * 2})
    .block_width(${TILE_WIDTH})
    .block_height(${TILE_HEIGHT})
    .iterations(1)
    .Test(${KERNEL});
}

TEST(${TEST_NAME}, bh_${TILE_HEIGHT}_bw_${TILE_WIDTH}_is_${TILE_WIDTH * 2}_os_${TILE_HEIGHT * 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .input_stride(${TILE_WIDTH * 2})
    .output_stride(${TILE_HEIGHT * 2})
    .block_width(${TILE_WIDTH})
    .block_height(${TILE_HEIGHT})
    .iterations(1)
    .Test(${KERNEL});
}
"""


def generate_test_cases(ukernel, tile_height, tile_width, isa):
  """Generates all tests cases for a Vector Convert Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    tile_height: Number of vertical elements processed by the ukernel.
    tile_width: Number of horizontal elements processed by the ukernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  test_args = [ukernel]
  return xngen.preprocess(
      TRANSPOSE_TEST_TEMPLATE, {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "KERNEL": ukernel,
          "TILE_HEIGHT": tile_height,
          "TILE_WIDTH": tile_width,
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

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"
""".format(
    specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      tile_height, tile_width, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, tile_height, tile_width, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != tests

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
