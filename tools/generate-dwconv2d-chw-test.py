#!/usr/bin/env python
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='Test generator for DWCONV2D CHW micro-kernels')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


TEST_TEMPLATE = """\
$if SUBSAMPLING == 1:
  TEST(${TEST_NAME}, output_width_eq_${WIDTH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    DWConv2DMicrokernelTester()
      .input_width(${(WIDTH_TILE - 1) * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING})
      .input_height(${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1})
      .kernel_height(${KERNEL_HEIGHT})
      .kernel_width(${KERNEL_WIDTH})
      .subsampling(${SUBSAMPLING})
      .padding_left(${PADDING})
      .padding_right(${PADDING})
      .padding_top(${PADDING})
      .padding_bottom(${PADDING})
      .Test(${", ".join(TEST_ARGS)});
  }
$else:
  TEST(${TEST_NAME}, output_width_eq_${WIDTH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_width = ${(WIDTH_TILE - 1) * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width < ${WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width++) {
      DWConv2DMicrokernelTester()
        .input_width(input_width)
        .input_height(${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1})
        .kernel_height(${KERNEL_HEIGHT})
        .kernel_width(${KERNEL_WIDTH})
        .subsampling(${SUBSAMPLING})
        .padding_left(${PADDING})
        .padding_right(${PADDING})
        .padding_top(${PADDING})
        .padding_bottom(${PADDING})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if WIDTH_TILE > 1:
  TEST(${TEST_NAME}, output_width_div_${WIDTH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_width = ${2 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING - 1}; input_width < ${8 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING - 1}; input_width += ${WIDTH_TILE * SUBSAMPLING}) {
      DWConv2DMicrokernelTester()
        .input_width(input_width)
        .input_height(${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1})
        .kernel_height(${KERNEL_HEIGHT})
        .kernel_width(${KERNEL_WIDTH})
        .subsampling(${SUBSAMPLING})
        .padding_left(${PADDING})
        .padding_right(${PADDING})
        .padding_top(${PADDING})
        .padding_bottom(${PADDING})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, output_width_lt_${WIDTH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_width = ${max(1, KERNEL_WIDTH - 2 * PADDING)}; input_width < ${(WIDTH_TILE - 1) * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width++) {
      DWConv2DMicrokernelTester()
        .input_width(${WIDTH_TILE * SUBSAMPLING})
        .input_height(${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1})
        .kernel_height(${KERNEL_HEIGHT})
        .kernel_width(${KERNEL_WIDTH})
        .subsampling(${SUBSAMPLING})
        .padding_left(${PADDING})
        .padding_right(${PADDING})
        .padding_top(${PADDING})
        .padding_bottom(${PADDING})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, output_width_gt_${WIDTH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t input_width = ${WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width < ${(5 if WIDTH_TILE == 1 else 2) * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width++) {
    DWConv2DMicrokernelTester()
      .input_width(input_width)
      .input_height(${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1})
      .kernel_height(${KERNEL_HEIGHT})
      .kernel_width(${KERNEL_WIDTH})
      .subsampling(${SUBSAMPLING})
      .padding_left(${PADDING})
      .padding_right(${PADDING})
      .padding_top(${PADDING})
      .padding_bottom(${PADDING})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if SUBSAMPLING > 1:
  TEST(${TEST_NAME}, output_height_eq_${HEIGHT_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_height = ${(HEIGHT_TILE - 1) * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING}; input_height < ${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING}; input_height++) {
      for (size_t input_width = 1; input_width < ${5 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width += ${max(1, WIDTH_TILE * SUBSAMPLING - 1)}) {
        DWConv2DMicrokernelTester()
          .input_width(input_width)
          .input_height(input_height)
          .kernel_height(${KERNEL_HEIGHT})
          .kernel_width(${KERNEL_WIDTH})
          .subsampling(${SUBSAMPLING})
          .padding_left(${PADDING})
          .padding_right(${PADDING})
          .padding_top(${PADDING})
          .padding_bottom(${PADDING})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if HEIGHT_TILE > 1:
  TEST(${TEST_NAME}, output_height_div_${HEIGHT_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_height = ${2 * HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1}; input_height < ${8 * HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING - 1}; input_height += ${HEIGHT_TILE * SUBSAMPLING}) {
      for (size_t input_width = 1; input_width < ${5 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width += ${max(1, WIDTH_TILE * SUBSAMPLING - 1)}) {
        DWConv2DMicrokernelTester()
          .input_width(input_width)
          .input_height(input_height)
          .kernel_height(${KERNEL_HEIGHT})
          .kernel_width(${KERNEL_WIDTH})
          .subsampling(${SUBSAMPLING})
          .padding_left(${PADDING})
          .padding_right(${PADDING})
          .padding_top(${PADDING})
          .padding_bottom(${PADDING})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, output_height_lt_${HEIGHT_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_height = ${max(1, KERNEL_HEIGHT - 2 * PADDING)}; input_height < ${(HEIGHT_TILE - 1) * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING}; input_height++) {
      for (size_t input_width = 1; input_width < ${5 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width += ${max(1, WIDTH_TILE * SUBSAMPLING - 1)}) {
        DWConv2DMicrokernelTester()
          .input_width(input_width)
          .input_height(input_height)
          .kernel_height(${KERNEL_HEIGHT})
          .kernel_width(${KERNEL_WIDTH})
          .subsampling(${SUBSAMPLING})
          .padding_left(${PADDING})
          .padding_right(${PADDING})
          .padding_top(${PADDING})
          .padding_bottom(${PADDING})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, output_height_gt_${HEIGHT_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t input_height = ${HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING}; input_height < ${(5 if WIDTH_TILE == 1 else 2) * HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING}; input_height++) {
    for (size_t input_width = 1; input_width < ${5 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width += ${max(1, WIDTH_TILE * SUBSAMPLING - 1)}) {
      DWConv2DMicrokernelTester()
        .input_width(input_width)
        .input_height(input_height)
        .kernel_height(${KERNEL_HEIGHT})
        .kernel_width(${KERNEL_WIDTH})
        .subsampling(${SUBSAMPLING})
        .padding_left(${PADDING})
        .padding_right(${PADDING})
        .padding_top(${PADDING})
        .padding_bottom(${PADDING})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if SUBSAMPLING > 1:
  TEST(${TEST_NAME}, padding_top_eq_${SUBSAMPLING - 1}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_height = ${max(1, KERNEL_HEIGHT - 2 * PADDING + 1)}; input_height < ${3 * HEIGHT_TILE * SUBSAMPLING + KERNEL_HEIGHT - 2 * PADDING + 1}; input_height++) {
      for (size_t input_width = 1; input_width < ${5 * WIDTH_TILE * SUBSAMPLING + KERNEL_WIDTH - 2 * PADDING}; input_width += ${max(1, WIDTH_TILE * SUBSAMPLING - 1)}) {
        DWConv2DMicrokernelTester()
          .input_width(input_width)
          .input_height(input_height)
          .kernel_height(${KERNEL_HEIGHT})
          .kernel_width(${KERNEL_WIDTH})
          .subsampling(${SUBSAMPLING})
          .padding_left(${PADDING})
          .padding_right(${PADDING})
          .padding_top(${PADDING - 1})
          .padding_bottom(${PADDING})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
"""

def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_dwconv2d_chw_ukernel_(\d+)x(\d+)(s2)?p(\d+)__(.+)_(\d+)x(\d+)(_acc\d+)?", name)
  assert match is not None
  kernel_height, kernel_width = int(match.group(2)), int(match.group(3))
  if match.group(4):
    assert match.group(4).startswith("s")
    stride = int(match.group(4)[1:])
  else:
    stride = 1
  padding = int(match.group(5))

  height_tile = int(match.group(7))
  width_tile = int(match.group(8))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(6))
  return kernel_height, kernel_width, stride, padding, arch, isa, \
         height_tile, width_tile


def generate_test_cases(ukernel, kernel_height, kernel_width, subsampling, \
  init_fn, padding, isa, height_tile, width_tile):
  """Generates all tests cases for a DWCONV2D CHW micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    kernel_height: convolution kernel height assumed by the micro-kernel.
    kernel_width: convolution kernel width assumed by the micro-kernel.
    subsampling: convolution subsampling (stride) assumed by the micro-kernel.
                 The same subsampling factor is assumed for both horizontal and
                 vertical directions.
    init_fn: C name of the function to initialize microkernel parameters.
    padding: convolution padding value assumed by the micro-kernel for right,
             bottom, and left padding. If convolution stride is 1, the same
             padding value is assumed for the top padding. If convolution stride
             is different than 1, top padding is specified via micro-kernel
             parameter, and can be either padding or (padding - 1).
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.
    height_tile: number of output rows processed in one iteration of the main
                 loop of the micro-kernel.
    width_tile: number of output columns processed in one iteration of the main
                loop of the micro-kernel.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel, init_fn]
  return xngen.preprocess(TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "KERNEL_HEIGHT": kernel_height,
      "KERNEL_WIDTH": kernel_width,
      "SUBSAMPLING": subsampling,
      "PADDING": padding,
      "HEIGHT_TILE": height_tile,
      "WIDTH_TILE": width_tile,
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

#include <xnnpack/dwconv.h>
#include "dwconv2d-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec["init"]
      pipelined = bool(ukernel_spec.get("pipelined", False))
      kernel_height, kernel_width, subsampling, padding, arch, isa, \
        height_tile, width_tile = split_ukernel_name(name)

      test_case = generate_test_cases(name, kernel_height, kernel_width, \
                                      subsampling, init_fn, padding, isa, \
                                      height_tile, width_tile)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
