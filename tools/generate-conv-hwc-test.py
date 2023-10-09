#!/usr/bin/env python
# Copyright 2023 Google LLC
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
    description="Test generator for CONV HWC micro-kernels"
)
parser.add_argument(
    "-s", "--spec", metavar="FILE", required=True, help="Spec (YAML) file"
)
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file",
)
parser.set_defaults(defines=list())


TEST_TEMPLATE = """\
TEST(${TEST_NAME}, input_width_eq_${INPUT_WIDTH}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  ConvHWCMicrokernelTester()
    .kernel_size(${KERNEL_SIZE})
    .subsampling(${SUBSAMPLING})
    $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
      .padding_width(${PADDING_RIGHT})
    $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
      .padding_right(${PADDING_RIGHT})
    .input_channels(${INPUT_CHANNELS})
    .output_channels_tile(${OUTPUT_CHANNELS_TILE})
    .output_channels(${OUTPUT_CHANNELS_TILE})
    .input_width(${INPUT_WIDTH})
    .input_height(${KERNEL_SIZE})
    .Test(${", ".join(TEST_ARGS)});
}

$if INPUT_WIDTH > 1:
  TEST(${TEST_NAME}, input_width_div_${INPUT_WIDTH}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_width = ${INPUT_WIDTH*2}; input_width <= ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*3}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(${OUTPUT_CHANNELS_TILE})
        .input_width(input_width)
        .input_height(${KERNEL_SIZE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, input_width_lt_${INPUT_WIDTH}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH}; input_width++) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(${OUTPUT_CHANNELS_TILE})
        .input_width(input_width)
        .input_height(${KERNEL_SIZE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, input_width_gt_${INPUT_WIDTH}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t input_width = ${INPUT_WIDTH+1}; input_width < ${INPUT_WIDTH*2}; input_width++) {
    ConvHWCMicrokernelTester()
      .kernel_size(${KERNEL_SIZE})
      .subsampling(${SUBSAMPLING})
      $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
        .padding_width(${PADDING_RIGHT})
      $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
        .padding_right(${PADDING_RIGHT})
      .input_channels(${INPUT_CHANNELS})
      .output_channels_tile(${OUTPUT_CHANNELS_TILE})
      .output_channels(${OUTPUT_CHANNELS_TILE})
      .input_width(input_width)
      .input_height(${KERNEL_SIZE})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, output_channels_lt_${OUTPUT_CHANNELS_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE}; output_channels++) {
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(${KERNEL_SIZE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, output_channels_div_${OUTPUT_CHANNELS_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_channels = ${OUTPUT_CHANNELS_TILE*2}; output_channels <= ${OUTPUT_CHANNELS_TILE*4}; output_channels += ${OUTPUT_CHANNELS_TILE}) {
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(${KERNEL_SIZE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, output_channels_gt_${OUTPUT_CHANNELS_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_channels = ${OUTPUT_CHANNELS_TILE+1}; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels++) {
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(${KERNEL_SIZE})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, input_height_lt_3) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t input_height = 1; input_height < 3; input_height++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 0:
            .padding_right(1)
            .padding_height(1) // padded input height of at least 3 required
          $else:
            .padding(1) // padded input height of at least 3 required
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(input_height)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, input_height_gt_3) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t input_height = 4; input_height <= 9; input_height++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
            .padding_width(${PADDING_RIGHT})
          $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
            .padding_right(${PADDING_RIGHT})
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(input_height)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, padding_top) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
            .padding_width(${PADDING_RIGHT})
          $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
            .padding_right(${PADDING_RIGHT})
          .padding_top(padding_top)
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, padding_bottom) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
            .padding_width(${PADDING_RIGHT})
          $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
            .padding_right(${PADDING_RIGHT})
          .padding_bottom(padding_bottom)
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, output_y_start) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_y_start = 1; output_y_start <= 3; output_y_start++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
            .padding_width(${PADDING_RIGHT})
          $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
            .padding_right(${PADDING_RIGHT})
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .output_y_start(output_y_start)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, output_y_end) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_y_end = 2; output_y_end < 5; output_y_end++) {
    for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
      for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
        ConvHWCMicrokernelTester()
          .kernel_size(${KERNEL_SIZE})
          .subsampling(${SUBSAMPLING})
          $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
            .padding_width(${PADDING_RIGHT})
          $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
            .padding_right(${PADDING_RIGHT})
          .input_channels(${INPUT_CHANNELS})
          .output_channels_tile(${OUTPUT_CHANNELS_TILE})
          .output_channels(output_channels)
          .input_width(input_width)
          .input_height(9)
          .output_y_end(output_y_end)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(6)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t output_channels = 1; output_channels < ${OUTPUT_CHANNELS_TILE*2}; output_channels += ${OUTPUT_CHANNELS_TILE-1}) {
    for (size_t input_width = ${1 if PADDING_LEFT else 2}; input_width < ${INPUT_WIDTH*8}; input_width += ${INPUT_WIDTH*2-1}) {
      ConvHWCMicrokernelTester()
        .kernel_size(${KERNEL_SIZE})
        .subsampling(${SUBSAMPLING})
        $if PADDING_LEFT == 1 and PADDING_RIGHT == 1:
          .padding_width(${PADDING_RIGHT})
        $elif PADDING_LEFT == 0 and PADDING_RIGHT == 1:
          .padding_right(${PADDING_RIGHT})
        .input_channels(${INPUT_CHANNELS})
        .output_channels_tile(${OUTPUT_CHANNELS_TILE})
        .output_channels(output_channels)
        .input_width(input_width)
        .input_height(6)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def split_ukernel_name(name):
  match = re.fullmatch(
      r"xnn_f32_conv_hwc_ukernel_(\d+)x(\d+)s(\d+)(p0)?(p1)c(\d+)x(\d+)__(.+)_(\d+)x(\d+)?",
      name,
  )
  assert match is not None
  kernel_height, kernel_width = int(match.group(1)), int(match.group(2))
  assert kernel_height == kernel_width
  subsampling = int(match.group(3))
  padding_right = 1
  padding_left = 0 if match.group(4) else 1
  input_channels = int(match.group(6))
  channel_tile = int(match.group(7))
  height_tile = int(match.group(9))
  width_tile = int(match.group(10))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(8))
  return (
      kernel_height,
      kernel_width,
      subsampling,
      padding_left,
      padding_right,
      input_channels,
      channel_tile,
      height_tile,
      width_tile,
      arch,
      isa,
  )


def generate_test_cases(
    ukernel,
    kernel_height,
    kernel_width,
    subsampling,
    padding_left,
    padding_right,
    input_channels,
    channel_tile,
    height_tile,
    width_tile,
    init_fn,
    isa,
):
  """Generates all tests cases for a CONV HWC micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    kernel_height: convolution kernel height assumed by the micro-kernel.
    kernel_width: convolution kernel width assumed by the micro-kernel.
    subsampling: convolution subsampling (stride) assumed by the micro-kernel.
      The same subsampling factor is assumed for both horizontal and vertical
      directions.
    padding_left: input padding on the left assumed by micro-kernel, 0 or 1
    padding_right: input padding on the right assumed by micro-kernel, current
      microkernels always assume this is 1
    input_channels: number of input channels assumed by micro-kernel.
    channel_tile: number of output channels processed in one iteration of the
      main loop of the micro-kernel.
    height_tile: number of output rows processed in one iteration of the main
      loop of the micro-kernel.
    width_tile: number of output columns processed in one iteration of the main
      loop of the micro-kernel.
    init_fn: C name of the function to initialize microkernel parameters.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel, init_fn]
  return xngen.preprocess(
      TEST_TEMPLATE,
      {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": test_args,
          "UKERNEL_TYPE": ukernel_type.upper(),
          "DATATYPE": datatype,
          "INPUT_WIDTH": width_tile * 2,
          "KERNEL_SIZE": kernel_height,
          "KERNEL_HEIGHT": kernel_height,
          "KERNEL_WIDTH": kernel_width,
          "SUBSAMPLING": subsampling,
          "PADDING_LEFT": padding_left,
          "PADDING_RIGHT": padding_right,
          "INPUT_CHANNELS": input_channels,
          "OUTPUT_CHANNELS_TILE": channel_tile,
          "HEIGHT_TILE": height_tile,
          "WIDTH_TILE": width_tile,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "next_prime": next_prime,
      },
  )


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

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/conv.h>
#include "conv-hwc-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec["init"]
      (
          kernel_height,
          kernel_width,
          subsampling,
          padding_left,
          padding_right,
          input_channels,
          channel_tile,
          height_tile,
          width_tile,
          arch,
          isa,
      ) = split_ukernel_name(name)

      test_case = generate_test_cases(
          name,
          kernel_height,
          kernel_width,
          subsampling,
          padding_left,
          padding_right,
          input_channels,
          channel_tile,
          height_tile,
          width_tile,
          init_fn,
          isa,
      )
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
