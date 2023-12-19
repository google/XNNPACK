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
  description='Pad microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_xx_pad_ukernel_p(\d+)__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  padding_tile = int(match.group(1))
  channel_tile = int(match.group(3))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return padding_tile, channel_tile, arch, isa


PAD_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, fulltile_copy_channels_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(${CHANNEL_TILE})
    .Test(${UKERNEL_NAME});
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, fulltile_copy_channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE*2}; channels <= ${CHANNEL_TILE*3}; channels += ${CHANNEL_TILE}) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, fulltile_copy_channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(channels)
        .Test(${UKERNEL_NAME});
    }
  }

TEST(${TEST_NAME}, fulltile_copy_channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(channels)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, fulltile_pre_padding_eq_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .pre_padding(1)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, fulltile_pre_padding_eq_2) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .pre_padding(2)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, fulltile_pre_padding_eq_4) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .pre_padding(4)
    .Test(${UKERNEL_NAME});
}

$if PADDING_TILE != 4:
  TEST(${TEST_NAME}, fulltile_pre_padding_eq_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(${PADDING_TILE})
      .Test(${UKERNEL_NAME});
  }

$if PADDING_TILE > 1:
  TEST(${TEST_NAME}, fulltile_pre_padding_div_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pre_padding = ${PADDING_TILE*2}; pre_padding <= ${PADDING_TILE*3}; pre_padding += ${PADDING_TILE}) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, fulltile_pre_padding_lt_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pre_padding = 1; pre_padding < ${PADDING_TILE}; pre_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .pre_padding(pre_padding)
        .Test(${UKERNEL_NAME});
    }
  }

TEST(${TEST_NAME}, fulltile_pre_padding_gt_${PADDING_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pre_padding = ${PADDING_TILE+1}; pre_padding < ${10 if PADDING_TILE == 1 else PADDING_TILE * 2}; pre_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .pre_padding(pre_padding)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, fulltile_post_padding_eq_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .post_padding(1)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, fulltile_post_padding_eq_2) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .post_padding(2)
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, fulltile_post_padding_eq_4) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PadMicrokernelTester()
    .rows(1)
    .input_channels(1)
    .post_padding(4)
    .Test(${UKERNEL_NAME});
}

$if PADDING_TILE > 4:
  TEST(${TEST_NAME}, fulltile_post_padding_eq_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .post_padding(${PADDING_TILE})
      .Test(${UKERNEL_NAME});
  }

$if PADDING_TILE > 1:
  TEST(${TEST_NAME}, fulltile_post_padding_div_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t post_padding = ${PADDING_TILE*2}; post_padding <= ${PADDING_TILE*3}; post_padding += ${PADDING_TILE}) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, fulltile_post_padding_lt_${PADDING_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t post_padding = 1; post_padding < ${PADDING_TILE}; post_padding++) {
      PadMicrokernelTester()
        .rows(1)
        .input_channels(1)
        .post_padding(post_padding)
        .Test(${UKERNEL_NAME});
    }
  }

TEST(${TEST_NAME}, fulltile_post_padding_gt_${PADDING_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t post_padding = ${PADDING_TILE+1}; post_padding < ${10 if PADDING_TILE == 1 else PADDING_TILE * 2}; post_padding++) {
    PadMicrokernelTester()
      .rows(1)
      .input_channels(1)
      .post_padding(post_padding)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, multitile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < ${PADDING_TILE*3}; channels += 3) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .Test(${UKERNEL_NAME});
    }
  }
}

TEST(${TEST_NAME}, multitile_with_input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < ${PADDING_TILE*3}; channels += 3) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(channels)
        .pre_padding(channels)
        .post_padding(channels)
        .input_stride(51)
        .Test(${UKERNEL_NAME});
    }
  }
}

TEST(${TEST_NAME}, multitile_with_output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 2; rows <= 5; rows++) {
    for (size_t channels = 1; channels < ${PADDING_TILE*3}; channels += 3) {
      PadMicrokernelTester()
        .rows(rows)
        .input_channels(2 * channels)
        .pre_padding(channels)
        .post_padding(channels)
        .output_stride(193)
        .Test(${UKERNEL_NAME});
    }
  }
}
"""


def generate_test_cases(ukernel, padding_tile, channel_tile, isa):
  """Generates all tests cases for a Pad micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    padding_tile: Number of channels processed per one iteration of the
                inner loop of the pre/post padding part of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the
                inner loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  return xngen.preprocess(PAD_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "PADDING_TILE": padding_tile,
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

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/pad.h>
#include "pad-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      padding_tile, channel_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, padding_tile, channel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
