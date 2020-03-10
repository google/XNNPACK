#!/usr/bin/env python
# Copyright 2019 Google LLC
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
  description='IBILINEAR microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"^xnn_(f16|f32)_ibilinear_ukernel__(.+)_c(\d+)$", name)
  assert match is not None
  channel_tile = int(match.group(3))
  pixel_tile = 1

  arch, isa = xnncommon.parse_target_name(target_name=match.group(2))
  return channel_tile, pixel_tile, arch, isa


IBILINEAR_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  IBilinearMicrokernelTester()
    .pixels(${PIXEL_TILE})
    .channels(${CHANNEL_TILE})
    .Test(${TEST_FUNC});
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*10}; channels += ${CHANNEL_TILE}) {
      IBilinearMicrokernelTester()
        .pixels(${PIXEL_TILE})
        .channels(channels)
        .Test(${TEST_FUNC});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      IBilinearMicrokernelTester()
        .pixels(${PIXEL_TILE})
        .channels(channels)
        .Test(${TEST_FUNC});
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    IBilinearMicrokernelTester()
      .pixels(${PIXEL_TILE})
      .channels(channels)
      .Test(${TEST_FUNC});
  }
}

$if PIXEL_TILE > 1:
  TEST(${TEST_NAME}, pixels_div_${PIXEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pixels = ${PIXEL_TILE*2}; pixels < ${PIXEL_TILE*10}; pixels += ${PIXEL_TILE}) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE * 5}; channels += ${max(1, CHANNEL_TILE - 1)}) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(${TEST_FUNC});
      }
    }
  }

  TEST(${TEST_NAME}, pixels_lt_${PIXEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t pixels = 1; pixels < ${PIXEL_TILE}; pixels++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE * 5}; channels += ${max(1, CHANNEL_TILE - 1)}) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(${TEST_FUNC});
      }
    }
  }

TEST(${TEST_NAME}, pixels_gt_${PIXEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pixels = ${PIXEL_TILE+1}; pixels < ${max(PIXEL_TILE*2, 3)}; pixels++) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE * 5}; channels += ${max(1, CHANNEL_TILE - 1)}) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .Test(${TEST_FUNC});
    }
  }
}

TEST(${TEST_NAME}, input_offset) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pixels = 1; pixels < ${PIXEL_TILE * 5}; pixels += ${max(1, PIXEL_TILE - 1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE * 5}; channels += ${max(1, CHANNEL_TILE - 1)}) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .input_offset(${next_prime(CHANNEL_TILE * 5 + 1)})
        .Test(${TEST_FUNC});
    }
  }
}
TEST(${TEST_NAME}, output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t pixels = 1; pixels < ${PIXEL_TILE * 5}; pixels += ${max(1, PIXEL_TILE - 1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE * 5}; channels += ${max(1, CHANNEL_TILE - 1)}) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(channels)
        .output_stride(${next_prime(CHANNEL_TILE * 5 + 1)})
        .Test(${TEST_FUNC});
    }
  }
}
"""


def generate_test_cases(ukernel, channel_tile, pixel_tile, isa):
  """Generates all tests cases for a BILINEAR micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: Number of channels processed per one iteration of the inner
                  loop of the micro-kernel.
    pixel_tile: Number of pixels processed per one iteration of the outer loop
                of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  return xngen.preprocess(IBILINEAR_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_FUNC": ukernel,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "CHANNEL_TILE": channel_tile,
      "PIXEL_TILE": pixel_tile,
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
// Copyright 2019 Google LLC
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

#include <xnnpack/ibilinear.h>
#include "ibilinear-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      channel_tile, pixel_tile, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, channel_tile, pixel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
