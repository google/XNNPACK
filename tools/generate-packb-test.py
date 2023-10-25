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


parser = argparse.ArgumentParser(description='PackB/ZeroB microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(x8|x16|x32)_(packb|zerob)_gemm_ukernel_(\d+)c(\d+)s(\d+)r__(.+)", name)
  assert match is not None
  channel_tile = int(match.group(3))
  channel_subtile = int(match.group(4))
  channel_round = int(match.group(5))
  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(4))
  return channel_tile, channel_subtile, channel_round, arch, isa


PACKB_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, n_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(${CHANNEL_TILE})
      .kernel_tile(k)
      .channel_tile(${CHANNEL_TILE})
      .channel_subtile(${CHANNEL_SUBTILE})
      .channel_round(${CHANNEL_ROUND})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, n_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < 4; k++) {
      PackBMicrokernelTester()
        .channels(${CHANNEL_TILE*2})
        .kernel_tile(k)
        .channel_tile(${CHANNEL_TILE})
        .channel_subtile(${CHANNEL_SUBTILE})
        .channel_round(${CHANNEL_ROUND})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, n_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < 4; k++) {
      for (size_t n = 1; n < ${CHANNEL_TILE}; n++) {
        PackBMicrokernelTester()
          .channels(n)
          .kernel_tile(k)
          .channel_tile(${CHANNEL_TILE})
          .channel_subtile(${CHANNEL_SUBTILE})
          .channel_round(${CHANNEL_ROUND})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, n_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = ${CHANNEL_TILE+1}; n < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(${CHANNEL_TILE})
        .channel_subtile(${CHANNEL_SUBTILE})
        .channel_round(${CHANNEL_ROUND})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, groups_gt_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = ${CHANNEL_TILE+1}; n < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(${CHANNEL_TILE})
          .channel_subtile(${CHANNEL_SUBTILE})
          .channel_round(${CHANNEL_ROUND})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}
"""


def generate_test_cases(ukernel, channel_tile, channel_subtile, channel_round, isa):
  """Generates all tests cases for a PACKB/ZEROB micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    channel_tile: channel_tile parameter of the PACKB/ZEROB micro-kernel.
    channel_subtile: channel_subtile parameter of the PACKB/ZEROB micro-kernel.
    channel_round: channel_round parameter of the PACKB/ZEROB micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(PACKB_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "CHANNEL_TILE": channel_tile,
      "CHANNEL_SUBTILE": channel_subtile,
      "CHANNEL_ROUND": channel_round,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")
    is_packb = 'packb' in options.spec

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

#include <xnnpack/{packb}.h>
#include "packb-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0], packb=("packb" if is_packb else "zerob"))

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      channel_tile, channel_subtile, channel_round, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, channel_tile, channel_subtile, channel_round, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
