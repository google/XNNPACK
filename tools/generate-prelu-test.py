#!/usr/bin/env python
# Copyright 2019 Google LLC
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


parser = argparse.ArgumentParser(description='PReLU microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(f16|f32)_prelu_ukernel__(.+)_(\d+)x(\d+)", name)
  assert match is not None
  row_tile = int(match.group(3))
  channel_tile = int(match.group(4))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(2))
  return row_tile, channel_tile, arch, isa


PRELU_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PReLUMicrokernelTester()
    .rows(${ROW_TILE})
    .channels(${CHANNEL_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

$if CHANNEL_TILE > 1:
  TEST(${TEST_NAME}, channels_div_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${CHANNEL_TILE*2}; channels < ${CHANNEL_TILE*10}; channels += ${CHANNEL_TILE}) {
      PReLUMicrokernelTester()
        .rows(${ROW_TILE})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${CHANNEL_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${CHANNEL_TILE}; channels++) {
      PReLUMicrokernelTester()
        .rows(${ROW_TILE})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, channels_gt_${CHANNEL_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${CHANNEL_TILE+1}; channels < ${10 if CHANNEL_TILE == 1 else CHANNEL_TILE*2}; channels++) {
    PReLUMicrokernelTester()
      .rows(${ROW_TILE})
      .channels(channels)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ROW_TILE > 1:
  TEST(${TEST_NAME}, rows_lt_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = 1; rows < ${ROW_TILE}; rows++) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, rows_div_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${ROW_TILE*2}; rows <= ${ROW_TILE*4}; rows += ${ROW_TILE}) {
      for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, rows_gt_${ROW_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = ${ROW_TILE+1}; rows < ${ROW_TILE*2}; rows++) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, input_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(${next_prime(CHANNEL_TILE*5+1)})
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(${next_prime(CHANNEL_TILE*5+1)})
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${CHANNEL_TILE*5}; channels += ${max(1, CHANNEL_TILE-1)}) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def generate_test_cases(ukernel, row_tile, channel_tile, isa):
  """Generates all tests cases for a PRELU micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    row_tile: Number of rows (pixels) processed per one iteration of the outer
              loop of the micro-kernel.
    channel_tile: Number of channels processed per one iteration of the inner
                  loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(PRELU_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "DATATYPE": datatype,
      "ROW_TILE": row_tile,
      "CHANNEL_TILE": channel_tile,
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
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/prelu.h"
#include "prelu-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      row_tile, channel_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, row_tile, channel_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
