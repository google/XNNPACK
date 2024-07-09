#!/usr/bin/env python
# Copyright 2022 Google LLC
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


parser = argparse.ArgumentParser(description='Window microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  shift = 0
  row_tile = 1
  match = re.fullmatch(r"xnn_s16_window(_shift(\d+))?_ukernel__(.+)_u(\d+)(v)?", name)
  assert match is not None
  if match.group(2):
    shift = int(match.group(2))
  channels_tile = int(match.group(4))

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(3))
  return shift, row_tile, channels_tile, arch, isa


WINDOW_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, channels_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  WindowMicrokernelTester()
    .rows(1)
    .channels(${BATCH_TILE})
    .shift(${SHIFT})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, channels_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = ${BATCH_TILE*2}; channels < ${BATCH_TILE*10}; channels += ${BATCH_TILE}) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(${SHIFT})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, channels_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels < ${BATCH_TILE}; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(${SHIFT})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, channels_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = ${BATCH_TILE+1}; channels < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(${SHIFT})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ROW_TILE > 1:
  TEST(${TEST_NAME}, rows_lt_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = 1; rows < ${ROW_TILE}; rows++) {
      for (size_t channels = 1; channels <= ${BATCH_TILE*5}; channels += ${max(1, BATCH_TILE-1)}) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(${SHIFT})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, rows_div_${ROW_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${ROW_TILE*2}; rows <= ${ROW_TILE*4}; rows += ${ROW_TILE}) {
      for (size_t channels = 1; channels <= ${BATCH_TILE*5}; channels += ${max(1, BATCH_TILE-1)}) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(${SHIFT})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, rows_gt_${ROW_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = ${ROW_TILE+1}; rows < ${ROW_TILE*2}; rows++) {
    for (size_t channels = 1; channels <= ${BATCH_TILE*5}; channels += ${max(1, BATCH_TILE-1)}) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(${SHIFT})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${ROW_TILE*3}; rows += ${max(1, ROW_TILE-1)}) {
    for (size_t channels = 1; channels <= ${BATCH_TILE*5}; channels += ${max(1, BATCH_TILE-1)}) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(${SHIFT})
        .inplace(true)
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if SHIFT == 0:
  TEST(${TEST_NAME}, shift) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(${ROW_TILE})
        .channels(${BATCH_TILE})
        .shift(shift)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
"""


def generate_test_cases(ukernel, shift, row_tile, channels_tile, isa):
  """Generates all tests cases for a Window micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    shift: Shift by constant value.
    row_tile: Number of rows (pixels) processed per one iteration of the outer
              loop of the micro-kernel.
    channels_tile: Number of channels processed per one iteration of the inner
                  loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(WINDOW_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "DATATYPE": datatype,
      "SHIFT": shift,
      "ROW_TILE": row_tile,
      "BATCH_TILE": channels_tile,
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
// Copyright 2022 Google LLC
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
#include "xnnpack/window.h"
#include "window-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      shift, row_tile, channels_tile, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, shift, row_tile, channels_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
