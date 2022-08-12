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


parser = argparse.ArgumentParser(description='VWindow microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_s16_vwindow_ukernel__(.+)_x(\d+)", name)
  assert match is not None
  row_tile = 1
  batch_tile = int(match.group(2))

  arch, isa = xnncommon.parse_target_name(target_name=match.group(1))
  return row_tile, batch_tile, arch, isa


VWINDOW_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VWindowMicrokernelTester()
    .batch(${BATCH_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch = ${BATCH_TILE*2}; batch < ${BATCH_TILE*10}; batch += ${BATCH_TILE}) {
      VWindowMicrokernelTester()
        .batch(batch)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch = 1; batch < ${BATCH_TILE}; batch++) {
      VWindowMicrokernelTester()
        .batch(batch)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch = ${BATCH_TILE+1}; batch < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch++) {
    VWindowMicrokernelTester()
      .batch(batch)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, rows_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = 1; rows < ${BATCH_TILE}; rows++) {
      for (size_t batch = 1; batch <= ${BATCH_TILE*5}; batch += ${max(1, BATCH_TILE-1)}) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, rows_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t rows = ${BATCH_TILE*2}; rows <= ${BATCH_TILE*4}; rows += ${BATCH_TILE}) {
      for (size_t batch = 1; batch <= ${BATCH_TILE*5}; batch += ${max(1, BATCH_TILE-1)}) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, rows_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = ${BATCH_TILE+1}; rows < ${BATCH_TILE*2}; rows++) {
    for (size_t batch = 1; batch <= ${BATCH_TILE*5}; batch += ${max(1, BATCH_TILE-1)}) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t rows = 1; rows <= ${BATCH_TILE*3}; rows += ${max(1, BATCH_TILE-1)}) {
    for (size_t batch = 1; batch <= ${BATCH_TILE*5}; batch += ${max(1, BATCH_TILE-1)}) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, shift) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t shift = 0; shift < 32; shift++) {
    VWindowMicrokernelTester()
      .rows(${BATCH_TILE})
      .batch(${BATCH_TILE})
      .shift(shift)
      .Test(${", ".join(TEST_ARGS)});
  }
}

"""


def generate_test_cases(ukernel, row_tile, batch_tile, isa):
  """Generates all tests cases for a Window micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    row_tile: Number of rows (pixels) processed per one iteration of the outer
              loop of the micro-kernel.
    batch_tile: Number of batch processed per one iteration of the inner
                  loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  return xngen.preprocess(VWINDOW_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "DATATYPE": datatype,
      "BATCH_TILE": row_tile,
      "BATCH_TILE": batch_tile,
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

#include <xnnpack/vwindow.h>
#include "vwindow-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      row_tile, batch_tile, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, row_tile, batch_tile, isa)
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
