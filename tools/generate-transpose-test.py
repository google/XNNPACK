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
  match = re.match(r"^xnn_(x32)_transpose_ukernel__(\d+)x(\d+)_(.+)$", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  block_height = int(match.group(2))
  block_width = int(match.group(3))

  arch, isa = xnncommon.parse_target_name(target_name=match.group(4))
  return block_height, block_width, arch, isa


TRANSPOSE_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, offset_${BLOCK_HEIGHT * 3}_${BLOCK_WIDTH * 3}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
      TransposeMicrokernelTester()
        .height(${BLOCK_HEIGHT * 3})
        .width(${BLOCK_WIDTH * 3})
        .h_start(2)
        .h_end(${BLOCK_HEIGHT * 3 - 1})
        .w_start(1)
        .w_end(${BLOCK_HEIGHT * 2 + 1})
        .iterations(1)
        .Test(${KERNEL});
}

TEST(${TEST_NAME}, offset_${BLOCK_HEIGHT * 4 + 1}_${BLOCK_WIDTH * 3 + 1}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
      TransposeMicrokernelTester()
        .height(${BLOCK_HEIGHT * 4 + 1})
        .width(${BLOCK_WIDTH * 3 + 1})
        .h_start(${BLOCK_HEIGHT + 1})
        .h_end(${BLOCK_HEIGHT * 3 + 1})
        .w_start(${BLOCK_WIDTH})
        .w_end(${BLOCK_WIDTH * 3})
        .iterations(1)
        .Test(${KERNEL});
}


TEST(${TEST_NAME}, offset_${BLOCK_HEIGHT * 2 + 1}_${BLOCK_WIDTH * 10 - 1}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
      TransposeMicrokernelTester()
        .height(${BLOCK_HEIGHT * 2 + 1})
        .width(${BLOCK_WIDTH * 10 - 1})
        .h_start(1)
        .h_end(${BLOCK_HEIGHT * 2 + 1})
        .w_start(${BLOCK_WIDTH * 7 + 2})
        .w_end(${BLOCK_WIDTH * 10 - 2})
        .iterations(1)
        .Test(${KERNEL});
}

TEST(${TEST_NAME}, offset_${BLOCK_HEIGHT * 4}_${BLOCK_WIDTH * 4}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
      TransposeMicrokernelTester()
        .height(${BLOCK_HEIGHT * 4})
        .width(${BLOCK_WIDTH * 4})
        .h_start(2)
        .h_end(${BLOCK_HEIGHT * 2 + 2})
        .w_start(3)
        .w_end(${BLOCK_WIDTH * 2 + 3})
        .iterations(1)
        .Test(${KERNEL});
}

TEST(${TEST_NAME}, offset_loop_${BLOCK_HEIGHT * 8}_${BLOCK_WIDTH * 8}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT * 2}; i < ${BLOCK_HEIGHT * 8}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH * 2}; j < ${BLOCK_WIDTH * 8}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(${BLOCK_HEIGHT - 2})
        .h_end(i - 2)
        .w_start(${BLOCK_WIDTH - 1})
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}

TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 8}_${BLOCK_WIDTH * 8}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 8}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 8}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}

TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 8 + 1}_${BLOCK_WIDTH * 8}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT} + 1; i < ${BLOCK_HEIGHT * 8 + 1}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 8}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}



TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 16}_${BLOCK_WIDTH * 8}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 16}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 8}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 8}_${BLOCK_WIDTH * 16}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 8}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 16}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 4 + 1}_${BLOCK_WIDTH * 8 + 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 4 + 1}; i += ${BLOCK_HEIGHT - 1}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 8 + 2}; j += ${BLOCK_WIDTH + 1}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 8}_${BLOCK_WIDTH * 6 + 1}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 8}; i += ${BLOCK_HEIGHT * 2 - 1}){
    for(size_t j = ${BLOCK_WIDTH}; j < ${BLOCK_WIDTH * 6 + 1}; j += ${BLOCK_WIDTH + 1}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 4 + 3}_${BLOCK_WIDTH * 7 + 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT + 1}; i < ${BLOCK_HEIGHT * 4 + 3}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH + 2}; j < ${BLOCK_WIDTH * 7 + 2}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 5}_${BLOCK_WIDTH * 7 + 2}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT}; i < ${BLOCK_HEIGHT * 5}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH + 2}; j < ${BLOCK_WIDTH * 7 + 2}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 3 + 1}_${BLOCK_WIDTH * 4 + 3}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT + 3}; i < ${BLOCK_HEIGHT * 3 + 1}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH + 3}; j < ${BLOCK_WIDTH * 4 + 3}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}


TEST(${TEST_NAME}, loop_${BLOCK_HEIGHT * 256 + 3}_${BLOCK_WIDTH * 31 + 3}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for(size_t i = ${BLOCK_HEIGHT + 3}; i < ${BLOCK_HEIGHT * 256 + 3}; i += ${BLOCK_HEIGHT}){
    for(size_t j = ${BLOCK_WIDTH + 3}; j < ${BLOCK_WIDTH * 31 + 3}; j += ${BLOCK_WIDTH}){
      TransposeMicrokernelTester()
        .height(i)
        .width(j)
        .h_start(0)
        .h_end(i)
        .w_start(0)
        .w_end(j)
        .iterations(1)
        .Test(${KERNEL});
    }
  }
}
"""


def generate_test_cases(ukernel, block_height, block_width, isa):
  """Generates all tests cases for a Vector Convert Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    block_height: Number of vertical elements processed by the ukernel.
    block_width: Number of horizontal elements processed by the ukernel.
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
          "BLOCK_HEIGHT": block_height,
          "BLOCK_WIDTH": block_width,
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
      block_height, block_width, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, block_height, block_width, isa)
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
