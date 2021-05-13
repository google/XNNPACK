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
import xngen
import xnncommon


parser = argparse.ArgumentParser(
  description='Vector binary operation microkernel test generator')
parser.add_argument("-t", "--tester", metavar="TESTER", required=True,
                    choices=["VAddMicrokernelTester", "VAddCMicrokernelTester",
                    "VBinaryMicrokernelTester", "VBinaryCMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"^xnn_(qs8|f16|f32)_v(add|div|max|min|mul|sqrdiff|sub|addc|divc|rdivc|maxc|minc|mulc|sqrdiffc|subc|rsubc)(_(minmax|relu))?_ukernel__(.+)_x(\d+)$", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  op_type = {
    "add": "Add",
    "div": "Div",
    "max": "Max",
    "min": "Min",
    "mul": "Mul",
    "sqrdiff": "SqrDiff",
    "sub": "Sub",
    "addc": "AddC",
    "divc": "DivC",
    "rdivc": "RDivC",
    "maxc": "MaxC",
    "minc": "MinC",
    "mulc": "MulC",
    "sqrdiffc": "SqrDiffC",
    "subc": "SubC",
    "rsubc": "RSubC",
  }[match.group(2)]
  batch_tile = int(match.group(6))

  activation_type = match.group(4)
  if activation_type is None:
    activation_type = "LINEAR"
  else:
    activation_type = activation_type.upper()

  arch, isa = xnncommon.parse_target_name(target_name=match.group(5))
  return op_type, activation_type, batch_tile, arch, isa


BINOP_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  ${TESTER}()
    .batch_size(${BATCH_TILE})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = ${BATCH_TILE*2}; batch_size < ${BATCH_TILE*10}; batch_size += ${BATCH_TILE}) {
      ${TESTER}()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size < ${BATCH_TILE}; batch_size++) {
      ${TESTER}()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t batch_size = ${BATCH_TILE+1}; batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch_size++) {
    ${TESTER}()
      .batch_size(batch_size)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if TESTER in ["VAddCMicrokernelTester", "VBinaryCMicrokernelTester"]:
  TEST(${TEST_NAME}, inplace) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .inplace(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
$else:
  TEST(${TEST_NAME}, inplace_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, inplace_b) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, inplace_a_and_b) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if DATATYPE == "QS8":
  TEST(${TEST_NAME}, a_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        ${TESTER}()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, b_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        ${TESTER}()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, y_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        ${TESTER}()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, a_scale) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        ${TESTER}()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, b_scale) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        ${TESTER}()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, y_scale) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        ${TESTER}()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if ACTIVATION_TYPE == "MINMAX":
  TEST(${TEST_NAME}, qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
      ${TESTER}()
        .batch_size(batch_size)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
"""


def generate_test_cases(ukernel, op_type, activation_type, tester, batch_tile, isa):
  """Generates all tests cases for a Vector Binary Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    op_type: Operation type (ADD/MUL/SUB/etc).
    activation_type: Activation type (LINEAR/MINMAX/RELU).
    tester: C++ name of the tester class.
    batch_tile: Number of batch elements processed per one iteration of the
                inner loop of the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  test_args = [ukernel]
  if tester in ["VBinaryMicrokernelTester", "VBinaryCMicrokernelTester"]:
    test_args.append("%s::OpType::%s" % (tester, op_type))
  if not isa:
    test_args.append("%s::Variant::Scalar" % tester)
  return xngen.preprocess(BINOP_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "TESTER": tester,
      "DATATYPE": datatype.upper(),
      "BATCH_TILE": batch_tile,
      "OP_TYPE": op_type,
      "ACTIVATION_TYPE": activation_type,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    spec_name = os.path.splitext(os.path.split(options.spec)[1])[0]
    microkernel_header = {
      "VAddMicrokernelTester": "xnnpack/vadd.h",
      "VAddCMicrokernelTester": "xnnpack/vadd.h",
      "VBinaryMicrokernelTester": "xnnpack/vbinary.h",
      "VBinaryCMicrokernelTester": "xnnpack/vbinary.h",
    }[options.tester]
    tester_header = {
      "VAddMicrokernelTester": "vadd-microkernel-tester.h",
      "VAddCMicrokernelTester": "vaddc-microkernel-tester.h",
      "VBinaryMicrokernelTester": "vbinary-microkernel-tester.h",
      "VBinaryCMicrokernelTester": "vbinaryc-microkernel-tester.h",
    }[options.tester]
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

#include <{microkernel_header}>
#include "{tester_header}"
""".format(specification=options.spec, generator=sys.argv[0],
           microkernel_header=microkernel_header, tester_header=tester_header)

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      op_type, activation_type, batch_tile, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, op_type, activation_type,
                                      options.tester, batch_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
