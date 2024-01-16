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
                    choices=[
                    "VCMulMicrokernelTester",
                    "VBinaryMicrokernelTester", "VBinaryCMicrokernelTester"],
                    help="Tester class to be used in the generated test")
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(qu8|qs8|f16|f32)_v(add|cmul|div|max|min|mul|sqrdiff|sub|addc|divc|rdivc|maxc|minc|mulc|sqrdiffc|subc|rsubc)(_(minmax|relu)(_(fp32|rndnu))?)?_ukernel__(.+)_u(\d+)(v)?", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  op_type = {
    "add": "Add",
    "cmul": "CMul",
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
  batch_tile = int(match.group(8))
  vector_tile = bool(match.group(9))
  activation_type = match.group(4)
  if activation_type is None:
    activation_type = "LINEAR"
  else:
    activation_type = activation_type.upper()

  requantization_type = match.group(6)
  if not requantization_type:
    requantization_type = None
  else:
    requantization_type = requantization_type.upper()

  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(7))
  return op_type, activation_type, requantization_type, batch_tile, vector_tile, arch, isa


BINOP_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, batch_eq_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  ${TESTER}()
    .batch_size(${BATCH_TILE}${BATCH_SCALE})
    .Test(${", ".join(TEST_ARGS)});
}

$if BATCH_TILE > 1:
  TEST(${TEST_NAME}, batch_div_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = ${BATCH_TILE*2}; batch_size < ${BATCH_TILE*10}; batch_size += ${BATCH_TILE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = ${BATCH_TILE*2}${BATCH_SCALE};
                  batch_size < ${BATCH_TILE*10}${BATCH_SCALE};
                  batch_size += ${BATCH_TILE}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, batch_lt_${BATCH_TILE}${BATCH_SUFFIX}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size < ${BATCH_TILE}; batch_size++) {
        ${TESTER}()
          .batch_size(batch_size)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1${BATCH_SCALE};
                  batch_size < ${BATCH_TILE}${BATCH_SCALE};
                  batch_size++) {
        ${TESTER}()
          .batch_size(batch_size)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

TEST(${TEST_NAME}, batch_gt_${BATCH_TILE}${BATCH_SUFFIX}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  $if BATCH_SCALE == "":
    for (size_t batch_size = ${BATCH_TILE+1}; batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}; batch_size++) {
      ${TESTER}()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
  $else:
    for (size_t batch_size = ${BATCH_TILE+1}${BATCH_SCALE};
                batch_size < ${10 if BATCH_TILE == 1 else BATCH_TILE*2}${BATCH_SCALE};
                batch_size += ${BATCH_TILE*2}) {
      ${TESTER}()
        .batch_size(batch_size)
        .Test(${", ".join(TEST_ARGS)});
    }
}

$if TESTER in ["VMulCMicrokernelTester", "VBinaryCMicrokernelTester"]:
  TEST(${TEST_NAME}, inplace) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace(true)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_a(true)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_a(true)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, inplace_b) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_b(true)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_b(true)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, inplace_a_and_b) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_a(true)
          .inplace_b(true)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .inplace_a(true)
          .inplace_b(true)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

$if DATATYPE.startswith("Q"):
  TEST(${TEST_NAME}, a_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
          ${TESTER}()
            .batch_size(batch_size)
            .a_zero_point(a_zero_point)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
          ${TESTER}()
            .batch_size(batch_size)
            .b_zero_point(b_zero_point)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
          ${TESTER}()
            .batch_size(batch_size)
            .y_zero_point(y_zero_point)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
          ${TESTER}()
            .batch_size(batch_size)
            .a_scale(a_scale)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
          ${TESTER}()
            .batch_size(batch_size)
            .b_scale(b_scale)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
          ${TESTER}()
            .batch_size(batch_size)
            .y_scale(y_scale)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
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
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }

  TEST(${TEST_NAME}, qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    $if BATCH_SCALE == "":
      for (size_t batch_size = 1; batch_size <= ${BATCH_TILE*5}; batch_size += ${max(1, BATCH_TILE-1)}) {
        ${TESTER}()
          .batch_size(batch_size)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    $else:
      for (size_t batch_size = 1;
                  batch_size <= ${BATCH_TILE*5}${BATCH_SCALE};
                  batch_size += ${max(1, BATCH_TILE-1)}${BATCH_SCALE}) {
        ${TESTER}()
          .batch_size(batch_size)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
  }
"""


def generate_test_cases(ukernel, op_type, init_fn, activation_type,
                        requantization_type, tester, batch_tile, vector_tile, isa):
  """Generates all tests cases for a Vector Binary Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    op_type: Operation type (ADD/MUL/SUB/etc).
    init_fn: C name of the function to initialize microkernel parameters.
    activation_type: Activation type (LINEAR/MINMAX/RELU).
    requantization_type: Requantization type (FP32/RNDNU).
    tester: C++ name of the tester class.
    batch_tile: Number of batch elements processed per one iteration of the
                inner loop of the micro-kernel.
    vector_tile: Indicates if batch tile is specified in vectors rather than
                 elements.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  test_args = [ukernel]
  if tester in ["VBinaryMicrokernelTester", "VBinaryCMicrokernelTester"] and not datatype in ['qs8', 'qu8']:
    test_args.append("%s::OpType::%s" % (tester, op_type))
  if init_fn:
    test_args.append(init_fn)
    if requantization_type:
      test_args.append("xnn_%s_requantize_%s" % \
        (datatype.lower(), requantization_type.lower()))
  batch_scale = ""
  if vector_tile:
    ctype = {"qs8": "int8_t", "qu8": "uint8_t", "f16": "uint16_t", "f32": "float"}[datatype]
    batch_scale = {
      "rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
      "rvvfp16arith": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype,
    }[isa]
  return xngen.preprocess(BINOP_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "TESTER": tester,
      "DATATYPE": datatype.upper(),
      "BATCH_TILE": batch_tile,
      "BATCH_SCALE": batch_scale,
      "BATCH_SUFFIX": "v" if vector_tile else "",
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
    microkernel_header = "xnnpack/vbinary.h"
    tester_header = {
      "VCMulMicrokernelTester": "vcmul-microkernel-tester.h",
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

#include <xnnpack/microparams-init.h>
#include <{microkernel_header}>
#include "{tester_header}"
""".format(specification=options.spec, generator=sys.argv[0],
           microkernel_header=microkernel_header, tester_header=tester_header)

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      op_type, activation_type, requantization_type, batch_tile, vector_tile, arch, isa = \
        split_ukernel_name(name)

      test_case = generate_test_cases(name, op_type, init_fn, activation_type,
                                      requantization_type, options.tester,
                                      batch_tile, vector_tile, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
