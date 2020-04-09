#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import codecs
import math
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]
  assert param_spec.startswith("up")
  cr, kr = map(int, param_spec[2:].split("x"))
  arch, isa = xnncommon.parse_target_name(target_name)
  return cr, kr, arch, isa


DWCONV_TEST_CODE = """\
TEST(${TEST_NAME}, c_eq_${CBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  DWConvMicrokernelTester()
    .cr(${CR})
    .kr(${KR})
    .channels(${CBLOCK})
    .Test(${", ".join(TEST_ARGS)});
}

$if IS_PIPELINED:
  TEST(${TEST_NAME}, c_eq_${CBLOCK * 2}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    DWConvMicrokernelTester()
      .cr(${CR})
      .kr(${KR})
      .channels(${CBLOCK * 2})
      .Test(${", ".join(TEST_ARGS)});
  }

$if CBLOCK > 1:
  TEST(${TEST_NAME}, c_div_${CBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t channels = ${ADJCBLOCK + CBLOCK}; channels < ${CR * 16}; channels += ${CR * 3}) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if ACTIVATION == "MINMAX":
    TEST(${TEST_NAME}, c_div_${CBLOCK}_with_qmin) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (uint32_t channels = ${ADJCBLOCK + CBLOCK}; channels < ${CR * 16}; channels += ${CR * 3}) {
        DWConvMicrokernelTester()
          .cr(${CR})
          .kr(${KR})
          .channels(channels)
          .qmin(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

    TEST(${TEST_NAME}, c_div_${CBLOCK}_with_qmax) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (uint32_t channels = ${ADJCBLOCK + CBLOCK}; channels < ${CR * 16}; channels += ${CR * 3}) {
        DWConvMicrokernelTester()
          .cr(${CR})
          .kr(${KR})
          .channels(channels)
          .qmax(128)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, c_lt_${ADJCBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t channels = 1; channels < ${ADJCBLOCK}; channels++) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, c_gt_${ADJCBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t channels = ${ADJCBLOCK + 1}; channels < ${10 if CBLOCK == 1 else ADJCBLOCK + CBLOCK}; channels++) {
    DWConvMicrokernelTester()
      .cr(${CR})
      .kr(${KR})
      .channels(channels)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ACTIVATION == "MINMAX":
  TEST(${TEST_NAME}, c_gt_${ADJCBLOCK}_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t channels = ${ADJCBLOCK + 1}; channels < ${10 if CBLOCK == 1 else ADJCBLOCK + CBLOCK}; channels++) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, c_gt_${ADJCBLOCK}_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t channels = ${ADJCBLOCK + 1}; channels < ${10 if CBLOCK == 1 else ADJCBLOCK + CBLOCK}; channels++) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, multipixel) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
    DWConvMicrokernelTester()
      .cr(${CR})
      .kr(${KR})
      .channels(channels)
      .width(3)
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, multipixel_with_step) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
    for (size_t step = 2; step <= ${KR}; step++) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .width(3)
        .step(step)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, multipixel_with_output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
    DWConvMicrokernelTester()
      .cr(${CR})
      .kr(${KR})
      .channels(${CR})
      .width(5)
      .output_stride(${next_prime(CR * 5 + 1)})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if ACTIVATION == "MINMAX":
  TEST(${TEST_NAME}, multipixel_with_qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, multipixel_with_qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if DATATYPE == "q8":
  TEST(${TEST_NAME}, input_zero_point_only) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .width(3)
        .input_zero_point(255)
        .kernel_zero_point(0)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, kernel_zero_point_only) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t channels = 1; channels <= ${CBLOCK * 5}; channels += ${max(1, CBLOCK - 1)}) {
      DWConvMicrokernelTester()
        .cr(${CR})
        .kr(${KR})
        .channels(channels)
        .width(3)
        .input_zero_point(0)
        .kernel_zero_point(255)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
"""


def generate_test_cases(ukernel, cr, kr, c_block, is_pipelined, isa):
  """Generates all tests cases for a DWCONV micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    cr: CR parameter of the DWCONV micro-kernel.
    kr: KR parameter of the DWCONV micro-kernel.
    k_block: Number of C values processed per one iteration of the main loop of
             the micro-kernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
                  pipelining. Additional test cases are generated for software
                  pipelined micro-kernels to separately test prologue + epiloque
                  of the pipelined loop and iteration of the pipelined loop.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, activation, _ = ukernel.split("_", 4)
  if activation == "ukernel":
    activation = "linear"
  test_args = [ukernel]
  if activation != "linear" and (not isa or isa == "psimd"):
    test_args.append("DWConvMicrokernelTester::Variant::Scalar")
  return xngen.preprocess(DWCONV_TEST_CODE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "ACTIVATION": activation.upper(),
      "CR": cr,
      "KR": kr,
      "CBLOCK": c_block,
      "ADJCBLOCK": 2 * c_block if is_pipelined else c_block,
      "IS_PIPELINED": is_pipelined,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
      "sqrt": math.sqrt,
    })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      pipelined = bool(ukernel_spec.get("pipelined", False))
      assembly = bool(ukernel_spec.get("assembly", False))
      cr, kr, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, cr, kr, cr, pipelined, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa, assembly)

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
