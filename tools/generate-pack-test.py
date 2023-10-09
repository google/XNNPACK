#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import codecs
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
  param_spec = common_parts[-1].split("x")
  mr = int(param_spec[0])
  arch, isa, assembly = xnncommon.parse_target_name(target_name)
  return mr, arch, isa


PACK_TEST_CODE = """\
TEST(${TEST_NAME}, k_eq_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PackMicrokernelTester()
    .mr(${MR})
    .m(${MR})
    .k(${KBLOCK})
    .Test(${UKERNEL_NAME});
}

TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m <= ${MR}; m++) {
    PackMicrokernelTester()
      .mr(${MR})
      .m(m)
      .k(${KBLOCK})
      .Test(${UKERNEL_NAME});
  }
}

$if KBLOCK != 1:
  TEST(${TEST_NAME}, k_lt_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${KBLOCK}; k++) {
      PackMicrokernelTester()
        .mr(${MR})
        .m(${MR})
        .k(k)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, k_lt_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${KBLOCK}; k++) {
      for (size_t m = 1; m <= ${MR}; m++) {
        PackMicrokernelTester()
          .mr(${MR})
          .m(m)
          .k(k)
          .Test(${UKERNEL_NAME});
      }
    }
  }

TEST(${TEST_NAME}, k_gt_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${KBLOCK + 1}; k < ${10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
    PackMicrokernelTester()
      .mr(${MR})
      .m(${MR})
      .k(k)
      .Test(${UKERNEL_NAME});
  }
}

TEST(${TEST_NAME}, k_gt_${KBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${KBLOCK + 1}; k < ${10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
    for (size_t m = 1; m <= ${MR}; m++) {
      PackMicrokernelTester()
        .mr(${MR})
        .m(m)
        .k(k)
        .Test(${UKERNEL_NAME});
    }
  }
}

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_div_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${KBLOCK * 2}; k < ${KBLOCK * 10}; k += ${KBLOCK}) {
      PackMicrokernelTester()
        .mr(${MR})
        .m(${MR})
        .k(k)
        .Test(${UKERNEL_NAME});
    }
  }

  TEST(${TEST_NAME}, k_div_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${KBLOCK * 2}; k < ${KBLOCK * 10}; k += ${KBLOCK}) {
      for (size_t m = 1; m <= ${MR}; m++) {
        PackMicrokernelTester()
          .mr(${MR})
          .m(m)
          .k(k)
          .Test(${UKERNEL_NAME});
      }
    }
  }

TEST(${TEST_NAME}, strided_x) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
    PackMicrokernelTester()
      .mr(${MR})
      .m(${MR})
      .k(k)
      .x_stride(${next_prime(KBLOCK * 5 + 1)})
      .Test(${UKERNEL_NAME});
  }
}
"""


def generate_test_cases(ukernel, mr, k_block, isa):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    mr: MR parameter of the PACK micro-kernel.
    k_block: Number of K values processed per one iteration of the main loop of
             the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  return xngen.preprocess(PACK_TEST_CODE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "UKERNEL_TYPE": ukernel_type.upper(),
      "UKERNEL_NAME": ukernel,
      "DATATYPE": datatype,
      "MR": mr,
      "KBLOCK": k_block,
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

#include <xnnpack/packx.h>
#include "pack-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      k_block = int(ukernel_spec["k-block"])
      mr, arch, isa = split_ukernel_name(name)

      test_case = generate_test_cases(name, mr, k_block, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)


if __name__ == "__main__":
  main(sys.argv[1:])
