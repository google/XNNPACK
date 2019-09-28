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


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.set_defaults(defines=list())


def indent(text):
  return "\n".join(map(lambda t: "  " + t if t else t, text.splitlines()))


def remove_duplicate_newlines(text):
  filtered_lines = list()
  last_newline = False
  for line in text.splitlines():
    is_newline = len(line.strip()) == 0
    if not is_newline or not last_newline:
      filtered_lines.append(line)
    last_newline = is_newline
  return "\n".join(filtered_lines)


ARCH_TO_MACRO_MAP = {
  "aarch32": "CPUINFO_ARCH_ARM",
  "aarch64": "CPUINFO_ARCH_ARM64",
  "x86": "CPUINFO_ARCH_X86",
  "x86-64": "CPUINFO_ARCH_X86_64",
}

ISA_TO_ARCH_MAP = {
  "neon": ["aarch32", "aarch64"],
  "neonfma": ["aarch32", "aarch64"],
  "neonfp16arith": ["aarch32", "aarch64"],
  "sse": ["x86", "x86-64"],
  "sse2": ["x86", "x86-64"],
  "avx": ["x86", "x86-64"],
  "avx512f": ["x86", "x86-64"],
  "psimd": [],
}

ISA_TO_CHECK_MAP = {
  "neon": "TEST_REQUIRES_ARM_NEON",
  "neonfma": "TEST_REQUIRES_ARM_NEON_FMA",
  "neonfp16arith": "TEST_REQUIRES_ARM_NEON_FP16_ARITH",
  "sse": "TEST_REQUIRES_X86_SSE",
  "sse2": "TEST_REQUIRES_X86_SSE2",
  "avx": "TEST_REQUIRES_X86_AVX",
  "avx512f": "TEST_REQUIRES_X86_AVX512F",
  "psimd": "TEST_REQUIRES_PSIMD",
}


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1].split("x")
  mr = int(param_spec[0])
  arch = list()
  isa = None
  for target_part in target_name.split("_"):
    if target_part in ARCH_TO_MACRO_MAP:
      arch = [target_part]
    elif target_part in ISA_TO_ARCH_MAP:
      isa = target_part
  if isa and not arch:
    arch = ISA_TO_ARCH_MAP[isa]
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
      "ISA_CHECK": ISA_TO_CHECK_MAP.get(isa, ""),
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


#include <cpuinfo.h>
#include "testing/base/public/gunit.h"

#include <xnnpack/packx.h>
#include <xnnpack/isa-checks.h>

#include "pack-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      k_block = int(ukernel_spec["k-block"])
      mr, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, mr, k_block, isa)
      test_case = remove_duplicate_newlines(test_case)
      tests += "\n\n"
      if arch:
        guard_macro = " || ".join(map(ARCH_TO_MACRO_MAP.get, arch))
        tests += "#if %s\n" % guard_macro
        tests += indent(test_case) + "\n"
        tests += "#endif  // %s\n" % guard_macro
      elif isa == "psimd":
        guard_macro = "!CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM"
        tests += "#if %s\n" % guard_macro
        tests += indent(test_case) + "\n"
        tests += "#endif  // %s\n" % guard_macro
      else:
        tests += test_case

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
