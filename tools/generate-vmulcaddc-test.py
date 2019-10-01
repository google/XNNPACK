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
  match = re.match(r"^xnn_(f16|f32)_vmulcaddc_ukernel_c(\d+)__(.+)$", name)
  assert match is not None
  cr = int(match.group(2))

  arch = list()
  isa = None
  target_name = match.group(3)
  for target_part in target_name.split("_"):
    if target_part in ARCH_TO_MACRO_MAP:
      arch = [target_part]
    elif target_part in ISA_TO_ARCH_MAP:
      isa = target_part
  if isa and not arch:
    arch = ISA_TO_ARCH_MAP[isa]
  return cr, arch, isa


VMULCADDC_TEST_CODE = """\
TEST(${TEST_NAME}, c_eq_${CBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  VMulCAddCMicrokernelTester()
    .cr(${CR})
    .c(${CBLOCK})
    .m(${MBLOCK})
    .Test(${", ".join(TEST_ARGS)});
}

$if CBLOCK > 1:
  TEST(${TEST_NAME}, c_div_${CBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t c = ${CBLOCK * 2}; c < ${CBLOCK * 16}; c += ${CBLOCK * 3}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(${MBLOCK})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, c_gt_${CBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t c = ${CBLOCK}; c < ${10 if CBLOCK == 1 else CBLOCK * 2}; c++) {
    VMulCAddCMicrokernelTester()
      .cr(${CR})
      .c(c)
      .m(${MBLOCK})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if CBLOCK > 1:
  TEST(${TEST_NAME}, c_lt_${CBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t c = 1; c < ${CBLOCK}; c++) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(${MBLOCK})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK}; m++) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, multitile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = ${MBLOCK + 1}; m < ${MBLOCK * 4}; m++) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, x_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK * 3}; m += ${max(1, MBLOCK - 1)}) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .x_stride(${next_prime(CBLOCK * 5 + 1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, y_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK * 3}; m += ${max(1, MBLOCK - 1)}) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .y_stride(${next_prime(CBLOCK * 5 + 1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, inplace) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK * 3}; m += ${max(1, MBLOCK - 1)}) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .inplace(true)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK * 3}; m += ${max(1, MBLOCK - 1)}) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t m = 1; m < ${MBLOCK * 3}; m += ${max(1, MBLOCK - 1)}) {
    for (size_t c = 1; c <= ${CBLOCK * 5}; c += ${max(1, CBLOCK - 1)}) {
      VMulCAddCMicrokernelTester()
        .cr(${CR})
        .c(c)
        .m(m)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def generate_test_cases(ukernel, cr, c_block, m_block, isa):
  """Generates all tests cases for a VMULCADDC micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    cr: CR parameter of the DWCONV micro-kernel.
    c_block: Number of C values processed per one iteration of the inner loop of
             the micro-kernel.
    m_block: Number of M values processed per one iteration of the outer loop of
             the micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel]
  if not isa or isa == "psimd":
    test_args.append("VMulCAddCMicrokernelTester::Variant::Scalar")
  return xngen.preprocess(VMULCADDC_TEST_CODE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "DATATYPE": datatype,
      "CR": cr,
      "CBLOCK": c_block,
      "MBLOCK": m_block,
      "ISA_CHECK": ISA_TO_CHECK_MAP.get(isa, ""),
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
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/vmulcaddc.h>
#include <xnnpack/isa-checks.h>

#include "vmulcaddc-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      m_block = int(ukernel_spec["m-block"])
      cr, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, cr, cr, m_block, isa)
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
