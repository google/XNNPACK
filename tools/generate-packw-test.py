#!/usr/bin/env python
# Copyright 2023 Google LLC
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


parser = argparse.ArgumentParser(description='PackW microkernel test generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Specification (YAML) file")
parser.add_argument("-o", "--output", metavar="FILE", required=True,
                    help='Output (C++ source) file')
parser.add_argument(
    "-b",
    "--output-bench",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  match = re.fullmatch(r"xnn_(x8|x16|x32)_packw_gemm_goi_ukernel_x(\d+)(c(\d+))?(s(\d+))?__(.+)_u(\d+)(_(.+))?", name)
  assert match is not None
  nr = int(match.group(2))
  if match.group(3):
    kr = int(match.group(4))
  else:
    kr = 1
  if match.group(5):
    sr = int(match.group(6))
  else:
    sr = 1
  kblock = int(match.group(8))
  arch, isa, assembly = xnncommon.parse_target_name(target_name=match.group(7))
  return nr, kr, sr, kblock, arch, isa

PACKW_BENCHMARK_TEMPLATE = """\
static void ${BENCHMARK_NAME}(benchmark::State& state, const char* net) {
  ${DATATYPE}_packw(state,
    ${UKERNEL_NAME},
    $if ISA_CHECK:
      /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR},
      benchmark::utils::${ISA_CHECK});
    $else:
      /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR});
}
BENCHMARK_BGEMM(${BENCHMARK_NAME})
"""

PACKW_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, k_eq_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  PackWMicrokernelTester()
    .n(${NR})
    .k(${KBLOCK})
    .nr(${NR})
    .kr(${KR})
    .sr(${SR})
    .Test(${", ".join(TEST_ARGS)});
}

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_div_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    PackWMicrokernelTester()
      .n(${NR})
      .k(${KBLOCK*5})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, k_lt_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${KBLOCK}; k++) {
      PackWMicrokernelTester()
        .n(${NR})
        .k(k)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, k_gt_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${KBLOCK+1}; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
    PackWMicrokernelTester()
      .n(${NR})
      .k(k)
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .Test(${", ".join(TEST_ARGS)});
  }
}

TEST(${TEST_NAME}, n_eq_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
    PackWMicrokernelTester()
      .n(${NR})
      .k(k)
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if NR > 1:
  TEST(${TEST_NAME}, n_div_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
      PackWMicrokernelTester()
        .n(${NR*2})
        .k(k)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, n_lt_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
      for (size_t n = 1; n < ${NR}; n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, n_gt_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
    for (size_t n = ${NR+1}; n < ${4 if NR == 1 else NR*2}; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, g_gt_1) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
      for (size_t n = ${NR+1}; n < ${4 if NR == 1 else NR*2}; n++) {
        PackWMicrokernelTester()
          .g(g)
          .n(n)
          .k(k)
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, null_bias) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < ${4 if KBLOCK == 1 else KBLOCK*2}; k++) {
      for (size_t n = ${NR+1}; n < ${4 if NR == 1 else NR*2}; n++) {
        PackWMicrokernelTester()
          .nullbias(true)
          .g(g)
          .n(n)
          .k(k)
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

"""


def generate_test_cases(ukernel, nr, kr, sr, kblock, isa):
  """Generates all tests cases for a PACKW micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    nr: NR parameter of the PACKW micro-kernel.
    kr: KR parameter of the PACKW micro-kernel.
    sr: SR parameter of the PACKW micro-kernel.
    kblock: unroll factor along the K dimension.
    isa: instruction set required to run the micro-kernel. Generated unit test
         will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_case = xngen.preprocess(PACKW_TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": [ukernel],
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "KBLOCK": kblock,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })

  benchmark = xngen.preprocess(PACKW_BENCHMARK_TEMPLATE, {
      "DATATYPE": datatype,
      "BENCHMARK_NAME": test_name,
      "UKERNEL_NAME": ukernel,
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "KBLOCK": kblock,
      "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
      "next_prime": next_prime,
    })

  return test_case, benchmark


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2023 Google LLC
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

#include <xnnpack/packw.h>
#include "packw-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    bench_output = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <benchmark/benchmark.h>
#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/common.h>
#include <xnnpack/packw.h>
""".format(specification=options.spec, generator=sys.argv[0])

    isa_hierarchy = xnncommon._ISA_HIERARCHY_MAP
    benches = [""] * len(isa_hierarchy)

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      nr, kr, sr, kblock, arch, isa = split_ukernel_name(name)

      test_case, benchmark = generate_test_cases(name, nr, kr, sr, kblock, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

      benches[isa_hierarchy.get(isa, 0)] += \
        "\n\n" + xnncommon.postprocess_test_case(benchmark, arch, isa)

    xnncommon.overwrite_if_changed(options.output, tests)

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_output += benches[arch_idx]

    bench_output += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

    if options.output_bench:
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_output)

if __name__ == "__main__":
  main(sys.argv[1:])
