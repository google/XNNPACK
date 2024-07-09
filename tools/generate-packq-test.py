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
  match = re.fullmatch(
      r"xnn_(x8|x16|x32)_packq_(f32|f16)qp8_ukernel__(.+)_u(\d+)", name
  )
  assert match is not None
  unroll = int(match.group(4))
  arch, isa, _ = xnncommon.parse_target_name(target_name=match.group(3))
  return arch, isa, unroll

PACKQ_BENCHMARK_TEMPLATE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
$for MR in (1, 2, 4):
  $for KR in (1, 2, 4):
    static void ${BENCHMARK_NAME}_mr_${MR}_kr_${KR}(
        benchmark::State& state, const char* net) {
      ${DATATYPE}_packq(state,
        ${UKERNEL_NAME},
        $if ISA_CHECK:
          /*mr=*/${MR}, /*kr=*/${KR}, /*sr=*/1,
          benchmark::utils::${ISA_CHECK});
        $else:
          /*mr=*/${MR}, /*kr=*/${KR}, /*sr=*/1);
    }
    BENCHMARK_BGEMM(${BENCHMARK_NAME}_mr_${MR}_kr_${KR})
$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""

PACKQ_TEST_TEMPLATE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
TEST(${TEST_NAME}, k_div_kr_m_div_mr) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 1; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr * ${UNROLL * 10})
          .k(kr * ${UNROLL * 10})
          .mr(mr)
          .kr(kr)
          .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, k_div_kr_m_div_mr_kr_div_sr) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t sr = 1; sr <= 4; sr++) {
    for (size_t kr = sr; kr <= 4 * sr; kr += sr) {
      for (size_t mr = 1; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(mr * ${UNROLL * 10})
            .k(kr * ${UNROLL * 10})
            .mr(mr)
            .kr(kr)
            .sr(sr)
            .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, k_div_kr_m_lt_mr) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr - 1)
          .k(kr * ${UNROLL * 10})
          .mr(mr)
          .kr(kr)
          .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, k_div_kr_m_gt_mr) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(2 * mr + 1)
          .k(kr * ${UNROLL * 10})
          .mr(mr)
          .kr(kr)
          .Test(${", ".join(TEST_ARGS)});
    }
  }
}
$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""


def generate_cases(ukernel, cpp_check,isa, unroll):
  """Generates all tests cases for a PACKQ micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    cpp_check: Optional preprocessor macro to check for the availability of the
      micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.
    unroll: The number of inputs processed per step.

  Returns:
    Code for the test and benchmark cases.
  """
  _, test_name = ukernel.split("_", 1)
  _, datatype, _ = ukernel.split("_", 2)
  test_case = xngen.preprocess(
      PACKQ_TEST_TEMPLATE,
      {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": [ukernel],
          "UNROLL": unroll,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "next_prime": next_prime,
          "CPP_CHECK": cpp_check,
      },
  )

  benchmark = xngen.preprocess(
      PACKQ_BENCHMARK_TEMPLATE,
      {
          "DATATYPE": datatype,
          "BENCHMARK_NAME": test_name,
          "UKERNEL_NAME": ukernel,
          "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
          "next_prime": next_prime,
          "CPP_CHECK": cpp_check,
      },
  )

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


#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packq.h"
#include "packq-microkernel-tester.h"


namespace xnnpack {{""".format(
        specification=options.spec, generator=sys.argv[0]
    )

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
#include "bench/packq-benchmark.h"
#include "xnnpack/common.h"
#include "xnnpack/packq.h"
""".format(specification=options.spec, generator=sys.argv[0])

    isa_hierarchy = xnncommon._ISA_HIERARCHY_MAP
    benches = [""] * len(isa_hierarchy)

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      cpp_check = ukernel_spec.get("cpp-check")
      arch, isa, unroll = split_ukernel_name(name)

      test_case, benchmark = generate_cases(name, cpp_check,isa, unroll)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

      benches[isa_hierarchy.get(isa, 0)] += \
        "\n\n" + xnncommon.postprocess_test_case(benchmark, arch, isa)

    tests += "\n\n};  // namespace xnnpack\n"

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
