#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import codecs
import collections
import os
import sys
import yaml
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primes import next_prime
import xngen
import xnncommon


parser = argparse.ArgumentParser(description='XNNPACK generator')
parser.add_argument("-s", "--spec", metavar="FILE", required=True,
                    help="Spec (YAML) file")
parser.add_argument(
    "-o",
    "--output-test",
    action="append",
    metavar="FILE",
    required=True,
    help="Test output (C++ source) file(s)")
parser.add_argument( "-b", "--output-bench",
                    metavar="FILE", required=False,
                    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  param_spec = common_parts[-1]
  mr, nr = map(int, param_spec.split("x"))
  arch, isa, assembly = xnncommon.parse_target_name(target_name)
  return mr, nr, arch, isa

SPMM_BENCH_CODE = """\
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  f32_spmm(state, ${SPMM}, ${MR}, ${NR},
    /*sparsity=*/0.8f, ${INIT_PARAMS},
  $if ISA_CHECK:
    benchmark::utils::${ISA_CHECK}
  $else:
    /*isa_check=*/nullptr
  );
}\n
BENCHMARK_SPMM(${UKERNEL_NAME})
"""

TEST_TEMPLATE = """\
TEST(${TEST_NAME}, k_eq_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  SpMMMicrokernelTester()
    .mr(${MR})
    .nr(${NR})
    .m(${MR})
    .n(${NR})
    .k(${KBLOCK})
    .sparsity(0.0f)
    .Test(${", ".join(TEST_ARGS)});
}

$if NR > 1:
  TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = 1; n <= ${NR}; n++) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR})
        .n(n)
        .k(${KBLOCK})
        .sparsity(0.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if IS_PIPELINED:
  TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    SpMMMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK * 2})
      .sparsity(0.0f)
      .Test(${", ".join(TEST_ARGS)});
  }

  $if NR > 1:
    TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (uint32_t n = 1; n <= ${NR}; n++) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(${MR})
          .n(n)
          .k(${KBLOCK * 2})
          .sparsity(0.0f)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .sparsity(0.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if NR > 1:
    TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
        for (uint32_t n = 1; n <= ${NR}; n++) {
          SpMMMicrokernelTester()
            .mr(${MR})
            .nr(${NR})
            .m(${MR})
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

TEST(${TEST_NAME}, k_gt_${ADJKBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${ADJKBLOCK + 1}; k < ${KBLOCK * 10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
    SpMMMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .m(${MR})
      .n(${NR})
      .k(k)
      .sparsity(0.0f)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if NR > 1:
  TEST(${TEST_NAME}, k_gt_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + 1}; k < ${10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(${MR})
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_div_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .sparsity(0.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if NR > 1:
    TEST(${TEST_NAME}, k_div_${KBLOCK}_subtile) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
        for (uint32_t n = 1; n <= ${NR}; n++) {
          SpMMMicrokernelTester()
            .mr(${MR})
            .nr(${NR})
            .m(${MR})
            .n(n)
            .k(k)
            .sparsity(0.0f)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }

TEST(${TEST_NAME}, n_gt_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${NR + 1}; n < ${max(10, NR * 2)}; n++) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR})
        .n(n)
        .k(k)
        .sparsity(0.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if NR > 1:
  TEST(${TEST_NAME}, n_div_${NR}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(${MR})
          .n(n)
          .k(k)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, m_lt_${MR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = ${1}; m < ${MR}; m++) {
    for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(m)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, m_div_${MR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = ${MR * 2}; m <= ${MR * 3}; m += ${MR}) {
    for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(m)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, m_gt_${MR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = ${MR + 1}; m < ${MR * 2}; m++) {
    for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        SpMMMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .m(m)
          .n(n)
          .k(k)
          .sparsity(0.0f)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, output_stride) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR * 2})
        .n(n)
        .k(k)
        .output_stride(${next_prime(MR * 2 + 1)})
        .sparsity(0.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmin) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR * 2})
        .n(n)
        .k(k)
        .sparsity(0.0f)
        .qmin(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, qmax) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR * 2})
        .n(n)
        .k(k)
        .sparsity(0.0f)
        .qmax(128)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, half_sparse) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR * 2})
        .n(n)
        .k(k)
        .sparsity(0.5f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, zero_weights) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n < ${max(10, NR * 5)}; n += ${NR + 1}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      SpMMMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .m(${MR * 2})
        .n(n)
        .k(k)
        .sparsity(1.0f)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}
"""


def generate_test_cases(ukernel, init_fn, mr, nr, k_block, is_pipelined, isa):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    k_block: Number of K values processed per one iteration of the main loop of
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
  _, ukernel_name = ukernel.split("_", 1)
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, _ = ukernel.split("_", 3)
  test_args = [ukernel, init_fn]
  test_case = xngen.preprocess(TEST_TEMPLATE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "MR": mr,
      "NR": nr,
      "KBLOCK": k_block,
      "ADJKBLOCK": 2 * k_block if is_pipelined else k_block,
      "IS_PIPELINED": is_pipelined,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
    })

  benchmark = xngen.preprocess(SPMM_BENCH_CODE, {
      "UKERNEL_NAME": ukernel_name,
      "SPMM": ukernel,
      "MR": mr,
      "NR": nr,
      "INIT_PARAMS": init_fn,
      "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
      "next_prime": next_prime,
    })
  return test_case, benchmark


def main(args):
  options = parser.parse_args(args)
  num_output_files = len(options.output_test)

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

#include <xnnpack/spmm.h>
#include "spmm-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    benches = """\
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}

#include <benchmark/benchmark.h>
#include "bench/spmm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
""".format(specification=options.spec, generator=sys.argv[0])

    test_outputs = collections.defaultdict(lambda: tests)
    bench_outputs = benches
    sorted_spec_yaml = collections.defaultdict(list)
    isa_hierarchy = xnncommon._ISA_HIERARCHY_MAP

    benches = [""] * len(isa_hierarchy)
    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec["init"]
      k_block = int(ukernel_spec["k-block"])
      pipelined = bool(ukernel_spec.get("pipelined", False))
      mr, nr, arch, isa = split_ukernel_name(name)

      test_case, bench_case = generate_test_cases(name, init_fn, mr, nr, k_block,
                                      pipelined, isa)
      # Hash the name of each microkernel and figure out which output file to
      # write it to.
      output_index = zlib.crc32(bytes(name, "utf-8")) % num_output_files
      test_outputs[options.output_test[output_index]] += "\n\n" + xnncommon.postprocess_test_case(
          test_case, arch, isa)
      benches[isa_hierarchy.get(isa, 0)] +=  "\n\n" + xnncommon.postprocess_test_case(bench_case, arch, isa)

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_outputs += benches[arch_idx]

    bench_outputs += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""
    for output_name in options.output_test:
      xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])

    if options.output_bench:
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_outputs)


if __name__ == "__main__":
  main(sys.argv[1:])
