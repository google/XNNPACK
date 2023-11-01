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

parser = argparse.ArgumentParser(description="XNNPACK generator")
parser.add_argument(
    "-s", "--spec", metavar="FILE", required=True, help="Spec (YAML) file")
parser.add_argument(
    "-o",
    "--output-test",
    action="append",
    metavar="FILE",
    required=True,
    help="Test output (C++ source) file(s)")
parser.add_argument(
    "-b",
    "--output-bench",
    metavar="FILE",
    required=False,
    help="Benchmark output (C++ source) file(s)")
parser.set_defaults(defines=list())

def split_ukernel_name(name):
  common_name, target_name = name.split("__", 1)
  common_parts = common_name.split("_")
  xw = "gemm_xw_" in common_name
  param_spec = common_parts[-1]
  if "s" in param_spec:
    param_spec, sr = param_spec.split("s", 1)
    sr = int(sr)
  else:
    sr = 1
  if "c" in param_spec:
    param_spec, kr = param_spec.split("c", 1)
    kr = int(kr)
  else:
    kr = 1
  mr, nr = map(int, param_spec.split("x"))
  arch, isa, assembly = xnncommon.parse_target_name(target_name)

  requantization = common_parts[-3]
  if requantization not in ["fp32", "rndnu"]:
    requantization = None

  return mr, nr, kr, sr, xw, requantization, arch, isa, assembly


GEMM_BENCH_CODE_XW = """\
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    ${GEMM},
    $if INIT_PARAMS is not None:
      ${INIT_PARAMS},
    $if PACK_FN is not None:
      ${PACK_FN},
    /*mr=*/${MR}, /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR},
    $if ISA_CHECK:
      benchmark::utils::${ISA_CHECK},
    $else:
      /*isa_check=*/nullptr,
    /*extended_weights=*/true);
}\n
BENCHMARK_GEMM(${UKERNEL_NAME})
"""

GEMM_BENCH_CODE = """\
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    ${GEMM},
    $if INIT_PARAMS is not None:
      ${INIT_PARAMS},
    $if PACK_FN is not None:
      ${PACK_FN},
    /*mr=*/${MR}, /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR},
    $if ISA_CHECK:
      benchmark::utils::${ISA_CHECK});
    $else:
      /*isa_check=*/nullptr);
}\n
BENCHMARK_GEMM(${UKERNEL_NAME})
"""

GEMM_TEST_CODE = """\
TEST(${TEST_NAME}, k_eq_${KBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  GemmMicrokernelTester()
    $if EXTENDED_WEIGHTS:
      .extended_weights(true)
    .mr(${MR})
    .nr(${NR})
    .kr(${KR})
    .sr(${SR})
    .m(${MR})
    .n(${NR})
    .k(${KBLOCK})
    $if KERNELTYPE == 'qc4w':
      .b_zero_point(8)
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, strided_cn) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  GemmMicrokernelTester()
    $if EXTENDED_WEIGHTS:
      .extended_weights(true)
    .mr(${MR})
    .nr(${NR})
    .kr(${KR})
    .sr(${SR})
    .m(${MR})
    .n(${NR})
    .k(${KBLOCK})
    .cn_stride(${next_prime(NR + 1)})
    $if KERNELTYPE == 'qc4w':
      .b_zero_point(8)
    .Test(${", ".join(TEST_ARGS)});
}

$if UKERNEL_TYPE != "IGEMM":
  TEST(${TEST_NAME}, k_eq_${KBLOCK}_strided_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK})
      .a_stride(${next_prime(KBLOCK + 1)})
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n <= ${NR}; n++) {
    for (uint32_t m = 1; m <= ${MR}; m++) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(m)
        .n(n)
        .k(${KBLOCK})
        .iterations(1)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile_m) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = 1; m <= ${MR}; m++) {
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(m)
      .n(${NR})
      .k(${KBLOCK})
      .iterations(1)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }
}


TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile_n) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n <= ${NR}; n++) {
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(n)
      .k(${KBLOCK})
      .iterations(1)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if IS_PIPELINED:
  TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK * 2})
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }

  $if UKERNEL_TYPE != "IGEMM":
    TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(${KBLOCK * 2})
        .a_stride(${next_prime(KBLOCK * 2 + 1)})
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }

  TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = 1; n <= ${NR}; n++) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(${KBLOCK * 2})
          .iterations(1)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if UKERNEL_TYPE != "IGEMM":
    TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .a_stride(${next_prime(ADJKBLOCK + 1)})
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        for (uint32_t m = 1; m <= ${MR}; m++) {
          GemmMicrokernelTester()
            $if EXTENDED_WEIGHTS:
              .extended_weights(true)
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            $if KERNELTYPE == 'qc4w':
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

TEST(${TEST_NAME}, k_gt_${ADJKBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${ADJKBLOCK + 1}; k < ${ADJKBLOCK * 10 if ADJKBLOCK == 1 else ADJKBLOCK * 2}; k++) {
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(k)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if UKERNEL_TYPE.startswith("GEMM"):
  TEST(${TEST_NAME}, k_gt_${ADJKBLOCK}_strided_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + 1}; k < ${10 if ADJKBLOCK == 1 else ADJKBLOCK * 2}; k++) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .a_stride(${next_prime(10 if ADJKBLOCK == 1 else ADJKBLOCK * 2 + 1)})
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, k_gt_${ADJKBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${ADJKBLOCK + 1}; k < ${10 if ADJKBLOCK == 1 else ADJKBLOCK * 2}; k++) {
    for (uint32_t n = 1; n <= ${NR}; n++) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

$if KBLOCK > 1:
  TEST(${TEST_NAME}, k_div_${KBLOCK}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if UKERNEL_TYPE.startswith("GEMM"):
    TEST(${TEST_NAME}, k_div_${KBLOCK}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .a_stride(${next_prime(KBLOCK * 10 + 1)})
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, k_div_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        for (uint32_t m = 1; m <= ${MR}; m++) {
          GemmMicrokernelTester()
            $if EXTENDED_WEIGHTS:
              .extended_weights(true)
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            $if KERNELTYPE == 'qc4w':
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

TEST(${TEST_NAME}, n_gt_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(n)
        .k(k)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if JIT:
  TEST(${TEST_NAME}, unknown_nc_mod_nr) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = 1; n < ${NR * 2}; n++) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .known_nc_mod_nr(false)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, relu) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK})
      .relu(true)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, n_gt_${NR}_strided_cn) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(n)
        .k(k)
        .cn_stride(${next_prime(NR + 1)})
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if UKERNEL_TYPE != "IGEMM":
  TEST(${TEST_NAME}, n_gt_${NR}_strided_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .a_stride(${next_prime(KBLOCK * 5 + 1)})
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, n_gt_${NR}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

TEST(${TEST_NAME}, n_div_${NR}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(n)
        .k(k)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, n_div_${NR}_strided_cn) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(n)
        .k(k)
        .cn_stride(${next_prime(NR + 1)})
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

$if UKERNEL_TYPE != "IGEMM":
  TEST(${TEST_NAME}, n_div_${NR}_strided_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .a_stride(${next_prime(KBLOCK * 5 + 1)})
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, n_div_${NR}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

$if UKERNEL_TYPE.startswith("IGEMM"):
  TEST(${TEST_NAME}, small_kernel) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .ks(3)
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, small_kernel_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        for (uint32_t m = 1; m <= ${MR}; m++) {
          GemmMicrokernelTester()
            $if EXTENDED_WEIGHTS:
              .extended_weights(true)
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
            $if KERNELTYPE == 'qc4w':
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

  TEST(${TEST_NAME}, n_gt_${NR}_small_kernel) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .ks(3)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

  TEST(${TEST_NAME}, n_div_${NR}_small_kernel) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t n = ${2 * NR}; n <= ${3 * NR}; n += ${NR}) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .ks(3)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, strided_cm_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
    for (uint32_t n = 1; n <= ${NR}; n++) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(${next_prime(NR + 1)})
          .iterations(1)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }
}

$if UKERNEL_TYPE.startswith("IGEMM"):
  TEST(${TEST_NAME}, a_offset) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .ks(3)
        .a_offset(${next_prime(MR * KBLOCK * 5 + 1)})
        $if KERNELTYPE == 'qc4w':
          .b_zero_point(8)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      for (uint32_t mz = 0; mz < ${MR}; mz++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .ks(3)
          .a_offset(${next_prime(MR * KBLOCK * 5 + 1)})
          .zero_index(mz)
          $if KERNELTYPE == 'qc4w':
            .b_zero_point(8)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if ACTIVATION == "MINMAX":
  TEST(${TEST_NAME}, qmin) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK})
      .qmin(128)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, qmax) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK})
      .qmax(128)
      $if KERNELTYPE == 'qc4w':
        .b_zero_point(8)
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, strided_cm) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  GemmMicrokernelTester()
    $if EXTENDED_WEIGHTS:
      .extended_weights(true)
    .mr(${MR})
    .nr(${NR})
    .kr(${KR})
    .sr(${SR})
    .m(${MR})
    .n(${NR})
    .k(${KBLOCK})
    .cm_stride(${next_prime(NR + 1)})
    $if KERNELTYPE == 'qc4w':
      .b_zero_point(8)
    .Test(${", ".join(TEST_ARGS)});
}

$if DATATYPE == "qu8":
  TEST(${TEST_NAME}, no_a_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .a_zero_point(0)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if DATATYPE == "qu8":
  TEST(${TEST_NAME}, no_b_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .b_zero_point(0)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, b_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint16_t b_zero_point = 0; b_zero_point <= 255; ++b_zero_point) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(${KBLOCK})
        .b_zero_point(b_zero_point)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, no_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        $if EXTENDED_WEIGHTS:
          .extended_weights(true)
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .a_zero_point(0)
        .b_zero_point(0)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16']:
  TEST(${TEST_NAME}, subtile_m_upto_mr) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t max_mr = 1; max_mr <= ${MR}; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= ${KBLOCK * 2}; k += 1) {
          GemmMicrokernelTester()
            $if EXTENDED_WEIGHTS:
              .extended_weights(true)
            .mr(max_mr)
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(${NR})
            .k(k)
            .iterations(1)
            $if KERNELTYPE == 'qc4w':
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

$if TEST_NAME.startswith('GENERATE') and DATATYPE == 'f32' and POST_OP:
  TEST(${TEST_NAME}, hardswish) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      $if EXTENDED_WEIGHTS:
        .extended_weights(true)
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK})
      .Test(
          ${", ".join(TEST_ARGS)},
          fused_operators);
  }
  $if MR > 1:
    TEST(${TEST_NAME}, hardswish_max_mr_lt_${MR}) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
      for (uint32_t max_mr = 1; max_mr < ${MR}; max_mr++) {
        GemmMicrokernelTester()
          $if EXTENDED_WEIGHTS:
            .extended_weights(true)
          .mr(max_mr)
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(max_mr)
          .n(${NR})
          .k(${KBLOCK})
          .Test(
              ${", ".join(TEST_ARGS)},
              fused_operators);
      }
    }

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16'] and PROTOTYPE is not None:
  #if XNN_ENABLE_ASSEMBLY
    TEST(${TEST_NAME}, matches_assembly) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      GemmMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(${KBLOCK})
        .Test(
            ${", ".join(TEST_ARGS)},
            &${PROTOTYPE});
    }
  #endif // XNN_ENABLE_ASSEMBLY

"""


def generate_test_cases(ukernel, mr, nr, kr, sr, xw, k_block, init_fn, pack_fn,
                        requantization, is_pipelined, isa, jit, prototype, post_op):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    kr: KR parameter of the GEMM micro-kernel.
    sr: SR parameter of the GEMM micro-kernel.
    xw: boolean indicator for microkernel with extended weights.
    k_block: Number of K values processed per one iteration of the main loop of
      the micro-kernel.
    init_fn: C name of the function to initialize microkernel parameters.
    pack_fn: C name of the function to pack the weights.
    requantization: name of the requantization scheme used by the microkernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
      pipelining. Additional test cases are generated for software pipelined
      micro-kernels to separately test prologue + epiloque of the pipelined loop
      and iteration of the pipelined loop.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.
    jit: if we are generating test code for JIT codegen.
    post_op: if post operation is supported (only for JIT).

  Returns:
    Code for the test case.
  """
  _, ukernel_name = ukernel.split("_", 1)

  if jit:
    _, _, datatype, ukernel_type, _ = ukernel.split("_", 4)
    kerneltype = datatype
    activation = None
  else:
    _, datatype, ukernel_type, activation, _ = ukernel.split("_", 4)
    kerneltype = datatype
    if datatype in ["f16", "f32"] and ukernel_type in ["qc8w", "qc4w"]:
      _, datatype, kerneltype, ukernel_type, activation, _ = ukernel.split("_", 5)
      datatype = datatype + "_" + kerneltype
    if datatype == "qd8" and ukernel_type in ["f16", "f32"] and activation in ["qc8w", "qc4w"]:
      _, datatype, _, kerneltype, ukernel_type, activation, _ = ukernel.split("_", 6)

  if activation == "ukernel":
    activation = "linear"
  if activation in ["qs8w"]:
    _, _, _, _, _, activation, _ = ukernel.split("_", 6)
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)

  if pack_fn:
    test_args.append(pack_fn)

  if init_fn and requantization:
    requantization_datatype = {"qc8": "qs8"}.get(datatype, datatype)
    test_args.append("xnn_%s_requantize_%s" % (requantization_datatype, requantization))

  if jit:
    if "minmax" in init_fn:
      activation = "minmax"

  test_case = xngen.preprocess(
      GEMM_TEST_CODE, {
          "TEST_NAME": ukernel_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": test_args,
          "UKERNEL_TYPE": ukernel_type.upper(),
          "DATATYPE": datatype,
          "KERNELTYPE": kerneltype,
          "ACTIVATION": activation.upper(),
          "MR": mr,
          "NR": nr,
          "KR": kr,
          "SR": sr,
          "EXTENDED_WEIGHTS": xw,
          "KBLOCK": k_block,
          "ADJKBLOCK": 2 * k_block if is_pipelined else k_block,
          "IS_PIPELINED": is_pipelined,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
          "next_prime": next_prime,
          "POST_OP": post_op,
          "PROTOTYPE": prototype,
          "JIT": jit,
      })

  benchmark = xngen.preprocess(
      GEMM_BENCH_CODE_XW if xw else GEMM_BENCH_CODE, {
          "UKERNEL_NAME": ukernel_name,
          "GEMM": ukernel,
          "INIT_PARAMS": init_fn,
          "PACK_FN": pack_fn,
          "MR": mr,
          "NR": nr,
          "KR": kr,
          "SR": sr,
          "EXTENDED_WEIGHTS": xw,
          "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
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

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"
""".format(
    specification=options.spec, generator=sys.argv[0])

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
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/isa-checks.h>
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
      k_block = int(ukernel_spec["k-block"])
      init_fn = ukernel_spec.get("init")
      pack_fn = ukernel_spec.get("pack")
      pipelined = bool(ukernel_spec.get("pipelined", False))
      jit = name.startswith("xnn_generate")
      prototype = ukernel_spec.get("prototype")
      post_op = ukernel_spec.get("post-op", True)
      mr, nr, kr, sr, xw, requantization, arch, isa, assembly = \
        split_ukernel_name(name)

      test_case, bench_case = generate_test_cases(name, mr, nr, kr, sr, xw, k_block,
                                      init_fn, pack_fn, requantization, pipelined, isa,
                                      jit, prototype, post_op)

      # Hash the name of each microkernel and figure out which output file to
      # write it to.
      output_index = zlib.crc32(bytes(name, "utf-8")) % num_output_files
      test_outputs[options.output_test[output_index]] += "\n\n" + xnncommon.postprocess_test_case(
                  test_case, arch, isa, assembly, jit)
      benches[isa_hierarchy.get(isa, 0)] +=  "\n\n" + xnncommon.postprocess_test_case(bench_case, arch, isa, assembly, jit)

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
