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
  arch, isa = xnncommon.parse_target_name(target_name)
  return mr, nr, kr, sr, arch, isa


GEMM_TEST_CODE = """\
TEST(${TEST_NAME}, k_eq_${KBLOCK}) {
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
    .Test(${", ".join(TEST_ARGS)});
}

TEST(${TEST_NAME}, strided_cn) {
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
    .cn_stride(${next_prime(NR + 1)})
    .Test(${", ".join(TEST_ARGS)});
}

$if UKERNEL_TYPE != "IGEMM":
  TEST(${TEST_NAME}, k_eq_${KBLOCK}_strided_a) {
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
      .a_stride(${next_prime(KBLOCK + 1)})
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = 1; m <= ${MR}; m++) {
    for (uint32_t n = 1; n <= ${NR}; n++) {
      GemmMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(m)
        .n(n)
        .k(${KBLOCK})
        .iterations(1)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile_m) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t m = 1; m <= ${MR}; m++) {
    GemmMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(m)
      .n(${NR})
      .k(${KBLOCK})
      .iterations(1)
      .Test(${", ".join(TEST_ARGS)});
  }
}


TEST(${TEST_NAME}, k_eq_${KBLOCK}_subtile_n) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = 1; n <= ${NR}; n++) {
    GemmMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(n)
      .k(${KBLOCK})
      .iterations(1)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if IS_PIPELINED:
  TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    GemmMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(${KBLOCK * 2})
      .Test(${", ".join(TEST_ARGS)});
  }

  $if UKERNEL_TYPE != "IGEMM":
    TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      GemmMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(${KBLOCK * 2})
        .a_stride(${next_prime(KBLOCK * 2 + 1)})
        .Test(${", ".join(TEST_ARGS)});
    }

  TEST(${TEST_NAME}, k_eq_${KBLOCK * 2}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t m = 1; m <= ${MR}; m++) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        GemmMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(${KBLOCK * 2})
          .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if UKERNEL_TYPE != "IGEMM":
    TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
        GemmMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .a_stride(${next_prime(ADJKBLOCK + 1)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, k_lt_${ADJKBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k < ${ADJKBLOCK}; k++) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        for (uint32_t n = 1; n <= ${NR}; n++) {
          GemmMicrokernelTester()
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

TEST(${TEST_NAME}, k_gt_${ADJKBLOCK}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${ADJKBLOCK + 1}; k < ${KBLOCK * 10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
    GemmMicrokernelTester()
      .mr(${MR})
      .nr(${NR})
      .kr(${KR})
      .sr(${SR})
      .m(${MR})
      .n(${NR})
      .k(k)
      .Test(${", ".join(TEST_ARGS)});
  }
}

$if UKERNEL_TYPE.startswith("GEMM"):
  TEST(${TEST_NAME}, k_gt_${KBLOCK}_strided_a) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + 1}; k < ${10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
      GemmMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .a_stride(${next_prime(10 if KBLOCK == 1 else KBLOCK * 2 + 1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

TEST(${TEST_NAME}, k_gt_${KBLOCK}_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = ${ADJKBLOCK + 1}; k < ${10 if KBLOCK == 1 else KBLOCK * 2}; k++) {
    for (uint32_t m = 1; m <= ${MR}; m++) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        GemmMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  $if UKERNEL_TYPE.startswith("GEMM"):
    TEST(${TEST_NAME}, k_div_${KBLOCK}_strided_a) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
        GemmMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .a_stride(${next_prime(KBLOCK * 10 + 1)})
          .Test(${", ".join(TEST_ARGS)});
      }
    }

  TEST(${TEST_NAME}, k_div_${KBLOCK}_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = ${ADJKBLOCK + KBLOCK}; k <= ${KBLOCK * 10}; k += ${KBLOCK}) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        for (uint32_t n = 1; n <= ${NR}; n++) {
          GemmMicrokernelTester()
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .Test(${", ".join(TEST_ARGS)});
    }
  }
}

TEST(${TEST_NAME}, n_gt_${NR}_strided_cn) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (uint32_t n = ${NR + 1}; n < ${NR * 2}; n++) {
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .cn_stride(${next_prime(NR + 1)})
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .a_stride(${next_prime(KBLOCK * 5 + 1)})
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(n)
        .k(k)
        .cn_stride(${next_prime(NR + 1)})
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(n)
          .k(k)
          .a_stride(${next_prime(KBLOCK * 5 + 1)})
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .ks(3)
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, small_kernel_subtile) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      for (uint32_t m = 1; m <= ${MR}; m++) {
        for (uint32_t n = 1; n <= ${NR}; n++) {
          GemmMicrokernelTester()
            .mr(${MR})
            .nr(${NR})
            .kr(${KR})
            .sr(${SR})
            .m(m)
            .n(n)
            .k(k)
            .ks(3)
            .iterations(1)
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .ks(3)
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
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(${MR})
          .n(${NR})
          .k(k)
          .ks(3)
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

TEST(${TEST_NAME}, strided_cm_subtile) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
    for (uint32_t m = 1; m <= ${MR}; m++) {
      for (uint32_t n = 1; n <= ${NR}; n++) {
        GemmMicrokernelTester()
          .mr(${MR})
          .nr(${NR})
          .kr(${KR})
          .sr(${SR})
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(${next_prime(NR + 1)})
          .iterations(1)
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
        .mr(${MR})
        .nr(${NR})
        .kr(${KR})
        .sr(${SR})
        .m(${MR})
        .n(${NR})
        .k(k)
        .ks(3)
        .a_offset(${next_prime(MR * KBLOCK * 5 + 1)})
        .Test(${", ".join(TEST_ARGS)});
    }
  }

  TEST(${TEST_NAME}, zero) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t mz = 0; mz < ${MR}; mz++) {
      for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
        GemmMicrokernelTester()
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
          .Test(${", ".join(TEST_ARGS)});
      }
    }
  }

$if ACTIVATION == "MINMAX":
  TEST(${TEST_NAME}, qmin) {
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
      .qmin(128)
      .Test(${", ".join(TEST_ARGS)});
  }

  TEST(${TEST_NAME}, qmax) {
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
      .qmax(128)
      .Test(${", ".join(TEST_ARGS)});
  }

TEST(${TEST_NAME}, strided_cm) {
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
    .cm_stride(${next_prime(NR + 1)})
    .Test(${", ".join(TEST_ARGS)});
}

$if DATATYPE == "q8":
  TEST(${TEST_NAME}, no_a_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
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

  TEST(${TEST_NAME}, no_b_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
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

  TEST(${TEST_NAME}, no_zero_point) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (size_t k = 1; k <= ${KBLOCK * 5}; k += ${KBLOCK + 1}) {
      GemmMicrokernelTester()
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
"""


def generate_test_cases(ukernel, mr, nr, kr, sr,
                        k_block, is_pipelined, isa):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    kr: KR parameter of the GEMM micro-kernel.
    sr: SR parameter of the GEMM micro-kernel.
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
  _, test_name = ukernel.split("_", 1)
  _, datatype, ukernel_type, activation, _ = ukernel.split("_", 4)
  if activation == "ukernel":
    activation = "linear"
  test_args = [ukernel]
  if activation != "linear" and (not isa or isa == "psimd"):
    test_args.append("GemmMicrokernelTester::Variant::Scalar")
  return xngen.preprocess(GEMM_TEST_CODE, {
      "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
      "TEST_ARGS": test_args,
      "UKERNEL_TYPE": ukernel_type.upper(),
      "DATATYPE": datatype,
      "ACTIVATION": activation.upper(),
      "MR": mr,
      "NR": nr,
      "KR": kr,
      "SR": sr,
      "KBLOCK": k_block,
      "ADJKBLOCK": 2 * k_block if is_pipelined else k_block,
      "IS_PIPELINED": is_pipelined,
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

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"
""".format(specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      k_block = int(ukernel_spec["k-block"])
      pipelined = bool(ukernel_spec.get("pipelined", False))
      assembly = bool(ukernel_spec.get("assembly", False))
      mr, nr, kr, sr, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(
        name, mr, nr, kr, sr, k_block, pipelined, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa, assembly)

    with codecs.open(options.output, "w", encoding="utf-8") as output_file:
      output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
