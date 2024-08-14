#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import collections
import os
import re
import sys
import zlib
import yaml

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
  if "v" in param_spec:
    vector_tile = True
    param_spec, _ = param_spec.split("v", 1)
  else:
    vector_tile = False
  mr, nr = map(int, param_spec.split("x"))
  arch, isa, assembly = xnncommon.parse_target_name(target_name)
  mr_packed = re.search(r"mstep([0-9]+)", target_name)
  if mr_packed:
    mr_packed = mr // int(mr_packed.group(1))
  else:
    mr_packed = mr

  requantization = common_parts[-3]
  if requantization not in ["fp32", "rndnu"]:
    requantization = None
  return mr, nr, kr, sr, mr_packed, vector_tile, requantization, arch, isa, assembly

GEMM_BENCH_CODE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
static void ${UKERNEL_NAME}(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    ${GEMM},
    $if INIT_PARAMS is not None:
      ${INIT_PARAMS},
    $if PACK_FN is not None:
      ${PACK_FN},
    $if PACKED_STRIDE_FN is not None:
      ${PACKED_STRIDE_FN},
    /*mr=*/${MR}, /*nr=*/${NR}${NR_SCALE}, /*kr=*/${KR}, /*sr=*/${SR},
    $if DATATYPE in ('qp8',):
      /*mr_packed=*/${MR_PACKED},
    $if ISA_CHECK:
      benchmark::utils::${ISA_CHECK});
    $else:
      /*isa_check=*/nullptr);
}\n
$if KERNELTYPE in ['qb4w']:
  BENCHMARK_GEMM_BL(${UKERNEL_NAME})
$else:
  BENCHMARK_GEMM(${UKERNEL_NAME})
$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""

GEMM_CREATE_TESTS_CODE = """\
std::vector<GemmTestParams> CreateTests(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    $if DATATYPE in ('qp8'):
      size_t mr_packed,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  $if NR_SCALE != "":
    nr = nr${NR_SCALE};
  std::string nrs = std::to_string(nr);

  $if DATATYPE in ('qp8',):
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed);
  $else:
    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func, isa_check));
  $if DATATYPE != "qp8":
    gemm_tests.push_back(GemmTestParams(
        "strided_cn",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .cn_stride(xnnpack::NextPrime(nr + 1))
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
      , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block).iterations(1)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func, isa_check)
      .loop_n(1, nr));
  $if IS_PIPELINED:
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s,
        tester.clone()
          .m(mr).n(nr).k(k_block * 2)
          $if KERNELTYPE in ['qb4w', 'qc4w']:
            .b_zero_point(8)
          $if KERNELTYPE in ['qb4w']:
            .bl(32)
      , test_func, isa_check));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kb2s + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block * 2)
              .a_stride(xnnpack::NextPrime(k_block * 2 + 1))
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
          , test_func, isa_check));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_subtile",
        tester.clone()
            .k(k_block * 2).iterations(1)
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            $if KERNELTYPE in ['qb4w']:
              .bl(32)
        , test_func, isa_check)
        .loop_n(1, nr)
        .loop_m(1, mr));
  $if KERNELTYPE not in ['qb4w']:
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs,
            tester.clone()
                .m(mr).n(nr)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, adj_k_block - 1));
        if (!is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "k_lt_" + akbs + "_strided_a",
              tester.clone()
                  .m(mr).n(nr)
                  .a_stride(xnnpack::NextPrime(adj_k_block + 1))
                  $if KERNELTYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if KERNELTYPE in ['qb4w']:
                    .bl(32)
              , test_func, isa_check)
              .loop_k(1, adj_k_block - 1));
        }
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs + "_subtile",
            tester.clone()
                .iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, adj_k_block - 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs,
          tester.clone()
              .m(mr).n(nr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_gt_" + akbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (k_block > 1) {
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs,
            tester.clone()
                .m(mr).n(nr)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block));
        if (is_igemm) {
          gemm_tests.push_back(GemmTestParams(
              "k_div_" + kbs + "_strided_a",
              tester.clone()
                  .m(mr).n(nr)
                  .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                  $if KERNELTYPE in ['qb4w', 'qc4w']:
                    .b_zero_point(8)
                  $if KERNELTYPE in ['qb4w']:
                    .bl(32)
              , test_func, isa_check)
              .loop_k(adj_k_block + k_block, k_block * 3, k_block));
        }
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs + "_subtile",
            tester.clone()
                .iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(adj_k_block + k_block, k_block * 5, k_block)
            .loop_n(1, nr)
            .loop_m(1, mr));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs,
          tester.clone()
              .m(mr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          $if NR_SCALE != "":
            .loop_n(nr + 1, nr * 2 - 1, 4)
          $else:
            .loop_n(nr + 1, nr * 2 - 1)
          .loop_k(1, k_block * 3, k_block + 1));
      $if DATATYPE != "qp8":
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_strided_cn",
            tester.clone()
                .m(mr)
                .cn_stride(xnnpack::NextPrime(nr + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            $if NR_SCALE != "":
              .loop_n(nr + 1, nr * 2 - 1, 4)
            $else:
              .loop_n(nr + 1, nr * 2 - 1)
            .loop_k(1, k_block * 3, k_block + 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_strided_a",
            tester.clone()
                .m(mr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            $if NR_SCALE != "":
              .loop_n(nr + 1, nr * 2 - 1, 4)
            $else:
              .loop_n(nr + 1, nr * 2 - 1)
            .loop_k(1, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          $if NR_SCALE != "":
            .loop_n(nr + 1, nr * 2 - 1, 4)
          $else:
            .loop_n(nr + 1, nr * 2 - 1)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs,
          tester.clone()
              .m(mr)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1));
      $if DATATYPE != "qp8":
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_strided_cn",
            tester.clone()
                .m(mr)
                .cn_stride(xnnpack::NextPrime(nr + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block + 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_strided_a",
            tester.clone()
                .m(mr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_subtile",
          tester.clone()
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "small_kernel",
            tester.clone()
                .m(mr).n(nr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "small_kernel_subtile",
            tester.clone()
                .ks(3).iterations(1)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_n(1, nr)
            .loop_m(1, mr));
        gemm_tests.push_back(GemmTestParams(
            "n_gt_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            $if NR_SCALE != "":
              .loop_n(nr + 1, nr * 2 - 1, 4)
            $else:
              .loop_n(nr + 1, nr * 2 - 1)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "n_div_" + nrs + "_small_kernel",
            tester.clone()
                .m(mr).ks(3)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_n(nr * 2, nr * 3, nr)
            .loop_k(1, k_block * 3, k_block + 1));
      }
      gemm_tests.push_back(GemmTestParams(
          "strided_cm_subtile",
          tester.clone()
              .mr(mr).nr(nr).kr(kr).sr(sr)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              .iterations(1)
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "a_offset",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "zero",
            tester.clone()
                .m(mr).n(nr).ks(3)
                .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1)
            .loop_zi(0, mr - 1));
      }
      $if ACTIVATION == "MINMAX":
        gemm_tests.push_back(GemmTestParams(
            "qmin",
            tester.clone()
                .m(mr).n(nr).k(k_block).qmin(128)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check));
        gemm_tests.push_back(GemmTestParams(
            "qmax",
            tester.clone()
                .m(mr).n(nr).k(k_block).qmax(128)
                $if KERNELTYPE in ['qb4w', 'qc4w']:
                  .b_zero_point(8)
                $if KERNELTYPE in ['qb4w']:
                  .bl(32)
            , test_func, isa_check));
      gemm_tests.push_back(GemmTestParams(
          "strided_cm",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .cm_stride(xnnpack::NextPrime(nr + 1))
              $if KERNELTYPE in ['qb4w', 'qc4w']:
                .b_zero_point(8)
              $if KERNELTYPE in ['qb4w']:
                .bl(32)
          , test_func, isa_check));
      $if DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_a_zero_point",
            tester.clone()
                .m(mr).n(nr).a_zero_point(0)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1));
      $if DATATYPE == "qu8":
        gemm_tests.push_back(GemmTestParams(
            "no_b_zero_point",
            tester.clone()
                .m(mr).n(nr).b_zero_point(0)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1));
        gemm_tests.push_back(GemmTestParams(
            "b_zero_point",
            tester.clone()
                .m(mr).n(nr).k(k_block)
            , test_func, isa_check)
            .loop_bzp(0, 255));
        gemm_tests.push_back(GemmTestParams(
            "no_zero_point",
            tester.clone()
                .m(mr).n(nr)
                .a_zero_point(0)
                .b_zero_point(0)
            , test_func, isa_check)
            .loop_k(1, k_block * 3, k_block + 1));
  $if KERNELTYPE in ['qb4w']:
    gemm_tests.push_back(GemmTestParams(
        "bl",
        tester.clone()
            .m(mr).n(nr).k(k_block * 12)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
        .loop_bl(32, k_block * 32, 32));

  return gemm_tests;
}
"""

GEMM_TEST_CODE = """\
$if CPP_CHECK:
  #if ${CPP_CHECK}
INSTANTIATE_TEST_SUITE_P(
    ${TEST_NAME}, GemmTest,
    testing::ValuesIn(CreateTests(
        /*k_block=*/${KBLOCK},
        /*adj_k_block=*/${ADJKBLOCK},
        /*mr=*/${MR}, /*nr=*/${NR}, /*kr=*/${KR}, /*sr=*/${SR},
        $if DATATYPE in ('qp8',):
          /*mr_packed=*/${MR_PACKED},
        /*is_igemm=*/${"true" if UKERNEL_TYPE.startswith("IGEMM") else "false"},
        [](GemmMicrokernelTester& tester) {
          tester.Test(${",\\n                      ".join(TEST_ARGS)});
        $if ISA_CHECK:
          },
          []() {
            ${ISA_CHECK};
          })),
        $else:
          })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16']:
  TEST(${TEST_NAME}, subtile_m_upto_mr) {
    $if ISA_CHECK:
      ${ISA_CHECK};
    for (uint32_t max_mr = 1; max_mr <= ${MR}; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= ${KBLOCK * 2}; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            $if NR > 1:
              .nr(${NR})
            $if KR > 1:
              .kr(${KR})
            $if SR > 1:
              .sr(${SR})
            .m(m)
            $if NR > 1:
              .n(${NR})
            .k(k)
            .iterations(1)
            $if KERNELTYPE in ['qb4w', 'qc4w']:
              .b_zero_point(8)
            .Test(${", ".join(TEST_ARGS)});
        }
      }
    }
  }

$if TEST_NAME.startswith('GENERATE') and DATATYPE in ['f32', 'f16'] and PROTOTYPE is not None:
  #if XNN_ENABLE_ASSEMBLY
    TEST(${TEST_NAME}, matches_assembly) {
      $if ISA_CHECK:
        ${ISA_CHECK};
      GemmMicrokernelTester()
        $if MR > 1:
          .mr(${MR})
        $if NR > 1:
          .nr(${NR})
        $if KR > 1:
          .kr(${KR})
        $if SR > 1:
          .sr(${SR})
        $if MR > 1:
          .m(${MR})
        $if NR > 1:
          .n(${NR})
        .k(${KBLOCK})
        .Test(
            ${", ".join(TEST_ARGS)},
            &${PROTOTYPE});
    }
  #endif // XNN_ENABLE_ASSEMBLY
$if CPP_CHECK:
  #endif  // ${CPP_CHECK}
"""


def generate_test_cases(
    ukernel,
    mr,
    nr,
    kr,
    sr,
    mr_packed,
    k_block,
    vector_tile,
    init_fn,
    pack_fn,
    packed_stride_fn,
    requantization,
    is_pipelined,
    cpp_check,
    isa,
    prototype,
):
  """Generates all tests cases for a GEMM micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    mr: MR parameter of the GEMM micro-kernel.
    nr: NR parameter of the GEMM micro-kernel.
    kr: KR parameter of the GEMM micro-kernel.
    sr: SR parameter of the GEMM micro-kernel.
    mr_packed: Optional MR parameter for the left-hand packing function.
    k_block: Number of K values processed per one iteration of the main loop of
      the micro-kernel.
    vector_tile: Indicates if vector tile for NR is specified in vectors rather
      than elements.
    init_fn: C name of the function to initialize microkernel parameters.
    pack_fn: C name of the function to pack the weights.
    packed_stride_fn: C name of the function to compute the packed weights
      stride.
    requantization: name of the requantization scheme used by the microkernel.
    is_pipelined: Indicates if the micro-kernel is implemented with software
      pipelining. Additional test cases are generated for software pipelined
      micro-kernels to separately test prologue + epiloque of the pipelined loop
      and iteration of the pipelined loop.
    cpp_check: Optional preprocessor macro to check for the availability of the
      micro-kernel.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, ukernel_name = ukernel.split("_", 1)

  _, datatype, ukernel_type, activation, _ = ukernel.split("_", 4)
  kerneltype = datatype
  if datatype in ["f16", "f32"] and ukernel_type in ["qc8w", "qc4w"]:
    _, datatype, kerneltype, ukernel_type, activation, _ = ukernel.split(
        "_", 5
    )
    datatype = datatype + "_" + kerneltype
  if (
      datatype in ("qd8", "qp8")
      and ukernel_type in ["f16", "f32"]
      and activation in ["qc8w", "qc4w", "qb4w"]
  ):
    _, datatype, _, kerneltype, ukernel_type, activation, _ = ukernel.split(
        "_", 6
    )

  if activation == "ukernel":
    activation = "linear"
  if activation in ["qs8w"]:
    _, _, _, _, _, activation, _ = ukernel.split("_", 6)
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)

  if pack_fn:
    test_args.append(pack_fn)
  if packed_stride_fn:
    test_args.append(packed_stride_fn)

  if init_fn and requantization:
    requantization_datatype = {"qc8": "qs8"}.get(datatype, datatype)
    test_args.append(
        "xnn_%s_requantize_%s" % (requantization_datatype, requantization)
    )

  nr_scale = ""
  if vector_tile:
    ctype = {
        "qs8": "int8_t",
        "qd8": "int32_t",
        "qp8": "int8_t",
        "qu8": "uint8_t",
        "f16": "uint16_t",
        "f32": "float",
    }[datatype]
    nr_scale = {"rvv": " * xnn_init_hardware_config()->vlenb / sizeof(%s)" % ctype}[isa]
  test_args = {
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
      "MR_PACKED": mr_packed,
      "KBLOCK": k_block,
      "NR_SCALE": nr_scale,
      "ADJKBLOCK": 2 * k_block if is_pipelined else k_block,
      "IS_PIPELINED": is_pipelined,
      "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      "next_prime": next_prime,
      "PROTOTYPE": prototype,
      "CPP_CHECK": cpp_check,
  }

  create_test_case = xngen.preprocess(GEMM_CREATE_TESTS_CODE, test_args)

  test_case = xngen.preprocess(GEMM_TEST_CODE, test_args)

  benchmark = xngen.preprocess(
      GEMM_BENCH_CODE,
      {
          "UKERNEL_NAME": ukernel_name,
          "GEMM": ukernel,
          "KERNELTYPE": kerneltype,
          "DATATYPE": datatype,
          "INIT_PARAMS": init_fn,
          "PACK_FN": pack_fn,
          "PACKED_STRIDE_FN": packed_stride_fn,
          "MR": mr,
          "NR": nr,
          "KR": kr,
          "SR": sr,
          "MR_PACKED": mr_packed,
          "NR_SCALE": nr_scale,
          "ISA_CHECK": xnncommon.generate_isa_utilcheck_macro(isa),
          "CPP_CHECK": cpp_check,
      },
  )
  return create_test_case, test_case, benchmark


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

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"
#include "xnnpack/ppmm.h"
#include "xnnpack/requantization.h"
#include "gemm-microkernel-tester.h"
#include "next_prime.h"
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
#include "bench/gemm-benchmark.h"
#include "bench/utils.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"
""".format(specification=options.spec, generator=sys.argv[0])

    test_outputs = collections.defaultdict(str)
    bench_outputs = benches

    isa_hierarchy = xnncommon._ISA_HIERARCHY_MAP

    # Cached `CreateTests` functions.
    idx_from_create_tests_hash = collections.defaultdict(
        lambda: len(idx_from_create_tests_hash) + 1
    )
    create_tests_from_idx = {}

    benches = [""] * len(isa_hierarchy)
    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      k_block = int(ukernel_spec["k-block"])
      init_fn = ukernel_spec.get("init")
      pack_fn = ukernel_spec.get("pack")
      packed_stride_fn = ukernel_spec.get("packed-stride")
      pipelined = bool(ukernel_spec.get("pipelined", False))
      cpp_check = ukernel_spec.get("cpp-check", False)
      prototype = ukernel_spec.get("prototype")
      (
          mr,
          nr,
          kr,
          sr,
          mr_packed,
          vector_tile,
          requantization,
          arch,
          isa,
          assembly,
      ) = split_ukernel_name(name)

      create_tests, test_case, bench_case = generate_test_cases(
          name,
          mr,
          nr,
          kr,
          sr,
          mr_packed,
          k_block,
          vector_tile,
          init_fn,
          pack_fn,
          packed_stride_fn,
          requantization,
          pipelined,
          cpp_check,
          isa,
          prototype,
      )

      # Store or reuse the `CreateTests` function?
      create_tests_hash = hash(create_tests)
      create_tests_idx = idx_from_create_tests_hash[create_tests_hash]
      if create_tests_idx not in create_tests_from_idx:
        create_tests_from_idx[create_tests_idx] = create_tests.replace(
            "CreateTests(", f"CreateTests{create_tests_idx}("
        )
        if isa == 'rvv':
          create_tests_from_idx[create_tests_idx] = xnncommon.postprocess_test_case(
            create_tests_from_idx[create_tests_idx], arch, isa, assembly)
      test_case = test_case.replace(
          "CreateTests(", f"CreateTests{create_tests_idx}("
      )

      # Hash the name of each microkernel and figure out which output file to
      # write it to.
      output_index = zlib.crc32(bytes(name, "utf-8")) % num_output_files
      test_outputs[
          options.output_test[output_index]
      ] += "\n\n" + xnncommon.postprocess_test_case(
          test_case, arch, isa, assembly
      )
      benches[
          isa_hierarchy.get(isa, 0)
      ] += "\n\n" + xnncommon.postprocess_test_case(
          bench_case, arch, isa, assembly
      )

    for arch_idx in reversed(range(len(isa_hierarchy))):
      bench_outputs += benches[arch_idx]

    bench_outputs += """\n
#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
"""

    if options.output_bench:
      # Strip out consecutive preprocessor `endif`/`if` pairs.
      for _ in range(2):
        bench_outputs = re.sub(
            r"^ *\#endif  // ([^\n]+)\n+ *\#if \1\n",
            "\n",
            bench_outputs,
            flags=re.MULTILINE,
        )
      output_name = options.output_bench
      xnncommon.overwrite_if_changed(output_name, bench_outputs)

    create_tests = (
        "namespace {\n\n"
        + "\n".join(create_tests_from_idx.values())
        + "\n}  // namespace\n"
    )
    test_outputs = {
        k: tests + "\n" + create_tests + v for k, v in test_outputs.items()
    }

    # Strip out consecutive preprocessor `endif`/`if` pairs.
    for _ in range(2):
      test_outputs = {
          k: re.sub(
              r"^ *\#endif  // ([^\n]+)\n+ *\#if \1\n",
              "\n",
              v,
              flags=re.MULTILINE,
          )
          for k, v in test_outputs.items()
      }

    for output_name in options.output_test:
      xnncommon.overwrite_if_changed(output_name, test_outputs[output_name])


if __name__ == "__main__":
  main(sys.argv[1:])
