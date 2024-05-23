// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>
#include <xnnpack/ppmm.h>
#include <xnnpack/requantization.h>

#include "gemm-microkernel-tester.h"
#include <gtest/gtest.h>

namespace {

std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cn_stride(NextPrime(nr + 1))
          .b_zero_point(8)
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
            .b_zero_point(8)
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).n(nr).k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(adj_k_block + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
            .a_stride(NextPrime(adj_k_block * 2 + 1))
            .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).ks(3).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(NextPrime(nr + 1))
          .iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "a_offset",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmin(128)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmax(128)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cm_stride(NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f16_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_bl_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });
