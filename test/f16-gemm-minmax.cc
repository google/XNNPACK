// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

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

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cn_stride(xnnpack::NextPrime(nr + 1))
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block + 1))
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        tester.clone()
            .iterations(1)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      tester.clone()
          .m(mr).n(nr)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr)
            .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      tester.clone()
          .iterations(1)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        tester.clone()
            .m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        tester.clone()
            .iterations(1)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      tester.clone()
          .m(mr)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      tester.clone()
          .m(mr)
          .cn_stride(xnnpack::NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      tester.clone()
          .iterations(1)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      tester.clone()
          .m(mr)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_strided_cn",
      tester.clone()
          .m(mr)
          .cn_stride(xnnpack::NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      tester.clone()
          .iterations(1)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        tester.clone()
            .m(mr).n(nr).ks(3)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        tester.clone()
            .ks(3).iterations(1)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
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
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmin(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmax(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_8X8__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X16__ASM_AARCH64_NEONFP16ARITH_LD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X16__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X16__ASM_AARCH64_NEONFP16ARITH_LD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X16__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__ASM_AARCH64_NEONFP16ARITH_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__ASM_AARCH64_NEONFP16ARITH_CORTEX_A55R0, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__ASM_AARCH64_NEONFP16ARITH_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__ASM_AARCH64_NEONFP16ARITH_LD32, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__ASM_AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X8__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X8__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X8__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_8X8__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X16__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X16__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X16__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_8X16__NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_5X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_5x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_7X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_7x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_8X32__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_8x32__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x32__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_5X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_5x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_7X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_8X64__AVX512FP16_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/64, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_x16_packw_gemm_goi_ukernel_x64__scalar_int_u4);
          },
          []() {
            TEST_REQUIRES_X86_AVX512FP16;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_5X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_6X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_7X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x8__avx2_u16);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_1X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_3X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_4X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F16_GEMM_MINMAX_5X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast,
                        xnn_init_f16_minmax_avx_params,
                        xnn_x16_packw_gemm_goi_ukernel_x16__avx2_u16_prfm);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
