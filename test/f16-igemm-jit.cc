// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-igemm-jit.yaml
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
          .cn_stride(NextPrime(nr + 1))
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
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
              .a_stride(NextPrime(adj_k_block + 1))
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
            .a_stride(NextPrime(adj_k_block * 2 + 1))
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
              .a_stride(NextPrime(k_block * 3 + 1))
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
      "unknown_nc_mod_nr",
      tester.clone()
          .m(mr).known_nc_mod_nr(false)
      , test_func, isa_check)
      .loop_n(1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "relu",
      tester.clone()
          .m(mr).n(nr).k(k_block).relu(true)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      tester.clone()
          .m(mr)
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(NextPrime(k_block * 3 + 1))
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
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(NextPrime(k_block * 3 + 1))
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
          .cm_stride(NextPrime(nr + 1))
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
            .a_offset(NextPrime(mr * k_block * 3 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 3 + 1))
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
          .cm_stride(NextPrime(nr + 1))
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_1X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .nr(16)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_1x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY

  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_4X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .m(4)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY

  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .m(6)
        .n(16)
        .k(2)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY

  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A55R0, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .m(6)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
    }
  #endif // XNN_ENABLE_ASSEMBLY

  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .m(6)
        .n(16)
        .k(2)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY

  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/true,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64,
                        xnn_init_f16_minmax_fp16arith_params,
                        xnn_pack_f16_conv_goki_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(16)
            .m(m)
            .n(16)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F16_IGEMM_6X16__AARCH64_NEONFP16ARITH_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FP16_ARITH;
      GemmMicrokernelTester()
        .mr(6)
        .nr(16)
        .m(6)
        .n(16)
        .k(4)
        .Test(
            xnn_generate_f16_igemm_ukernel_6x16__aarch64_neonfp16arith_ld64, xnn_init_f16_minmax_fp16arith_params, xnn_pack_f16_conv_goki_w,
            &xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
