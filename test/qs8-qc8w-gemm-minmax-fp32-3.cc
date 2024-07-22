// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-gemm-minmax-fp32.yaml
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
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_7X64C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x64c4__avx512amx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512AMX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8__ASM_AARCH32_NEONV8_MLAL_LANE_CORTEX_A35, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__ASM_AARCH32_NEONV8_MLAL_LANE_LD64_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C4__ASM_AARCH32_NEONDOT_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C4__ASM_AARCH32_NEONDOT_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__ASM_AARCH64_NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X16C4__ASM_AARCH64_NEONDOT_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__ASM_AARCH64_NEON_MLAL_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X16__ASM_AARCH64_NEON_MLAL_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_6X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_6X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEON_MLAL_LD2R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C2__NEONV8_MLAL_LD4R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C2S4__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C4__NEON_MLAL_LD1R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__NEONV8_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X16C8__AARCH64_NEONDOT_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C2__NEONV8_MLAL_LD4R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C4__NEON_MLAL_LD1R, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__NEON_MLAL, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X16__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X8__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X16__NEONV8_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_6X8__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_6X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_6X16__NEONV8_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2S4__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx256skx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx256skx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X16C8__AVX512SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_9X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_5X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_9X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_10X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_10x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_28X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/28, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_28x16c4__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_12X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_14X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_9X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_12X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_14X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_14x16c8__avx512vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx256vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx256vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_9X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_9x8c8__avx256vnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_7X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx256vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_12X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_12x8c8__avx256vnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_7X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_5X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_8X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C2__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C8__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C2S4__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4C8__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C8__WASMSIMD_DOT16X2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C2__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_1X4C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_WASM_SDOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_WASM_SDOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X4C16__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_WASM_USDOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_3X8C16__WASMSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c16__wasmsdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_WASM_SDOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X8C16__WASMUSDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c16__wasmusdot,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_pack_qs8_to_qu8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_WASM_USDOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_2X4__WASM_FMAGIC, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_GEMM_MINMAX_FP32_4X2__WASM_FMAGIC, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_pack_qs8_gemm_goi_w,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_1X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_2X2__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_2X4__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_3X2__SCALAR_FMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_3X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/3, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_IMAGIC, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_4X2__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_GEMM_MINMAX_FP32_4X4__SCALAR_LRINTF, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_pack_qs8_gemm_goi_w,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });
