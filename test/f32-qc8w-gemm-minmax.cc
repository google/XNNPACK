// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qc8w-gemm-minmax.yaml
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


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEON_LD128_ACC2_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC2_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_ACC4_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD64_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC2_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_ACC4_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__ASM_AARCH64_NEONFMA_LD128_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X1__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X1__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/1, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__ASM_AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__NEON_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__NEONFMA_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__NEON_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__NEONFMA_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X2__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X2__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__AARCH64_NEONFMA_LANE_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__NEON_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__NEON_LANE_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__NEONFMA_DUP_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__NEONFMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__NEONFMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__NEONFMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__SSE41_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__SSE41_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_load1,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__SSE41_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_dup,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__SSE41_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_load1,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8S4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2C4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__SSE41_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__SSE41_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_load1,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__SSE41_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_dup,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__SSE41_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_load1,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8S4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__SSE41_DUP, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_dup,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__SSE41_LOAD1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_load1,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__SSE41, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__sse41,
                        xnn_init_f32_minmax_sse_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X16__AVX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X16__FMA3_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__fma3_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X16S4__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x16s4__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X8__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x8__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X16__AVX2_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx2_broadcast,
                        xnn_init_f32_minmax_avx_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_7X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/7, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X16__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_8X32__AVX512SKX_BROADCAST, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/8, /*nr=*/32, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_8x32__avx512skx_broadcast,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2C4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2C4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMSIMD_ARM_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMSIMD_ARM_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMSIMD_X86_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMSIMD_X86_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__WASMSIMD_ARM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__WASMSIMD_X86, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_FMA_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMRELAXEDSIMD_FMA_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_3X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2C4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2C4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/2, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMRELAXEDSIMD_FMA_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMRELAXEDSIMD_FMA_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_5X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMRELAXEDSIMD_FMA_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMRELAXEDSIMD_FMA_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMRELAXEDSIMD_LOADSPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8__WASMRELAXEDSIMD_SPLAT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__WASMRELAXEDSIMD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_6X8S4__WASMRELAXEDSIMD_FMA, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_1X4__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_2X4__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X2__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      F32_QC8W_GEMM_MINMAX_4X4__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_qs8w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    F32_QC8W_GEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_qs8w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    F32_QC8W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_qs8w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    F32_QC8W_GEMM_MINMAX_4X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_qs8w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    F32_QC8W_GEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/1,
        /*adj_k_block=*/1,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f32_minmax_scalar_params,
                      xnn_pack_f32_qs8w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });
