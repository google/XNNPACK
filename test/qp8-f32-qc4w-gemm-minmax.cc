// clang-format off
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qp8-f32-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/ppmm.h"
#include "src/xnnpack/requantization.h"
#include "test/gemm-microkernel-tester.h"
#include "test/next_prime.h"

namespace {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    size_t mr_packed,
    bool is_igemm,
    bool unsigned_inputs,
    uint8_t planes,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr).mr_packed(mr_packed).unsigned_inputs(unsigned_inputs).planes(planes);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      tester.clone()
          .m(mr).n(nr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        tester.clone()
            .ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      tester.clone()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
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
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "min",
      tester.clone()
          .m(mr).n(nr).k(k_block).min(0.0f)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "max",
      tester.clone()
          .m(mr).n(nr).k(k_block).max(0.0f)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_1X4C16S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_1X8C16S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_1X4C8S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_16X4C8S2__AARCH64_NEONDOT_MSTEP4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/16, /*nr=*/4, /*kr=*/8, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x4c8s2__aarch64_neondot_mstep4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_4X4C16S2__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x4c16s2__neoni8mm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_4X8C16S2__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_8X4C16S2__NEONI8MM_MSTEP2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/8, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_8X8C16S2__NEONI8MM_MSTEP2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/8, /*nr=*/8, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_8x8c16s2__neoni8mm_mstep2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases,
                        xnn_packed_stride_kai_qs4_weights_and_biases);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_1X128C4__NEONSME2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/128, /*kr=*/4, /*sr=*/1,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x128c4__neonsme2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases_sme,
                        xnn_packed_stride_kai_qs4_weights_and_biases_sme);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_SME2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QC4W_GEMM_MINMAX_32X128C4__NEONSME2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/32, /*nr=*/128, /*kr=*/4, /*sr=*/1,
          /*mr_packed=*/32,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x128c4__neonsme2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_kai_qs4_weights_and_biases_sme,
                        xnn_packed_stride_kai_qs4_weights_and_biases_sme);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_SME2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64

