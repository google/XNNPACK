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
//   Specification: test/qp8-f32-qb4w-gemm-minmax.yaml
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
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/ppmm.h"
#include "src/xnnpack/requantization.h"
#include "test/gemm-microkernel-tester.h"
#include "test/next_prime.h"

namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace


namespace {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
    ConstantOrFunction mr_packed,
    bool is_igemm,
    bool unsigned_inputs,
    uint8_t planes,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    uint64_t arch_flags = 0) {
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
          .bl(32)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
          .b_zero_point(8)
          .bl(32)
      , test_func, arch_flags)
      .loop_n(1, nr));
  gemm_tests.push_back(GemmTestParams(
      "bl",
      tester.clone()
          .m(mr).n(nr).k(k_block * 12)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
      .loop_bl(32, k_block * 32, 32));

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_1X4C16S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x4c16s2__aarch64_neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_1X8C16S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/8, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x8c16s2__aarch64_neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_1X4C8S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/2,
          /*mr_packed=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x4c8s2__aarch64_neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_4X4C8S2__AARCH64_NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_4x4c8s2__aarch64_neondot,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_4X8C16S2__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/4, /*nr=*/8, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_4x8c16s2__neoni8mm,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_8X4C16S2__NEONI8MM_MSTEP2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/8, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_8x4c16s2__neoni8mm_mstep2,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      QP8_F32_QB4W_GEMM_MINMAX_16X4C16S2__NEONI8MM_MSTEP4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/16, /*nr=*/4, /*kr=*/16, /*sr=*/2,
          /*mr_packed=*/4,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qp8_f32_qb4w_gemm_minmax_ukernel_16x4c16s2__neoni8mm_mstep4,
                        xnn_init_f32_qb4w_minmax_scalar_params,
                        xnn_pack_kai_qb4_weights_and_biases,
                        xnn_packed_stride_kai_qb4_weights_and_biases);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

