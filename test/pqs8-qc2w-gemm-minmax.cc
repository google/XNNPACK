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
//   Specification: test/pqs8-qc2w-gemm-minmax.yaml
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
  (void) adj_k_block;
  (void) is_igemm;
  const size_t mr_value = mr;
  const size_t nr_value = nr;
  if (mr_value == 0 || nr_value == 0) {
    // Keep the parameterized suite instantiated when this binary runs on a
    // host without the required architecture. GemmTest checks arch_flags and
    // skips the case before invoking the microkernel.
    return {GemmTestParams(
        "unsupported_hardware",
        GemmMicrokernelTester()
            .mr(1).nr(1).kr(kr).sr(sr).mr_packed(1)
            .unsigned_inputs(unsigned_inputs).planes(planes).b_zero_point(0),
        test_func, arch_flags)};
  }
  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr_value).nr(nr_value).kr(kr).sr(sr).mr_packed(mr_packed)
      .unsigned_inputs(unsigned_inputs).planes(planes).b_zero_point(0);

  std::vector<GemmTestParams> gemm_tests;
  auto add_test = [&](const std::string& name, size_t m, size_t n, size_t k) {
    gemm_tests.emplace_back(
        name, tester.clone().m(m).n(n).k(k), test_func, arch_flags);
  };

  if (mr_value == 1) {
    // The DOT/GEMV kernel has 64-column physical RHS panels on a 64-byte SVL,
    // but advertises a preferred scheduling step of four panels. Exercise
    // tails on both boundaries while keeping every K supported by the kernel.
    const size_t n_values[] = {
        nr_value - 1, nr_value, nr_value + 1,
        4 * nr_value - 1, 4 * nr_value, 4 * nr_value + 1,
    };
    for (const size_t n_value : n_values) {
      add_test("n_eq_" + std::to_string(n_value) + "_k_eq_" +
                   std::to_string(k_block),
               1, n_value, k_block);
    }
    add_test("k_eq_" + std::to_string(2 * k_block), 1, nr_value + 1,
             2 * k_block);
    add_test("k_eq_" + std::to_string(3 * k_block), 1, nr_value + 1,
             3 * k_block);
  } else {
    // Exercise row and column tails of the packed-LHS MOPA kernel, including
    // the single-row tail even though normal dispatch selects DOT for M=1.
    const size_t m_values[] = {1, mr_value - 1, mr_value};
    const size_t n_values[] = {1, nr_value - 1, nr_value, nr_value + 1};
    for (const size_t m_value : m_values) {
      for (const size_t n_value : n_values) {
        add_test("m_eq_" + std::to_string(m_value) + "_n_eq_" +
                     std::to_string(n_value),
                 m_value, n_value, k_block);
      }
    }
    add_test("m_tail_n_eq_" + std::to_string(2 * nr_value - 1),
             mr_value - 1, 2 * nr_value - 1, k_block);
    add_test("m_tail_n_eq_" + std::to_string(2 * nr_value + 1),
             mr_value - 1, 2 * nr_value + 1, k_block);
    add_test("k_eq_" + std::to_string(2 * k_block), mr_value - 1,
             nr_value + 1, 2 * k_block);
    add_test("k_eq_" + std::to_string(3 * k_block), mr_value - 1,
             nr_value + 1, 3 * k_block);
  }

  gemm_tests.emplace_back(
      "input_zero_point_min",
      tester.clone().m(mr_value).n(nr_value + 1).k(k_block).a_zero_point(0),
      test_func, arch_flags);
  gemm_tests.emplace_back(
      "input_zero_point_max",
      tester.clone().m(mr_value).n(nr_value + 1).k(k_block).a_zero_point(255),
      test_func, arch_flags);
  gemm_tests.emplace_back(
      "qmin",
      tester.clone().m(mr_value).n(nr_value).k(k_block).qmin(128),
      test_func, arch_flags);
  gemm_tests.emplace_back(
      "qmax",
      tester.clone().m(mr_value).n(nr_value).k(k_block).qmax(128),
      test_func, arch_flags);
  gemm_tests.emplace_back(
      "strided_cm",
      tester.clone()
          .m(mr_value)
          .n(nr_value + 1)
          .k(k_block)
          .cm_stride(xnnpack::NextPrime(nr_value + 2)),
      test_func, arch_flags);

  return gemm_tests;
}

}  // namespace


#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
  #if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_KLEIDIAI
  INSTANTIATE_TEST_SUITE_P(
      PQS8_QC2W_GEMM_MINMAX_FP32_1X64C4__NEONSME2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  , /*nr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_nr();
        } else {
          return 0;
        }
      }
  , /*kr=*/4, /*sr=*/1,
          /*mr_packed=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  ,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/4,
          [](GemmMicrokernelTester& tester) {
            tester.Test_PQS8QC2W(xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_kai_qs8_qc2w_weights_and_biases_sme2,
                        xnn_packed_stride_kai_qs8_qc2w_weights_and_biases_sme2,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_sme2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });


  INSTANTIATE_TEST_SUITE_P(
      PQS8_QC2W_GEMM_MINMAX_FP32_32X64C4__NEONSME2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/32,
          /*adj_k_block=*/32,
          /*mr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  , /*nr=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2_get_nr();
        } else {
          return 0;
        }
      }
  , /*kr=*/4, /*sr=*/1,
          /*mr_packed=*/[]() -> size_t {
        const struct xnn_hardware_config* hardware_config =
              xnn_init_hardware_config();
        if (hardware_config != nullptr && (hardware_config->arch_flags & xnn_arch_arm_sme2) == xnn_arch_arm_sme2) {
          return xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2_get_mr();
        } else {
          return 0;
        }
      }
  ,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/4,
          [](GemmMicrokernelTester& tester) {
            tester.Test_PQS8QC2W(xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params,
                        xnn_pack_kai_qs8_qc2w_weights_and_biases_sme2,
                        xnn_packed_stride_kai_qs8_qc2w_weights_and_biases_sme2,
                        xnn_qs8_requantize_fp32);
          },
          xnn_arch_arm_sme2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  #endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_KLEIDIAI
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64


}  // namespace
