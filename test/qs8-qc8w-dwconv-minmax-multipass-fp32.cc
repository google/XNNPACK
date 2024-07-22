// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-dwconv-minmax-multipass-fp32.yaml
//   Generator: tools/generate-dwconv-multipass-test.py


#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/requantization.h"
#include "dwconv-microkernel-tester.h"
#include "next_prime.h"

namespace {

std::vector<DWConvTestParams> CreateTests1(
    size_t c_block, size_t adj_c_block, size_t cr, size_t kr,
    size_t first_pass_tile, size_t middle_pass_tile, size_t last_pass_tile,
    size_t channel_subtile, size_t channel_round,
    std::function<void(DWConvMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  const std::string cbs = std::to_string(c_block);
  const std::string acbs = std::to_string(adj_c_block);

  std::vector<DWConvTestParams> tests;
  tests.reserve(17);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_plus_one",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
          .channels(c_block)
      , test_func, isa_check));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_and_last_pass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
          .channels(c_block)
      , test_func, isa_check));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_multipass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .channels(c_block)
      , test_func, isa_check)
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_first_pass_plus_one",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + 1)
        , test_func, isa_check)
        .loop_channels(adj_c_block = c_block, cr * 16, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_first_pass_and_last_pass",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + last_pass_tile)
        , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_multipass",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16, cr * 3)
        .loop_kernel_size(
            first_pass_tile + middle_pass_tile + last_pass_tile,
            first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_first_pass_plus_one",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_first_pass_and_last_pass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_multipass",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, c_block == 1 ? 10 : adj_c_block + c_block)
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_plus_one_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + 1)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_first_pass_and_last_pass_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .kernel_size(first_pass_tile + last_pass_tile)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs + "_multipass_multipixel",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(
          first_pass_tile + middle_pass_tile + last_pass_tile,
          first_pass_tile + middle_pass_tile * 2 + last_pass_tile)
      .loop_step(2, kr + 1));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .width(5)
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
      , test_func, isa_check)
      .loop_channels(1, c_block * 5 + 1, std::max(size_t(1), c_block - 1))
      .loop_kernel_size(first_pass_tile + middle_pass_tile + last_pass_tile,
                        first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .first_pass_tile(first_pass_tile)
          .middle_pass_tile(middle_pass_tile)
          .last_pass_tile(last_pass_tile)
          .channel_tile(cr)
          .channel_subtile(channel_subtile)
          .channel_round(channel_round)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16, cr * 3)
      .loop_kernel_size(first_pass_tile + middle_pass_tile + last_pass_tile,
                        first_pass_tile + middle_pass_tile * 2 + last_pass_tile));

  return tests;
}

}  // namespace


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neon_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neon_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEONV8_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neon_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neon_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEONV8_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neon_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neon_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neon_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neon_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neonv8_mla8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neonv8_mla8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neonv8_mul8_ld64,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEONV8_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neonv8_mul8_ld128,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C8S8R__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c8s8r__neon_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C8S8R__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c8s8r__neonv8_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C4S4R__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c4s4r__sse41_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse2_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse2_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse41_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse41_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C4S4R__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c4s4r__avx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C8S8R__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c8s8r__avx2_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C16S16R__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c16s16r__avx2_mul16_add16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C16S16R__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c16s16r__avx2_mul16_vpmovsx,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C16S16R__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/16,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c16s16r__avx2_mul16_vpunpck,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L32C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l32c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L32C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l32c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L32C16S1R__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l32c16s1r__avx512skx_mul32,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L8C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L16C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L8C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L16C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L8C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L16C8S8R__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/8,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L1C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L2C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L4C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L1C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L2C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l2c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L4C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L1C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l1c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L2C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L4C1S1R__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l4c1s1r__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L1C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L1C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L1C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L2C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L2C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L2C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L4C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L4C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_5F5M5L4C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L1C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L1C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L1C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L2C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l2c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L2C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l2c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L2C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l2c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L4C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L4C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_6F6M7L4C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L1C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l1c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L1C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l1c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L1C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l1c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L2C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L2C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L2C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L4C1S1R__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l4c1s1r__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L4C1S1R__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l4c1s1r__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_8F8M9L4C1S1R__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l4c1s1r__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });