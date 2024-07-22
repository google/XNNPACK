// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-dwconv-minmax-multipass.yaml
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

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmin",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + last_pass_tile)
            .qmin(128)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmax",
        DWConvMicrokernelTester()
            .first_pass_tile(first_pass_tile)
            .middle_pass_tile(middle_pass_tile)
            .last_pass_tile(last_pass_tile)
            .channel_tile(cr)
            .channel_subtile(channel_subtile)
            .channel_round(channel_round)
            .kernel_size(first_pass_tile + last_pass_tile)
            .qmax(128)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16, cr * 3));
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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L8C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L8C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L16C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L16C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L32C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L32C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L8C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L8C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L16C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L16C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L32C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L32C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L8C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L8C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L16C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L16C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L32C8S4R__NEONFP16ARITH, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L32C8S4R__NEONFP16ARITH_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith_acc2,
                        xnn_init_f16_minmax_fp16arith_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L8C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L8C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L16C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L16C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L32C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_6F6M7L32C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L8C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L8C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L16C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L16C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L32C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F16_DWCONV_MINMAX_8F8M9L32C8S4R__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3_acc2,
                        xnn_init_f16_minmax_avx_params);
          },
          []() {
            TEST_REQUIRES_X86_FMA3;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
