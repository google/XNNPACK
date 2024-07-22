// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-minmax-multipass.yaml
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2,
                        xnn_init_f32_minmax_scalar_params);
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
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2,
                        xnn_init_f32_minmax_sse_params);
          },
          []() {
            TEST_REQUIRES_X86_SSE;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/7,
          /*first_pass_tile=*/7, /*middle_pass_tile=*/6, /*last_pass_tile=*/6,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/7,
          /*first_pass_tile=*/7, /*middle_pass_tile=*/6, /*last_pass_tile=*/6,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/7,
          /*first_pass_tile=*/7, /*middle_pass_tile=*/6, /*last_pass_tile=*/6,
          /*channel_subtile=*/8, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3,
                        xnn_init_f32_minmax_avx_params);
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
      F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_X86_AVX512F;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_X86_AVX512F;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_X86_AVX512F;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/16, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_X86_AVX512F;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/4, /*channel_round=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
          /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
          /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
          /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
          /*channel_subtile=*/1, /*channel_round=*/1,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/2,
        /*first_pass_tile=*/2, /*middle_pass_tile=*/2, /*last_pass_tile=*/2,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/2,
        /*first_pass_tile=*/2, /*middle_pass_tile=*/2, /*last_pass_tile=*/2,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/2,
        /*first_pass_tile=*/2, /*middle_pass_tile=*/2, /*last_pass_tile=*/2,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/2,
        /*first_pass_tile=*/2, /*middle_pass_tile=*/2, /*last_pass_tile=*/2,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/5,
        /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/6,
        /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/8,
        /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
        /*channel_subtile=*/1, /*channel_round=*/1,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });