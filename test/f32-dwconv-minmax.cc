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
//   Microkernel: f32-dwconv-minmax
//   Generator: tools/generate-dwconv-test.py


#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/kai-dwconv.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/requantization.h"
#include "test/dwconv-microkernel-tester.h"
#include "test/next_prime.h"

namespace {

std::vector<DWConvTestParams> CreateTests(
    size_t c_block, bool is_pipelined, size_t cr, size_t kr,
    std::function<void(DWConvMicrokernelTester& tester)> test_func) {
  size_t adj_c_block = is_pipelined ? c_block * 2 : c_block;
  const std::string cbs = std::to_string(c_block);
  const std::string acbs = std::to_string(adj_c_block);

  std::vector<DWConvTestParams> tests;
  tests.reserve(18);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .channels(c_block)
      , test_func));

  if (is_pipelined) {
    tests.push_back(DWConvTestParams(
        "c_eq_" + std::to_string(c_block * 2),
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .channels(c_block * 2)
        , test_func));
  }

  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmin",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmin(128)
        , test_func)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmax",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmax(128)
        , test_func)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_lt_" + acbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func)
      .loop_channels(1, adj_c_block - 1));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
      , test_func)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_with_qmin",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .qmin(128)
      , test_func)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_with_qmax",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .qmax(128)
      , test_func)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "multipixel",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
      , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
        , test_func)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1))
        .loop_step(2, kr));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(5)
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
      , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_qmin",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
          .qmin(128)
      , test_func)
    .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_qmax",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
          .qmax(128)
      , test_func)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));


  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func)
      .loop_zi(0, kr - 1, std::max<size_t>(1, kr / 3))
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}


#define XNN_UKERNEL(arch_flags, ukernel, c_block, is_pipelined, cr, kr, datatype, weights_type, params_type, init_params)\
INSTANTIATE_TEST_SUITE_P(                                                                                                \
    ukernel, DWConvTest,                                                                                                 \
    testing::ValuesIn(CreateTests(                                                                                       \
        c_block, is_pipelined, cr, kr,                                                                                   \
        [](DWConvMicrokernelTester& tester) {                                                                            \
          TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
          tester.Test(ukernel, init_params);                                                                             \
        })),                                                                                                             \
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {                                                      \
      return info.param.test_name;                                                                                       \
    });
#include "src/f32-dwconv/f32-dwconv-minmax.inc"
#undef XNN_UKERNEL

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && \
    (XNN_ENABLE_KLEIDIAI || XNN_ENABLE_ARM_SME_ACLE)
std::vector<DWConvTestParams> CreateKaiTests(
    size_t cr, size_t output_height_tile,
    std::function<void(DWConvMicrokernelTester& tester)> test_func) {
  const std::string cbs = std::to_string(cr);
  std::vector<DWConvTestParams> tests;
  tests.reserve(45);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(9)
          .output_height_tile(output_height_tile)
          .height(9)
          .width(5)
          .channels(cr),
      test_func));

  if (cr > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(9)
            .output_height_tile(output_height_tile)
            .height(8)
            .width(5),
        test_func)
        .loop_channels(cr * 2, cr * 8, cr * 2));

    tests.push_back(DWConvTestParams(
        "c_lt_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(9)
            .output_height_tile(output_height_tile)
            .height(9)
            .width(5),
        test_func)
        .loop_channels(1, cr - 1));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + cbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(9)
          .output_height_tile(output_height_tile)
          .height(9)
          .width(5),
      test_func)
      .loop_channels(cr + 1, cr * 2 - 1));

  tests.push_back(DWConvTestParams(
      "c_gt_" + cbs + "_with_qmin",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(9)
          .output_height_tile(output_height_tile)
          .height(6)
          .width(5)
          .qmin(128),
      test_func)
      .loop_channels(cr + 1, cr * 2 - 1, std::max<size_t>(1, cr / 2)));

  tests.push_back(DWConvTestParams(
      "c_gt_" + cbs + "_with_qmax",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(9)
          .output_height_tile(output_height_tile)
          .height(6)
          .width(5)
          .qmax(128),
      test_func)
      .loop_channels(cr + 1, cr * 2 - 1, std::max<size_t>(1, cr / 2)));

  for (size_t width : {size_t{1}, size_t{2}, size_t{27}, size_t{28},
                       size_t{29}, size_t{32}}) {
    for (size_t height : {size_t{1}, size_t{2}, size_t{3}, size_t{4},
                          size_t{7}, size_t{9}}) {
      tests.push_back(DWConvTestParams(
          "h_" + std::to_string(height) + "_w_" + std::to_string(width),
          DWConvMicrokernelTester()
              .channel_tile(cr)
              .kernel_tile(9)
              .output_height_tile(output_height_tile)
              .height(height)
              .width(width),
          test_func)
          .loop_channels(1, cr * 3, std::max<size_t>(1, cr - 1)));
    }
  }

  for (size_t width : {size_t{5}, size_t{28}, size_t{29}}) {
    tests.push_back(DWConvTestParams(
        "no_padding_w_" + std::to_string(width),
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(9)
            .output_height_tile(output_height_tile)
            .height(6)
            .width(width)
            .padding_left(0)
            .padding_right(0)
            .padding_top(0)
            .padding_bottom(0),
        test_func)
        .loop_channels(1, cr * 2 + 1, std::max<size_t>(1, cr - 1)));
  }

  return tests;
}

#if XNN_ENABLE_KLEIDIAI
INSTANTIATE_TEST_SUITE_P(
    xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2, DWConvTest,
    testing::ValuesIn(CreateKaiTests(
        (xnn_init_hardware_config() &&
                 (xnn_init_hardware_config()->arch_flags & xnn_arch_arm_sme2)
             ? xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_channel_tile()
             : 16),
        (xnn_init_hardware_config() &&
                 (xnn_init_hardware_config()->arch_flags & xnn_arch_arm_sme2)
             ? xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_output_height_tile()
             : 4),
        [](DWConvMicrokernelTester& tester) {
          TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
          tester.Test(xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });
#endif  // XNN_ENABLE_KLEIDIAI

#if XNN_ENABLE_ARM_SME_ACLE
INSTANTIATE_TEST_SUITE_P(
    xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2, DWConvTest,
    testing::ValuesIn(CreateKaiTests(
        (xnn_init_hardware_config() &&
                 (xnn_init_hardware_config()->arch_flags & xnn_arch_arm_sme2)
             ? xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2_get_channel_tile()
             : 16),
        (xnn_init_hardware_config() &&
                 (xnn_init_hardware_config()->arch_flags & xnn_arch_arm_sme2)
             ? xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2_get_output_height_tile()
             : 4),
        [](DWConvMicrokernelTester& tester) {
          TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
          tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });
#endif  // XNN_ENABLE_ARM_SME_ACLE
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 &&
        // (XNN_ENABLE_KLEIDIAI || XNN_ENABLE_ARM_SME_ACLE)

}  // namespace
