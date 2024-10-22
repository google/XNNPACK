// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: qs8-qc8w-dwconv-minmax-unipass-fp32
//   Generator: tools/generate-dwconv-unipass-test.py


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
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}

}  // namespace

#define XNN_DWCONV_UNIPASS(arch_flags, ukernel, c_block, is_pipelined, cr, kr, datatype, weights_type, params_type, init_params)\
INSTANTIATE_TEST_SUITE_P(                                                                                                       \
    ukernel, DWConvTest,                                                                                                        \
    testing::ValuesIn(CreateTests(                                                                                              \
        c_block, is_pipelined, cr, kr,                                                                                          \
        [](DWConvMicrokernelTester& tester) {                                                                                   \
          TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
          tester.Test(ukernel, init_params, xnn_qs8_requantize_fp32);                                                           \
        })),                                                                                                                    \
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {                                                             \
      return info.param.test_name;                                                                                              \
    });
#include "qs8-qc8w-dwconv/qs8-qc8w-dwconv-minmax-unipass-fp32.h"
#undef XNN_UKERNEL_WITH_PARAMS
