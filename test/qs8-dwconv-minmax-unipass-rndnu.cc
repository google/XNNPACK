// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-dwconv-minmax-unipass-rndnu.yaml
//   Generator: tools/generate-dwconv-unipass-test.py


#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/requantization.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "dwconv-microkernel-tester.h"
#include <gtest/gtest.h>

namespace {

std::vector<DWConvTestParams> CreateTests1(
    size_t c_block, size_t adj_c_block, size_t cr, size_t kr,
    std::function<void(DWConvMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
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
      , test_func, isa_check));


  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmin",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmin(128)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_div_" + cbs + "_with_qmax",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .qmax(128)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

    tests.push_back(DWConvTestParams(
        "c_lt_" + acbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func, isa_check)
      .loop_channels(1, adj_c_block - 1));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_with_qmin",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .qmin(128)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs + "_with_qmax",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .qmax(128)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));

  tests.push_back(DWConvTestParams(
      "multipixel",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
        , test_func, isa_check)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1))
        .loop_step(2, kr));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(5)
          .output_stride(NextPrime(cr * 5 + 1))
      , test_func, isa_check)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_qmin",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
          .qmin(128)
      , test_func, isa_check)
    .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_qmax",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
          .qmax(128)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));


  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}

}  // namespace


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
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
      QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16,
                        xnn_init_qs8_conv_minmax_rndnu_neon_params,
                        xnn_qs8_requantize_rndnu);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


INSTANTIATE_TEST_SUITE_P(
    QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar,
                      xnn_init_qs8_conv_minmax_rndnu_scalar_params,
                      xnn_qs8_requantize_rndnu);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar,
                      xnn_init_qs8_conv_minmax_rndnu_scalar_params,
                      xnn_qs8_requantize_rndnu);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar,
                      xnn_init_qs8_conv_minmax_rndnu_scalar_params,
                      xnn_qs8_requantize_rndnu);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });