// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-minmax-unipass.yaml
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
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
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
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}

std::vector<DWConvTestParams> CreateTests2(
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

  tests.push_back(DWConvTestParams(
      "c_eq_" + std::to_string(c_block * 2),
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .channels(c_block * 2)
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
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
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
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}

}  // namespace


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__ASM_AARCH64_NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__ASM_AARCH64_NEONFMA_CORTEX_A55, DWConvTest,
      testing::ValuesIn(CreateTests2(
          /*c_block=*/4, /*adj_c_block=*/8, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55,
                        xnn_init_f32_minmax_scalar_params);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__neon,
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
      F32_DWCONV_MINMAX_3P4C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__neon_acc2,
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
      F32_DWCONV_MINMAX_3P4C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__neonfma,
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
      F32_DWCONV_MINMAX_3P4C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_3P8C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__neon,
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
      F32_DWCONV_MINMAX_3P8C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__neon_acc2,
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
      F32_DWCONV_MINMAX_3P8C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__neonfma,
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
      F32_DWCONV_MINMAX_3P8C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_3P16C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__neon,
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
      F32_DWCONV_MINMAX_3P16C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__neon_acc2,
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
      F32_DWCONV_MINMAX_3P16C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__neonfma,
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
      F32_DWCONV_MINMAX_3P16C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_4P4C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__neon,
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
      F32_DWCONV_MINMAX_4P4C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__neon_acc2,
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
      F32_DWCONV_MINMAX_4P4C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__neonfma,
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
      F32_DWCONV_MINMAX_4P4C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_4P8C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__neon,
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
      F32_DWCONV_MINMAX_4P8C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__neon_acc2,
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
      F32_DWCONV_MINMAX_4P8C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma,
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
      F32_DWCONV_MINMAX_4P8C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_4P16C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__neon,
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
      F32_DWCONV_MINMAX_4P16C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__neon_acc2,
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
      F32_DWCONV_MINMAX_4P16C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__neonfma,
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
      F32_DWCONV_MINMAX_4P16C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_9P4C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__neon,
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
      F32_DWCONV_MINMAX_9P4C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__neon_acc2,
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
      F32_DWCONV_MINMAX_9P4C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma,
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
      F32_DWCONV_MINMAX_9P4C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_9P8C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__neon,
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
      F32_DWCONV_MINMAX_9P8C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__neon_acc2,
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
      F32_DWCONV_MINMAX_9P8C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma,
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
      F32_DWCONV_MINMAX_9P8C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_9P16C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__neon,
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
      F32_DWCONV_MINMAX_9P16C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__neon_acc2,
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
      F32_DWCONV_MINMAX_9P16C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma,
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
      F32_DWCONV_MINMAX_9P16C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_25P4C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__neon,
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
      F32_DWCONV_MINMAX_25P4C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__neon_acc2,
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
      F32_DWCONV_MINMAX_25P4C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__neonfma,
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
      F32_DWCONV_MINMAX_25P4C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_25P8C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__neon,
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
      F32_DWCONV_MINMAX_25P8C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__neon_acc2,
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
      F32_DWCONV_MINMAX_25P8C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma,
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
      F32_DWCONV_MINMAX_25P8C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_25P16C__NEON, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__neon,
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
      F32_DWCONV_MINMAX_25P16C__NEON_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__neon_acc2,
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
      F32_DWCONV_MINMAX_25P16C__NEONFMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__neonfma,
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
      F32_DWCONV_MINMAX_25P16C__NEONFMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__neonfma_acc2,
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
      F32_DWCONV_MINMAX_3P4C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__sse,
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
      F32_DWCONV_MINMAX_3P4C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__sse_acc2,
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
      F32_DWCONV_MINMAX_3P8C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__sse,
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
      F32_DWCONV_MINMAX_3P8C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__sse_acc2,
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
      F32_DWCONV_MINMAX_4P4C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__sse,
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
      F32_DWCONV_MINMAX_4P4C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__sse_acc2,
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
      F32_DWCONV_MINMAX_4P8C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__sse,
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
      F32_DWCONV_MINMAX_4P8C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__sse_acc2,
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
      F32_DWCONV_MINMAX_9P4C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__sse,
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
      F32_DWCONV_MINMAX_9P4C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__sse_acc2,
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
      F32_DWCONV_MINMAX_9P8C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__sse,
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
      F32_DWCONV_MINMAX_9P8C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__sse_acc2,
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
      F32_DWCONV_MINMAX_25P4C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__sse,
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
      F32_DWCONV_MINMAX_25P4C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__sse_acc2,
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
      F32_DWCONV_MINMAX_25P8C__SSE, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__sse,
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
      F32_DWCONV_MINMAX_25P8C__SSE_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__sse_acc2,
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
      F32_DWCONV_MINMAX_3P8C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__avx,
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
      F32_DWCONV_MINMAX_3P8C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__avx_acc2,
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
      F32_DWCONV_MINMAX_3P16C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__avx,
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
      F32_DWCONV_MINMAX_3P16C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__avx_acc2,
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
      F32_DWCONV_MINMAX_4P8C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__avx,
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
      F32_DWCONV_MINMAX_4P8C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__avx_acc2,
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
      F32_DWCONV_MINMAX_4P16C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__avx,
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
      F32_DWCONV_MINMAX_4P16C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__avx_acc2,
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
      F32_DWCONV_MINMAX_9P8C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__avx,
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
      F32_DWCONV_MINMAX_9P8C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__avx_acc2,
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
      F32_DWCONV_MINMAX_9P16C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__avx,
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
      F32_DWCONV_MINMAX_9P16C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__avx_acc2,
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
      F32_DWCONV_MINMAX_25P8C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__avx,
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
      F32_DWCONV_MINMAX_25P8C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__avx_acc2,
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
      F32_DWCONV_MINMAX_25P16C__AVX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__avx,
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
      F32_DWCONV_MINMAX_25P16C__AVX_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__avx_acc2,
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
      F32_DWCONV_MINMAX_3P8C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__fma3,
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
      F32_DWCONV_MINMAX_3P8C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__fma3_acc2,
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
      F32_DWCONV_MINMAX_3P16C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__fma3,
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
      F32_DWCONV_MINMAX_3P16C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__fma3_acc2,
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
      F32_DWCONV_MINMAX_4P8C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__fma3,
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
      F32_DWCONV_MINMAX_4P8C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__fma3_acc2,
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
      F32_DWCONV_MINMAX_4P16C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__fma3,
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
      F32_DWCONV_MINMAX_4P16C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__fma3_acc2,
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
      F32_DWCONV_MINMAX_9P8C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__fma3,
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
      F32_DWCONV_MINMAX_9P8C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__fma3_acc2,
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
      F32_DWCONV_MINMAX_9P16C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__fma3,
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
      F32_DWCONV_MINMAX_9P16C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__fma3_acc2,
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
      F32_DWCONV_MINMAX_25P8C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__fma3,
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
      F32_DWCONV_MINMAX_25P8C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__fma3_acc2,
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
      F32_DWCONV_MINMAX_25P16C__FMA3, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__fma3,
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
      F32_DWCONV_MINMAX_25P16C__FMA3_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__fma3_acc2,
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
      F32_DWCONV_MINMAX_3P16C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__avx512f,
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
      F32_DWCONV_MINMAX_3P16C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p16c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_3P32C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p32c__avx512f,
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
      F32_DWCONV_MINMAX_3P32C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p32c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_4P16C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__avx512f,
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
      F32_DWCONV_MINMAX_4P16C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p16c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_4P32C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p32c__avx512f,
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
      F32_DWCONV_MINMAX_4P32C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p32c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_9P16C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f,
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
      F32_DWCONV_MINMAX_9P16C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_9P32C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p32c__avx512f,
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
      F32_DWCONV_MINMAX_9P32C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p32c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_25P16C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f,
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
      F32_DWCONV_MINMAX_25P16C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_25P32C__AVX512F, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p32c__avx512f,
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
      F32_DWCONV_MINMAX_25P32C__AVX512F_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p32c__avx512f_acc2,
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
      F32_DWCONV_MINMAX_3P4C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMSIMD_ARM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_arm,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMSIMD_ARM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_arm_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMSIMD_X86, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_x86,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMSIMD_X86_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_x86_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P4C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p4c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P8C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P4C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p4c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P8C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P4C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p4c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P8C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P4C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMRELAXEDSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMRELAXEDSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P8C__WASMRELAXEDSIMD_FMA_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma_acc2,
                        xnn_init_f32_minmax_wasmsimd_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P1C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p1c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P1C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p1c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P2C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p2c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_3P2C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_3p2c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P1C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p1c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P1C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p1c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P2C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p2c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_4P2C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_4p2c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P1C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p1c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P1C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p1c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P2C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p2c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_9P2C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_9p2c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P1C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p1c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P1C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p1c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P2C__WASM, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p2c__wasm,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_MINMAX_25P2C__WASM_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_minmax_ukernel_25p2c__wasm_acc2,
                        xnn_init_f32_minmax_scalar_params);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_3P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_3p1c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_3P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_3p1c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_3P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_3p2c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_3P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_3p2c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_4P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_4p1c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_4P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_4P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_4p2c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_4P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_4p2c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_9P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_9p1c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_9P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_9P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_9p2c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_9P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_9p2c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_25P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_25p1c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_25P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_25p1c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_25P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_25p2c__scalar,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_MINMAX_25P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_minmax_ukernel_25p2c__scalar_acc2,
                      xnn_init_f32_minmax_scalar_params);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });