// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-qc8w-dwconv-minmax-unipass-fp32.yaml
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


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_3P8C__ASM_AARCH32_NEONV8_MLA8_CORTEX_A35, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__asm_aarch32_neonv8_mla8_cortex_a35,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_3P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P8C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__neonv8_mla8_ld64,
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


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__ASM_AARCH32_NEONV8_MLA8_CORTEX_A35, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__asm_aarch32_neonv8_mla8_cortex_a35,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params,
                        xnn_qs8_requantize_fp32);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_V8;
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neon_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neonv8_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neonv8_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_4P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_4p8c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neon_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neonv8_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neonv8_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mul8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEONV8_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mul8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neon_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEON_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neon_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEON_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neon_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEON_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neon_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEON_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neon_mul8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEONV8_MLA8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mla8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEONV8_MLA8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mla8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEONV8_MUL8_LD64, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mul8_ld64,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEONV8_MUL8_LD128, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mul8_ld128,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__NEON_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__neon_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__NEONV8_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__neonv8_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P8C__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse2_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P8C__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__sse41_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__sse2_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__sse2_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__SSE2_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__SSE2_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__SSE41_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__SSE41_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__SSE41_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__AVX_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__avx_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__AVX_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__avx_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__AVX_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__avx_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__avx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_add16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_vpmovsx,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_add16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_vpmovsx,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__AVX_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__avx_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__AVX_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__avx_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__avx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx_mul16_add16,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul16_add16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul16_vpmovsx,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__AVX2_MUL16_ADD16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul16_add16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__AVX2_MUL16_VPMOVSX, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul16_vpmovsx,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__AVX2_MUL16_VPUNPCK, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul16_vpunpck,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__AVX2_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx2_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P32C__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p32c__avx512skx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__avx512skx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_9P32C__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__avx512skx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_25P32C__AVX512SKX_MUL32, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/32, /*adj_c_block=*/32, /*cr=*/32, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32,
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
      QS8_QC8W_DWCONV_MINMAX_FP32_3P16C__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P8C__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P16C__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P8C__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__WASMSIMD_MUL16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P16C__WASMSIMD_MUL16_ADD16, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/16, /*adj_c_block=*/16, /*cr=*/16, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16_add16,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_3P2C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P1C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p1c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P2C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_9P4C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p4c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P1C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P2C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QS8_QC8W_DWCONV_MINMAX_FP32_25P4C__WASM_FMAGIC, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p4c__wasm_fmagic,
                        xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                        xnn_qs8_requantize_fp32);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_3P1C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p1c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_3P2C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_3P2C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p2c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_4P2C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_4p2c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P1C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P1C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p1c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P1C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p1c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P2C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P2C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P2C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P4C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p4c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P4C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p4c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_9P4C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p4c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P1C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P1C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P1C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p1c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P2C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P2C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P2C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p2c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P4C__SCALAR_FMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p4c__scalar_fmagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P4C__SCALAR_IMAGIC, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p4c__scalar_imagic,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    QS8_QC8W_DWCONV_MINMAX_FP32_25P4C__SCALAR_LRINTF, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p4c__scalar_lrintf,
                      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
                      xnn_qs8_requantize_fp32);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });