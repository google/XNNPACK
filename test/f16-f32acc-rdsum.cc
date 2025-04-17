// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32acc-rdsum.yaml
//   Generator: tools/generate-reduce-discontiguous-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/rdsum-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C16, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C32, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__NEONFP16ARITH_C64, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__neonfp16arith_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_eq_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_div_128_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({256, 1024, 128})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_div_128_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_div_128_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_div_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 36, 7})
      .input_stride(2053)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_lt_128_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_lt_128_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_lt_128_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_lt_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_gt_128_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({129, 256})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_gt_128_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_gt_128_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, channels_gt_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 35, 14})
      .input_stride(269)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__F16C_C128, overflow_accumulator) {
    TEST_REQUIRES_X86_F16C;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({32, 128, 16})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({32, 128, 16})
      .rows({1, 36, 7})
      .input_stride(263)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(19)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({17, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({17, 32})
      .rows({1, 35, 14})
      .input_stride(47)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 16;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c16, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({64, 256, 32})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 36, 7})
      .input_stride(521)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(37)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({33, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({33, 64})
      .rows({1, 35, 14})
      .input_stride(79)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C32, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 32;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c32, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_eq_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_div_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({128, 512, 64})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_div_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_div_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_div_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({128, 512, 64})
      .rows({1, 36, 7})
      .input_stride(1031)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_lt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_lt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_lt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_lt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(67)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_gt_64_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({65, 128})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_gt_64_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_gt_64_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, channels_gt_64_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({65, 128})
      .rows({1, 35, 14})
      .input_stride(149)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C64, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 64;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 14})
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_eq_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows({1, 36, 7})
      .channels(channel_tile)
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_div_128_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({256, 1024, 128})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_div_128_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_div_128_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_div_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({256, 1024, 128})
      .rows({1, 36, 7})
      .input_stride(2053)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_lt_128_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({1, channel_tile})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_lt_128_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_lt_128_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_lt_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 36, 7})
      .input_stride(131)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_gt_128_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .rows(14)
      .channels({129, 256})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_gt_128_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_gt_128_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 35, 14})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, channels_gt_128_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512SKX;
    RDSumMicrokernelTester()
      .channels({129, 256})
      .rows({1, 35, 14})
      .input_stride(269)
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RDSUM_7P7X__AVX512SKX_C128, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t channel_tile = 128;
    RDSumMicrokernelTester()
      .rows(264)
      .channels({1, channel_tile*2})
      .Test(xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c128, xnn_init_f16_f32acc_scale_scalar_params);
  }
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
