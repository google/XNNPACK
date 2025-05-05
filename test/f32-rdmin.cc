// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rdmin.yaml
//   Generator: tools/generate-reduce-discontiguous-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/reduce-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__NEON_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__neon_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__SSE2_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__sse2_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__AVX512F_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__avx512f_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_2pass_fulltile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_2pass_subtile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_multipass_fulltile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_div_32_2pass_fulltile) {
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_div_32_2pass_subtile) {
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_div_32_multipass_fulltile) {
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_lt_32_2pass_fulltile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_lt_32_2pass_subtile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_lt_32_multipass_fulltile) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_gt_32_2pass_fulltile) {
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_gt_32_2pass_subtile) {
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_gt_32_multipass_fulltile) {
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__WASMSIMD_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__wasmsimd_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 4})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows({1, 11, 2})
      .channels(channel_tile)
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({64, 256, 32})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({64, 256, 32})
      .rows({1, 11, 2})
      .input_stride(521)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({1, channel_tile})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_HVX;
    const size_t channel_tile = 32;
    ReduceMicrokernelTester()
      .channels({1, channel_tile})
      .rows({1, 11, 2})
      .input_stride(37)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .rows(4)
      .channels({33, 64})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RDMIN_2P2X__HVX_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_HVX;
    ReduceMicrokernelTester()
      .channels({33, 64})
      .rows({1, 10, 4})
      .input_stride(79)
      .Test(xnn_f32_rdmin_ukernel_2p2x__hvx_c32, ReduceMicrokernelTester::OpType::Min);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_2pass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels(channel_tile)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_2pass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_2pass_subtile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 4})
    .channels(channel_tile)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_2pass_subtile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 4})
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_multipass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 11, 2})
    .channels(channel_tile)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_eq_2_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows({1, 11, 2})
    .channels(channel_tile)
    .input_stride(5)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_div_2_2pass_fulltile) {
  ReduceMicrokernelTester()
    .rows(4)
    .channels({4, 16, 2})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_div_2_2pass_subtile) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 4})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_div_2_multipass_fulltile) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 11, 2})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_div_2_multipass_fulltile_with_input_stride) {
  ReduceMicrokernelTester()
    .channels({4, 16, 2})
    .rows({1, 11, 2})
    .input_stride(37)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_lt_2_2pass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .rows(4)
    .channels({1, channel_tile})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_lt_2_2pass_subtile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 4})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_lt_2_multipass_fulltile) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 11, 2})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_lt_2_multipass_fulltile_with_input_stride) {
  const size_t channel_tile = 2;
  ReduceMicrokernelTester()
    .channels({1, channel_tile})
    .rows({1, 11, 2})
    .input_stride(5)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_gt_2_2pass_fulltile) {
  ReduceMicrokernelTester()
    .rows(4)
    .channels({3, 4})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_gt_2_2pass_subtile) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 4})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_gt_2_multipass_fulltile) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 10, 4})
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RDMIN_2P2X__SCALAR_C2, channels_gt_2_multipass_fulltile_with_input_stride) {
  ReduceMicrokernelTester()
    .channels({3, 4})
    .rows({1, 10, 4})
    .input_stride(17)
    .Test(xnn_f32_rdmin_ukernel_2p2x__scalar_c2, ReduceMicrokernelTester::OpType::Min);
}
