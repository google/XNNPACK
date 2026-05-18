// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/isa-checks.h"
#include "test/ibilinear-microkernel-tester.h"
#include "test/next_prime.h"

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
namespace {

inline size_t ChannelTile(size_t vector_tile) {
  return vector_tile * xnn_init_hardware_config()->vlenb / sizeof(float);
}

inline size_t StridedChannels(size_t channel_tile) {
  return xnnpack::NextPrime(channel_tile * 5 + 1);
}

#define XNN_TEST_F32_IBILINEAR_RVV_UKERNEL(UKERNEL, TEST_NAME, VECTOR_TILE, CHANNEL_STEP) \
  TEST(TEST_NAME, channels_eq_##VECTOR_TILE##v) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    IBilinearMicrokernelTester() \
      .pixels(1) \
      .channels(channel_tile) \
      .Test(UKERNEL); \
  } \
  \
  TEST(TEST_NAME, channels_div_##VECTOR_TILE##v) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    for (size_t channels = channel_tile * 2; channels < channel_tile * 10; channels += channel_tile) { \
      IBilinearMicrokernelTester() \
        .pixels(1) \
        .channels(channels) \
        .Test(UKERNEL); \
    } \
  } \
  \
  TEST(TEST_NAME, channels_lt_##VECTOR_TILE##v) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    for (size_t channels = 1; channels < channel_tile; channels++) { \
      IBilinearMicrokernelTester() \
        .pixels(1) \
        .channels(channels) \
        .Test(UKERNEL); \
    } \
  } \
  \
  TEST(TEST_NAME, channels_gt_##VECTOR_TILE##v) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    for (size_t channels = channel_tile + 1; channels < channel_tile * 2; channels++) { \
      IBilinearMicrokernelTester() \
        .pixels(1) \
        .channels(channels) \
        .Test(UKERNEL); \
    } \
  } \
  \
  TEST(TEST_NAME, pixels_gt_1) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    for (size_t pixels = 2; pixels < 3; pixels++) { \
      for (size_t channels = 1; channels <= channel_tile * 5; channels += CHANNEL_STEP) { \
        IBilinearMicrokernelTester() \
          .pixels(pixels) \
          .channels(channels) \
          .Test(UKERNEL); \
      } \
    } \
  } \
  \
  TEST(TEST_NAME, input_offset) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    const size_t input_offset = StridedChannels(channel_tile); \
    for (size_t pixels = 1; pixels < 5; pixels++) { \
      for (size_t channels = 1; channels <= channel_tile * 5; channels += CHANNEL_STEP) { \
        IBilinearMicrokernelTester() \
          .pixels(pixels) \
          .channels(channels) \
          .input_offset(input_offset) \
          .Test(UKERNEL); \
      } \
    } \
  } \
  \
  TEST(TEST_NAME, output_stride) { \
    TEST_REQUIRES_ARCH_FLAGS(xnn_arch_riscv_vector); \
    const size_t channel_tile = ChannelTile(VECTOR_TILE); \
    const size_t output_stride = StridedChannels(channel_tile); \
    for (size_t pixels = 1; pixels < 5; pixels++) { \
      for (size_t channels = 1; channels <= channel_tile * 5; channels += CHANNEL_STEP) { \
        IBilinearMicrokernelTester() \
          .pixels(pixels) \
          .channels(channels) \
          .output_stride(output_stride) \
          .Test(UKERNEL); \
      } \
    } \
  }

XNN_TEST_F32_IBILINEAR_RVV_UKERNEL(
    xnn_f32_ibilinear_ukernel__rvv_u1v, F32_IBILINEAR__RVV_U1V, 1, 1)
XNN_TEST_F32_IBILINEAR_RVV_UKERNEL(
    xnn_f32_ibilinear_ukernel__rvv_u2v, F32_IBILINEAR__RVV_U2V, 2, 1)
XNN_TEST_F32_IBILINEAR_RVV_UKERNEL(
    xnn_f32_ibilinear_ukernel__rvv_u4v, F32_IBILINEAR__RVV_U4V, 4, 3)

#undef XNN_TEST_F32_IBILINEAR_RVV_UKERNEL

}  // namespace
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
