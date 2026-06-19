// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

#ifdef YNN_ARCH_X86

// Enable us to refer to kernels by name instead of by function pointer.
std::map<dot_kernel_fn, std::string> kernels = {
#define YNN_DOT_KERNEL(arch_flags, kernel, block_m, block_n, block_k, tile_m, \
                       tile_n, tile_k, flags, a_type, b_type, c_type)         \
  {kernel, #kernel},
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL
};

const std::string& get_dot_kernel_name(
    const dot_type& type, const dot_shape& shape, uint64_t arch_flags,
    const dot_packed_shape* packed_shape = nullptr) {
  return kernels[get_dot_kernel(type, shape, packed_shape,
                                /*consistent_arithmetic=*/false,
                                /*transpose_a=*/std::nullopt, arch_flags)
                     .kernel];
}

// We use a large highly composite value when we want to test large shapes, so
// it is unlikely that block shapes do not divide this extent.
const int large_shape = 3 * 5 * 7 * 64;

constexpr uint64_t arch_flags_sse2 = arch_flag::sse2;
constexpr uint64_t arch_flags_avx = arch_flag::avx | arch_flags_sse2;
constexpr uint64_t arch_flags_avx2 = arch_flag::avx2 | arch_flags_avx;
constexpr uint64_t arch_flags_fma3 = arch_flag::fma3 | arch_flags_avx;
constexpr uint64_t arch_flags_avx2_fma3 = arch_flags_avx2 | arch_flags_fma3;
constexpr uint64_t arch_flags_avx512 =
    arch_flag::avx512 | arch_flags_fma3 | arch_flags_avx2;

TEST(get_dot_kernel, small_m) {
  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  // Test small m, large k, n
  auto fp32_1x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {1, large_shape, large_shape}, arch_flags);
  };
  auto fp32_2x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {2, large_shape, large_shape}, arch_flags);
  };
  auto fp32_3x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {3, large_shape, large_shape}, arch_flags);
  };
  auto fp32_4x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {4, large_shape, large_shape}, arch_flags);
  };
  auto fp32_6x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {6, large_shape, large_shape}, arch_flags);
  };
  auto fp32_8x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {8, large_shape, large_shape}, arch_flags);
  };

  ASSERT_EQ(fp32_1x(arch_flags_sse2), "dot_fp32_1x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_1x(arch_flags_avx), "dot_fp32_1x32x1_1x8x1_avx");
  ASSERT_EQ(fp32_1x(arch_flags_fma3), "dot_fp32_1x32x1_1x8x1_fma3");
  ASSERT_EQ(fp32_1x(arch_flags_avx512), "dot_fp32_1x32x4_1x4x4_avx512");
  ASSERT_EQ(fp32_2x(arch_flags_sse2), "dot_fp32_2x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_2x(arch_flags_avx), "dot_fp32_2x32x1_1x8x1_avx");
  ASSERT_EQ(fp32_2x(arch_flags_fma3), "dot_fp32_2x32x1_1x8x1_fma3");
  ASSERT_EQ(fp32_2x(arch_flags_avx512), "dot_fp32_2x32x4_1x4x4_avx512");
  ASSERT_EQ(fp32_3x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_3x(arch_flags_avx), "dot_fp32_3x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_3x(arch_flags_fma3), "dot_fp32_3x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_3x(arch_flags_avx512), "dot_fp32_3x64x1_1x16x1_avx512");
  ASSERT_EQ(fp32_4x(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  ASSERT_EQ(fp32_4x(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_4x(arch_flags_fma3), "dot_fp32_4x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_4x(arch_flags_avx512), "dot_fp32_4x64x1_1x16x1_avx512");
  ASSERT_EQ(fp32_6x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_6x(arch_flags_avx), "dot_fp32_3x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_6x(arch_flags_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_6x(arch_flags_avx512), "dot_fp32_6x8x4_1x4x4_avx512");
  ASSERT_EQ(fp32_8x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_8x(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_8x(arch_flags_fma3), "dot_fp32_4x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_8x(arch_flags_avx512), "dot_fp32_4x64x1_1x16x1_avx512");
}

TEST(get_dot_kernel, small_n) {
  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  // Test small n, large m, k
  auto fp32_x1 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 1, large_shape}, arch_flags);
  };
  auto fp32_x2 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 2, large_shape}, arch_flags);
  };
  auto fp32_x3 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 3, large_shape}, arch_flags);
  };
  auto fp32_x4 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 4, large_shape}, arch_flags);
  };
  auto fp32_x6 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 6, large_shape}, arch_flags);
  };
  auto fp32_x8 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 8, large_shape}, arch_flags);
  };
  ASSERT_EQ(fp32_x1(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x1(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  ASSERT_EQ(fp32_x1(arch_flags_avx2_fma3), "dot_fp32_8x4x2_1x4x2_avx2_fma3");
  ASSERT_EQ(fp32_x1(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  ASSERT_EQ(fp32_x2(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x2(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  ASSERT_EQ(fp32_x2(arch_flags_avx2_fma3), "dot_fp32_8x4x2_1x4x2_avx2_fma3");
  ASSERT_EQ(fp32_x2(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  ASSERT_EQ(fp32_x3(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x3(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  ASSERT_EQ(fp32_x3(arch_flags_avx2_fma3), "dot_fp32_8x4x2_1x4x2_avx2_fma3");
  ASSERT_EQ(fp32_x3(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  ASSERT_EQ(fp32_x4(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x4(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  ASSERT_EQ(fp32_x4(arch_flags_avx2_fma3), "dot_fp32_8x4x2_1x4x2_avx2_fma3");
  ASSERT_EQ(fp32_x4(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  ASSERT_EQ(fp32_x6(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x6(arch_flags_avx), "dot_fp32_8x8x1_1x8x1_avx");
  ASSERT_EQ(fp32_x6(arch_flags_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  ASSERT_EQ(fp32_x6(arch_flags_avx512), "dot_fp32_6x8x4_1x4x4_avx512");
  ASSERT_EQ(fp32_x8(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x8(arch_flags_avx), "dot_fp32_8x8x1_1x8x1_avx");
  ASSERT_EQ(fp32_x8(arch_flags_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  ASSERT_EQ(fp32_x8(arch_flags_avx512), "dot_fp32_6x8x4_1x4x4_avx512");
}

TEST(get_dot_kernel, large) {
  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};
  auto fp32_large = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, large_shape, large_shape},
                               arch_flags);
  };
  ASSERT_EQ(fp32_large(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_large(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_large(arch_flags_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_large(arch_flags_avx512), "dot_fp32_5x64x1_1x16x1_avx512");
}

TEST(get_dot_kernel, small_n_tile_k_1) {
  dot_packed_shape no_tile_k = {0, 1};

  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  auto fp32_x8 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, 8, large_shape}, arch_flags,
                               &no_tile_k);
  };
  ASSERT_EQ(fp32_x8(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  ASSERT_EQ(fp32_x8(arch_flags_avx2), "dot_fp32_8x8x1_1x8x1_avx");
  ASSERT_EQ(fp32_x8(arch_flags_avx2_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  ASSERT_EQ(fp32_x8(arch_flags_avx512), "dot_fp32_8x8x1_1x8x1_fma3");
}

TEST(get_dot_kernel, large_tile_k_1) {
  dot_packed_shape no_tile_k = {0, 1};

  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  auto fp32_large = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {large_shape, large_shape, large_shape},
                               arch_flags, &no_tile_k);
  };
  ASSERT_EQ(fp32_large(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  ASSERT_EQ(fp32_large(arch_flags_avx2), "dot_fp32_4x16x1_1x8x1_avx");
  ASSERT_EQ(fp32_large(arch_flags_avx2_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  ASSERT_EQ(fp32_large(arch_flags_avx512), "dot_fp32_5x64x1_1x16x1_avx512");
}

#endif  // YNN_ARCH_X86

}  // namespace ynn
