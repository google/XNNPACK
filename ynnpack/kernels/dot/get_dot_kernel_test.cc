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
#include "ynnpack/kernels/dot/cost_model/cost_model.h"
#include "ynnpack/kernels/dot/dot.h"
#if defined(YNN_ARCH_X86)
#include "ynnpack/kernels/dot/cost_model/redwood_cove.h"
#elif defined(YNN_ARCH_ARM64)
#include "ynnpack/kernels/dot/cost_model/donan_everest.h"
#elif defined(YNN_ARCH_ARM)
#include "ynnpack/kernels/dot/cost_model/cortex_a510.h"
#endif

namespace ynn {

#if defined(YNN_ARCH_X86)
static const dot_cost_models& test_cost_models = redwood_cove;
#elif defined(YNN_ARCH_ARM64)
static const dot_cost_models& test_cost_models = donan_everest;
#elif defined(YNN_ARCH_ARM)
static const dot_cost_models& test_cost_models = cortex_a510;
#else
static const dot_cost_models test_cost_models = {};
#endif

// Enable us to refer to kernels by name instead of by function pointer.
std::map<dot_kernel_fn, std::string> kernels = {
#define YNN_DOT_KERNEL(arch_flags, kernel, block_m, block_n, block_k, tile_m, \
                       tile_n, tile_k, flags, a_type, b_type, c_type,         \
                       cost_model)                                            \
  {kernel, #kernel},
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL
};

const std::string& get_dot_kernel_name(
    const dot_type& type, const dot_shape& shape, uint64_t arch_flags,
    int thread_count = 1, const dot_packed_shape& packed_shape = {}) {
  return kernels[get_dot_kernel(type, test_cost_models, shape, packed_shape,
                                /*consistent_arithmetic=*/false,
                                /*transpose_a=*/std::nullopt, arch_flags,
                                thread_count)
                     .kernel];
}

const std::string& get_dot_kernel_name(const dot_type& type,
                                       const dot_shape& shape,
                                       uint64_t arch_flags,
                                       const dot_packed_shape& packed_shape,
                                       int thread_count = 1) {
  return get_dot_kernel_name(type, shape, arch_flags, thread_count,
                             packed_shape);
}

#ifdef YNN_ARCH_X86

// We use a large highly composite value when we want to test large shapes, so
// it is unlikely that block shapes do not divide this extent.
const int m = 3 * 5 * 32;
const int n = 256;
const int k = 256;

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
    return get_dot_kernel_name(fp32, {1, n, k}, arch_flags);
  };
  auto fp32_2x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {2, n, k}, arch_flags);
  };
  auto fp32_3x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {3, n, k}, arch_flags);
  };
  auto fp32_4x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {4, n, k}, arch_flags);
  };
  auto fp32_6x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {6, n, k}, arch_flags);
  };
  auto fp32_8x = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {8, n, k}, arch_flags);
  };

  EXPECT_EQ(fp32_1x(arch_flags_sse2), "dot_fp32_1x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_1x(arch_flags_avx), "dot_fp32_1x32x1_1x8x1_avx");
  EXPECT_EQ(fp32_1x(arch_flags_fma3), "dot_fp32_1x32x1_1x8x1_fma3");
  EXPECT_EQ(fp32_1x(arch_flags_avx512), "dot_fp32_1x64x1_1x16x1_avx512");
  EXPECT_EQ(fp32_2x(arch_flags_sse2), "dot_fp32_2x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_2x(arch_flags_avx), "dot_fp32_2x32x1_1x8x1_avx");
  EXPECT_EQ(fp32_2x(arch_flags_fma3), "dot_fp32_2x32x1_1x8x1_fma3");
  EXPECT_EQ(fp32_2x(arch_flags_avx512), "dot_fp32_2x32x4_1x4x4_avx512");
  EXPECT_EQ(fp32_3x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_3x(arch_flags_avx), "dot_fp32_3x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_3x(arch_flags_fma3), "dot_fp32_3x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_3x(arch_flags_avx512), "dot_fp32_3x64x1_1x16x1_avx512");
  EXPECT_EQ(fp32_4x(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  EXPECT_EQ(fp32_4x(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_4x(arch_flags_fma3), "dot_fp32_4x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_4x(arch_flags_avx512), "dot_fp32_4x64x1_1x16x1_avx512");
  EXPECT_EQ(fp32_6x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_6x(arch_flags_avx), "dot_fp32_3x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_6x(arch_flags_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_6x(arch_flags_avx512), "dot_fp32_3x64x1_1x16x1_avx512");
  EXPECT_EQ(fp32_8x(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_8x(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_8x(arch_flags_fma3), "dot_fp32_4x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_8x(arch_flags_avx512), "dot_fp32_4x64x1_1x16x1_avx512");
}

TEST(get_dot_kernel, small_n) {
  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  // Test small n, large m, k
  auto fp32_x1 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 1, k}, arch_flags);
  };
  auto fp32_x2 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 2, k}, arch_flags);
  };
  auto fp32_x3 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 3, k}, arch_flags);
  };
  auto fp32_x4 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 4, k}, arch_flags);
  };
  auto fp32_x6 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 6, k}, arch_flags);
  };
  auto fp32_x8 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 8, k}, arch_flags);
  };
  EXPECT_EQ(fp32_x1(arch_flags_avx2_fma3), "dot_fp32_8x1x8_1x1x8_avx2_fma3");
  EXPECT_EQ(fp32_x1(arch_flags_avx512), "dot_fp32_8x1x8_1x1x8_avx2_fma3");
  EXPECT_EQ(fp32_x2(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x2(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  EXPECT_EQ(fp32_x2(arch_flags_avx2_fma3), "dot_fp32_8x1x8_1x1x8_avx2_fma3");
  EXPECT_EQ(fp32_x2(arch_flags_avx512), "dot_fp32_8x1x8_1x1x8_avx2_fma3");
  EXPECT_EQ(fp32_x3(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x3(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  EXPECT_EQ(fp32_x3(arch_flags_avx2_fma3), "dot_fp32_8x1x8_1x1x8_avx2_fma3");
  EXPECT_EQ(fp32_x3(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  EXPECT_EQ(fp32_x4(arch_flags_sse2), "dot_fp32_8x4x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x4(arch_flags_avx2), "dot_fp32_8x4x2_1x4x2_avx2");
  EXPECT_EQ(fp32_x4(arch_flags_avx2_fma3), "dot_fp32_8x4x2_1x4x2_avx2_fma3");
  EXPECT_EQ(fp32_x4(arch_flags_avx512), "dot_fp32_8x4x4_1x4x4_avx512");
  EXPECT_EQ(fp32_x6(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x6(arch_flags_avx), "dot_fp32_8x8x1_1x8x1_avx");
  EXPECT_EQ(fp32_x6(arch_flags_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  EXPECT_EQ(fp32_x6(arch_flags_avx512), "dot_fp32_6x8x4_1x4x4_avx512");
  EXPECT_EQ(fp32_x8(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x8(arch_flags_avx), "dot_fp32_8x8x1_1x8x1_avx");
  EXPECT_EQ(fp32_x8(arch_flags_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  EXPECT_EQ(fp32_x8(arch_flags_avx512), "dot_fp32_6x8x4_1x4x4_avx512");
}

TEST(get_dot_kernel, large) {
  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};
  auto fp32_large = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, n, k}, arch_flags);
  };
  EXPECT_EQ(fp32_large(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_large(arch_flags_avx), "dot_fp32_4x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_large(arch_flags_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_large(arch_flags_avx512), "dot_fp32_5x64x1_1x16x1_avx512");
}

TEST(get_dot_kernel, small_n_tile_k_1) {
  dot_packed_shape no_tile_k = {0, 1};

  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  auto fp32_x8 = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, 8, k}, arch_flags, no_tile_k);
  };
  EXPECT_EQ(fp32_x8(arch_flags_sse2), "dot_fp32_4x8x1_1x4x1_sse2");
  EXPECT_EQ(fp32_x8(arch_flags_avx2), "dot_fp32_8x8x1_1x8x1_avx");
  EXPECT_EQ(fp32_x8(arch_flags_avx2_fma3), "dot_fp32_8x8x1_1x8x1_fma3");
  EXPECT_EQ(fp32_x8(arch_flags_avx512), "dot_fp32_8x8x1_1x8x1_fma3");
}

TEST(get_dot_kernel, large_tile_k_1) {
  dot_packed_shape no_tile_k = {0, 1};

  dot_type fp32 = {ynn_type_fp32, ynn_type_fp32, ynn_type_fp32};

  auto fp32_large = [=](uint64_t arch_flags) {
    return get_dot_kernel_name(fp32, {m, n, k}, arch_flags, no_tile_k);
  };
  EXPECT_EQ(fp32_large(arch_flags_sse2), "dot_fp32_3x16x1_1x4x1_sse2");
  EXPECT_EQ(fp32_large(arch_flags_avx2), "dot_fp32_4x16x1_1x8x1_avx");
  EXPECT_EQ(fp32_large(arch_flags_avx2_fma3), "dot_fp32_6x16x1_1x8x1_fma3");
  EXPECT_EQ(fp32_large(arch_flags_avx512), "dot_fp32_5x64x1_1x16x1_avx512");
}

TEST(dot_kernel_state, destructor) {
  static bool destroyed = false;
  destroyed = false;
  {
    dot_kernel_state state;
    state.destroy = [](dot_kernel_state*) { destroyed = true; };
    EXPECT_FALSE(destroyed);
  }
  EXPECT_TRUE(destroyed);
}

#endif  // YNN_ARCH_X86

#ifdef YNN_ARCH_ARM64
TEST(get_dot_kernel_arm, int8_int4_int32) {
#if !defined(YNN_ARCH_ARM64_SME) || defined(YNN_DISABLE_SME)
  GTEST_SKIP() << "SME is not enabled in this build";
#else
  if (!is_arch_supported(arch_flag::sme)) {
    GTEST_SKIP() << "SME is not supported on this hardware";
  }

  dot_type int8_int4 = {ynn_type_int8, ynn_type_int4, ynn_type_int32};
  uint64_t arch = arch_flag::neon | arch_flag::neondot | arch_flag::neoni8mm |
                  arch_flag::sme;

  dot_kernel k_prefill =
      get_dot_kernel(int8_int4, test_cost_models, {512, 2048, 2048}, {},
                     /*required_flags=*/0, std::nullopt, arch);
  dot_packed_shape packed_shape = {k_prefill.block_n, k_prefill.tile_k};

  // For large m cases, we expect to use SME for 1 or many threads.
  EXPECT_EQ(get_dot_kernel_name(int8_int4, {512, 2048, 2048}, arch,
                                /*thread_count=*/1),
            "dot_int8_int4_int32_sme");
  EXPECT_EQ(get_dot_kernel_name(int8_int4, {512, 2048, 2048}, arch,
                                /*thread_count=*/4),
            "dot_int8_int4_int32_sme");

  // For small m cases, we expect to use SME for 1 thread, but neondot for many
  // threads.
  EXPECT_EQ(get_dot_kernel_name(int8_int4, {1, 2048, 2048}, arch, packed_shape,
                                /*thread_count=*/1),
            "dot_int8_int4_int32_sme");

  EXPECT_EQ(get_dot_kernel_name(int8_int4, {1, 2048, 2048}, arch, packed_shape,
                                /*thread_count=*/4),
            "dot_int8_int4_int32_1x32x8_1x4x8_neondot");
#endif
}

TEST(get_dot_kernel_arm, int8_int2_int32) {
#if !defined(YNN_ARCH_ARM64_SME) || defined(YNN_DISABLE_SME)
  GTEST_SKIP() << "SME is not enabled in this build";
#else
  if (!is_arch_supported(arch_flag::sme)) {
    GTEST_SKIP() << "SME is not supported on this hardware";
  }

  dot_type int8_int2 = {ynn_type_int8, ynn_type_int2, ynn_type_int32};
  uint64_t arch = arch_flag::neon | arch_flag::neondot | arch_flag::neoni8mm |
                  arch_flag::sme;

  dot_kernel k_prefill =
      get_dot_kernel(int8_int2, test_cost_models, {512, 2048, 2048}, {},
                     /*required_flags=*/0, std::nullopt, arch);
  dot_packed_shape packed_shape = {k_prefill.block_n, k_prefill.tile_k};

  // For large m cases, we expect to use SME for 1 or many threads.
  EXPECT_EQ(get_dot_kernel_name(int8_int2, {512, 2048, 2048}, arch,
                                /*thread_count=*/1),
            "dot_int8_int2_int32_sme");
  EXPECT_EQ(get_dot_kernel_name(int8_int2, {512, 2048, 2048}, arch,
                                /*thread_count=*/4),
            "dot_int8_int2_int32_sme");

  // For small m cases, we expect to use SME for 1 thread, but neondot for many
  // threads.
  EXPECT_EQ(get_dot_kernel_name(int8_int2, {1, 2048, 2048}, arch, packed_shape,
                                /*thread_count=*/1),
            "dot_int8_int2_int32_sme");
  EXPECT_EQ(get_dot_kernel_name(int8_int2, {1, 2048, 2048}, arch, packed_shape,
                                /*thread_count=*/4),
            "dot_int8_int2_int32_1x32x16_1x4x16_neondot");
#endif
}
#endif  // YNN_ARCH_ARM64

}  // namespace ynn
