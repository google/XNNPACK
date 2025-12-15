// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse2.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(x86_sse2, u8x16, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, s8x16, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, s16x8, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, f16x8, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, bf16x8, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, f32x4, arch_flag::sse2);
TEST_BROADCAST(x86_sse2, s32x4, arch_flag::sse2);

TEST_LOAD_STORE(x86_sse2, u8x16, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, s8x16, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, s16x8, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, f16x8, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, bf16x8, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, f32x4, arch_flag::sse2);
TEST_LOAD_STORE(x86_sse2, s32x4, arch_flag::sse2);

TEST_ALIGNED_LOAD_STORE(x86_sse2, u8x16, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s8x16, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s16x8, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, f16x8, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, bf16x8, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, f32x4, arch_flag::sse2);
TEST_ALIGNED_LOAD_STORE(x86_sse2, s32x4, arch_flag::sse2);

TEST_PARTIAL_LOAD_STORE(x86_sse2, u8x16, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s8x16, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s16x8, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, f16x8, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, bf16x8, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, f32x4, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(x86_sse2, s32x4, arch_flag::sse2);

TEST_ADD(x86_sse2, u8x16, arch_flag::sse2);
TEST_ADD(x86_sse2, s8x16, arch_flag::sse2);
TEST_ADD(x86_sse2, s16x8, arch_flag::sse2);
TEST_ADD(x86_sse2, f32x4, arch_flag::sse2);
TEST_ADD(x86_sse2, s32x4, arch_flag::sse2);

TEST_SUBTRACT(x86_sse2, u8x16, arch_flag::sse2);
TEST_SUBTRACT(x86_sse2, s8x16, arch_flag::sse2);
TEST_SUBTRACT(x86_sse2, s16x8, arch_flag::sse2);
TEST_SUBTRACT(x86_sse2, f32x4, arch_flag::sse2);
TEST_SUBTRACT(x86_sse2, s32x4, arch_flag::sse2);

TEST_MULTIPLY(x86_sse2, f32x4, arch_flag::sse2);

TEST_MIN(x86_sse2, u8x16, arch_flag::sse2);
TEST_MIN(x86_sse2, s16x8, arch_flag::sse2);
TEST_MIN(x86_sse2, f32x4, arch_flag::sse2);

TEST_MAX(x86_sse2, u8x16, arch_flag::sse2);
TEST_MAX(x86_sse2, s16x8, arch_flag::sse2);
TEST_MAX(x86_sse2, f32x4, arch_flag::sse2);

TEST_HORIZONTAL_MIN(x86_sse2, u8x16, arch_flag::sse2);
TEST_HORIZONTAL_MIN(x86_sse2, s16x8, arch_flag::sse2);
TEST_HORIZONTAL_MIN(x86_sse2, f32x4, arch_flag::sse2);

TEST_HORIZONTAL_MAX(x86_sse2, u8x16, arch_flag::sse2);
TEST_HORIZONTAL_MAX(x86_sse2, s16x8, arch_flag::sse2);
TEST_HORIZONTAL_MAX(x86_sse2, f32x4, arch_flag::sse2);

TEST_CONVERT(x86_sse2, f32x8, bf16x8, arch_flag::sse2);
TEST_CONVERT(x86_sse2, s32x16, u8x16, arch_flag::sse2);
TEST_CONVERT(x86_sse2, s32x16, s8x16, arch_flag::sse2);

}  // namespace simd
}  // namespace ynn
