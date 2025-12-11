// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(x86_avx, u8x32, arch_flag::avx);
TEST_BROADCAST(x86_avx, s8x32, arch_flag::avx);
TEST_BROADCAST(x86_avx, s16x16, arch_flag::avx);
TEST_BROADCAST(x86_avx, f16x16, arch_flag::avx);
TEST_BROADCAST(x86_avx, bf16x16, arch_flag::avx);
TEST_BROADCAST(x86_avx, f32x8, arch_flag::avx);
TEST_BROADCAST(x86_avx, s32x8, arch_flag::avx);

TEST_LOAD_STORE(x86_avx, u8x32, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, s8x32, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, s16x16, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, f16x16, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, bf16x16, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, f32x8, arch_flag::avx);
TEST_LOAD_STORE(x86_avx, s32x8, arch_flag::avx);

TEST_ALIGNED_LOAD_STORE(x86_avx, u8x32, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, s8x32, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, s16x16, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, f16x16, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, bf16x16, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, f32x8, arch_flag::avx);
TEST_ALIGNED_LOAD_STORE(x86_avx, s32x8, arch_flag::avx);

TEST_PARTIAL_LOAD_STORE(x86_avx, u8x32, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, s8x32, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, s16x16, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, f16x16, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, bf16x16, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, f32x8, arch_flag::avx);
TEST_PARTIAL_LOAD_STORE(x86_avx, s32x8, arch_flag::avx);

TEST_ADD(x86_avx, f32x8, arch_flag::avx);
TEST_SUBTRACT(x86_avx, f32x8, arch_flag::avx);
TEST_MULTIPLY(x86_avx, f32x8, arch_flag::avx);
TEST_MIN(x86_avx, f32x8, arch_flag::avx);
TEST_MAX(x86_avx, f32x8, arch_flag::avx);
TEST_HORIZONTAL_MIN(x86_avx, f32x8, arch_flag::avx);
TEST_HORIZONTAL_MAX(x86_avx, f32x8, arch_flag::avx);

}  // namespace simd
}  // namespace ynn
