// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/arch.h"

#include "ynnpack/base/half.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/test/generic.h"

#include "ynnpack/base/simd/x86_avx512f.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(x86_avx512f, u8x64, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, s8x64, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, s16x32, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, f16x32, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, bf16x32, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_BROADCAST(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_LOAD_STORE(x86_avx512f, u8x64, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, s8x64, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, s16x32, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, f16x32, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, bf16x32, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_LOAD_STORE(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_ALIGNED_LOAD_STORE(x86_avx512f, u8x64, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, s8x64, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, s16x32, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, f16x32, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, bf16x32, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_ALIGNED_LOAD_STORE(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_PARTIAL_LOAD_STORE(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_PARTIAL_LOAD_STORE(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_ADD(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_ADD(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_SUBTRACT(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_SUBTRACT(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_MULTIPLY(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_MULTIPLY(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_MIN(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_MIN(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_MAX(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_MAX(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_EXTRACT(x86_avx512f, s32x4, s32x16, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, f32x4, f32x16, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, s8x16, s8x64, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, u8x16, u8x64, arch_flag::avx512f);

TEST_EXTRACT(x86_avx512f, bf16x16, bf16x32, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, f16x16, f16x32, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, s8x32, s8x64, arch_flag::avx512f);
TEST_EXTRACT(x86_avx512f, u8x32, u8x64, arch_flag::avx512f);

TEST_CONCAT(x86_avx512f, bf16x16, arch_flag::avx512f);
TEST_CONCAT(x86_avx512f, f16x16, arch_flag::avx512f);
TEST_CONCAT(x86_avx512f, s8x32, arch_flag::avx512f);
TEST_CONCAT(x86_avx512f, u8x32, arch_flag::avx512f);

TEST_CONVERT(x86_avx512f, f32x16, bf16x16, arch_flag::avx512f);
TEST_CONVERT(x86_avx512f, f32x16, f16x16, arch_flag::avx512f);

TEST_HORIZONTAL_MIN(x86_avx512f, u8x64, arch_flag::avx512f);
TEST_HORIZONTAL_MIN(x86_avx512f, s8x64, arch_flag::avx512f);
TEST_HORIZONTAL_MIN(x86_avx512f, s16x32, arch_flag::avx512f);
TEST_HORIZONTAL_MIN(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_HORIZONTAL_MIN(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_HORIZONTAL_MAX(x86_avx512f, u8x64, arch_flag::avx512f);
TEST_HORIZONTAL_MAX(x86_avx512f, s8x64, arch_flag::avx512f);
TEST_HORIZONTAL_MAX(x86_avx512f, s16x32, arch_flag::avx512f);
TEST_HORIZONTAL_MAX(x86_avx512f, f32x16, arch_flag::avx512f);
TEST_HORIZONTAL_MAX(x86_avx512f, s32x16, arch_flag::avx512f);

TEST_FMA(x86_avx512f, f32x16, arch_flag::avx512f);

}  // namespace simd
}  // namespace ynn
