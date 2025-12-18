// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse2.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_BROADCAST(multi_vec, u8x32, arch_flag::sse2);
TEST_BROADCAST(multi_vec, s8x32, arch_flag::sse2);
TEST_BROADCAST(multi_vec, s16x16, arch_flag::sse2);
TEST_BROADCAST(multi_vec, f16x16, arch_flag::sse2);
TEST_BROADCAST(multi_vec, bf16x16, arch_flag::sse2);
TEST_BROADCAST(multi_vec, f32x8, arch_flag::sse2);
TEST_BROADCAST(multi_vec, s32x8, arch_flag::sse2);

TEST_LOAD_STORE(multi_vec, u8x32, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, s8x32, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, s16x16, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, f16x16, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, bf16x16, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, f32x8, arch_flag::sse2);
TEST_LOAD_STORE(multi_vec, s32x8, arch_flag::sse2);

TEST_PARTIAL_LOAD_STORE(multi_vec, u8x32, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, s8x32, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, s16x16, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, f16x16, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, bf16x16, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, f32x8, arch_flag::sse2);
TEST_PARTIAL_LOAD_STORE(multi_vec, s32x8, arch_flag::sse2);

TEST_ADD(multi_vec, u8x32, arch_flag::sse2);
TEST_ADD(multi_vec, s8x32, arch_flag::sse2);
TEST_ADD(multi_vec, s16x16, arch_flag::sse2);
TEST_ADD(multi_vec, f32x8, arch_flag::sse2);
TEST_ADD(multi_vec, s32x8, arch_flag::sse2);

TEST_MULTIPLY(multi_vec, f32x8, arch_flag::sse2);

TEST_EXTRACT(multi_vec, s32x4, s32x8, arch_flag::sse2);
TEST_EXTRACT(multi_vec, f32x4, f32x8, arch_flag::sse2);
TEST_EXTRACT(multi_vec, bf16x8, bf16x16, arch_flag::sse2);
TEST_EXTRACT(multi_vec, f16x8, f16x16, arch_flag::sse2);
TEST_EXTRACT(multi_vec, s8x16, s8x32, arch_flag::sse2);
TEST_EXTRACT(multi_vec, u8x16, u8x32, arch_flag::sse2);

TEST_CONCAT(multi_vec, s32x4, arch_flag::sse2);
TEST_CONCAT(multi_vec, f32x4, arch_flag::sse2);
TEST_CONCAT(multi_vec, bf16x8, arch_flag::sse2);
TEST_CONCAT(multi_vec, f16x8, arch_flag::sse2);
TEST_CONCAT(multi_vec, s8x16, arch_flag::sse2);
TEST_CONCAT(multi_vec, u8x16, arch_flag::sse2);

}  // namespace simd
}  // namespace ynn
