// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_avx512bw.h"

#include "ynnpack/base/arch.h"
#include "ynnpack/base/simd/test/generic.h"

namespace ynn {
namespace simd {

TEST_PARTIAL_LOAD_STORE(x86_avx512bw, u8x64, arch_flag::avx512bw);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, s8x64, arch_flag::avx512bw);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, s16x32, arch_flag::avx512bw);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, f16x32, arch_flag::avx512bw);
TEST_PARTIAL_LOAD_STORE(x86_avx512bw, bf16x32, arch_flag::avx512bw);

TEST_ADD(x86_avx512bw, u8x64, arch_flag::avx512bw);
TEST_ADD(x86_avx512bw, s8x64, arch_flag::avx512bw);

TEST_SUBTRACT(x86_avx512bw, u8x64, arch_flag::avx512bw);
TEST_SUBTRACT(x86_avx512bw, s8x64, arch_flag::avx512bw);

TEST_MULTIPLY(x86_avx512bw, f32x16, arch_flag::avx512bw);
TEST_MULTIPLY(x86_avx512bw, s32x16, arch_flag::avx512bw);

TEST_MIN(x86_avx512bw, u8x64, arch_flag::avx512bw);
TEST_MIN(x86_avx512bw, s8x64, arch_flag::avx512bw);
TEST_MIN(x86_avx512bw, s16x32, arch_flag::avx512bw);

TEST_MAX(x86_avx512bw, u8x64, arch_flag::avx512bw);
TEST_MAX(x86_avx512bw, s8x64, arch_flag::avx512bw);
TEST_MAX(x86_avx512bw, s16x32, arch_flag::avx512bw);

}  // namespace simd
}  // namespace ynn
