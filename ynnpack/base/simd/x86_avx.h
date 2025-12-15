// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_

#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

using f32x16 = multi_vec<f32x8, 2>;
using s32x16 = multi_vec<s32x8, 2>;
using bf16x32 = multi_vec<bf16x16, 2>;
using f16x32 = multi_vec<f16x16, 2>;
using s8x64 = multi_vec<s8x32, 2>;
using u8x64 = multi_vec<u8x32, 2>;

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_AVX_H_
