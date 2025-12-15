// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_F16C_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_F16C_H_

#include <immintrin.h>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/x86_avx.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

YNN_ALWAYS_INLINE f32x8 convert(f16x8 a, float) {
  return f32x8{_mm256_cvtph_ps(a.v)};
}

YNN_ALWAYS_INLINE f32x16 convert(f16x16 a, float) {
  return {
      convert(extract<0>(a, f16x8{}), float{}),
      convert(extract<1>(a, f16x8{}), float{}),
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_F16C_H_
