// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_X86_FMA3_H_
#define XNNPACK_YNNPACK_BASE_SIMD_X86_FMA3_H_

#include <immintrin.h>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_avx_base.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

YNN_ALWAYS_INLINE f32x8 fma(f32x8 a, f32x8 b, f32x8 acc) {
  return f32x8{_mm256_fmadd_ps(a.v, b.v, acc.v)};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_X86_FMA3_H_
