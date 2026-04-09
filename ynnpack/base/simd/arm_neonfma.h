// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFMA_H_
#define XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFMA_H_

#include <arm_neon.h>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/arm_neon.h"  // IWYU pragma: export

namespace ynn {

namespace simd {

YNN_ALWAYS_INLINE f32x4 fma(f32x4 a, f32x4 b, f32x4 acc) {
  return f32x4{vfmaq_f32(acc.v, a.v, b.v)};
}

#ifdef __aarch64__
YNN_ALWAYS_INLINE f64x2 fma(f64x2 a, f64x2 b, f64x2 acc) {
  return f64x2{vfmaq_f64(acc.v, a.v, b.v)};
}
#endif

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFMA_H_
