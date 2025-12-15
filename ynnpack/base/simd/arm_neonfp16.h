// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFP16_H_
#define XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFP16_H_

#include <arm_neon.h>

#include "ynnpack/base/simd/arm_neon.h"  // IWYU pragma: export
#include "ynnpack/base/simd/multi_vec.h"

namespace ynn {

namespace simd {

using f32x8 = multi_vec<f32x4, 2>;

YNN_ALWAYS_INLINE f32x8 convert(f16x8 a, float) {
  return {
      f32x4{vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(a.v)))},
      f32x4{vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(a.v)))},
  };
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFP16_H_
