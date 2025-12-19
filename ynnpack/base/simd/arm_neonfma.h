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

}  // namespace simd

}  // namespace ynn

#include "ynnpack/base/simd/generic.inc"  // IWYU pragma: export

#endif  // XNNPACK_YNNPACK_BASE_SIMD_ARM_NEONFMA_H_
