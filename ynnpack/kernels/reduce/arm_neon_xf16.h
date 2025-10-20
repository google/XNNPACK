#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/arm.h"
#include "ynnpack/base/simd/multi_vec.h"

namespace ynn {

namespace simd {

using f32x4x2 = multi_vec<f32x4, 2>;
using f32x4x16 = multi_vec<f32x4, 16>;

template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x4x2 x, f32x4) {
  return f32x4{x.v[Index]};
}

template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x4x16 x, f32x4) {
  return f32x4{x.v[Index]};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_
