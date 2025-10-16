#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_

#include "ynnpack/base/simd/arm.h"
#include "ynnpack/base/base.h"

namespace ynn {

namespace simd {

struct f32x4x2 {
  f32x4 v[2];

  using value_type = float;
  static constexpr std::integral_constant<size_t, 8> N = {};

  f32x4x2() = default;
  explicit f32x4x2(float x) : v{x, x} {};

  YNN_ALWAYS_INLINE f32x4x2 operator+(f32x4x2 a) const {
    f32x4x2 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

YNN_ALWAYS_INLINE f32x4x2 load(const float* ptr, f32x4x2,
                               decltype(f32x4x2::N)) {
  f32x4x2 x;

  x.v[0] = load(ptr, f32x4{}, f32x4::N);
  x.v[1] = load(ptr + f32x4::N, f32x4{}, f32x4::N);

  return x;
}

YNN_ALWAYS_INLINE f32x4x2 load(const float* ptr, f32x4x2, size_t n) {
  f32x4x2 x;

  if (n < f32x4::N) {
    x.v[0] = load(ptr, f32x4{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, f32x4{});
    x.v[1] = load(ptr + f32x4::N, f32x4{}, n - f32x4::N);
  }

  return x;
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4x2 b, decltype(f32x4x2::N)) {
  store(ptr, b.v[0], f32x4::N);
  store(ptr + f32x4::N, b.v[1], f32x4::N);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x4x2 b, size_t n) {
  if (n < f32x4::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0]);
    store(ptr + f32x4::N, b.v[1], n - f32x4::N);
  }
}

template <int Index>
YNN_ALWAYS_INLINE f32x4 extract(f32x4x2 x, f32x4) {
  return f32x4{x.v[Index]};
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_ARM_NEON_XF16_H_
