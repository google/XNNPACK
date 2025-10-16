#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_X86_AVX512_XF16_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_X86_AVX512_XF16_H_

#include <immintrin.h>

#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_avx512.h"

namespace ynn {

namespace simd {

struct f32x16x2 {
  f32x16 v[2];

  using value_type = float;
  static constexpr std::integral_constant<size_t, 32> N = {};

  f32x16x2() = default;
  explicit f32x16x2(float x) : v{x, x} {};

  YNN_ALWAYS_INLINE f32x16x2 operator+(f32x16x2 a) const {
    f32x16x2 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

YNN_ALWAYS_INLINE f32x16x2 load(const float* ptr, f32x16x2,
                                decltype(f32x16x2::N)) {
  f32x16x2 x;

  x.v[0] = load(ptr, f32x16{}, f32x16::N);
  x.v[1] = load(ptr + f32x16::N, f32x16{}, f32x16::N);

  return x;
}

YNN_ALWAYS_INLINE f32x16x2 load(const float* ptr, f32x16x2, size_t n) {
  f32x16x2 x;

  if (n <= f32x16::N) {
    x.v[0] = load(ptr, f32x16{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, f32x16{}, f32x16::N);
    x.v[1] = load(ptr + f32x16::N, f32x16{}, n - f32x16::N);
  }

  return x;
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x16x2 b, decltype(f32x16x2::N)) {
  store(ptr, b.v[0]);
  store(ptr + f32x16::N, b.v[1]);
}

YNN_ALWAYS_INLINE void store(float* ptr, f32x16x2 b, size_t n) {
  if (n <= f32x16::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0]);
    store(ptr + f32x16::N, b.v[1], n - f32x16::N);
  }
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_X86_AVX512_XF16_H_
