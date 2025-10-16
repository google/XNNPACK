#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/x86_sse.h"

namespace ynn {

namespace simd {

struct s32x4x2 {
  s32x4 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  s32x4x2() = default;
  explicit s32x4x2(int32_t x) : v{x, x} {}

  YNN_ALWAYS_INLINE s32x4x2 operator+(s32x4x2 a) const {
    s32x4x2 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

// TODO(b/441600173): we could accumulate to s16x16 first and later convert to
// s32 when needed.
struct s32x4x4 {
  s32x4x2 v[2];

  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  s32x4x4() = default;
  explicit s32x4x4(int32_t x) : v{s32x4x2(x), s32x4x2(x)} {}

  YNN_ALWAYS_INLINE s32x4x4 operator+(s32x4x4 a) const {
    s32x4x4 res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }
};

YNN_ALWAYS_INLINE s32x4x2 load(const int32_t* ptr, s32x4x2,
                               decltype(s32x4x2::N)) {
  s32x4x2 x;

  x.v[0] = load(ptr, s32x4{}, s32x4::N);
  x.v[1] = load(ptr + s32x4::N, s32x4{});

  return x;
}

YNN_ALWAYS_INLINE s32x4x2 load(const int32_t* ptr, s32x4x2, size_t n) {
  s32x4x2 x;

  if (n < s32x4::N) {
    x.v[0] = load(ptr, s32x4{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, s32x4{}, s32x4::N);
    x.v[1] = load(ptr + s32x4::N, s32x4{}, n - s32x4::N);
  }

  return x;
}

YNN_ALWAYS_INLINE s32x4x4 load(const int32_t* ptr, s32x4x4,
                               decltype(s32x4x4::N)) {
  s32x4x4 x;

  x.v[0] = load(ptr, s32x4x2{}, s32x4x2::N);
  x.v[1] = load(ptr + s32x4x2::N, s32x4x2{}, s32x4x2::N);

  return x;
}

YNN_ALWAYS_INLINE s32x4x4 load(const int32_t* ptr, s32x4x4, size_t n) {
  s32x4x4 x;

  if (n < s32x4x2::N) {
    x.v[0] = load(ptr, s32x4x2{}, n);
    x.v[1] = s32x4x2(0);
  } else {
    x.v[0] = load(ptr, s32x4x2{}, s32x4x2::N);
    x.v[1] = load(ptr + s32x4x2::N, s32x4x2{}, n - s32x4x2::N);
  }

  return x;
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4x2 b, decltype(s32x4x2::N)) {
  store(ptr, b.v[0], decltype(s32x4::N){});
  store(ptr + s32x4::N, b.v[1], decltype(s32x4::N){});
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4x2 b, size_t n) {
  if (n < s32x4::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x4::N);
    store(ptr + s32x4::N, b.v[1], n - s32x4::N);
  }
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4x4 b, decltype(s32x4x4::N)) {
  store(ptr, b.v[0], decltype(s32x4x2::N){});
  store(ptr + s32x4x2::N, b.v[1], decltype(s32x4x2::N){});
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x4x4 b, size_t n) {
  if (n < s32x4x2::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x4x2::N);
    store(ptr + s32x4x2::N, b.v[1], n - s32x4x2::N);
  }
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_
