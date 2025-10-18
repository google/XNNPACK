#ifndef XNNPACK_YNNPACK_BASE_SIMD_MULTIPLE_OF_H_
#define XNNPACK_YNNPACK_BASE_SIMD_MULTIPLE_OF_H_

#include <array>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/base.h"

namespace ynn {

namespace simd {

template <typename Vec, size_t N_>
struct multiple_of {
  std::array<Vec, N_> v;

  using value_type = typename Vec::value_type;
  static constexpr std::integral_constant<size_t, N_ * Vec::N> N = {};

  multiple_of() = default;
  explicit multiple_of(value_type x) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 999
#endif
    for (size_t i = 0; i < N_; ++i) {
      v[i] = Vec(x);
    }
  };

  YNN_ALWAYS_INLINE multiple_of operator+(multiple_of a) const {
    multiple_of res;

#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 999
#endif
    for (size_t i = 0; i < N_; ++i) {
      res.v[i] = this->v[i] + a.v[i];
    }

    return res;
  }
};

template <typename Vec, size_t N>
YNN_ALWAYS_INLINE multiple_of<Vec, N> load(const typename Vec::value_type* ptr,
    multiple_of<Vec, N>, decltype(multiple_of<Vec, N>::N)) {
  multiple_of<Vec, N> x;

#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 999
#endif
  for (size_t i = 0; i < N; ++i) {
    x.v[i] = load(ptr + i * Vec::N, Vec{});
  }

  return x;
}

template <typename Vec, size_t N>
YNN_ALWAYS_INLINE multiple_of<Vec, N> load(const typename Vec::value_type* ptr,
    multiple_of<Vec, N>, size_t n) {
  multiple_of<Vec, N> x;

  size_t i = 0;
  for (; n >= Vec::N; n -= Vec::N, i++) {
    x.v[i] = load(ptr + i * Vec::N, Vec{});
  }

  Vec zero(0);
  if (n > 0) {
    x.v[i] = load(ptr + i * Vec::N, zero, n % Vec::N);
    i++;
  }

  for (; i < N; ++i) {
    x.v[i] = zero;
  }

  return x;
}

template <typename Vec, size_t N>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr,
    multiple_of<Vec, N> b, decltype(multiple_of<Vec, N>::N)) {
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 999
#endif
  for (size_t i = 0; i < N; ++i) {
    store(ptr + i * Vec::N, b.v[i]);
  }
}

template <typename Vec, size_t N>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr,
    multiple_of<Vec, N> b, size_t n) {
  size_t i = 0;
  for (; n >= Vec::N; n -= Vec::N, i++) {
    store(ptr + i * Vec::N, b.v[i]);
  }

  if (n > 0) {
    store(ptr + i * Vec::N, b.v[i], n % Vec::N);
  }
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_MULTIPLE_OF_H_
