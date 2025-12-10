#ifndef XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"

namespace ynn {

namespace simd {

// This struct allows to easily create multiples of natural vectors: Vec X M.
template <typename Vec, size_t M>
struct multi_vec {
  std::array<Vec, M> v;

  using value_type = typename Vec::value_type;
  static constexpr std::integral_constant<size_t, M * Vec::N> N = {};

  multi_vec() = default;
  explicit multi_vec(value_type x) {
    for (size_t i = 0; i < M; ++i) {
      v[i] = Vec(x);
    }
  }

  YNN_ALWAYS_INLINE multi_vec operator+(multi_vec a) const {
    multi_vec res;

    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      res.v[i] = this->v[i] + a.v[i];
    }

    return res;
  }

  YNN_ALWAYS_INLINE multi_vec operator*(multi_vec a) const {
    multi_vec res;

    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      res.v[i] = this->v[i] * a.v[i];
    }

    return res;
  }

  YNN_ALWAYS_INLINE multi_vec operator+=(multi_vec a) {
    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      this->v[i] = this->v[i] + a.v[i];
    }

    return *this;
  }

  YNN_ALWAYS_INLINE multi_vec operator*=(multi_vec a) {
    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      this->v[i] = this->v[i] * a.v[i];
    }

    return *this;
  }
};

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE multi_vec<Vec, M> load(const typename Vec::value_type* ptr,
    multi_vec<Vec, M>, decltype(multi_vec<Vec, M>::N)) {
  multi_vec<Vec, M> x;

  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    x.v[i] = load(ptr + i * Vec::N, Vec{});
  }

  return x;
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE multi_vec<Vec, M> load(const typename Vec::value_type* ptr,
    multi_vec<Vec, M>, size_t n) {
  multi_vec<Vec, M> x;

  Vec zero(0);
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    if (Vec::N <= n) {
      x.v[i] = load(ptr + i * Vec::N, Vec{});
    } else if (n > 0) {
      x.v[i] = load(ptr + i * Vec::N, zero, n);
    }
    n = sub_sat(n, Vec::N);
  }

  return x;
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr,
    multi_vec<Vec, M> b, decltype(multi_vec<Vec, M>::N)) {
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    store(ptr + i * Vec::N, b.v[i]);
  }
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr,
    multi_vec<Vec, M> b, size_t n) {
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    if (Vec::N <= n) {
      store(ptr + i * Vec::N, b.v[i]);
    } else if (n > 0) {
      store(ptr + i * Vec::N, b.v[i], n % Vec::N);
    }
    n = sub_sat(n, Vec::N);
  }
}

template <int Index, typename Vec>
Vec extract(Vec x, Vec) {
  static_assert(Index == 0, "");
  return x;
}

template <int Index, typename Vec, typename ResultVec, size_t M>
ResultVec extract(multi_vec<Vec, M> b, ResultVec) {
  static_assert(Index * ResultVec::N < M * Vec::N);

  return extract<Index % (Vec::N / ResultVec::N)>(
      b.v[(Index * ResultVec::N) / Vec::N], ResultVec{});
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_
