#ifndef XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

namespace simd {

// This struct allows to easily create multiples of natural vectors: Vec X M.
template <typename Vec, size_t M>
struct multi_vec {
  std::array<Vec, M> v;

  using value_type = typename Vec::value_type;
  static constexpr std::integral_constant<size_t, M * Vec::N> N = {};

  multi_vec() = default;
  YNN_ALWAYS_INLINE explicit multi_vec(Vec x) {
    for (size_t i = 0; i < M; ++i) {
      v[i] = x;
    }
  }
  template <typename... Args>
  YNN_ALWAYS_INLINE multi_vec(Vec v0, Vec v1, Args... args)
      : v{v0, v1, args...} {}

  YNN_ALWAYS_INLINE multi_vec& operator=(Vec x) {
    for (size_t i = 0; i < M; ++i) {
      v[i] = x;
    }
    return *this;
  }

  YNN_ALWAYS_INLINE Vec& operator[](size_t i) { return v[i]; }
  YNN_ALWAYS_INLINE Vec operator[](size_t i) const { return v[i]; }

  YNN_ALWAYS_INLINE multi_vec& operator+=(multi_vec a) {
    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      v[i] = v[i] + a[i];
    }

    return *this;
  }

  YNN_ALWAYS_INLINE multi_vec& operator*=(multi_vec a) {
    YNN_UNROLL
    for (size_t i = 0; i < M; ++i) {
      v[i] = v[i] * a[i];
    }

    return *this;
  }

  YNN_ALWAYS_INLINE multi_vec operator+(multi_vec a) const {
    multi_vec res(*this);
    res += a;
    return res;
  }

  YNN_ALWAYS_INLINE multi_vec operator*(multi_vec a) const {
    multi_vec res(*this);
    res *= a;
    return res;
  }
};

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE multi_vec<Vec, M> load(const typename Vec::value_type* ptr,
                                         multi_vec<Vec, M>,
                                         decltype(multi_vec<Vec, M>::N) = {}) {
  multi_vec<Vec, M> x;

  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    x[i] = load(ptr + i * Vec::N, Vec{});
  }

  return x;
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE multi_vec<Vec, M> load(const typename Vec::value_type* ptr,
    multi_vec<Vec, M> src, size_t n) {
  multi_vec<Vec, M> x;

  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    if (Vec::N <= n) {
      x[i] = load(ptr + i * Vec::N, Vec{});
    } else if (n >= 0) {
      x[i] = load(ptr + i * Vec::N, src[i], n);
    }
    n = sub_sat(n, Vec::N);
  }

  return x;
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr, multi_vec<Vec, M> b,
                             decltype(multi_vec<Vec, M>::N) = {}) {
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    store(ptr + i * Vec::N, b[i]);
  }
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE void store(typename Vec::value_type* ptr,
    multi_vec<Vec, M> b, size_t n) {
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    if (Vec::N <= n) {
      store(ptr + i * Vec::N, b[i]);
    } else if (n > 0) {
      store(ptr + i * Vec::N, b[i], n % Vec::N);
    }
    n = sub_sat(n, Vec::N);
  }
}

template <typename Vec, size_t M>
YNN_ALWAYS_INLINE multi_vec<Vec, M> fma(multi_vec<Vec, M> a,
                                        multi_vec<Vec, M> b,
                                        multi_vec<Vec, M> acc) {
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    acc[i] = fma(a[i], b[i], acc[i]);
  }
  return acc;
}

template <int Index, typename Vec, typename ResultVec, size_t M>
YNN_ALWAYS_INLINE ResultVec extract(multi_vec<Vec, M> b, ResultVec) {
  static_assert(Index * ResultVec::N < M * Vec::N);

  return extract<Index % (Vec::N / ResultVec::N)>(
      b[(Index * ResultVec::N) / Vec::N], ResultVec{});
}

template <typename T, size_t N, typename... Args>
YNN_ALWAYS_INLINE multi_vec<vec<T, N>, sizeof...(Args) + 2> concat(
    vec<T, N> x, vec<T, N> y, Args... args) {
  return {x, y, args...};
}

template <typename To, typename Vec, size_t M>
auto convert(multi_vec<Vec, M> x, To) {
  using ResultVec = decltype(convert(x[0], To{}));
  multi_vec<ResultVec, M> result;
  YNN_UNROLL
  for (size_t i = 0; i < M; ++i) {
    result[i] = convert(x[i], To{});
  }
  return result;
}

template <typename T, size_t N, size_t M>
YNN_ALWAYS_INLINE multi_vec<vec<T, N>, M> convert(multi_vec<vec<T, N>, M> x,
                                                  T) {
  return x;
}

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_MULTI_VEC_H_
