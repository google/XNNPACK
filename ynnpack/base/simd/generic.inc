// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_
#define THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"

// This header provides generic implementations of vec<> operations that work by
// splitting the vec into two smaller vectors and executing the operation on
// those. It should be included after implementing specializations of vec<>.

namespace ynn {

namespace simd {

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 vec<T, N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  return {load(ptr, n2), load(ptr + n2, n2)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load_aligned(const T* ptr,
                                         std::integral_constant<size_t, N> n,
                                         vec<T, N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  return {load_aligned(ptr, n2), load_aligned(ptr + n2, n2)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, vec<T, N> src) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, src.lo()), src.hi()};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, src.hi())};
  }
}

// Store `N` elements of `T` to `ptr`.
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value,
                             std::integral_constant<size_t, N> n) {
  store(ptr, value.lo());
  store(ptr + N / 2, value.hi());
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* ptr, vec<T, N> value,
                                     std::integral_constant<size_t, N> n) {
  store_aligned(ptr, value.lo());
  store_aligned(ptr + N / 2, value.hi());
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value, size_t n) {
  if (n < N / 2) {
    store(ptr, value.lo(), n);
  } else {
    store(ptr, value.lo());
    store(ptr + N / 2, value.hi(), n - N / 2);
  }
}

// Arithmetic operators.
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator+=(vec<T, N>& a, vec<T, N> b) {
  a.lo() += b.lo();
  a.hi() += b.hi();
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator-=(vec<T, N>& a, vec<T, N> b) {
  a.lo() -= b.lo();
  a.hi() -= b.hi();
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator*=(vec<T, N>& a, vec<T, N> b) {
  a.lo() *= b.lo();
  a.hi() *= b.hi();
  return a;
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator+(vec<T, N> a, vec<T, N> b) {
  return a += b;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a, vec<T, N> b) {
  return a -= b;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator*(vec<T, N> a, vec<T, N> b) {
  return a *= b;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, vec<T, N> b) {
  return {min(a.lo(), b.lo()), min(a.hi(), b.hi())};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return {max(a.lo(), b.lo()), max(a.hi(), b.hi())};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> abs(vec<T, N> a) {
  return {abs(a.lo()), abs(a.hi())};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> fma(vec<T, N> a, vec<T, N> b, vec<T, N> acc) {
  return {fma(a.lo(), b.lo(), acc.lo()), fma(a.hi(), b.hi(), acc.hi())};
}

template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> extract(vec<T, N> x,
                                    std::integral_constant<size_t, N>) {
  static_assert(Index == 0, "");
  return x;
}
template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N / 2> extract(vec<T, N> x,
                                        std::integral_constant<size_t, N / 2>) {
  static_assert(Index == 0 || Index == 1, "");
  return Index == 0 ? x.lo() : x.hi();
}
template <int Index, typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N / 4> extract(vec<T, N> x,
                                        std::integral_constant<size_t, N / 4>) {
  constexpr std::integral_constant<size_t, N / 2> n2 = {};
  constexpr std::integral_constant<size_t, N / 4> n4 = {};
  return extract<Index % 2>(extract<Index / 2>(x, n2), n4);
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N*2> concat(vec<T, N> a, vec<T, N> b) {
  return {a, b};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> convert(vec<T, N> from, T) {
  return from;
}
template <typename To, typename From, size_t N>
YNN_ALWAYS_INLINE vec<To, N> convert(vec<From, N> from, To) {
  return {convert(from.lo(), To()), convert(from.hi(), To())};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_min(vec<T, N> x) {
  return horizontal_min(min(x.lo(), x.hi()));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_max(vec<T, N> x) {
  return horizontal_max(max(x.lo(), x.hi()));
}

}  // namespace simd

}  // namespace ynn

#endif  // THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_VEC_INC_
