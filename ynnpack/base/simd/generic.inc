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
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 zeros<N>) {
  return load(ptr, n, vec<T, N>{});
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 undef<N>) {
  return load(ptr, n, vec<T, N>{});
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
    return {load(ptr, n, lo(src)), hi(src)};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, hi(src))};
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, zeros<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, zeros<N / 2>{}), vec<T, N / 2>{T{0}}};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, zeros<N / 2>{})};
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, undef<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return {load(ptr, n, undef<N / 2>{}), vec<T, N / 2>{}};
  } else {
    return {load(ptr, n2), load(ptr + n2, n - n2, undef<N / 2>{})};
  }
}

// Store `N` elements of `T` to `ptr`.
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value,
                             std::integral_constant<size_t, N> n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  store(ptr, lo(value), n2);
  store(ptr + n2, hi(value), n2);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store_aligned(T* ptr, vec<T, N> value,
                                     std::integral_constant<size_t, N> n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  store_aligned(ptr, lo(value), n2);
  store_aligned(ptr + n2, hi(value), n2);
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE void store(T* ptr, vec<T, N> value, size_t n) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    store(ptr, lo(value), n);
  } else {
    store(ptr, lo(value), n2);
    store(ptr + n2, hi(value), n - n2);
  }
}

// Arithmetic operators.
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator+(vec<T, N> a, vec<T, N> b) {
  return {lo(a) + lo(b), hi(a) + hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a, vec<T, N> b) {
  return {lo(a) - lo(b), hi(a) - hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator*(vec<T, N> a, vec<T, N> b) {
  return {lo(a) * lo(b), hi(a) * hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator/(vec<T, N> a, vec<T, N> b) {
  return {lo(a) / lo(b), hi(a) / hi(b)};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator+=(vec<T, N>& a, vec<T, N> b) {
  a = a + b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator-=(vec<T, N>& a, vec<T, N> b) {
  a = a - b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator*=(vec<T, N>& a, vec<T, N> b) {
  a = a * b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator/=(vec<T, N>& a, vec<T, N> b) {
  a = a / b;
  return a;
}

// Boolean operators.
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator&(vec<T, N> a, vec<T, N> b) {
  return {lo(a) & lo(b), hi(a) & hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator|(vec<T, N> a, vec<T, N> b) {
  return {lo(a) | lo(b), hi(a) | hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator^(vec<T, N> a, vec<T, N> b) {
  return {lo(a) ^ lo(b), hi(a) ^ hi(b)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator~(vec<T, N> a) {
  return {~lo(a), ~hi(a)};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator<<(vec<T, N> a, int b) {
  return {lo(a) << b, hi(a) << b};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator&=(vec<T, N>& a, vec<T, N> b) {
  a = a & b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator|=(vec<T, N>& a, vec<T, N> b) {
  a = a | b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N>& operator^=(vec<T, N>& a, vec<T, N> b) {
  a = a ^ b;
  return a;
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, vec<T, N> b) {
  return {min(lo(a), lo(b)), min(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return {max(lo(a), lo(b)), max(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copysign(vec<T, N> mag, vec<T, N> sgn) {
  return {copysign(lo(mag), lo(sgn)), copysign(hi(mag), hi(sgn))};
};
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> abs(vec<T, N> a) {
  return {abs(lo(a)), abs(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> add_sat(vec<T, N> a, vec<T, N> b) {
  return {add_sat(lo(a), lo(b)), add_sat(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sub_sat(vec<T, N> a, vec<T, N> b) {
  return {sub_sat(lo(a), lo(b)), sub_sat(hi(a), hi(b))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor(vec<T, N> a) {
  return {floor(lo(a)), floor(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor_log2(vec<T, N> a) {
  return {floor_log2(lo(a)), floor_log2(hi(a))};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> exp2_round(vec<T, N> a) {
  return {exp2_round(lo(a)), exp2_round(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copynan(vec<T, N> x, vec<T, N> nan) {
  return {copynan(lo(x), lo(nan)), copynan(hi(x), hi(nan))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> ceil(vec<T, N> a) {
  return {ceil(lo(a)), ceil(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> round(vec<T, N> a) {
  return {round(lo(a)), round(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sqrt(vec<T, N> a) {
  return {sqrt(lo(a)), sqrt(hi(a))};
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> fma(vec<T, N> a, vec<T, N> b, vec<T, N> acc) {
  return {fma(lo(a), lo(b), lo(acc)), fma(hi(a), hi(b), hi(acc))};
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
  return Index == 0 ? lo(x) : hi(x);
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

template <typename To, typename From, size_t N>
YNN_ALWAYS_INLINE vec<To, N> cast(vec<From, N> from, To) {
  return {cast(lo(from), To()), cast(hi(from), To())};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_sum(vec<T, N> x) {
  return horizontal_sum(lo(x) + hi(x));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_min(vec<T, N> x) {
  return horizontal_min(min(lo(x), hi(x)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE T horizontal_max(vec<T, N> x) {
  return horizontal_max(max(lo(x), hi(x)));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE void kahan_sum(vec<T, N> a, vec<T, N>& acc,
                                 vec<T, N>& error) {
  vec<T, N / 2> acc_lo = lo(acc);
  vec<T, N / 2> acc_hi = hi(acc);
  vec<T, N / 2> error_lo = lo(error);
  vec<T, N / 2> error_hi = hi(error);
  kahan_sum(lo(a), acc_lo, error_lo);
  kahan_sum(hi(a), acc_hi, error_hi);
  acc = concat(acc_lo, acc_hi);
  error = concat(error_lo, error_hi);
}

}  // namespace simd

}  // namespace ynn

#endif  // THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_VEC_INC_
