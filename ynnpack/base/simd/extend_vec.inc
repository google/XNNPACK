// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_EXTEND_VEC_INC_
#define THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_EXTEND_VEC_INC_

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
YNN_ALWAYS_INLINE vec<T, N * 2> concat(vec<T, N> a, vec<T, N> b) {
  return vec<T, N * 2>{a, b};
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr,
                                 std::integral_constant<size_t, N> n,
                                 vec<T, N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  return concat(load(ptr, n2), load(ptr + n2, n2));
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
  return concat(load_aligned(ptr, n2), load_aligned(ptr + n2, n2));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, vec<T, N> src) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return concat(load(ptr, n, lo(src)), hi(src));
  } else {
    return concat(load(ptr, n2), load(ptr + n2, n - n2, hi(src)));
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, zeros<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return concat(load(ptr, n, zeros<N / 2>{}), vec<T, N / 2>{T{0}});
  } else {
    return concat(load(ptr, n2), load(ptr + n2, n - n2, zeros<N / 2>{}));
  }
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> load(const T* ptr, size_t n, undef<N>) {
  std::integral_constant<size_t, N / 2> n2 = {};
  if (n < n2) {
    return concat(load(ptr, n, undef<N / 2>{}), vec<T, N / 2>{});
  } else {
    return concat(load(ptr, n2), load(ptr + n2, n - n2, undef<N / 2>{}));
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
  return concat(lo(a) + lo(b), hi(a) + hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) - lo(b), hi(a) - hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator*(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) * lo(b), hi(a) * hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator/(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) / lo(b), hi(a) / hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator-(vec<T, N> a) {
  return concat(-lo(a), -hi(a));
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
  return concat(lo(a) & lo(b), hi(a) & hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator|(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) | lo(b), hi(a) | hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator^(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) ^ lo(b), hi(a) ^ hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator~(vec<T, N> a) {
  return concat(~lo(a), ~hi(a));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> operator<<(vec<T, N> a, int b) {
  return concat(lo(a) << b, hi(a) << b);
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
  return concat(min(lo(a), lo(b)), min(hi(a), hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(vec<T, N> a, T b) {
  return concat(min(lo(a), b), min(hi(a), b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> min(T a, vec<T, N> b) {
  return concat(min(a, lo(b)), min(a, hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, vec<T, N> b) {
  return concat(max(lo(a), lo(b)), max(hi(a), hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(vec<T, N> a, T b) {
  return concat(max(lo(a), b), max(hi(a), b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> max(T a, vec<T, N> b) {
  return concat(max(a, lo(b)), max(a, hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> copysign(vec<T, N> mag, vec<T, N> sgn) {
  return concat(copysign(lo(mag), lo(sgn)), copysign(hi(mag), hi(sgn)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> abs(vec<T, N> a) {
  return concat(abs(lo(a)), abs(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> add_sat(vec<T, N> a, vec<T, N> b) {
  return concat(add_sat(lo(a), lo(b)), add_sat(hi(a), hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sub_sat(vec<T, N> a, vec<T, N> b) {
  return concat(sub_sat(lo(a), lo(b)), sub_sat(hi(a), hi(b)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor(vec<T, N> a) {
  return concat(floor(lo(a)), floor(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> floor_log2(vec<T, N> a) {
  return concat(floor_log2(lo(a)), floor_log2(hi(a)));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> exp2_round(vec<T, N> a) {
  return concat(exp2_round(lo(a)), exp2_round(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> ceil(vec<T, N> a) {
  return concat(ceil(lo(a)), ceil(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> round(vec<T, N> a) {
  return concat(round(lo(a)), round(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE vec<T, N> sqrt(vec<T, N> a) {
  return concat(sqrt(lo(a)), sqrt(hi(a)));
}

template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator==(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) == lo(b), hi(a) == hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator!=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) != lo(b), hi(a) != hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator<(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) < lo(b), hi(a) < hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator<=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) <= lo(b), hi(a) <= hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator>(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) > lo(b), hi(a) > hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto operator>=(vec<T, N> a, vec<T, N> b) {
  return concat(lo(a) >= lo(b), hi(a) >= hi(b));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isinf(vec<T, N> a) {
  return concat(isinf(lo(a)), isinf(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isnan(vec<T, N> a) {
  return concat(isnan(lo(a)), isnan(hi(a)));
}
template <typename T, size_t N>
YNN_ALWAYS_INLINE auto isfinite(vec<T, N> a) {
  return concat(isfinite(lo(a)), isfinite(hi(a)));
}

template <typename T, size_t N, typename M>
YNN_ALWAYS_INLINE vec<T, N> select(M cond, vec<T, N> a, vec<T, N> b) {
  return concat(select(lo(cond), lo(a), lo(b)), select(hi(cond), hi(a), hi(b)));
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
YNN_ALWAYS_INLINE vec<T, N * 4> concat(vec<T, N> a, vec<T, N> b, vec<T, N> c,
                                       vec<T, N> d) {
  return concat(concat(a, b), concat(c, d));
}

template <typename To, typename From, size_t N>
YNN_ALWAYS_INLINE vec<To, N> cast(vec<From, N> from, To) {
  return concat(cast(lo(from), To()), cast(hi(from), To()));
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

}  // namespace simd

}  // namespace ynn

#endif  // THIRD_PARTY_XNNPACK_YNNPACK_BASE_SIMD_GENERIC_INC_
