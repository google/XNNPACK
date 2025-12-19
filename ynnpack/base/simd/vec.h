// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_VEC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_VEC_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/base.h"

namespace ynn {

namespace simd {

// The idea here is to provide the minimal wrappers around various platform
// specific intrinsics that allow overloading behavior based on type and vector
// length. For example, suppose you want to implement the following generic
// function.
//
//   template <typename T>
//   T f(T x) {
//     return g(x, 2);
//   }
//
// With x86 intrinsics, it is not possible to implement `g` for overloaded types
// because x86 intrinsics use `__mXYZi` for all integer types. And even if it
// did offer overloadable types like ARM, ARM's intrinsics are not overloads
// either, so it requires making wrappers for every operation that are
// (statically) dispatched based on types.
//
// This set of headers is intended to be those wrappers for basic operations:
// load, store, and basic arithmetic. If an operation is needed in multiple
// places, it may make sense to promote it to be a globally visible wrapper
// here.

// This represents a vector of N elements of type T. By default, it is
// implemented by splitting into two halves of N/2 elements each. The recursive
// nature of this implementation enables users of this type to write SIMD code
// that is portable to many SIMD instruction sets.
template <typename T, size_t N_>
struct vec {
  static constexpr std::integral_constant<size_t, N_> N = {};
  using subvec = vec<T, N_ / 2>;
  using value_type = typename subvec::value_type;

  subvec v[2];

  subvec& lo() { return v[0]; }
  const subvec& lo() const { return v[0]; }
  subvec& hi() { return v[1]; }
  const subvec& hi() const { return v[1]; }

  vec() = default;
  YNN_ALWAYS_INLINE explicit vec(value_type x) : v{subvec{x}, subvec{x}} {}
  YNN_ALWAYS_INLINE vec(subvec v0, subvec v1) : v{v0, v1} {}

  YNN_ALWAYS_INLINE subvec& operator[](size_t i) { return v[i]; }
  YNN_ALWAYS_INLINE const subvec& operator[](size_t i) const { return v[i]; }
};

template <size_t N, typename T>
YNN_ALWAYS_INLINE vec<T, N> broadcast(T x) {
  return vec<T, N>{x};
}

// Load `n` elements of `T` from `ptr`. Values not loaded from `ptr` are taken
// from `src` instead. When `n` is the constant value `N`, the value from
// `src` is unused.
template <typename T, size_t N>
vec<T, N> load(const T* ptr, std::integral_constant<size_t, N> n,
               vec<T, N> = {});
template <typename T, size_t N>
vec<T, N> load_aligned(const T* ptr, std::integral_constant<size_t, N> n,
                       vec<T, N> = {});
template <typename T, size_t N>
vec<T, N> load(const T* ptr, size_t n, vec<T, N> src);

// Store `N` elements of `T` to `ptr`.
template <typename T, size_t N>
void store(T* ptr, vec<T, N> value, std::integral_constant<size_t, N> n = {});
template <typename T, size_t N>
void store_aligned(T* ptr, vec<T, N> value,
                   std::integral_constant<size_t, N> n = {});
template <typename T, size_t N>
void store(T* ptr, vec<T, N> value, size_t n);

// Arithmetic operators.
template <typename T, size_t N>
vec<T, N> operator+(vec<T, N> a, vec<T, N> b);
template <typename T, size_t N>
vec<T, N> operator-(vec<T, N> a, vec<T, N> b);
template <typename T, size_t N>
vec<T, N> operator*(vec<T, N> a, vec<T, N> b);
template <typename T, size_t N>
vec<T, N> min(vec<T, N> a, vec<T, N> b);
template <typename T, size_t N>
vec<T, N> max(vec<T, N> a, vec<T, N> b);

template <typename T>
std::array<vec<T, 4>, 4> transpose(std::array<vec<T, 4>, 4> x);
template <int Index, typename T, size_t N, typename SliceN>
auto extract(vec<T, N>, SliceN);

template <typename To, typename From, size_t N>
vec<To, N> convert(vec<From, N> from, To);

namespace internal {

template <typename T, size_t N>
vec<T, N> partial_load_memcpy(const T* ptr, vec<T, N> src, size_t n) {
  assert(n <= N);
  memcpy(&src, ptr, sizeof(T) * n);
  return src;
}

template <typename T, size_t N>
void partial_store_memcpy(T* ptr, vec<T, N> value, size_t n) {
  assert(n <= N);
  memcpy(ptr, &value, sizeof(T) * n);
}

}  // namespace internal

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_VEC_H_
