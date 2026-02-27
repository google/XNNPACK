// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_ACCUMULATOR_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_ACCUMULATOR_H_

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

using simd::transpose;
using simd::extract;

struct Identity {
  template <typename T>
  T operator()(T x) {
    return x;
  }
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  }
};

template <typename AccT, typename AT, typename MapFn>
YNN_ALWAYS_INLINE AccT
reduce_add(AccT acc, AT a, MapFn map_fn,
           std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  return acc += map_fn(convert(a, typename AccT::value_type{}));
}

template <typename AccT>
YNN_ALWAYS_INLINE auto sum_rows(const AccT* acc,
                                std::integral_constant<size_t, 16> /*K*/,
                                std::integral_constant<size_t, 4> /*N*/) {
  std::integral_constant<size_t, 4> cols = {};
  auto v_0 = (extract<0>(acc[0], cols) + extract<1>(acc[0], cols)) +
             (extract<2>(acc[0], cols) + extract<3>(acc[0], cols));
  auto v_1 = (extract<0>(acc[1], cols) + extract<1>(acc[1], cols)) +
             (extract<2>(acc[1], cols) + extract<3>(acc[1], cols));
  auto v_2 = (extract<0>(acc[2], cols) + extract<1>(acc[2], cols)) +
             (extract<2>(acc[2], cols) + extract<3>(acc[2], cols));
  auto v_3 = (extract<0>(acc[3], cols) + extract<1>(acc[3], cols)) +
             (extract<2>(acc[3], cols) + extract<3>(acc[3], cols));

  auto t = transpose<typename AccT::value_type>(
      {{v_0, v_1, v_2, v_3}});
  return (t[0] + t[1]) + (t[2] + t[3]);
}

template <typename AccT>
YNN_ALWAYS_INLINE auto sum_rows(const AccT* acc,
                                std::integral_constant<size_t, 16> /*K*/,
                                std::integral_constant<size_t, 2> /*N*/) {
  std::integral_constant<size_t, 4> cols = {};
  auto v_0 = (extract<0>(acc[0], cols) + extract<1>(acc[0], cols)) +
             (extract<2>(acc[0], cols) + extract<3>(acc[0], cols));
  auto v_1 = (extract<0>(acc[1], cols) + extract<1>(acc[1], cols)) +
             (extract<2>(acc[1], cols) + extract<3>(acc[1], cols));

  // TODO(dsharlet): This returns a vector of 4 values, when it should return
  // a vector of 2 values.
  auto zero = decltype(v_0)(0);
  auto t = transpose<typename AccT::value_type>({{v_0, v_1, zero, zero}});
  return (t[0] + t[1]) + (t[2] + t[3]);
}

template <typename AccT>
YNN_ALWAYS_INLINE auto sum_rows(const AccT* acc,
                                std::integral_constant<size_t, 8> /*K*/,
                                std::integral_constant<size_t, 4> /*N*/) {
  std::integral_constant<size_t, 4> cols = {};
  auto v_0 = (extract<0>(acc[0], cols) + extract<1>(acc[0], cols));
  auto v_1 = (extract<0>(acc[1], cols) + extract<1>(acc[1], cols));
  auto v_2 = (extract<0>(acc[2], cols) + extract<1>(acc[2], cols));
  auto v_3 = (extract<0>(acc[3], cols) + extract<1>(acc[3], cols));

  auto t = transpose<typename AccT::value_type>({{v_0, v_1, v_2, v_3}});
  return (t[0] + t[1]) + (t[2] + t[3]);
}

template <typename AccT>
YNN_ALWAYS_INLINE auto sum_rows(const AccT* acc,
                                std::integral_constant<size_t, 4> /*K*/,
                                std::integral_constant<size_t, 4> /*N*/) {
  auto t = transpose<typename AccT::value_type>(
      {acc[0], acc[1], acc[2], acc[3]});

  return (t[0] + t[1]) + (t[2] + t[3]);
}

#ifndef YNN_ARCH_X86
// This is not numerically consistent, don't let it be used on x86.
template <typename AccT, size_t K, size_t N>
YNN_ALWAYS_INLINE auto sum_rows(const AccT* __restrict acc,
                                std::integral_constant<size_t, K> /*K*/,
                                std::integral_constant<size_t, N> /*N*/) {
  using scalar = typename AccT::value_type;
  scalar result[N];
  YNN_UNROLL
  for (size_t i = 0; i < N; ++i) {
    result[i] = simd::horizontal_sum(acc[i]);
  }
  // TODO(dsharlet): This returns a vector of 4 values to meet the assumptions
  // of the callers below. It should return a vector of N values.
  static_assert(N <= 4);
  return simd::load(result, N, simd::vec<scalar, 4>{});
}
#endif  // YNN_ARCH_X86

template <typename AccT, size_t K_, typename MapFn = Identity, size_t N_ = 4>
struct sum_accumulator_x32 {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, K_> K = {};
  static constexpr std::integral_constant<size_t, K / AccT::N>
      horizontal_factor = {};

  AccT acc[N];
  MapFn map_fn;

  sum_accumulator_x32() = default;

  YNN_ALWAYS_INLINE explicit sum_accumulator_x32(size_t) {
    const AccT zero(0);

    for (size_t i = 0; i < N; ++i) {
      acc[i] = zero;
    }
  }

  template <typename AT, typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n,
                                NT n, KT k) {
    const simd::vec<AT, K> zero(0);
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), k, zero);
    acc[0] = reduce_add(acc[0], a_0, map_fn, horizontal_factor);
    if constexpr (N >= 2) {
      auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), k, zero) : zero;
      acc[1] = reduce_add(acc[1], a_1, map_fn, horizontal_factor);
    }
    if constexpr (N >= 4) {
      auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), k, zero) : zero;
      auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), k, zero) : zero;
      acc[2] = reduce_add(acc[2], a_2, map_fn, horizontal_factor);
      acc[3] = reduce_add(acc[3], a_3, map_fn, horizontal_factor);
    }
    static_assert(N <= 4, "");
  }

  template <typename T, typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, T* __restrict C,
                                    NT n) {
    static_assert(N <= 4);
    store(C, load(C, n, simd::undef<4>{}) + sum_rows<AccT>(acc, AccT::N, N), n);
  }
};

// We attempt to make all reductions numerically consistent across all x86
// instruction sets (SSE, AVX, AVX512). We do this by always reassociating
// reductions with the same tile size, and using multiple vectors when required.
constexpr size_t consistent_tile_k = 16;

template <size_t horizontal_factor = 1, typename MapFn = Identity,
          size_t N_ = 4>
using sum_accumulator_fp32 =
    sum_accumulator_x32<simd::vec<float, consistent_tile_k>,
                        consistent_tile_k * horizontal_factor, MapFn, N_>;

template <typename AccT, typename MapFn = Identity>
struct sum_accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, 4> K2 = {};
  static constexpr std::integral_constant<size_t, AccT::N> N = {};
  static constexpr std::integral_constant<size_t, 1> horizontal_factor = {};

  MapFn map_fn;

  template <typename NT, typename K2T, typename AT>
  YNN_ALWAYS_INLINE void reduce_accumulate(
      const AT* __restrict A, NT n, size_t A_stride_k2, K2T k2,
      size_t /*C_stride_m*/, typename AccT::value_type* __restrict C) {
    assert(k2 <= K2);
    assert(n <= N);
    assert(n > 0);

    const simd::vec<AT, N> zero{0};
    auto a_0 = load(offset_bytes(A, 0 * A_stride_k2), n, zero);
    auto a_1 = 1 < k2 ? load(offset_bytes(A, 1 * A_stride_k2), n, zero) : zero;
    auto a_2 = 2 < k2 ? load(offset_bytes(A, 2 * A_stride_k2), n, zero) : zero;
    auto a_3 = 3 < k2 ? load(offset_bytes(A, 3 * A_stride_k2), n, zero) : zero;

    AccT acc = load(C, n, simd::undef<N>{});

    acc = reduce_add(acc, a_0, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_1, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_2, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_3, map_fn, horizontal_factor);

    store(C, acc, n);
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_ACCUMULATOR_H_
