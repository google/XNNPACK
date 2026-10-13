// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_PROPAGATE_NAN_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_PROPAGATE_NAN_H_

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/generic.h"

namespace ynn {

using simd::any;
using simd::broadcast;
using simd::isnan;
using simd::load;
using simd::select;
using simd::store;

template <typename T, size_t K_>
struct propagate_nan_accumulator {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, K_> K = {};

  using vec_bool = decltype(isnan(std::declval<simd::vec<T, K_>>()));

  vec_bool is_nan[N];

  propagate_nan_accumulator() = default;

  YNN_ALWAYS_INLINE explicit propagate_nan_accumulator(size_t) {
    for (size_t i = 0; i < N; ++i) {
      is_nan[i] = vec_bool{false};
    }
  }

  template <typename AT, typename NT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n, NT n,
                                decltype(K)) {
    auto undef = simd::undef<K_>{};
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), K, undef);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), K, undef) : a_0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), K, undef) : a_1;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), K, undef) : a_2;
    is_nan[0] = is_nan[0] | isnan(a_0);
    is_nan[1] = is_nan[1] | isnan(a_1);
    is_nan[2] = is_nan[2] | isnan(a_2);
    is_nan[3] = is_nan[3] | isnan(a_3);
  }

  template <typename AT, typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n, NT n, KT k) {
    auto id = broadcast<K>(static_cast<T>(0));
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), k, id);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), k, id) : a_0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), k, id) : a_1;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), k, id) : a_2;
    is_nan[0] = is_nan[0] | isnan(a_0);
    is_nan[1] = is_nan[1] | isnan(a_1);
    is_nan[2] = is_nan[2] | isnan(a_2);
    is_nan[3] = is_nan[3] | isnan(a_3);
  }

  YNN_ALWAYS_INLINE void accumulate(T* __restrict x0, T* __restrict x1,
                                    size_t n) {
    for (size_t i = 0; i < n; ++i) {
      if (any(is_nan[i])) {
        x0[i] = type_info<T>::nan();
        if (x1 && x1 != x0) {
          x1[i] = type_info<T>::nan();
        }
      }
    }
  }
};

template <typename T, size_t N_>
struct propagate_nan_accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, 4> K2 = {};
  static constexpr std::integral_constant<size_t, N_> N = {};

  template <typename AT, typename NT, typename K2T>
  YNN_ALWAYS_INLINE void reduce_accumulate(const AT* __restrict A, NT n,
                                           size_t A_stride_k2, K2T k2,
                                           T* __restrict x0, T* __restrict x1) {
    assert(k2 <= K2);
    assert(n <= N);
    auto id = broadcast<N>(static_cast<T>(0));
    auto a_0 = load(offset_bytes(A, 0 * A_stride_k2), n, id);
    auto a_1 = 1 < k2 ? load(offset_bytes(A, 1 * A_stride_k2), n, id) : a_0;
    auto a_2 = 2 < k2 ? load(offset_bytes(A, 2 * A_stride_k2), n, id) : a_0;
    auto a_3 = 3 < k2 ? load(offset_bytes(A, 3 * A_stride_k2), n, id) : a_0;

    using bool_vec = decltype(isnan(std::declval<simd::vec<T, N_>>()));
    bool_vec is_nan = isnan(a_0) | isnan(a_1) | isnan(a_2) | isnan(a_3);
    auto nan = broadcast<N>(type_info<T>::nan());
    store(x0, select(is_nan, nan, load(x0, n, simd::undef<N>{})), n);
    if (x1 && x1 != x0) {
      store(x1, select(is_nan, nan, load(x1, n, simd::undef<N>{})), n);
    }
  }
};

#define PROPAGATE_NAN_K1_KERNEL(name, scalar, N)                            \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, void* x0, \
            void* x1) {                                                     \
    tiled_reduce<propagate_nan_accumulator<scalar, N>, scalar, scalar>(     \
        n, 1, k, a_stride_n, 0, reinterpret_cast<const scalar*>(a),         \
        reinterpret_cast<scalar*>(x0), reinterpret_cast<scalar*>(x1));      \
  }

#define PROPAGATE_NAN_KN_KERNEL(name, scalar, N)                              \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a, void* x0,   \
            void* x1) {                                                       \
    stream_reduce<propagate_nan_accumulator_k1_1<scalar, N>, scalar, scalar>( \
        n, k, sizeof(scalar), a_stride_k, reinterpret_cast<const scalar*>(a), \
        reinterpret_cast<scalar*>(x0), reinterpret_cast<scalar*>(x1));        \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_PROPAGATE_NAN_H_
