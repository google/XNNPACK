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

template <typename AccT>
struct sum_accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, 4> K2 = {};
  static constexpr std::integral_constant<size_t, AccT::N> N = {};

  AccT acc[K2];

  sum_accumulator_k1_1() = default;

  YNN_ALWAYS_INLINE explicit sum_accumulator_k1_1(size_t) {
    AccT zero(0);
    for (size_t i = 0; i < K2; ++i) {
      acc[i] = zero;
    }
  }

  template <typename AT, typename NT, typename K2T>
  YNN_ALWAYS_INLINE void reduce(const AT* A, NT n, size_t A_stride_k2, K2T k2) {
    assert(k2 <= K2);
    assert(n <= N);
    assert(n > 0);

    const simd::vec<AT, AccT::N> zero(0);
    auto a_0 = load(offset_bytes(A, 0 * A_stride_k2), zero, n);
    auto a_1 = 1 < k2 ? load(offset_bytes(A, 1 * A_stride_k2), zero, n) : zero;
    auto a_2 = 2 < k2 ? load(offset_bytes(A, 2 * A_stride_k2), zero, n) : zero;
    auto a_3 = 3 < k2 ? load(offset_bytes(A, 3 * A_stride_k2), zero, n) : zero;

    acc[0] += a_0;
    acc[1] += a_1;
    acc[2] += a_2;
    acc[3] += a_3;
  }

  template <typename T, typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, T* __restrict C,
                                    NT n) {
    assert(n <= N);
    assert(n > 0);

    acc[0] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    store(C, load(C, AccT{}, n) + acc[0], n);
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_ACCUMULATOR_H_
