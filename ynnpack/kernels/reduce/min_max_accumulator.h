// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_MIN_MAX_ACCUMULATOR_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_MIN_MAX_ACCUMULATOR_H_

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/generic.h"

namespace ynn {

template <typename Float, typename Int, size_t N>
float16_wrapper<Float, Int> horizontal_min(
    float16_wrapper<simd::vec<Float, N>, simd::vec<Int, N>> x) {
  return horizontal_min(static_cast<simd::vec<Int, N>>(x));
}
template <typename Float, typename Int, size_t N>
float16_wrapper<Float, Int> horizontal_max(
    float16_wrapper<simd::vec<Float, N>, simd::vec<Int, N>> x) {
  return horizontal_max(static_cast<simd::vec<Int, N>>(x));
}

// Below, we define an accumulator for both min and max at the same time.
// This type allows indicating that the accumulator should only produce min or
// max, by setting the other value to this dummy type.
struct dummy_t {
  using value_type = int;
};

// We need to implement the operations min_max_accumulator expects, to be no-ops
// for the dummy type.
inline dummy_t min(dummy_t, dummy_t) { return {}; }
inline dummy_t max(dummy_t, dummy_t) { return {}; }
template <typename T>
dummy_t min(dummy_t, T) {
  return {};
}
template <typename T>
dummy_t max(dummy_t, T) {
  return {};
}
template <typename N>
dummy_t load(const void* ptr, N, dummy_t) {
  return {};
}
template <typename N>
void store(void* ptr, dummy_t, N) {}
YNN_ALWAYS_INLINE void store(void* ptr, dummy_t) {}

template <>
class type_info<dummy_t> {
 public:
  static constexpr dummy_t min_identity() { return {}; }
  static constexpr dummy_t max_identity() { return {}; }
};

template <typename AccMinT, typename AccMaxT, typename T, size_t K_>
struct min_max_accumulator {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, K_> K = {};

  AccMinT acc_min[N];
  AccMaxT acc_max[N];

  min_max_accumulator() = default;

  YNN_ALWAYS_INLINE explicit min_max_accumulator(size_t) {
    for (size_t i = 0; i < N; ++i) {
      acc_min[i] = type_info<AccMinT>::min_identity();
      acc_max[i] = type_info<AccMaxT>::max_identity();
    }
  }

  template <typename AT, typename NT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n, NT n,
                                decltype(K)) {
    simd::vec<AT, K> undef{};
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), K, undef);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), K, undef) : a_0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), K, undef) : a_1;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), K, undef) : a_2;
    acc_min[0] = min(acc_min[0], a_0);
    acc_min[1] = min(acc_min[1], a_1);
    acc_min[2] = min(acc_min[2], a_2);
    acc_min[3] = min(acc_min[3], a_3);
    acc_max[0] = max(acc_max[0], a_0);
    acc_max[1] = max(acc_max[1], a_1);
    acc_max[2] = max(acc_max[2], a_2);
    acc_max[3] = max(acc_max[3], a_3);
  }

  template <typename AT, typename NT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t A_stride_n, NT n,
                                size_t k) {
    AccMaxT id_max(type_info<AccMaxT>::max_identity());
    AccMinT id_min(type_info<AccMinT>::min_identity());

    auto a_0_min = load(offset_bytes(A, 0 * A_stride_n), k, id_min);
    auto a_0_max = load(offset_bytes(A, 0 * A_stride_n), k, id_max);
    auto a_1_min =
        1 < n ? load(offset_bytes(A, 1 * A_stride_n), k, id_min) : a_0_min;
    auto a_1_max =
        1 < n ? load(offset_bytes(A, 1 * A_stride_n), k, id_max) : a_0_max;
    auto a_2_min =
        2 < n ? load(offset_bytes(A, 2 * A_stride_n), k, id_min) : a_0_min;
    auto a_2_max =
        2 < n ? load(offset_bytes(A, 2 * A_stride_n), k, id_max) : a_0_max;
    auto a_3_min =
        3 < n ? load(offset_bytes(A, 3 * A_stride_n), k, id_min) : a_0_min;
    auto a_3_max =
        3 < n ? load(offset_bytes(A, 3 * A_stride_n), k, id_max) : a_0_max;

    acc_min[0] = min(acc_min[0], a_0_min);
    acc_max[0] = max(acc_max[0], a_0_max);
    acc_min[1] = min(acc_min[1], a_1_min);
    acc_max[1] = max(acc_max[1], a_1_max);
    acc_min[2] = min(acc_min[2], a_2_min);
    acc_max[2] = max(acc_max[2], a_2_max);
    acc_min[3] = min(acc_min[3], a_3_min);
    acc_max[3] = max(acc_max[3], a_3_max);
  }

  template <typename AccT>
  void accumulate_min(T* __restrict C, size_t n, const AccT* acc) {
    switch (n) {
      case 4:
        C[3] = min(C[3], horizontal_min(acc[3]));
        [[fallthrough]];
      case 3:
        C[2] = min(C[2], horizontal_min(acc[2]));
        [[fallthrough]];
      case 2:
        C[1] = min(C[1], horizontal_min(acc[1]));
        [[fallthrough]];
      case 1:
        C[0] = min(C[0], horizontal_min(acc[0]));
    }
  }

  template <typename AccT>
  void accumulate_max(T* __restrict C, size_t n, const AccT* acc) {
    switch (n) {
      case 4:
        C[3] = max(C[3], horizontal_max(acc[3]));
        [[fallthrough]];
      case 3:
        C[2] = max(C[2], horizontal_max(acc[2]));
        [[fallthrough]];
      case 2:
        C[1] = max(C[1], horizontal_max(acc[1]));
        [[fallthrough]];
      case 1:
        C[0] = max(C[0], horizontal_max(acc[0]));
    }
  }

  void accumulate_min(T* __restrict C, size_t n, const dummy_t* acc) {}
  void accumulate_max(T* __restrict C, size_t n, const dummy_t* acc) {}

  YNN_ALWAYS_INLINE void accumulate(size_t C_stride_m, T* __restrict C,
                                    size_t n) {
    accumulate_min(C, n, acc_min);
    if (!std::is_same<AccMinT, dummy_t>::value) {
      // The min was not a dummy, move to the next row.
      C = offset_bytes(C, C_stride_m);
    }
    accumulate_max(C, n, acc_max);
  }
};

template <typename AccMinT, typename AccMaxT, typename T, size_t N_>
struct min_max_accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, 4> K2 = {};
  static constexpr std::integral_constant<size_t, N_> N = {};

  template <typename NT>
  void accumulate_min(T* __restrict C, const AccMinT& acc, NT n) {
    AccMinT id_min(type_info<AccMinT>::min_identity());
    store(C, min(acc, load(C, n, id_min)), n);
  }

  template <typename NT>
  void accumulate_max(T* __restrict C, const AccMaxT& acc, NT n) {
    AccMaxT id_max(type_info<AccMaxT>::max_identity());
    store(C, max(acc, load(C, n, id_max)), n);
  }

  template <typename AT, typename NT, typename K2T>
  YNN_ALWAYS_INLINE void reduce_accumulate(const AT* __restrict A, NT n,
                                           size_t A_stride_k2, K2T k2,
                                           size_t C_stride_m, T* __restrict C) {
    assert(k2 <= K2);
    assert(n <= N);
    AccMaxT id_max(type_info<AccMaxT>::max_identity());
    AccMinT id_min(type_info<AccMinT>::min_identity());

    auto a_0_min = load(offset_bytes(A, 0 * A_stride_k2), n, id_min);
    auto a_0_max = load(offset_bytes(A, 0 * A_stride_k2), n, id_max);
    auto a_1_min =
        1 < k2 ? load(offset_bytes(A, 1 * A_stride_k2), n, id_min) : a_0_min;
    auto a_1_max =
        1 < k2 ? load(offset_bytes(A, 1 * A_stride_k2), n, id_max) : a_0_max;
    auto a_2_min =
        2 < k2 ? load(offset_bytes(A, 2 * A_stride_k2), n, id_min) : a_0_min;
    auto a_2_max =
        2 < k2 ? load(offset_bytes(A, 2 * A_stride_k2), n, id_max) : a_0_max;
    auto a_3_min =
        3 < k2 ? load(offset_bytes(A, 3 * A_stride_k2), n, id_min) : a_0_min;
    auto a_3_max =
        3 < k2 ? load(offset_bytes(A, 3 * A_stride_k2), n, id_max) : a_0_max;

    AccMinT acc_min = static_cast<AccMinT>(a_0_min);
    AccMaxT acc_max = static_cast<AccMaxT>(a_0_max);

    acc_min = min(acc_min, a_1_min);
    acc_max = max(acc_max, a_1_max);
    acc_min = min(acc_min, a_2_min);
    acc_max = max(acc_max, a_2_max);
    acc_min = min(acc_min, a_3_min);
    acc_max = max(acc_max, a_3_max);

    accumulate_min(C, acc_min, n);
    if (!std::is_same<AccMinT, dummy_t>::value) {
      // The min was not a dummy, move to the next row.
      C = offset_bytes(C, C_stride_m);
    }
    accumulate_max(C, acc_max, n);
  }
};

#define MIN_MAX_KERNEL(name, acc_min, acc_max, scalar, N)                      \
  void name(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,      \
            size_t a_stride_k3, size_t a_stride_k2, const void* a,             \
            size_t c_stride_m, void* c) {                                      \
    if (k1 == 1 && (a_stride_n == sizeof(scalar))) {                           \
      stream_reduce<min_max_accumulator_k1_1<acc_min, acc_max, scalar, N>,     \
                    scalar, scalar>(n, k3, k2, a_stride_k3, a_stride_k2,       \
                                    reinterpret_cast<const scalar*>(a),        \
                                    c_stride_m, reinterpret_cast<scalar*>(c)); \
    } else {                                                                   \
      tiled_reduce<min_max_accumulator<acc_min, acc_max, scalar, N>, scalar,   \
                   scalar>(n, k3, k2, k1, a_stride_n, a_stride_k3,             \
                           a_stride_k2, reinterpret_cast<const scalar*>(a),    \
                           c_stride_m, reinterpret_cast<scalar*>(c));          \
    }                                                                          \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_MIN_MAX_ACCUMULATOR_H_
