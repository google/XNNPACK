// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

using simd::transpose;
using simd::extract;

struct identity {
  template <typename T>
  T operator()(T x) {
    return x;
  }
};

struct square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  }
};

template <typename AccT, typename AT, typename MapFn>
YNN_ALWAYS_INLINE AccT static reduce_add(
    AccT acc, AT a, MapFn map_fn,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  return acc += map_fn(cast(a, typename AccT::value_type{}));
}

// Computes the sum of every K*4 elements of n, producing n/(K*4) outputs.
template <size_t K_, typename NT, typename AccT, typename T, size_t N,
          typename MapFn>
YNN_ALWAYS_INLINE static void partial_sum_k1(NT n, const T* __restrict a,
                                             simd::vec<AccT, N>& x0,
                                             MapFn map_fn) {
  constexpr std::integral_constant<size_t, K_> K = {};
  constexpr std::integral_constant<size_t, N * K> NK = {};

  simd::vec<T, NK * 4> a0123 = load(a, n, simd::zeros<NK * 4>{});

  simd::vec<AccT, N> local{0};
  local = reduce_add(local, extract<0>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<1>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<2>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<3>(a0123, NK), map_fn, K);
  x0 = x0 + local;
}

template <size_t K_, typename AccT, typename T, size_t N, typename MapFn>
static void sum_k1(size_t k, const T* __restrict a, simd::vec<AccT, N>& acc,
                   MapFn map_fn) {
  constexpr std::integral_constant<size_t, K_> K = {};
  constexpr std::integral_constant<size_t, N * K> NK = {};
  constexpr std::integral_constant<size_t, N * K * 4> NK4 = {};

  while (k >= NK * 16) {
    // Unroll level 1 of the tree while we can.
    partial_sum_k1<K>(NK4, a, acc, map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc, map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc, map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc, map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k -= NK * 16;
  }
  while (k >= NK * 4) {
    partial_sum_k1<K>(NK4, a, acc, map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k = sub_sat(k, NK * 4);
  }
  if (k > 0) {
    // Handle any remainder.
    partial_sum_k1<K>(k, a, acc, map_fn);
  }
}

template <size_t N, size_t K, typename AccT, typename MapFn, typename AT>
static void sum_k1(size_t n, size_t k, size_t a_stride_n,
                   const AT* __restrict a, AccT* __restrict x0, MapFn map_fn) {
  while (n > 0) {
    simd::vec<AccT, N> acc{0};
    sum_k1<K>(k, a, acc, map_fn);
    *x0++ += horizontal_sum(acc);
    a = offset_bytes(a, a_stride_n);
    --n;
  }
}

#define SUM_K1_KERNEL(name, type_a, type_c, N, K, map_fn)                   \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, void* x0, \
            void* x1) {                                                     \
    sum_k1<N, K>(n, k, a_stride_n, reinterpret_cast<const type_a*>(a),      \
                 reinterpret_cast<type_c*>(x0), map_fn{});                  \
  }

// This is similar to the above, but for doing kn reductions instead of k1.
template <size_t N_, typename KT, typename AT, typename AccT, typename MapFn>
static void partial_sum_kn(size_t n, KT k, size_t a_stride_k,
                           const AT* __restrict a, AccT* __restrict x0,
                           MapFn map_fn) {
  static constexpr std::integral_constant<size_t, 1> horizontal_factor = {};
  static constexpr std::integral_constant<size_t, N_> N = {};
  const simd::vec<AT, N> zero{0};
  while (n >= N) {
    auto a_0 = simd::load(offset_bytes(a, 0 * a_stride_k), N);
    auto a_1 = 1 < k ? simd::load(offset_bytes(a, 1 * a_stride_k), N) : zero;
    auto a_2 = 2 < k ? simd::load(offset_bytes(a, 2 * a_stride_k), N) : zero;
    auto a_3 = 3 < k ? simd::load(offset_bytes(a, 3 * a_stride_k), N) : zero;

    simd::vec<AccT, N> acc{static_cast<AccT>(0)};
    acc = reduce_add(acc, a_0, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_1, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_2, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_3, map_fn, horizontal_factor);
    store(x0, simd::load(x0, N) + acc, N);

    a += N;
    x0 += N;
    n -= N;
  }
  if (n > 0) {
    simd::zeros<N> id{};
    auto a_0 = load(offset_bytes(a, 0 * a_stride_k), n, id);
    auto a_1 = 1 < k ? load(offset_bytes(a, 1 * a_stride_k), n, id) : zero;
    auto a_2 = 2 < k ? load(offset_bytes(a, 2 * a_stride_k), n, id) : zero;
    auto a_3 = 3 < k ? load(offset_bytes(a, 3 * a_stride_k), n, id) : zero;

    simd::vec<AccT, N> acc{static_cast<AccT>(0)};
    acc = reduce_add(acc, a_0, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_1, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_2, map_fn, horizontal_factor);
    acc = reduce_add(acc, a_3, map_fn, horizontal_factor);
    store(x0, load(x0, n, id) + acc, n);
  }
}

template <size_t N, typename AccT, typename T, typename MapFn>
static void sum_kn(size_t n, size_t k, size_t a_stride_k, const T* a, AccT* acc,
                   MapFn map_fn) {
  static constexpr std::integral_constant<size_t, 4> K = {};
  while (k >= K) {
    partial_sum_kn<N>(n, K, a_stride_k, a, acc, map_fn);
    a = offset_bytes(a, a_stride_k * 4);
    k -= K;
  }
  if (k > 0) {
    // Handle any remainder.
    partial_sum_kn<N>(n, k, a_stride_k, a, acc, map_fn);
  }
}

#define SUM_KN_KERNEL(name, type_a, type_c, N, map_fn)                      \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a, void* x0, \
            void* x1) {                                                     \
    sum_kn<N>(n, k, a_stride_k, reinterpret_cast<const type_a*>(a),         \
              reinterpret_cast<type_c*>(x0), map_fn{});                     \
  }

// The following two functions implement an algorithm to compute summations with
// good numerical precision.
//
// A simple summation of n values in a loop would compute:
//
//   (((a_0 + a_1) + a_2) + a_3) ...
//
// This has poor numerical properties, because the length of the (one) sequence
// of adds is n. A better result would be to compute (a_0 + a_1) + (a_2 + a_3),
// and use this tree ordering recursively. This has a maximum sequence length of
// log2(n), much better than the simple summation.
//
// The simple implementation is a recursive function, but this is inefficient.
// The approach used here is equivalent, but with an explicitly managed stack.
// This is best visualized by thinking of the sequence of operations as a tree.
// Store an accumulator at each *level* of the tree. The leaves of the tree can
// be thought of as the "accumulators" at level 0. The algorithm is then:
// - Move across the tree from left to right.
// - When passing a node, add the accumulator to the next level's accumulator,
//   and reset that node's accumulator to 0.
// - At the end of the array, add all the accumulators together, this is the
//   result of the summation.
//
// This visits the memory in order, exactly as we would want to do for best
// performance.
//
// This approach gives good performance with little overhead. After a few levels
// of the tree, Kahan summation becomes negligible in cost, allowing us to
// truncate the tree while maintaining good numerical properties for large
// summations.
//
// In this implementation, there are some details to the above:
// - The accumulators are SIMD vectors. This doesn't change anything about the
//   algorithm above.
// - We use a degree 16 tree.
// - We have three levels in the tree.
constexpr size_t tree_degree = 16;

constexpr size_t level_mask(size_t level) {
  return pow(tree_degree, level) - 1;
}

template <size_t K_, typename AccT, typename T, size_t N, typename MapFn>
static void sum_float_k1(size_t k, const T* __restrict a,
                         simd::vec<AccT, N>& x0, MapFn map_fn) {
  constexpr std::integral_constant<size_t, K_> K = {};
  constexpr std::integral_constant<size_t, N * K * 4> NK4 = {};

  simd::vec<AccT, N> acc[] = {
      simd::vec<AccT, N>{0},
      simd::vec<AccT, N>{0},
      simd::vec<AccT, N>{0},
  };
  // Kahan summation error, applied to the last level of the tree of
  // accumulators.
  simd::vec<AccT, N> error{0};

  // Update the tree of accumulators according to the mask.
  auto accumulate = [&](size_t clock = 0) {
    if ((clock & level_mask(1)) == 0) {
      acc[1] += acc[0];
      acc[0] = simd::broadcast<N>(static_cast<AccT>(0));
      if ((clock & level_mask(2)) == 0) {
        // Use Kahan summation for the last level of the acc.
        kahan_sum(acc[1], acc[2], error);
        acc[1] = simd::broadcast<N>(static_cast<AccT>(0));
      }
    }
  };

  // The clock is based on the number of outputs, not the number of inputs
  // which would be N*K. This both avoids excessive accumulation for bf16 and
  // fp16 (since there is plenty of precision in the wider accumulator), and
  // makes the k1 reductions consistent with kn reductions, which don't have a K
  // factor.
  size_t clock = 0;
  while (k >= NK4 * 4) {
    // Unroll level 1 of the tree while we can.
    partial_sum_k1<K>(NK4, a, acc[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    partial_sum_k1<K>(NK4, a, acc[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k -= NK4 * 4;

    clock += N * 4;
    accumulate(clock);
  }
  while (k >= NK4) {
    partial_sum_k1<K>(NK4, a, acc[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k = sub_sat(k, NK4);

    clock += N;
    accumulate(clock);
  }
  if (k > 0) {
    // Handle any remainder.
    partial_sum_k1<K>(k, a, acc[0], map_fn);
  }

  // Flush the accumulators in the tree order.
  accumulate();

  x0 = x0 + acc[2];
}

template <size_t N, size_t K, typename AccT, typename MapFn, typename AT>
static void sum_float_k1(size_t n, size_t k, size_t a_stride_n,
                         const AT* __restrict a, AccT* __restrict x0,
                         MapFn map_fn) {
  // The N to use for consistent results.
  constexpr size_t consistent_n = 64 / sizeof(AccT);
  while (n > 0) {
    simd::vec<AccT, N ? N : consistent_n> acc{0};
    sum_float_k1<K>(k, a, acc, map_fn);
    *x0++ += horizontal_sum(acc);
    a = offset_bytes(a, a_stride_n);
    --n;
  }
}

#define SUM_FLOAT_K1_KERNEL(name, type_a, type_c, N, K, map_fn)              \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, void* x0,  \
            void* x1) {                                                      \
    sum_float_k1<N, K>(n, k, a_stride_n, reinterpret_cast<const type_a*>(a), \
                       reinterpret_cast<type_c*>(x0), map_fn{});             \
  }

// Compute x += a; a = 0
template <size_t N_, typename T>
YNN_ALWAYS_INLINE static void sum_n(size_t n, T* __restrict a,
                                    T* __restrict x) {
  static constexpr std::integral_constant<size_t, N_> N = {};
  while (n >= N) {
    store(x, simd::load(x, N) + simd::load(a, N), N);
    store(a, simd::broadcast<N>(static_cast<T>(0)), N);
    x += N;
    a += N;
    n -= N;
  }
  if (n > 0) {
    store(x, load(x, n, simd::undef<N>{}) + load(a, n, simd::undef<N>{}), n);
    store(a, simd::broadcast<N>(static_cast<T>(0)), n);
  }
}

// Calls kahan_sum on vectors of size N, with n lanes.
template <size_t N, typename NT, typename T>
YNN_ALWAYS_INLINE static void kahan_sum(NT n, T* __restrict a_ptr,
                                        T* __restrict acc_ptr,
                                        T* __restrict error_ptr) {
  simd::undef<N> undef{};
  auto acc = simd::load(acc_ptr, n, undef);
  auto error = simd::load(error_ptr, n, undef);
  kahan_sum(simd::load(a_ptr, n, undef), acc, error);
  simd::store(a_ptr, simd::broadcast<N>(static_cast<T>(0)), n);
  simd::store(acc_ptr, acc, n);
  simd::store(error_ptr, error, n);
}

template <size_t N_, typename T>
YNN_ALWAYS_INLINE static void kahan_sum_n(size_t n, T* __restrict a,
                                          T* __restrict acc,
                                          T* __restrict error) {
  static constexpr std::integral_constant<size_t, N_> N = {};
  while (n >= N) {
    kahan_sum<N>(N, a, acc, error);
    error += N;
    a += N;
    acc += N;
    n -= N;
  }
  if (n > 0) {
    kahan_sum<N>(n, a, acc, error);
  }
}

template <size_t N, size_t MaxN, typename AccT, typename T, typename MapFn>
static void sum_float_kn_tile(size_t n, size_t k, size_t a_stride_k, const T* a,
                              AccT* x0, MapFn map_fn) {
  assert(n <= MaxN);

  AccT acc[3][MaxN];
  std::fill_n(&acc[0][0], n, static_cast<AccT>(0));
  std::fill_n(&acc[1][0], n, static_cast<AccT>(0));
  std::fill_n(&acc[2][0], n, static_cast<AccT>(0));
  // Kahan summation error, applied to the last level of the tree of
  // accumulators.
  AccT error[MaxN];
  std::fill_n(error, n, static_cast<AccT>(0));

  // Update the tree of accumulators according to the mask.
  auto accumulate = [&](size_t clock = 0) {
    if ((clock & level_mask(1)) == 0) {
      sum_n<N>(n, acc[0], acc[1]);
      if ((clock & level_mask(2)) == 0) {
        // Use Kahan summation for the last level of the accumulators.
        kahan_sum_n<N>(n, acc[1], acc[2], error);
      }
    }
  };

  static constexpr std::integral_constant<size_t, 4> K = {};
  size_t clock = 0;
  while (k >= K) {
    partial_sum_kn<N>(n, K, a_stride_k, a, acc[0], map_fn);
    a = offset_bytes(a, a_stride_k * 4);
    k -= K;

    clock += K;
    accumulate(clock);
  }
  if (k > 0) {
    // Handle any remainder.
    partial_sum_kn<N>(n, k, a_stride_k, a, acc[0], map_fn);
  }

  // Flush the accumulators in the tree order.
  accumulate();

  sum_n<N>(n, acc[2], x0);
}

template <size_t N, typename AccT, typename T, typename MapFn>
static void sum_float_kn(size_t n, size_t k, size_t a_stride_k, const T* a,
                         AccT* x0, MapFn map_fn) {
  static constexpr std::integral_constant<size_t, 4> K = {};
  if (k < K) {
    partial_sum_kn<N>(n, k, a_stride_k, a, x0, map_fn);
  } else if (k == K) {
    partial_sum_kn<N>(n, K, a_stride_k, a, x0, map_fn);
  } else {
    constexpr size_t tile_n = 4096 / sizeof(AccT);
    while (n > 0) {
      sum_float_kn_tile<N, tile_n>(std::min(n, tile_n), k, a_stride_k, a, x0,
                                   map_fn);
      a = offset_bytes(a, tile_n * sizeof(T));
      x0 = offset_bytes(x0, tile_n * sizeof(AccT));
      n = sub_sat(n, tile_n);
    }
  }
}

#define SUM_FLOAT_KN_KERNEL(name, type_a, type_c, N, map_fn)                \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a, void* x0, \
            void* x1) {                                                     \
    sum_float_kn<N>(n, k, a_stride_k, reinterpret_cast<const type_a*>(a),   \
                    reinterpret_cast<type_c*>(x0), map_fn{});               \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_H_
