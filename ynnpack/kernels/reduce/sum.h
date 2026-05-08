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

// We attempt to make all reductions numerically consistent across all x86
// instruction sets (SSE, AVX, AVX512). We do this by always reassociating
// reductions with the same tile size, and using multiple vectors when required.
constexpr size_t consistent_tile_k_fp32 = 16;
constexpr size_t consistent_tile_k_fp64 = 8;

// The following two functions implement an algorithm to compute summations in a
// way that minimizes the maximum length of any sequence of adds.
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
// Implementing this ordering efficiently is tricky. The approach used here is
// best visualized by thinking of the sequence of operations as a tree. Store an
// accumulator at each *level* of the tree. The leaves of the tree can be
// thought of as the "accumulators" at level 0. The algorithm is then:
// - Move across the tree from left to right.
// - When passing a node, add the accumulator to the next level's accumulator,
//   and reset that node's accumulator to 0.
// - At the end of the array, add all the accumulators together, this is the
//   resulting summation.
//
// This visits the memory in order, exactly as we would want to do for best
// performance.
//
// This approach gives good performance with little overhead, and with a modest
// number of levels in the tree, delivers good numerical precision for even huge
// summations.
//
// In this implementation, there are some details to the above:
// - The accumulators are SIMD vectors. This doesn't change anything about the
//   algorithm above.
// - We use a degree 4 tree. This gives instruction level parallelism suitable
//   for most processors, and also reduces the depth of the tree required to
//   avoid numerical error for large summations.
template <size_t K_, typename KT, typename AccT, typename T, size_t N,
          typename MapFn>
YNN_ALWAYS_INLINE static void sum_k1_leaf(KT k, const T* __restrict a,
                                          simd::vec<AccT, N>& result,
                                          MapFn map_fn) {
  constexpr std::integral_constant<size_t, K_> K = {};
  constexpr std::integral_constant<size_t, N * K> NK = {};

  simd::vec<AccT, N> local{0};

  simd::vec<T, NK * 4> a0123 = load(a, k, simd::zeros<NK * 4>{});

  local = reduce_add(local, extract<0>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<1>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<2>(a0123, NK), map_fn, K);
  local = reduce_add(local, extract<3>(a0123, NK), map_fn, K);

  result = result + local;
}

template <size_t K_, typename AccT, typename T, size_t N, typename MapFn>
static void sum_k1(size_t k, const T* __restrict a, simd::vec<AccT, N>& result,
                   MapFn map_fn) {
  constexpr std::integral_constant<size_t, K_> K = {};
  constexpr std::integral_constant<size_t, N * K> NK = {};
  constexpr std::integral_constant<size_t, N * K * 4> NK4 = {};

  constexpr int levels = 6;
  simd::vec<AccT, N> accumulators[levels];
  for (int i = 0; i < levels; ++i) {
    accumulators[i] = simd::broadcast<N>(static_cast<AccT>(0));
  }

  size_t clock = 0;

  auto tick = [&](size_t n, size_t mask = -1) {
    clock += n;
    if (((clock & mask) & 3) == 0) {
      accumulators[1] += accumulators[0];
      accumulators[0] = simd::broadcast<N>(static_cast<AccT>(0));
      if (((clock & mask) & 15) == 0) {
        accumulators[2] += accumulators[1];
        accumulators[1] = simd::broadcast<N>(static_cast<AccT>(0));
        if (((clock & mask) & 63) == 0) {
          accumulators[3] += accumulators[2];
          accumulators[2] = simd::broadcast<N>(static_cast<AccT>(0));
          if (((clock & mask) & 255) == 0) {
            accumulators[4] += accumulators[3];
            accumulators[3] = simd::broadcast<N>(static_cast<AccT>(0));
            if (((clock & mask) & 1023) == 0) {
              accumulators[5] += accumulators[4];
              accumulators[4] = simd::broadcast<N>(static_cast<AccT>(0));
            }
          }
        }
      }
    }
  };

  while (k >= NK * 16) {
    // Unroll level 1 of the tree while we can.
    sum_k1_leaf<K_>(NK4, a, accumulators[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    sum_k1_leaf<K_>(NK4, a, accumulators[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    sum_k1_leaf<K_>(NK4, a, accumulators[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    sum_k1_leaf<K_>(NK4, a, accumulators[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k -= NK * 16;

    tick(4, ~0x3);
  }
  while (k >= NK * 4) {
    sum_k1_leaf<K_>(NK4, a, accumulators[0], map_fn);
    a = offset_bytes(a, NK4 * sizeof(T));
    k = sub_sat(k, NK * 4);

    tick(1);
  }
  if (k > 0) {
    // Handle any remainder.
    sum_k1_leaf<K_>(k, a, accumulators[0], map_fn);
  }

  // Now flush the accumulators in the order it would have been flushed.
  for (int i = 0; i < levels - 1; ++i) {
    accumulators[i + 1] += accumulators[i];
  }

  result = result + accumulators[levels - 1];
}

template <size_t N, size_t K, typename AccT, typename MapFn, typename AT>
static void sum_k1(size_t n, size_t k, size_t a_stride_n,
                   const AT* __restrict a, AccT* __restrict result,
                   MapFn map_fn) {
  while (n > 0) {
    simd::vec<AccT, N> acc{0};
    sum_k1<K>(k, a, acc, map_fn);
    *result++ += horizontal_sum(acc);
    a = offset_bytes(a, a_stride_n);
    --n;
  }
}

#define SUM_K1_KERNEL(name, type_a, type_c, N, K, map_fn)              \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a,      \
            size_t c_stride_m, void* c) {                              \
    sum_k1<N, K>(n, k, a_stride_n, reinterpret_cast<const type_a*>(a), \
                 reinterpret_cast<type_c*>(c), map_fn{});              \
  }

template <size_t N, typename NT, typename KT, typename AT, typename AccT,
          typename MapFn>
YNN_ALWAYS_INLINE static void sum_kn(const AT* __restrict a, NT n,
                                     size_t a_stride_k, KT k,
                                     AccT* __restrict result, MapFn map_fn) {
  assert(k <= 4);
  assert(n <= N);
  assert(n > 0);

  const simd::vec<AT, N> zero{0};
  auto a_0 = load(offset_bytes(a, 0 * a_stride_k), n, zero);
  auto a_1 = 1 < k ? load(offset_bytes(a, 1 * a_stride_k), n, zero) : zero;
  auto a_2 = 2 < k ? load(offset_bytes(a, 2 * a_stride_k), n, zero) : zero;
  auto a_3 = 3 < k ? load(offset_bytes(a, 3 * a_stride_k), n, zero) : zero;

  simd::vec<AccT, N> acc{0};

  static constexpr std::integral_constant<size_t, 1> horizontal_factor = {};
  acc = reduce_add(acc, a_0, map_fn, horizontal_factor);
  acc = reduce_add(acc, a_1, map_fn, horizontal_factor);
  acc = reduce_add(acc, a_2, map_fn, horizontal_factor);
  acc = reduce_add(acc, a_3, map_fn, horizontal_factor);

  store(result, load(result, n, simd::undef<N>{}) + acc, n);
}

template <size_t N_, typename AccT, typename T, typename MapFn>
static void sum_kn(size_t n, size_t k, size_t a_stride_k, const T* a, AccT* acc,
                   MapFn map_fn) {
  std::integral_constant<size_t, N_> N{};
  if (k <= 4) {
    while (n >= N) {
      sum_kn<N>(a, N, a_stride_k, k, acc, map_fn);
      a = offset_bytes(a, N * sizeof(T));
      acc = offset_bytes(acc, N * sizeof(AccT));
      n = sub_sat(n, N);
    }
    if (n > 0) {
      sum_kn<N>(a, n, a_stride_k, k, acc, map_fn);
    }
  } else {
    // Make a local accumulator on the stack and recurse.
    constexpr size_t tile_size = 4096 / sizeof(AccT);
    AccT local[tile_size];

    while (n > 0) {
      size_t n_chunk = std::min(n, (size_t)N);
      std::fill_n(local, n_chunk, static_cast<AccT>(0));
      size_t k_n = k;
      const T* a_n = a;
      while (k_n > 0) {
        size_t k_chunk = std::min(k_n, (size_t)4);
        sum_kn<N>(a_n, n_chunk, a_stride_k, k_chunk, local, map_fn);
        a_n = offset_bytes(a_n, k_chunk * a_stride_k);
        k_n = sub_sat(k_n, k_chunk);
      }
      for (size_t i = 0; i < n_chunk; ++i) {
        acc[i] += local[i];
      }
      n = sub_sat(n, N);
      a = offset_bytes(a, N * sizeof(T));
      acc = offset_bytes(acc, N * sizeof(AccT));
    }
  }
}

#define SUM_KN_KERNEL(name, type_a, type_c, N, map_fn)              \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a,   \
            size_t c_stride_m, void* c) {                           \
    sum_kn<N>(n, k, a_stride_k, reinterpret_cast<const type_a*>(a), \
              reinterpret_cast<type_c*>(c), map_fn{});              \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_SUM_H_
