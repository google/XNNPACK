// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_

#include <cstddef>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

// This function implements reductions in multiple dimensions using an
// `Accumulator` helper type.
//
// Conceptually, `Accumulator` is an object that represents a tile of `N` x `K`
// accumulators. It loads tiles of an input and does an elementwise reduction
// with the tile. The `K` dimension of the accumulators are always reduced, the
// `N` dimension is optionally reduced.
//
// It should have the following members:
// - (constructor): Initialize the accumulators to the identify value for the
//   reduction.
// - N: The number of columns it accumulates at once.
// - K: The number of K values it accumulates at once.
// - reduce(const AT* A, size_t A_stride_n, n, k):
//   This should accumulate a tile of nxk tile of values from A.
// - reduce(const AT* A, n):
//   Same as above, but assumes that A_stride_n == sizeof(AT) && K == 1.
// - accumulate(size_t c_stride_m, CT* c, n): write `n` of the accumulators to
//   `c`.
//
// In all of the above, the size parameters pass the original type of
// `Accumulator::N` and `Accumulator::K` when appropriate, so if those values
// have type `std::integral_constant<>` or similar, the behavior can be
// statically dispatched at compile time to handle the whole-tile vs. "tail"
// cases.
template <typename Accumulator, typename AT, typename CT>
static void tiled_reduce(size_t N, size_t K2, size_t K1, size_t A_stride_n,
                         size_t A_stride_k2, const AT* A, CT* x0, CT* x1) {
  Accumulator acc;
  while (N >= Accumulator::N) {
    acc = Accumulator(K2 * K1);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const AT* A_k1 = offset_bytes(A, k2 * A_stride_k2);
      size_t k1 = K1;
      while (k1 >= Accumulator::K) {
        acc.reduce(A_k1, A_stride_n, Accumulator::N, Accumulator::K);
        A_k1 = offset_bytes(A_k1, Accumulator::K * sizeof(AT));
        k1 -= Accumulator::K;
      }
      if (k1 > 0) {
        acc.reduce(A_k1, A_stride_n, Accumulator::N, k1);
      }
    }
    acc.accumulate(x0, x1, Accumulator::N);
    x0 = offset_bytes(x0, Accumulator::N * sizeof(CT));
    if (x1) x1 = offset_bytes(x1, Accumulator::N * sizeof(CT));
    A = offset_bytes(A, Accumulator::N * A_stride_n);
    N -= Accumulator::N;
  }
  if (N > 0) {
    acc = Accumulator(K2 * K1);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const AT* A_k1 = offset_bytes(A, k2 * A_stride_k2);
      size_t k1 = K1;
      while (k1 >= Accumulator::K) {
        acc.reduce(A_k1, A_stride_n, N, Accumulator::K);
        A_k1 = offset_bytes(A_k1, Accumulator::K * sizeof(AT));
        k1 -= Accumulator::K;
      }
      if (k1 > 0) {
        acc.reduce(A_k1, A_stride_n, N, k1);
      }
    }
    acc.accumulate(x0, x1, N);
  }
}

// The above is inefficient if K1 = 1. This function is similar, but assumes
// K1 = 1 and A_stride_n == sizeof(AT). It works in a more "streaming" fashion,
// where it maintains the reduction result in C itself instead of a local
// accumulator.
template <typename Accumulator, typename AT, typename CT>
static void stream_reduce(size_t N, size_t K2, size_t A_stride_n,
                          size_t A_stride_k2, const AT* A, CT* x0, CT* x1) {
  Accumulator acc;
  size_t k2 = K2;
  while (k2 >= Accumulator::K2) {
    const AT* a = A;
    CT* c0 = x0;
    CT* c1 = x1;
    size_t n = N;
    while (n >= Accumulator::N) {
      acc.reduce_accumulate(a, Accumulator::N, A_stride_k2, Accumulator::K2, c0,
                            c1);
      a = offset_bytes(a, Accumulator::N * A_stride_n);
      c0 = offset_bytes(c0, Accumulator::N * sizeof(CT));
      if (c1) c1 = offset_bytes(c1, Accumulator::N * sizeof(CT));
      n -= Accumulator::N;
    }
    if (n > 0) {
      acc.reduce_accumulate(a, n, A_stride_k2, Accumulator::K2, c0, c1);
    }
    A = offset_bytes(A, Accumulator::K2 * A_stride_k2);
    k2 -= Accumulator::K2;
  }
  if (k2 > 0) {
    const AT* a = A;
    CT* c0 = x0;
    CT* c1 = x1;
    size_t n = N;
    while (n >= Accumulator::N) {
      acc.reduce_accumulate(a, Accumulator::N, A_stride_k2, k2, c0, c1);
      a = offset_bytes(a, Accumulator::N * A_stride_n);
      c0 = offset_bytes(c0, Accumulator::N * sizeof(CT));
      if (c1) c1 = offset_bytes(c1, Accumulator::N * sizeof(CT));
      n -= Accumulator::N;
    }
    if (n > 0) {
      acc.reduce_accumulate(a, n, A_stride_k2, k2, c0, c1);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_
