// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/type.h"

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
static void tiled_reduce(size_t N, size_t K3, size_t K2, size_t K1,
                         size_t A_stride_n, size_t A_stride_k3,
                         size_t A_stride_k2, const AT* A, size_t C_stride_m,
                         CT* C) {
  Accumulator acc;
  while (N >= Accumulator::N) {
    acc = Accumulator(K3 * K2 * K1);
    for (size_t k3 = 0; k3 < K3; ++k3) {
      const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
      for (size_t k2 = 0; k2 < K2; ++k2) {
        const AT* A_k1 = offset_bytes(A_k3, k2 * A_stride_k2);
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
    }
    acc.accumulate(C_stride_m, C, Accumulator::N);
    C = offset_bytes(C, Accumulator::N * sizeof(CT));
    A = offset_bytes(A, Accumulator::N * A_stride_n);
    N -= Accumulator::N;
  }
  if (N > 0) {
    acc = Accumulator(K3 * K2 * K1);
    for (size_t k3 = 0; k3 < K3; ++k3) {
      const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
      for (size_t k2 = 0; k2 < K2; ++k2) {
        const AT* A_k1 = offset_bytes(A_k3, k2 * A_stride_k2);
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
    }
    acc.accumulate(C_stride_m, C, N);
  }
}

// The above is inefficient if K1 = 1. This function is similar, but assumes
// K1 = 1 and A_stride_n == sizeof(AT). It works in a more "streaming" fashion,
// where it maintains the reduction result in C itself instead of a local
// accumulator.
template <typename Accumulator, typename AT, typename CT>
static void stream_reduce(size_t N, size_t K3, size_t K2, size_t A_stride_k3,
                          size_t A_stride_k2, const AT* A, size_t C_stride_m,
                          CT* C) {
  Accumulator acc;
  for (size_t k3 = 0; k3 < K3; ++k3) {
    const AT* a_k3 = offset_bytes(A, k3 * A_stride_k3);
    size_t k2 = K2;
    while (k2 >= Accumulator::K2) {
      const AT* a = a_k3;
      CT* c = C;
      size_t n = N;
      while (n >= Accumulator::N) {
        acc.reduce_accumulate(a, Accumulator::N, A_stride_k2, Accumulator::K2,
                              C_stride_m, c);
        a = offset_bytes(a, Accumulator::N * sizeof(AT));
        c = offset_bytes(c, Accumulator::N * sizeof(CT));
        n -= Accumulator::N;
      }
      if (n > 0) {
        acc.reduce_accumulate(a, n, A_stride_k2, Accumulator::K2, C_stride_m,
                              c);
      }
      a_k3 = offset_bytes(a_k3, Accumulator::K2 * A_stride_k2);
      k2 -= Accumulator::K2;
    }
    if (k2 > 0) {
      const AT* a = a_k3;
      CT* c = C;
      size_t n = N;
      while (n >= Accumulator::N) {
        acc.reduce_accumulate(a, Accumulator::N, A_stride_k2, k2, C_stride_m,
                              c);
        a = offset_bytes(a, Accumulator::N * sizeof(AT));
        c = offset_bytes(c, Accumulator::N * sizeof(CT));
        n -= Accumulator::N;
      }
      if (n > 0) {
        acc.reduce_accumulate(a, n, A_stride_k2, k2, C_stride_m, c);
      }
    }
  }
}

using std::max;
using std::min;

// This class allows min/max reductions of sign-magnitude floating point types
// to be computed efficiently using integer arithmetic. It is implicitly
// convertible to-from Float, and supports computing min/max.
//
// The intended usage is:
//
//   float16_wrapper<Float, Int, Float::value_type> r(init_value);
//   for (int i = 0; i < n; ++i) {
//     r = min(r, floats[i]);
//   }
//   Float result = static_cast<Float>(r);
template <typename Float, typename Int>
class float16_wrapper {
 public:
  float16_wrapper() = default;
  float16_wrapper(Float value)  // NOLINT
      : value_(sign_complement(bit_cast<Int>(value))) {}
  constexpr float16_wrapper(Int value)  // NOLINT
      : value_(value) {}

  operator Float() const {  // NOLINT
    return bit_cast<Float>(sign_complement(value_));
  }

  operator Int() const {  // NOLINT
    return value_;
  }

  friend float16_wrapper min(float16_wrapper a, float16_wrapper b) {
    a.value_ = min(a.value_, b.value_);
    return a;
  }
  friend float16_wrapper max(float16_wrapper a, float16_wrapper b) {
    a.value_ = max(a.value_, b.value_);
    return a;
  }

 private:
  Int value_;

  static constexpr Int sign_complement(Int a) {
    return (a & 0x7FFF) ^ (a >> 15);
  }
};

using half_rvar = float16_wrapper<half, int16_t>;
using bfloat16_rvar = float16_wrapper<bfloat16, int16_t>;

// Forward type_info for float16_wrapper.
template <typename Float, typename Int>
class type_info<float16_wrapper<Float, Int>> {
 public:
  static constexpr float16_wrapper<Float, Int> min_identity() {
    return float16_wrapper<Float, Int>(static_cast<Int>(32767));
  }
  static constexpr float16_wrapper<Float, Int> max_identity() {
    return float16_wrapper<Float, Int>(static_cast<Int>(-32768));
  }
};

// Forward type_info for simd::vec.
template <typename T, size_t N>
class type_info<simd::vec<T, N>> {
 public:
  static constexpr simd::vec<T, N> min_identity() {
    return type_info<T>::min_identity();
  }
  static constexpr simd::vec<T, N> max_identity() {
    return type_info<T>::max_identity();
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_GENERIC_H_
