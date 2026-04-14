// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_GENERIC_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"

namespace ynn {

// If stride is 0, make a broadcasted copy of length N, and change `ptr` to
// point to the broadcasted copy.
template <size_t N, typename T>
struct broadcast_to_load {
  broadcast_to_load(const T*& ptr, size_t stride) {
    if (stride == 0) {
      std::fill_n(data, N, *ptr);
      ptr = data;
    } else {
      assert(stride == sizeof(T));
    }
  }

 private:
  T data[N];
};

template <size_t Vectorize, size_t Unroll, typename Output>
static void dequantize_dot(size_t m, size_t n, size_t stride_dot_m,
                           const int32_t* dot, size_t stride_a_offset_m,
                           const int32_t* a_offset, size_t stride_b_offset_n,
                           const int32_t* b_offset, size_t stride_offset_n,
                           const float* offset, size_t stride_a_scale_m,
                           const float* a_scale, size_t stride_b_scale_n,
                           const float* b_scale, size_t stride_output_m,
                           Output* output, const dequantize_dot_params*) {
  while (m > 0) {
    const int32_t* dot_m = dot;
    const int32_t* b_offset_m = b_offset;
    const float* offset_m = offset;
    const float* b_scale_m = b_scale;
    Output* output_m = output;

    const int32_t a_offset_m = *a_offset;
    const float a_scale_m = *a_scale;

    // TODO(b/383769559): This broadcasts for the unrolling too, which shouldn't
    // be necessary.
    broadcast_to_load<Vectorize * Unroll, float> b_scale_broadcast(
        b_scale_m, stride_b_scale_n);
    broadcast_to_load<Vectorize * Unroll, int32_t> b_offset_broadcast(
        b_offset_m, stride_b_offset_n);

    if (stride_offset_n == 0) {
      size_t j = n;

      // Process n elements, with a constant upper bound of N.
      auto body = [&](auto n, auto N) {
        assert(n <= N);
        auto dot_n = simd::load(dot_m, n, simd::undef<N>{});
        const auto a_offset_n = simd::broadcast<N>(a_offset_m);
        const auto a_scale_n = simd::broadcast<N>(a_scale_m);
        const auto b_offset_n = simd::load(b_offset_m, n, simd::undef<N>{});
        const auto b_scale_n = simd::load(b_scale_m, n, simd::undef<N>{});
        const auto offset_n = simd::broadcast<N>(*offset_m);

        dot_n -= a_offset_n * b_offset_n;
        auto output_n =
            cast(dot_n, float{}) * (a_scale_n * b_scale_n) + offset_n;

        store(output_m, cast(output_n, Output{}), n);
      };

      while (j >= Vectorize * Unroll) {
        constexpr std::integral_constant<size_t, Vectorize * Unroll> N = {};
        body(N, N);

        j -= N;
        dot_m = offset_bytes(dot_m, N * sizeof(int32_t));
        b_offset_m = offset_bytes(b_offset_m, N * stride_b_offset_n);
        b_scale_m = offset_bytes(b_scale_m, N * stride_b_scale_n);
        output_m = offset_bytes(output_m, N * sizeof(Output));
      }
      while (j >= Vectorize) {
        constexpr std::integral_constant<size_t, Vectorize> N = {};
        body(N, N);

        j -= N;
        dot_m = offset_bytes(dot_m, N * sizeof(int32_t));
        b_offset_m = offset_bytes(b_offset_m, N * stride_b_offset_n);
        b_scale_m = offset_bytes(b_scale_m, N * stride_b_scale_n);
        output_m = offset_bytes(output_m, N * sizeof(Output));
      }
      if (j > 0) {
        constexpr std::integral_constant<size_t, Vectorize> N = {};
        body(j, N);
      }
    } else {
      assert(stride_offset_n == sizeof(float));
      size_t j = n;

      // Process n elements, with a constant upper bound of N.
      auto body = [&](auto n, auto N) {
        assert(n <= N);
        auto dot_n = simd::load(dot_m, n, simd::undef<N>{});
        const auto a_offset_n = simd::broadcast<N>(a_offset_m);
        const auto a_scale_n = simd::broadcast<N>(a_scale_m);
        const auto b_offset_n = simd::load(b_offset_m, n, simd::undef<N>{});
        const auto b_scale_n = simd::load(b_scale_m, n, simd::undef<N>{});
        const auto offset_n = simd::load(offset_m, n, simd::undef<N>{});

        dot_n -= a_offset_n * b_offset_n;
        auto output_n =
            cast(dot_n, float{}) * (a_scale_n * b_scale_n) + offset_n;

        store(output_m, cast(output_n, Output{}), n);
      };

      while (j >= Vectorize * Unroll) {
        constexpr std::integral_constant<size_t, Vectorize * Unroll> N = {};
        body(N, N);

        j -= N;
        dot_m = offset_bytes(dot_m, N * sizeof(int32_t));
        b_offset_m = offset_bytes(b_offset_m, N * stride_b_offset_n);
        b_scale_m = offset_bytes(b_scale_m, N * stride_b_scale_n);
        offset_m = offset_bytes(offset_m, N * stride_offset_n);
        output_m = offset_bytes(output_m, N * sizeof(Output));
      }
      while (j >= Vectorize) {
        constexpr std::integral_constant<size_t, Vectorize> N = {};
        body(N, N);

        j -= N;
        dot_m = offset_bytes(dot_m, N * sizeof(int32_t));
        b_offset_m = offset_bytes(b_offset_m, N * stride_b_offset_n);
        b_scale_m = offset_bytes(b_scale_m, N * stride_b_scale_n);
        offset_m = offset_bytes(offset_m, N * stride_offset_n);
        output_m = offset_bytes(output_m, N * sizeof(Output));
      }
      if (j > 0) {
        constexpr std::integral_constant<size_t, Vectorize> N = {};
        body(j, N);
      }
    }

    --m;
    dot = offset_bytes(dot, stride_dot_m);
    a_offset = offset_bytes(a_offset, stride_a_offset_m);
    a_scale = offset_bytes(a_scale, stride_a_scale_m);
    output = offset_bytes(output, stride_output_m);
  }
}

#define YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(name, type, vectorize, unroll)       \
  void name(                                                                  \
      size_t m, size_t n, size_t stride_dot_m, const void* dot,               \
      size_t stride_a_offset_m, const void* a_offset,                         \
      size_t stride_b_offset_n, const void* b_offset, size_t stride_offset_n, \
      const void* offset, size_t stride_a_scale_m, const void* a_scale,       \
      size_t stride_b_scale_n, const void* b_scale, size_t stride_output_m,   \
      void* output, const dequantize_dot_params* params) {                    \
    dequantize_dot<vectorize, unroll>(                                        \
        m, n, stride_dot_m, static_cast<const int32_t*>(dot),                 \
        stride_a_offset_m, static_cast<const int32_t*>(a_offset),             \
        stride_b_offset_n, static_cast<const int32_t*>(b_offset),             \
        stride_offset_n, static_cast<const float*>(offset), stride_a_scale_m, \
        static_cast<const float*>(a_scale), stride_b_scale_n,                 \
        static_cast<const float*>(b_scale), stride_output_m,                  \
        static_cast<type*>(output), params);                                  \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_GENERIC_H_
