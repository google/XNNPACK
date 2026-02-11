// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HEXAGON_HVX_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HEXAGON_HVX_H_

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/simd/hexagon_hvx.h"
#include "ynnpack/base/simd/vec.h"

namespace ynn {

using simd::u8x128;

template <typename ElemSizeBits>
static std::tuple<u8x128, u8x128> interleave(ElemSizeBits elem_size_bits,
                                             u8x128 x0, u8x128 x1) {
  HVX_VectorPair x01 = Q6_W_vshuff_VVR(x1.v, x0.v, -(elem_size_bits / 8));
  return {u8x128{Q6_V_lo_W(x01)}, u8x128{Q6_V_hi_W(x01)}};
}

template <size_t M, typename NBytes>
YNN_ALWAYS_INLINE static std::array<u8x128, M> load(std::array<u8x128, M>,
                                                    const void* a,
                                                    size_t stride, size_t m,
                                                    NBytes n_bytes) {
  assert(m > 0);
  assert(m <= M);
  std::array<u8x128, M> x;
  x[0] =
      simd::load(static_cast<const uint8_t*>(a), n_bytes, simd::zeros<128>{});
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      x[i] =
          simd::load(static_cast<const uint8_t*>(offset_bytes(a, i * stride)),
                     n_bytes, simd::zeros<128>{});
    } else {
      x[i] = u8x128{Q6_V_vzero()};
    }
  }
  return x;
}

template <size_t M, typename NBytes>
YNN_ALWAYS_INLINE static void store(std::array<u8x128, M> x, void* a,
                                    size_t stride, size_t m, NBytes n_bytes) {
  assert(m > 0);
  assert(m <= M);
  simd::store(static_cast<uint8_t*>(a), x[0], n_bytes);
  for (size_t i = 1; i < M; ++i) {
    if (i < m) {
      simd::store(static_cast<uint8_t*>(offset_bytes(a, i * stride)), x[i],
                  n_bytes);
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_HEXAGON_HVX_H_
