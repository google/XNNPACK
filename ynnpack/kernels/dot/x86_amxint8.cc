// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include "ynnpack/kernels/dot/x86_amx.h"

namespace ynn {

template <int c, int a, int b>
struct dpbssd {
  void operator()() {
    YNN_TILE_DPBSSD(c, a, b);
  }
};

template <int c, int a, int b>
struct dpbusd {
  void operator()() {
    YNN_TILE_DPBUSD(c, a, b);
  }
};

void dot_int8_int8_int32_16x64x64_16x16x4_amxint8(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  x86_amx_dot<int8_t, int32_t, dpbssd>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_uint8_int8_int32_16x64x64_16x16x4_amxint8(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  x86_amx_dot<int8_t, int32_t, dpbusd>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn
