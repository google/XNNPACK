// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/kernels/dot/x86_amx.h"

namespace ynn {

template <int c, int a, int b>
struct dpbf16ps {
  void operator()() {
    YNN_TILE_DPBF16PS(c, a, b);
  }
};

void dot_bf16_bf16_fp32_16x64x32_16x16x2_amxbf16(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  x86_amx_dot<bfloat16, float, dpbf16ps>(
      M, N, K3, K2, K1 & ~1, A_stride_m, A_stride_k3, A_stride_k2, A,
      B_stride_k3, B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in,
      C_out_stride_m, C_out);
}

}  // namespace ynn
