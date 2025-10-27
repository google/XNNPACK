// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cstddef>

#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/kernels/dot/x86_amx.h"

namespace ynn {

template <int c, int a, int b>
struct dpfp16ps {
  void operator()() {
    // This should be _tile_dpfp16ps(c, a, b), but GCC has a ridiculous bug:
    // https://github.com/google/XNNPACK/issues/9000#issuecomment-3449425946
    if (c == 0 && a == 4 && b == 5) {
      _tile_dpfp16ps(0, 4, 5);
    } else if (c == 1 && a == 4 && b == 5) {
      _tile_dpfp16ps(1, 4, 5);
    } else if (c == 2 && a == 4 && b == 5) {
      _tile_dpfp16ps(2, 4, 5);
    } else if (c == 3 && a == 4 && b == 5) {
      _tile_dpfp16ps(3, 4, 5);
    } else if (c == 0 && a == 6 && b == 7) {
      _tile_dpfp16ps(0, 6, 7);
    } else if (c == 1 && a == 6 && b == 7) {
      _tile_dpfp16ps(1, 6, 7);
    } else if (c == 2 && a == 6 && b == 7) {
      _tile_dpfp16ps(2, 6, 7);
    } else if (c == 3 && a == 6 && b == 7) {
      _tile_dpfp16ps(3, 6, 7);
    } else {
      YNN_UNREACHABLE;
    }
  }
};

void dot_fp16_fp16_fp32_16x64x32_16x16x2_amxfp16(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  x86_amx_dot<half, float, dpfp16ps>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn
