// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/x86_vec256.h"
#include "ynnpack/kernels/dequantize_dot/generic.h"

namespace ynn {

YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(dequantize_dot_f32_avx2, float, /*n=*/8,
                                 /*unroll=*/4);
YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(dequantize_dot_bf16_avx2, bfloat16, /*n=*/16,
                                 /*unroll=*/2);

}  // namespace ynn
