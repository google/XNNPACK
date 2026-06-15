// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/simd/arm_vec128.h"
#include "ynnpack/kernels/dequantize_dot/generic.h"

namespace ynn {

YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(dequantize_dot_f32_neon, float,
                                 /*n=*/4, /*unroll=*/4);
YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(dequantize_dot_bf16_neon, bfloat16,
                                 /*n=*/8, /*unroll=*/2);

}  // namespace ynn
