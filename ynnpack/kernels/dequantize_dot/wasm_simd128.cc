// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/wasm_simd128.h"

#include "ynnpack/kernels/dequantize_dot/generic.h"

namespace ynn {

YNN_DEFINE_DEQUANTIZE_DOT_KERNEL(dequantize_dot_f32_1x16_simd128, float,
                                 /*n=*/4, /*unroll=*/4);

}  // namespace ynn
