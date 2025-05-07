// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-sse2.h"

static XNN_INLINE float load_tail_reduce_add_f32(xnn_simd_f32_t acc,
                                                 const float* input,
                                                 size_t num_elements) {
  assert(num_elements < xnn_simd_size_f32);
  for (; num_elements > 0; num_elements -= 1) {
    const __m128 vt = _mm_load_ss(input);
    input += 1;
    acc = _mm_add_ss(acc, vt);
  }
  return xnn_reduce_add_f32(acc);
}

void xnn_f32_rsum_ukernel__sse_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    vacc0 = xnn_add_f32(vacc0, vt);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_f32(vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}
