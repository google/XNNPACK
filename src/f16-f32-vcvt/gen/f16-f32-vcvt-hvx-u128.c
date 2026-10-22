// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__hvx_u128(
    size_t batch,
    const xnn_float16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 128 * sizeof(xnn_float16); batch -= 128 * sizeof(xnn_float16)) {
    #if __HVX_ARCH__ >= 81
    const xnn_simd_f16_t vx0 = xnn_loadu_f16(input + 0);
    const xnn_simd_f16_t vx1 = xnn_loadu_f16(input + 64);
    input += 128;

    const HVX_VectorPair vw0 = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vx0));
    const HVX_VectorPair vw1 = Q6_Wsf_vcvt_Vhf(Q6_Vh_vshuff_Vh(vx1));

    xnn_storeu_f32(output + 0, Q6_V_lo_W(vw0));
    xnn_storeu_f32(output + 32, Q6_V_hi_W(vw0));
    xnn_storeu_f32(output + 64, Q6_V_lo_W(vw1));
    xnn_storeu_f32(output + 96, Q6_V_hi_W(vw1));
    #else
    const xnn_simd_f32_t vy0 = xnn_cvt_f32_f16(xnn_loadu_f16(input + 0));
    const xnn_simd_f32_t vy1 = xnn_cvt_f32_f16(xnn_loadu_f16(input + 32));
    const xnn_simd_f32_t vy2 = xnn_cvt_f32_f16(xnn_loadu_f16(input + 64));
    const xnn_simd_f32_t vy3 = xnn_cvt_f32_f16(xnn_loadu_f16(input + 96));
    input += 128;

    xnn_storeu_f32(output + 0, vy0);
    xnn_storeu_f32(output + 32, vy1);
    xnn_storeu_f32(output + 64, vy2);
    xnn_storeu_f32(output + 96, vy3);
    #endif
    output += 128;
  }
  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    const xnn_simd_f32_t vy = xnn_cvt_f32_f16(xnn_loadu_f16(input));
    input += 32;

    xnn_storeu_f32(output, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_float16));
    assert(batch <= 31 * sizeof(xnn_float16));

    const size_t num_elements = batch >> XNN_LOG2_SIZEOF_FLOAT16;
    const xnn_simd_f32_t vy = xnn_cvt_f32_f16(xnn_load_tail_f16(input, num_elements));
    xnn_store_tail_f32(output, vy, num_elements);
  }
}
