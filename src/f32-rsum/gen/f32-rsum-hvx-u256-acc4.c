// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"


void xnn_f32_rsum_ukernel__hvx_u256_acc4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  xnn_simd_f32_t vacc2 = xnn_zero_f32();
  xnn_simd_f32_t vacc3 = xnn_zero_f32();
  for (; batch >= 256 * sizeof(float); batch -= 256 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 32);
    const xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 64);
    const xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 96);
    const xnn_simd_f32_t vt4 = xnn_loadu_f32(input + 128);
    const xnn_simd_f32_t vt5 = xnn_loadu_f32(input + 160);
    const xnn_simd_f32_t vt6 = xnn_loadu_f32(input + 192);
    const xnn_simd_f32_t vt7 = xnn_loadu_f32(input + 224);
    input += 256;

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, vt0);
    vacc1 = Q6_Vqf32_vadd_Vqf32Vsf(vacc1, vt1);
    vacc2 = Q6_Vqf32_vadd_Vqf32Vsf(vacc2, vt2);
    vacc3 = Q6_Vqf32_vadd_Vqf32Vsf(vacc3, vt3);
    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, vt4);
    vacc1 = Q6_Vqf32_vadd_Vqf32Vsf(vacc1, vt5);
    vacc2 = Q6_Vqf32_vadd_Vqf32Vsf(vacc2, vt6);
    vacc3 = Q6_Vqf32_vadd_Vqf32Vsf(vacc3, vt7);
  }
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, vacc1);
  vacc2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2, vacc3);
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, vacc2);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, vt);
  }

  if XNN_UNLIKELY(batch) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, Q6_V_vmux_QVV(mask, vt, xnn_zero_f32()));
  }

  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_V_vror_VR(vacc0, 64));
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_V_vror_VR(vacc0, 32));
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_V_vror_VR(vacc0, 16));
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_V_vror_VR(vacc0, 8));
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_V_vror_VR(vacc0, 4));
  vacc0 = Q6_Vsf_equals_Vqf32(vacc0);

  float partial_sum = *((float*) &vacc0);

  const float vscale = params->scalar.scale;
  *output += partial_sum * vscale;
}
