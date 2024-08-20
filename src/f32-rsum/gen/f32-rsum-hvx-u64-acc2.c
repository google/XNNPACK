// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "xnnpack/simd/f32-hvx.h"

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"


void xnn_f32_rsum_ukernel__hvx_u64_acc2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 32);
    input += 64;

    vacc0 = xnn_add_f32(vacc0, vt0);
    vacc1 = xnn_add_f32(vacc1, vt1);
  }
  vacc0 = xnn_add_f32(vacc0, vacc1);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vacc0 = xnn_add_f32(vacc0, vt);
  }

  if XNN_UNLIKELY(batch) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);

    vacc0 = xnn_add_f32(vacc0, Q6_V_vmux_QVV(mask, vt, xnn_zero_f32()));
  }

  vacc0 = xnn_add_f32(vacc0, Q6_V_vror_VR(vacc0, 64));
  vacc0 = xnn_add_f32(vacc0, Q6_V_vror_VR(vacc0, 32));
  vacc0 = xnn_add_f32(vacc0, Q6_V_vror_VR(vacc0, 16));
  vacc0 = xnn_add_f32(vacc0, Q6_V_vror_VR(vacc0, 8));
  vacc0 = xnn_add_f32(vacc0, Q6_V_vror_VR(vacc0, 4));

  float result = *((float*) &vacc0);

  const float vscale = params->scale;
  result = result * vscale;
  result = math_max_f32(result, params->scalar.min);
  result = math_min_f32(result, params->scalar.max);
  *output += result;
}
