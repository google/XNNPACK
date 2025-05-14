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


void xnn_f32_rsum_ukernel__hvx_u64_acc2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert((size_t)input % sizeof(float) == 0);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  xnn_simd_f32_t vacc1 = xnn_zero_f32();

  const size_t alignment_size = -(size_t)input & 127;
  if XNN_UNLIKELY(alignment_size) {
    const size_t head_size = alignment_size > batch ? batch : alignment_size;
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    HVX_VectorPred mask = Q6_Q_vsetq_R(head_size);
    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, Q6_V_vmux_QVV(mask, vt, xnn_zero_f32()));
    batch -= head_size;
    input = (const float*)((intptr_t)input + head_size);
  }

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_load_f32(input + 0);
    const xnn_simd_f32_t vt1 = xnn_load_f32(input + 32);
    input += 64;

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, vt0);
    vacc1 = Q6_Vqf32_vadd_Vqf32Vsf(vacc1, vt1);
  }
  vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, vacc1);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_load_f32(input);
    input += 32;

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, vt);
  }

  if XNN_UNLIKELY(batch) {
    const xnn_simd_f32_t vt = xnn_load_tail_f32(input, batch >> 2);
    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);

    vacc0 = Q6_Vqf32_vadd_Vqf32Vsf(vacc0, Q6_V_vand_QV(mask, vt));
  }

  float partial_sum = xnn_reduce_add_f32(Q6_Vsf_equals_Vqf32(vacc0));
  const float vscale = params->scalar.scale;
  *output += partial_sum * vscale;
}
