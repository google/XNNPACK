// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f16_f32acc_rsum_ukernel__neonfp16arith_u4(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  const float32x2_t vscale = vld1_dup_f32(&params->scale);
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}
