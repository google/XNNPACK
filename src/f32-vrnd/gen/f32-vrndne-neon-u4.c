// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndne-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrndne_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vmagic_number = vreinterpretq_f32_u32(vmovq_n_u32(UINT32_C(0x4B000000)));
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;

    const float32x4_t vabsx0123 = vabsq_f32(vx0123);
    uint32x4_t vrndmask0123 = vcaltq_f32(vmagic_number, vx0123);

    float32x4_t vrndabsx0123 = vaddq_f32(vabsx0123, vmagic_number);

    vrndmask0123 = vorrq_u32(vrndmask0123, vmovq_n_u32(UINT32_C(0x80000000)));

    vrndabsx0123 = vsubq_f32(vrndabsx0123, vmagic_number);

    const float32x4_t vy0123 = vbslq_f32(vrndmask0123, vx0123, vrndabsx0123);

    vst1q_f32(output, vy0123); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x4_t vabsx = vabsq_f32(vx);
    uint32x4_t vrndmask = vcaltq_f32(vmagic_number, vx);
    float32x4_t vrndabsx = vaddq_f32(vabsx, vmagic_number);
    vrndmask = vorrq_u32(vrndmask, vmovq_n_u32(UINT32_C(0x80000000)));
    vrndabsx = vsubq_f32(vrndabsx, vmagic_number);
    const float32x4_t vy = vbslq_f32(vrndmask, vx, vrndabsx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
