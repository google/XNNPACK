// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x8_t vpositive_infinity = vmovq_n_u16(UINT16_C(0x7C00));
  const float16x8_t vhalf = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3800)));  // 0.5h
  const uint16x8_t vexp4_mask = vmovq_n_u16(UINT16_C(0x7800));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx0 = vbslq_f16(vexp4_mask, vhalf, vi0);
    const int16x8_t vexp4i0 = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi0), vexp4_mask));
    const float16x8_t vx1 = vbslq_f16(vexp4_mask, vhalf, vi1);
    const int16x8_t vexp4i1 = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi1), vexp4_mask));
    const float16x8_t vx2 = vbslq_f16(vexp4_mask, vhalf, vi2);
    const int16x8_t vexp4i2 = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi2), vexp4_mask));
    const float16x8_t vx3 = vbslq_f16(vexp4_mask, vhalf, vi3);
    const int16x8_t vexp4i3 = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi3), vexp4_mask));

    const float16x8_t vrsqrtx0 = vrsqrteq_f16(vx0);
    const int16x8_t vpostscale0 = vhsubq_s16(vexp4i0, vreinterpretq_s16_f16(vhalf));
    const float16x8_t vrsqrtx1 = vrsqrteq_f16(vx1);
    const int16x8_t vpostscale1 = vhsubq_s16(vexp4i1, vreinterpretq_s16_f16(vhalf));
    const float16x8_t vrsqrtx2 = vrsqrteq_f16(vx2);
    const int16x8_t vpostscale2 = vhsubq_s16(vexp4i2, vreinterpretq_s16_f16(vhalf));
    const float16x8_t vrsqrtx3 = vrsqrteq_f16(vx3);
    const int16x8_t vpostscale3 = vhsubq_s16(vexp4i3, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx0 = vmulq_f16(vrsqrtx0, vx0);
    const float16x8_t vhalfrsqrtx0 = vmulq_f16(vrsqrtx0, vhalf);
    uint16x8_t vspecial_mask0 = vcgeq_u16(vreinterpretq_u16_f16(vi0), vpositive_infinity);
    float16x8_t vsqrtx1 = vmulq_f16(vrsqrtx1, vx1);
    const float16x8_t vhalfrsqrtx1 = vmulq_f16(vrsqrtx1, vhalf);
    uint16x8_t vspecial_mask1 = vcgeq_u16(vreinterpretq_u16_f16(vi1), vpositive_infinity);
    float16x8_t vsqrtx2 = vmulq_f16(vrsqrtx2, vx2);
    const float16x8_t vhalfrsqrtx2 = vmulq_f16(vrsqrtx2, vhalf);
    uint16x8_t vspecial_mask2 = vcgeq_u16(vreinterpretq_u16_f16(vi2), vpositive_infinity);
    float16x8_t vsqrtx3 = vmulq_f16(vrsqrtx3, vx3);
    const float16x8_t vhalfrsqrtx3 = vmulq_f16(vrsqrtx3, vhalf);
    uint16x8_t vspecial_mask3 = vcgeq_u16(vreinterpretq_u16_f16(vi3), vpositive_infinity);

    const float16x8_t vresidual0 = vfmsq_f16(vhalf, vsqrtx0, vhalfrsqrtx0);
    const uint16x8_t vzero_mask0 = vceqq_f16(vi0, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value0 = vmovq_n_u16(UINT16_C(0x7E00));
    const float16x8_t vresidual1 = vfmsq_f16(vhalf, vsqrtx1, vhalfrsqrtx1);
    const uint16x8_t vzero_mask1 = vceqq_f16(vi1, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value1 = vmovq_n_u16(UINT16_C(0x7E00));
    const float16x8_t vresidual2 = vfmsq_f16(vhalf, vsqrtx2, vhalfrsqrtx2);
    const uint16x8_t vzero_mask2 = vceqq_f16(vi2, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value2 = vmovq_n_u16(UINT16_C(0x7E00));
    const float16x8_t vresidual3 = vfmsq_f16(vhalf, vsqrtx3, vhalfrsqrtx3);
    const uint16x8_t vzero_mask3 = vceqq_f16(vi3, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value3 = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx0 = vfmaq_f16(vsqrtx0, vresidual0, vsqrtx0);
    vspecial_mask0 = vorrq_u16(vspecial_mask0, vzero_mask0);
    const uint16x8_t vinfinity_mask0 = vceqq_u16(vreinterpretq_u16_f16(vi0), vpositive_infinity);
    vsqrtx1 = vfmaq_f16(vsqrtx1, vresidual1, vsqrtx1);
    vspecial_mask1 = vorrq_u16(vspecial_mask1, vzero_mask1);
    const uint16x8_t vinfinity_mask1 = vceqq_u16(vreinterpretq_u16_f16(vi1), vpositive_infinity);
    vsqrtx2 = vfmaq_f16(vsqrtx2, vresidual2, vsqrtx2);
    vspecial_mask2 = vorrq_u16(vspecial_mask2, vzero_mask2);
    const uint16x8_t vinfinity_mask2 = vceqq_u16(vreinterpretq_u16_f16(vi2), vpositive_infinity);
    vsqrtx3 = vfmaq_f16(vsqrtx3, vresidual3, vsqrtx3);
    vspecial_mask3 = vorrq_u16(vspecial_mask3, vzero_mask3);
    const uint16x8_t vinfinity_mask3 = vceqq_u16(vreinterpretq_u16_f16(vi3), vpositive_infinity);

    const float16x8_t vadjustment0 = vfmsq_f16(vx0, vsqrtx0, vsqrtx0);
    const uint16x8_t vinput_mask0 = vorrq_u16(vinfinity_mask0, vzero_mask0);
    const float16x8_t vadjustment1 = vfmsq_f16(vx1, vsqrtx1, vsqrtx1);
    const uint16x8_t vinput_mask1 = vorrq_u16(vinfinity_mask1, vzero_mask1);
    const float16x8_t vadjustment2 = vfmsq_f16(vx2, vsqrtx2, vsqrtx2);
    const uint16x8_t vinput_mask2 = vorrq_u16(vinfinity_mask2, vzero_mask2);
    const float16x8_t vadjustment3 = vfmsq_f16(vx3, vsqrtx3, vsqrtx3);
    const uint16x8_t vinput_mask3 = vorrq_u16(vinfinity_mask3, vzero_mask3);

    vsqrtx0 = vfmaq_f16(vsqrtx0, vhalfrsqrtx0, vadjustment0);
    vspecial_value0 = vbslq_u16(vinput_mask0, vreinterpretq_u16_f16(vi0), vspecial_value0);
    vsqrtx1 = vfmaq_f16(vsqrtx1, vhalfrsqrtx1, vadjustment1);
    vspecial_value1 = vbslq_u16(vinput_mask1, vreinterpretq_u16_f16(vi1), vspecial_value1);
    vsqrtx2 = vfmaq_f16(vsqrtx2, vhalfrsqrtx2, vadjustment2);
    vspecial_value2 = vbslq_u16(vinput_mask2, vreinterpretq_u16_f16(vi2), vspecial_value2);
    vsqrtx3 = vfmaq_f16(vsqrtx3, vhalfrsqrtx3, vadjustment3);
    vspecial_value3 = vbslq_u16(vinput_mask3, vreinterpretq_u16_f16(vi3), vspecial_value3);

    float16x8_t vy0 = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx0), vpostscale0));
    float16x8_t vy1 = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx1), vpostscale1));
    float16x8_t vy2 = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx2), vpostscale2));
    float16x8_t vy3 = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx3), vpostscale3));

    vy0 = vbslq_f16(vspecial_mask0, vreinterpretq_f16_u16(vspecial_value0), vy0);
    vy1 = vbslq_f16(vspecial_mask1, vreinterpretq_f16_u16(vspecial_value1), vy1);
    vy2 = vbslq_f16(vspecial_mask2, vreinterpretq_f16_u16(vspecial_value2), vy2);
    vy3 = vbslq_f16(vspecial_mask3, vreinterpretq_f16_u16(vspecial_value3), vy3);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy3)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vbslq_f16(vexp4_mask, vhalf, vi);
    const int16x8_t vexp4i = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_f16(vi), vexp4_mask));

    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    const int16x8_t vpostscale = vhsubq_s16(vexp4i, vreinterpretq_s16_f16(vhalf));

    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);
    uint16x8_t vspecial_mask = vcgeq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    const uint16x8_t vzero_mask = vceqq_f16(vi, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    uint16x8_t vspecial_value = vmovq_n_u16(UINT16_C(0x7E00));

    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);
    vspecial_mask = vorrq_u16(vspecial_mask, vzero_mask);
    const uint16x8_t vinfinity_mask = vceqq_u16(vreinterpretq_u16_f16(vi), vpositive_infinity);

    const float16x8_t vadjustment = vfmsq_f16(vx, vsqrtx, vsqrtx);
    const uint16x8_t vinput_mask = vorrq_u16(vinfinity_mask, vzero_mask);

    vsqrtx = vfmaq_f16(vsqrtx, vhalfrsqrtx, vadjustment);
    vspecial_value = vbslq_u16(vinput_mask, vreinterpretq_u16_f16(vi), vspecial_value);

    float16x8_t vy = vreinterpretq_f16_s16(vaddq_s16(vreinterpretq_s16_f16(vsqrtx), vpostscale));

    vy = vbslq_f16(vspecial_mask, vreinterpretq_f16_u16(vspecial_value), vy);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
