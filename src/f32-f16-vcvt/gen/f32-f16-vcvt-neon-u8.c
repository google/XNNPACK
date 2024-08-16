// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_f32_f16_vcvt_ukernel__neon_u8(
    size_t batch,
    const float* input,
    void* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32x4_t vexp_bias = vdupq_n_u32(UINT32_C(0x07800000));
  const float32x4_t vscale_to_inf = vdupq_n_f32(0x1.0p+112f);
  const uint32x4_t vexpw_max = vdupq_n_u32(UINT32_C(0x7F800000));
  const float32x4_t vscale_to_zero = vdupq_n_f32(0x1.0p-110f);
  const uint32x4_t vbias_min = vdupq_n_u32(UINT32_C(0x40000000));
  const uint16x8_t vexph_mask = vdupq_n_u16(UINT16_C(0x7C00));
  const uint16x8_t vmanth_mask = vdupq_n_u16(UINT16_C(0x0FFF));
  const uint16x8_t vsignh_mask = vdupq_n_u16(UINT16_C(0x8000));
  const uint16x8_t vnanh = vdupq_n_u16(UINT16_C(0x7E00));

  // Only realizing a subset of these to match prior behavior.
  XNN_FORCE_REALIZATION(vexp_bias);
  XNN_FORCE_REALIZATION(vscale_to_inf);
  XNN_FORCE_REALIZATION(vexpw_max);
  XNN_FORCE_REALIZATION(vscale_to_zero);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0 = vld1q_f32(input); input += 4;
    const float32x4_t vx1 = vld1q_f32(input); input += 4;

    const float32x4_t vabsx0 = vabsq_f32(vx0);
    const float32x4_t vabsx1 = vabsq_f32(vx1);

    uint32x4_t vbias0 = vaddq_u32(vreinterpretq_u32_f32(vabsx0), vexp_bias);
    uint32x4_t vbias1 = vaddq_u32(vreinterpretq_u32_f32(vabsx1), vexp_bias);

    float32x4_t vf0 = vmulq_f32(vabsx0, vscale_to_inf);
    float32x4_t vf1 = vmulq_f32(vabsx1, vscale_to_inf);
    const uint32x4_t vnanmaskw0 = vcgtq_u32(vreinterpretq_u32_f32(vabsx0), vexpw_max);
    const uint32x4_t vnanmaskw1 = vcgtq_u32(vreinterpretq_u32_f32(vabsx1), vexpw_max);

    vbias0 = vandq_u32(vbias0, vexpw_max);
    vbias1 = vandq_u32(vbias1, vexpw_max);
    vf0 = vmulq_f32(vf0, vscale_to_zero);
    vf1 = vmulq_f32(vf1, vscale_to_zero);

    const uint16x8_t vnanmaskh0 = vcombine_u16(vmovn_u32(vnanmaskw0), vmovn_u32(vnanmaskw1));
    vbias0 = vmaxq_u32(vbias0, vbias_min);
    vbias1 = vmaxq_u32(vbias1, vbias_min);

    vf0 = vaddq_f32(vf0, vreinterpretq_f32_u32(vbias0));
    vf1 = vaddq_f32(vf1, vreinterpretq_f32_u32(vbias1));

    uint16x8_t vexph0 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf0), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf1), 13));
    uint16x8_t vmanth0 = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf0)), vmovn_u32(vreinterpretq_u32_f32(vf1)));
    uint16x8_t vsignh0 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx0), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx1), 16));

    vexph0 = vandq_u16(vexph0, vexph_mask);
    vmanth0 = vandq_u16(vmanth0, vmanth_mask);
    vsignh0 = vandq_u16(vsignh0, vsignh_mask);

    uint16x8_t vh0 = vaddq_u16(vmanth0, vexph0);

    vh0 = vbslq_u16(vnanmaskh0, vnanh, vh0);

    vh0 = vorrq_u16(vh0, vsignh0);

    vst1q_u16(o, vh0); o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vabsx = vabsq_f32(vx);

    uint32x4_t vbias = vaddq_u32(vreinterpretq_u32_f32(vabsx), vexp_bias);

    float32x4_t vf = vmulq_f32(vabsx, vscale_to_inf);
    const uint32x4_t vnanmaskw = vcgtq_u32(vreinterpretq_u32_f32(vabsx), vexpw_max);

    vbias = vandq_u32(vbias, vexpw_max);
    vf = vmulq_f32(vf, vscale_to_zero);

    const uint16x4_t vnanmaskh = vmovn_u32(vnanmaskw);
    vbias = vmaxq_u32(vbias, vbias_min);

    vf = vaddq_f32(vf, vreinterpretq_f32_u32(vbias));

    uint16x4_t vexph = vshrn_n_u32(vreinterpretq_u32_f32(vf), 13);
    uint16x4_t vmanth = vmovn_u32(vreinterpretq_u32_f32(vf));
    uint16x4_t vsignh = vshrn_n_u32(vreinterpretq_u32_f32(vx), 16);

    vexph = vand_u16(vexph, vget_low_u16(vexph_mask));
    vmanth = vand_u16(vmanth, vget_low_u16(vmanth_mask));
    vsignh = vand_u16(vsignh, vget_low_u16(vsignh_mask));

    uint16x4_t vh = vadd_u16(vmanth, vexph);

    vh = vbsl_u16(vnanmaskh, vget_low_u16(vnanh), vh);

    vh = vorr_u16(vh, vsignh);

    vst1_u16(o, vh); o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch % sizeof(float) == 0);
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vx = vld1q_f32(input);

    const float32x4_t vabsx = vabsq_f32(vx);

    uint32x4_t vbias = vaddq_u32(vreinterpretq_u32_f32(vabsx), vexp_bias);

    float32x4_t vf = vmulq_f32(vabsx, vscale_to_inf);
    const uint32x4_t vnanmaskw = vcgtq_u32(vreinterpretq_u32_f32(vabsx), vexpw_max);

    vbias = vandq_u32(vbias, vexpw_max);
    vf = vmulq_f32(vf, vscale_to_zero);

    const uint16x4_t vnanmaskh = vmovn_u32(vnanmaskw);
    vbias = vmaxq_u32(vbias, vbias_min);

    vf = vaddq_f32(vf, vreinterpretq_f32_u32(vbias));

    uint16x4_t vexph = vshrn_n_u32(vreinterpretq_u32_f32(vf), 13);
    uint16x4_t vmanth = vmovn_u32(vreinterpretq_u32_f32(vf));
    uint16x4_t vsignh = vshrn_n_u32(vreinterpretq_u32_f32(vx), 16);

    vexph = vand_u16(vexph, vget_low_u16(vexph_mask));
    vmanth = vand_u16(vmanth, vget_low_u16(vmanth_mask));
    vsignh = vand_u16(vsignh, vget_low_u16(vsignh_mask));

    uint16x4_t vh = vadd_u16(vmanth, vexph);

    vh = vbsl_u16(vnanmaskh, vget_low_u16(vnanh), vh);

    vh = vorr_u16(vh, vsignh);

    if (batch & (2 * sizeof(float))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vh), 0); o += 2;
      vh = vext_u16(vh, vh, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u16(o, vh, 0);
    }
  }
}
