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


void xnn_f32_f16_vcvt_ukernel__neon_u32(
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
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const float32x4_t vx0 = vld1q_f32(input); input += 4;
    const float32x4_t vx1 = vld1q_f32(input); input += 4;
    const float32x4_t vx2 = vld1q_f32(input); input += 4;
    const float32x4_t vx3 = vld1q_f32(input); input += 4;
    const float32x4_t vx4 = vld1q_f32(input); input += 4;
    const float32x4_t vx5 = vld1q_f32(input); input += 4;
    const float32x4_t vx6 = vld1q_f32(input); input += 4;
    const float32x4_t vx7 = vld1q_f32(input); input += 4;

    const float32x4_t vabsx0 = vabsq_f32(vx0);
    const float32x4_t vabsx1 = vabsq_f32(vx1);
    const float32x4_t vabsx2 = vabsq_f32(vx2);
    const float32x4_t vabsx3 = vabsq_f32(vx3);
    const float32x4_t vabsx4 = vabsq_f32(vx4);
    const float32x4_t vabsx5 = vabsq_f32(vx5);
    const float32x4_t vabsx6 = vabsq_f32(vx6);
    const float32x4_t vabsx7 = vabsq_f32(vx7);

    uint32x4_t vbias0 = vaddq_u32(vreinterpretq_u32_f32(vabsx0), vexp_bias);
    uint32x4_t vbias1 = vaddq_u32(vreinterpretq_u32_f32(vabsx1), vexp_bias);
    uint32x4_t vbias2 = vaddq_u32(vreinterpretq_u32_f32(vabsx2), vexp_bias);
    uint32x4_t vbias3 = vaddq_u32(vreinterpretq_u32_f32(vabsx3), vexp_bias);
    uint32x4_t vbias4 = vaddq_u32(vreinterpretq_u32_f32(vabsx4), vexp_bias);
    uint32x4_t vbias5 = vaddq_u32(vreinterpretq_u32_f32(vabsx5), vexp_bias);
    uint32x4_t vbias6 = vaddq_u32(vreinterpretq_u32_f32(vabsx6), vexp_bias);
    uint32x4_t vbias7 = vaddq_u32(vreinterpretq_u32_f32(vabsx7), vexp_bias);

    float32x4_t vf0 = vmulq_f32(vabsx0, vscale_to_inf);
    float32x4_t vf1 = vmulq_f32(vabsx1, vscale_to_inf);
    float32x4_t vf2 = vmulq_f32(vabsx2, vscale_to_inf);
    float32x4_t vf3 = vmulq_f32(vabsx3, vscale_to_inf);
    float32x4_t vf4 = vmulq_f32(vabsx4, vscale_to_inf);
    float32x4_t vf5 = vmulq_f32(vabsx5, vscale_to_inf);
    float32x4_t vf6 = vmulq_f32(vabsx6, vscale_to_inf);
    float32x4_t vf7 = vmulq_f32(vabsx7, vscale_to_inf);
    const uint32x4_t vnanmaskw0 = vcgtq_u32(vreinterpretq_u32_f32(vabsx0), vexpw_max);
    const uint32x4_t vnanmaskw1 = vcgtq_u32(vreinterpretq_u32_f32(vabsx1), vexpw_max);
    const uint32x4_t vnanmaskw2 = vcgtq_u32(vreinterpretq_u32_f32(vabsx2), vexpw_max);
    const uint32x4_t vnanmaskw3 = vcgtq_u32(vreinterpretq_u32_f32(vabsx3), vexpw_max);
    const uint32x4_t vnanmaskw4 = vcgtq_u32(vreinterpretq_u32_f32(vabsx4), vexpw_max);
    const uint32x4_t vnanmaskw5 = vcgtq_u32(vreinterpretq_u32_f32(vabsx5), vexpw_max);
    const uint32x4_t vnanmaskw6 = vcgtq_u32(vreinterpretq_u32_f32(vabsx6), vexpw_max);
    const uint32x4_t vnanmaskw7 = vcgtq_u32(vreinterpretq_u32_f32(vabsx7), vexpw_max);

    vbias0 = vandq_u32(vbias0, vexpw_max);
    vbias1 = vandq_u32(vbias1, vexpw_max);
    vbias2 = vandq_u32(vbias2, vexpw_max);
    vbias3 = vandq_u32(vbias3, vexpw_max);
    vbias4 = vandq_u32(vbias4, vexpw_max);
    vbias5 = vandq_u32(vbias5, vexpw_max);
    vbias6 = vandq_u32(vbias6, vexpw_max);
    vbias7 = vandq_u32(vbias7, vexpw_max);
    vf0 = vmulq_f32(vf0, vscale_to_zero);
    vf1 = vmulq_f32(vf1, vscale_to_zero);
    vf2 = vmulq_f32(vf2, vscale_to_zero);
    vf3 = vmulq_f32(vf3, vscale_to_zero);
    vf4 = vmulq_f32(vf4, vscale_to_zero);
    vf5 = vmulq_f32(vf5, vscale_to_zero);
    vf6 = vmulq_f32(vf6, vscale_to_zero);
    vf7 = vmulq_f32(vf7, vscale_to_zero);

    const uint16x8_t vnanmaskh0 = vcombine_u16(vmovn_u32(vnanmaskw0), vmovn_u32(vnanmaskw1));
    const uint16x8_t vnanmaskh1 = vcombine_u16(vmovn_u32(vnanmaskw2), vmovn_u32(vnanmaskw3));
    const uint16x8_t vnanmaskh2 = vcombine_u16(vmovn_u32(vnanmaskw4), vmovn_u32(vnanmaskw5));
    const uint16x8_t vnanmaskh3 = vcombine_u16(vmovn_u32(vnanmaskw6), vmovn_u32(vnanmaskw7));
    vbias0 = vmaxq_u32(vbias0, vbias_min);
    vbias1 = vmaxq_u32(vbias1, vbias_min);
    vbias2 = vmaxq_u32(vbias2, vbias_min);
    vbias3 = vmaxq_u32(vbias3, vbias_min);
    vbias4 = vmaxq_u32(vbias4, vbias_min);
    vbias5 = vmaxq_u32(vbias5, vbias_min);
    vbias6 = vmaxq_u32(vbias6, vbias_min);
    vbias7 = vmaxq_u32(vbias7, vbias_min);

    vf0 = vaddq_f32(vf0, vreinterpretq_f32_u32(vbias0));
    vf1 = vaddq_f32(vf1, vreinterpretq_f32_u32(vbias1));
    vf2 = vaddq_f32(vf2, vreinterpretq_f32_u32(vbias2));
    vf3 = vaddq_f32(vf3, vreinterpretq_f32_u32(vbias3));
    vf4 = vaddq_f32(vf4, vreinterpretq_f32_u32(vbias4));
    vf5 = vaddq_f32(vf5, vreinterpretq_f32_u32(vbias5));
    vf6 = vaddq_f32(vf6, vreinterpretq_f32_u32(vbias6));
    vf7 = vaddq_f32(vf7, vreinterpretq_f32_u32(vbias7));

    uint16x8_t vexph0 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf0), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf1), 13));
    uint16x8_t vexph1 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf2), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf3), 13));
    uint16x8_t vexph2 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf4), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf5), 13));
    uint16x8_t vexph3 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf6), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf7), 13));
    uint16x8_t vmanth0 = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf0)), vmovn_u32(vreinterpretq_u32_f32(vf1)));
    uint16x8_t vmanth1 = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf2)), vmovn_u32(vreinterpretq_u32_f32(vf3)));
    uint16x8_t vmanth2 = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf4)), vmovn_u32(vreinterpretq_u32_f32(vf5)));
    uint16x8_t vmanth3 = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf6)), vmovn_u32(vreinterpretq_u32_f32(vf7)));
    uint16x8_t vsignh0 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx0), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx1), 16));
    uint16x8_t vsignh1 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx2), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx3), 16));
    uint16x8_t vsignh2 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx4), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx5), 16));
    uint16x8_t vsignh3 = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx6), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx7), 16));

    vexph0 = vandq_u16(vexph0, vexph_mask);
    vexph1 = vandq_u16(vexph1, vexph_mask);
    vexph2 = vandq_u16(vexph2, vexph_mask);
    vexph3 = vandq_u16(vexph3, vexph_mask);
    vmanth0 = vandq_u16(vmanth0, vmanth_mask);
    vmanth1 = vandq_u16(vmanth1, vmanth_mask);
    vmanth2 = vandq_u16(vmanth2, vmanth_mask);
    vmanth3 = vandq_u16(vmanth3, vmanth_mask);
    vsignh0 = vandq_u16(vsignh0, vsignh_mask);
    vsignh1 = vandq_u16(vsignh1, vsignh_mask);
    vsignh2 = vandq_u16(vsignh2, vsignh_mask);
    vsignh3 = vandq_u16(vsignh3, vsignh_mask);

    uint16x8_t vh0 = vaddq_u16(vmanth0, vexph0);
    uint16x8_t vh1 = vaddq_u16(vmanth1, vexph1);
    uint16x8_t vh2 = vaddq_u16(vmanth2, vexph2);
    uint16x8_t vh3 = vaddq_u16(vmanth3, vexph3);

    vh0 = vbslq_u16(vnanmaskh0, vnanh, vh0);
    vh1 = vbslq_u16(vnanmaskh1, vnanh, vh1);
    vh2 = vbslq_u16(vnanmaskh2, vnanh, vh2);
    vh3 = vbslq_u16(vnanmaskh3, vnanh, vh3);

    vh0 = vorrq_u16(vh0, vsignh0);
    vh1 = vorrq_u16(vh1, vsignh1);
    vh2 = vorrq_u16(vh2, vsignh2);
    vh3 = vorrq_u16(vh3, vsignh3);

    vst1q_u16(o, vh0); o += 8;
    vst1q_u16(o, vh1); o += 8;
    vst1q_u16(o, vh2); o += 8;
    vst1q_u16(o, vh3); o += 8;
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
