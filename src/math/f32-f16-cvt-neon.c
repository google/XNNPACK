// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f32_f16_cvt__neon(
    size_t n,
    const float* input,
    void* output)
{
  assert(n % (8 * sizeof(uint16_t)) == 0);

  const uint32x4_t vexp_bias = vdupq_n_u32(UINT32_C(0x07800000));
  const float32x4_t vscale_to_inf = vdupq_n_f32(0x1.0p+112f);
  const uint32x4_t vexpw_max = vdupq_n_u32(UINT32_C(0x7F800000));
  const float32x4_t vscale_to_zero = vdupq_n_f32(0x1.0p-110f);
  const uint32x4_t vbias_min = vdupq_n_u32(UINT32_C(0x40000000));
  const uint16x8_t vexph_mask = vdupq_n_u16(UINT16_C(0x7C00));
  const uint16x8_t vmanth_mask = vdupq_n_u16(UINT16_C(0x0FFF));
  const uint16x8_t vsignh_mask = vdupq_n_u16(UINT16_C(0x8000));
  const uint16x8_t vnanh = vdupq_n_u16(UINT16_C(0x7E00));

  uint16_t* o = (uint16_t*) output;
  for (; n != 0; n -= 8 * sizeof(uint16_t)) {
    const float32x4_t vx_lo = vld1q_f32(input); input += 4;
    const float32x4_t vx_hi = vld1q_f32(input); input += 4;

    const float32x4_t vabsx_lo = vabsq_f32(vx_lo);
    const float32x4_t vabsx_hi = vabsq_f32(vx_hi);

    uint32x4_t vbias_lo = vaddq_u32(vreinterpretq_u32_f32(vabsx_lo), vexp_bias);
    uint32x4_t vbias_hi = vaddq_u32(vreinterpretq_u32_f32(vabsx_hi), vexp_bias);

    float32x4_t vf_lo = vmulq_f32(vabsx_lo, vscale_to_inf);
    float32x4_t vf_hi = vmulq_f32(vabsx_hi, vscale_to_inf);
    const uint32x4_t vnanmaskw_lo = vcgtq_u32(vreinterpretq_u32_f32(vabsx_lo), vexpw_max);
    const uint32x4_t vnanmaskw_hi = vcgtq_u32(vreinterpretq_u32_f32(vabsx_hi), vexpw_max);

    vbias_lo = vandq_u32(vbias_lo, vexpw_max);
    vbias_hi = vandq_u32(vbias_hi, vexpw_max);
    vf_lo = vmulq_f32(vf_lo, vscale_to_zero);
    vf_hi = vmulq_f32(vf_hi, vscale_to_zero);

    const uint16x8_t vnanmaskh = vcombine_u16(vmovn_u32(vnanmaskw_lo), vmovn_u32(vnanmaskw_hi));
    vbias_lo = vmaxq_u32(vbias_lo, vbias_min);
    vbias_hi = vmaxq_u32(vbias_hi, vbias_min);

    vf_lo = vaddq_f32(vf_lo, vreinterpretq_f32_u32(vbias_lo));
    vf_hi = vaddq_f32(vf_hi, vreinterpretq_f32_u32(vbias_hi));

    uint16x8_t vexph = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vf_lo), 13), vshrn_n_u32(vreinterpretq_u32_f32(vf_hi), 13));
    uint16x8_t vmanth = vcombine_u16(vmovn_u32(vreinterpretq_u32_f32(vf_lo)), vmovn_u32(vreinterpretq_u32_f32(vf_hi)));
    uint16x8_t vsignh = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_f32(vx_lo), 16), vshrn_n_u32(vreinterpretq_u32_f32(vx_hi), 16));

    vexph = vandq_u16(vexph, vexph_mask);
    vmanth = vandq_u16(vmanth, vmanth_mask);
    vsignh = vandq_u16(vsignh, vsignh_mask);

    uint16x8_t vh = vaddq_u16(vmanth, vexph);
    vh = vbslq_u16(vnanmaskh, vnanh, vh);
    vh = vorrq_u16(vh, vsignh);

    vst1q_u16(o, vh); o += 8;
  }
}
