// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__neon_int16(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const uint16x8_t vsign_mask = vmovq_n_u16(0x8000);
  const uint16x8_t vexp_offset = vmovq_n_u16(0x7000);
  const float32x4_t vexp_scale = vmovq_n_f32(0x1.0p-112f);
  const uint32x4_t vmagic_mask = vmovq_n_u32(0x3F000000);
  const float32x4_t vmagic_bias = vmovq_n_f32(0.5f);
  const uint16x8_t vdenorm_cutoff = vmovq_n_u16(0x0400);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const uint16x8_t vh = vld1q_u16(i); i += 8;

    const uint16x8_t vsign = vandq_u16(vh, vsign_mask);

    const uint16x8_t vnonsign = veorq_u16(vh, vsign);

    const uint16x8x2_t vprenorm = vzipq_u16(vshlq_n_u16(vnonsign, 13), vsraq_n_u16(vexp_offset, vnonsign, 3));
    const float32x4_t vnorm_lo = vmulq_f32(vreinterpretq_f32_u16(vprenorm.val[0]), vexp_scale);
    const float32x4_t vnorm_hi = vmulq_f32(vreinterpretq_f32_u16(vprenorm.val[1]), vexp_scale);

    const float32x4_t vdenorm_lo = vsubq_f32(vreinterpretq_f32_u32(vaddw_u16(vmagic_mask, vget_low_u16(vnonsign))), vmagic_bias);
    const float32x4_t vdenorm_hi = vsubq_f32(vreinterpretq_f32_u32(vaddw_u16(vmagic_mask, vget_high_u16(vnonsign))), vmagic_bias);

    const uint16x8_t vmask = vcgtq_u16(vnonsign, vdenorm_cutoff);
    const uint32x4_t vmask_lo = vreinterpretq_u32_s32(vmovl_s16(vreinterpret_s16_u16(vget_low_u16(vmask))));
    const uint32x4_t vmask_hi = vreinterpretq_u32_s32(vmovl_s16(vreinterpret_s16_u16(vget_high_u16(vmask))));

    const uint32x4_t vf_lo = vorrq_u32(vshll_n_u16(vget_low_u16(vsign), 16),
      vreinterpretq_u32_f32(vbslq_f32(vmask_lo, vnorm_lo, vdenorm_lo)));
    const uint32x4_t vf_hi = vorrq_u32(vshll_n_u16(vget_high_u16(vsign), 16),
      vreinterpretq_u32_f32(vbslq_f32(vmask_hi, vnorm_hi, vdenorm_hi)));

    vst1q_f32(output, vreinterpretq_f32_u32(vf_lo)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf_hi)); output += 4;
  }
}
