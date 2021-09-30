// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math-stubs.h>


void xnn_math_f16_f32_cvt__neon_int32(
    size_t n,
    const void* input,
    float* output)
{
  assert(n % (8 * sizeof(float)) == 0);

  const uint32x4_t vsign_mask = vmovq_n_u32(0x80000000);
  const uint32x4_t vexp_offset = vmovq_n_u32(0x70000000);
  const float32x4_t vexp_scale = vmovq_n_f32(0x1.0p-112f);
  const uint32x4_t vmagic_mask = vmovq_n_u32(0x3F000000);
  const float32x4_t vmagic_bias = vmovq_n_f32(0.5f);
  const uint32x4_t vdenorm_cutoff = vmovq_n_u32(0x04000000);

  const uint16_t* i = (const uint16_t*) input;
  for (; n != 0; n -= 8 * sizeof(float)) {
    const uint16x8_t vh = vld1q_u16(i); i += 8;

    const uint32x4_t vw_lo = vshll_n_u16(vget_low_u16(vh), 16);
    const uint32x4_t vw_hi = vshll_n_u16(vget_high_u16(vh), 16);

    const uint32x4_t vsign_lo = vandq_u32(vw_lo, vsign_mask);
    const uint32x4_t vsign_hi = vandq_u32(vw_hi, vsign_mask);

    const uint32x4_t vnonsign_lo = veorq_u32(vw_lo, vsign_lo);
    const uint32x4_t vnonsign_hi = veorq_u32(vw_hi, vsign_hi);

    const float32x4_t vnorm_lo = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_lo, 3)), vexp_scale);
    const float32x4_t vnorm_hi = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_hi, 3)), vexp_scale);

    const float32x4_t vdenorm_lo = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_mask, vnonsign_lo, 16)), vmagic_bias);
    const float32x4_t vdenorm_hi = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_mask, vnonsign_hi, 16)), vmagic_bias);

    const uint32x4_t vmask_lo = vcgtq_u32(vnonsign_lo, vdenorm_cutoff);
    const uint32x4_t vmask_hi = vcgtq_u32(vnonsign_hi, vdenorm_cutoff);

    const uint32x4_t vf_lo = vorrq_u32(vsign_lo, vreinterpretq_u32_f32(vbslq_f32(vmask_lo, vnorm_lo, vdenorm_lo)));
    const uint32x4_t vf_hi = vorrq_u32(vsign_hi, vreinterpretq_u32_f32(vbslq_f32(vmask_hi, vnorm_hi, vdenorm_hi)));

    vst1q_f32(output, vreinterpretq_f32_u32(vf_lo)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf_hi)); output += 4;
  }
}
