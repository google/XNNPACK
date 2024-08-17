// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/neon-int32.c.in
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


void xnn_f16_f32_vcvt_ukernel__neon_int32_u32(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));
  const uint32x4_t vexp_offset = vmovq_n_u32(UINT32_C(0x70000000));
  const float32x4_t vexp_scale = vmovq_n_f32(0x1.0p-112f);
  const uint32x4_t vmagic_bias = vmovq_n_u32(UINT32_C(0x3F000000));
  const uint32x4_t vdenorm_cutoff = vmovq_n_u32(UINT32_C(0x04000000));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const uint16x8_t vh0 = vld1q_u16(i); i += 8;
    const uint16x8_t vh1 = vld1q_u16(i); i += 8;
    const uint16x8_t vh2 = vld1q_u16(i); i += 8;
    const uint16x8_t vh3 = vld1q_u16(i); i += 8;

    const uint32x4_t vw0 = vshll_n_u16(vget_low_u16(vh0), 16);
    const uint32x4_t vw1 = vshll_n_u16(vget_high_u16(vh0), 16);
    const uint32x4_t vw2 = vshll_n_u16(vget_low_u16(vh1), 16);
    const uint32x4_t vw3 = vshll_n_u16(vget_high_u16(vh1), 16);
    const uint32x4_t vw4 = vshll_n_u16(vget_low_u16(vh2), 16);
    const uint32x4_t vw5 = vshll_n_u16(vget_high_u16(vh2), 16);
    const uint32x4_t vw6 = vshll_n_u16(vget_low_u16(vh3), 16);
    const uint32x4_t vw7 = vshll_n_u16(vget_high_u16(vh3), 16);

    const uint32x4_t vsign0 = vandq_u32(vw0, vsign_mask);
    const uint32x4_t vsign1 = vandq_u32(vw1, vsign_mask);
    const uint32x4_t vsign2 = vandq_u32(vw2, vsign_mask);
    const uint32x4_t vsign3 = vandq_u32(vw3, vsign_mask);
    const uint32x4_t vsign4 = vandq_u32(vw4, vsign_mask);
    const uint32x4_t vsign5 = vandq_u32(vw5, vsign_mask);
    const uint32x4_t vsign6 = vandq_u32(vw6, vsign_mask);
    const uint32x4_t vsign7 = vandq_u32(vw7, vsign_mask);

    const uint32x4_t vnonsign0 = veorq_u32(vw0, vsign0);
    const uint32x4_t vnonsign1 = veorq_u32(vw1, vsign1);
    const uint32x4_t vnonsign2 = veorq_u32(vw2, vsign2);
    const uint32x4_t vnonsign3 = veorq_u32(vw3, vsign3);
    const uint32x4_t vnonsign4 = veorq_u32(vw4, vsign4);
    const uint32x4_t vnonsign5 = veorq_u32(vw5, vsign5);
    const uint32x4_t vnonsign6 = veorq_u32(vw6, vsign6);
    const uint32x4_t vnonsign7 = veorq_u32(vw7, vsign7);

    const float32x4_t vnorm0 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign0, 3)), vexp_scale);
    const float32x4_t vnorm1 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign1, 3)), vexp_scale);
    const float32x4_t vnorm2 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign2, 3)), vexp_scale);
    const float32x4_t vnorm3 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign3, 3)), vexp_scale);
    const float32x4_t vnorm4 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign4, 3)), vexp_scale);
    const float32x4_t vnorm5 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign5, 3)), vexp_scale);
    const float32x4_t vnorm6 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign6, 3)), vexp_scale);
    const float32x4_t vnorm7 = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign7, 3)), vexp_scale);

    const float32x4_t vdenorm0 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign0, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm1 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign1, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm2 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign2, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm3 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign3, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm4 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign4, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm5 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign5, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm6 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign6, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm7 = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign7, 16)), vreinterpretq_f32_u32(vmagic_bias));

    const uint32x4_t vxmask0 = vcgtq_u32(vnonsign0, vdenorm_cutoff);
    const uint32x4_t vxmask1 = vcgtq_u32(vnonsign1, vdenorm_cutoff);
    const uint32x4_t vxmask2 = vcgtq_u32(vnonsign2, vdenorm_cutoff);
    const uint32x4_t vxmask3 = vcgtq_u32(vnonsign3, vdenorm_cutoff);
    const uint32x4_t vxmask4 = vcgtq_u32(vnonsign4, vdenorm_cutoff);
    const uint32x4_t vxmask5 = vcgtq_u32(vnonsign5, vdenorm_cutoff);
    const uint32x4_t vxmask6 = vcgtq_u32(vnonsign6, vdenorm_cutoff);
    const uint32x4_t vxmask7 = vcgtq_u32(vnonsign7, vdenorm_cutoff);

    const uint32x4_t vf0 = vorrq_u32(vsign0, vreinterpretq_u32_f32(vbslq_f32(vxmask0, vnorm0, vdenorm0)));
    const uint32x4_t vf1 = vorrq_u32(vsign1, vreinterpretq_u32_f32(vbslq_f32(vxmask1, vnorm1, vdenorm1)));
    const uint32x4_t vf2 = vorrq_u32(vsign2, vreinterpretq_u32_f32(vbslq_f32(vxmask2, vnorm2, vdenorm2)));
    const uint32x4_t vf3 = vorrq_u32(vsign3, vreinterpretq_u32_f32(vbslq_f32(vxmask3, vnorm3, vdenorm3)));
    const uint32x4_t vf4 = vorrq_u32(vsign4, vreinterpretq_u32_f32(vbslq_f32(vxmask4, vnorm4, vdenorm4)));
    const uint32x4_t vf5 = vorrq_u32(vsign5, vreinterpretq_u32_f32(vbslq_f32(vxmask5, vnorm5, vdenorm5)));
    const uint32x4_t vf6 = vorrq_u32(vsign6, vreinterpretq_u32_f32(vbslq_f32(vxmask6, vnorm6, vdenorm6)));
    const uint32x4_t vf7 = vorrq_u32(vsign7, vreinterpretq_u32_f32(vbslq_f32(vxmask7, vnorm7, vdenorm7)));

    vst1q_f32(output, vreinterpretq_f32_u32(vf0)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf1)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf2)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf3)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf4)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf5)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf6)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf7)); output += 4;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const uint16x8_t vh = vld1q_u16(i); i += 8;

    const uint32x4_t vw_lo = vshll_n_u16(vget_low_u16(vh), 16);
    const uint32x4_t vw_hi = vshll_n_u16(vget_high_u16(vh), 16);

    const uint32x4_t vsign_lo = vandq_u32(vw_lo, vsign_mask);
    const uint32x4_t vsign_hi = vandq_u32(vw_hi, vsign_mask);

    const uint32x4_t vnonsign_lo = veorq_u32(vw_lo, vsign_lo);
    const uint32x4_t vnonsign_hi = veorq_u32(vw_hi, vsign_hi);

    const float32x4_t vnorm_lo = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_lo, 3)), vexp_scale);
    const float32x4_t vnorm_hi = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_hi, 3)), vexp_scale);

    const float32x4_t vdenorm_lo = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign_lo, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm_hi = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign_hi, 16)), vreinterpretq_f32_u32(vmagic_bias));

    const uint32x4_t vxmask_lo = vcgtq_u32(vnonsign_lo, vdenorm_cutoff);
    const uint32x4_t vf_lo = vorrq_u32(vsign_lo, vreinterpretq_u32_f32(vbslq_f32(vxmask_lo, vnorm_lo, vdenorm_lo)));

    const uint32x4_t vxmask_hi = vcgtq_u32(vnonsign_hi, vdenorm_cutoff);
    const uint32x4_t vf_hi = vorrq_u32(vsign_hi, vreinterpretq_u32_f32(vbslq_f32(vxmask_hi, vnorm_hi, vdenorm_hi)));

    vst1q_f32(output, vreinterpretq_f32_u32(vf_lo)); output += 4;
    vst1q_f32(output, vreinterpretq_f32_u32(vf_hi)); output += 4;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const uint16x8_t vh = vld1q_u16(i); i += 8;

    const uint32x4_t vw_lo = vshll_n_u16(vget_low_u16(vh), 16);
    const uint32x4_t vw_hi = vshll_n_u16(vget_high_u16(vh), 16);

    const uint32x4_t vsign_lo = vandq_u32(vw_lo, vsign_mask);
    const uint32x4_t vsign_hi = vandq_u32(vw_hi, vsign_mask);

    const uint32x4_t vnonsign_lo = veorq_u32(vw_lo, vsign_lo);
    const uint32x4_t vnonsign_hi = veorq_u32(vw_hi, vsign_hi);

    const float32x4_t vnorm_lo = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_lo, 3)), vexp_scale);
    const float32x4_t vnorm_hi = vmulq_f32(vreinterpretq_f32_u32(vsraq_n_u32(vexp_offset, vnonsign_hi, 3)), vexp_scale);

    const float32x4_t vdenorm_lo = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign_lo, 16)), vreinterpretq_f32_u32(vmagic_bias));
    const float32x4_t vdenorm_hi = vsubq_f32(vreinterpretq_f32_u32(vsriq_n_u32(vmagic_bias, vnonsign_hi, 16)), vreinterpretq_f32_u32(vmagic_bias));

    const uint32x4_t vxmask_lo = vcgtq_u32(vnonsign_lo, vdenorm_cutoff);
    uint32x4_t vf = vorrq_u32(vsign_lo, vreinterpretq_u32_f32(vbslq_f32(vxmask_lo, vnorm_lo, vdenorm_lo)));

    if (batch & (4 * sizeof(uint16_t))) {
      vst1q_f32(output, vreinterpretq_f32_u32(vf)); output += 4;

      const uint32x4_t vxmask_hi = vcgtq_u32(vnonsign_hi, vdenorm_cutoff);
      vf = vorrq_u32(vsign_hi, vreinterpretq_u32_f32(vbslq_f32(vxmask_hi, vnorm_hi, vdenorm_hi)));
    }
    uint32x2_t vf_lo = vget_low_u32(vf);
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_f32(output, vreinterpret_f32_u32(vf_lo)); output += 2;
      vf_lo = vget_high_u32(vf);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_f32(output, vreinterpret_f32_u32(vf_lo), 0);
    }
  }
}
