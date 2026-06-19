// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__neon_u24(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32x4_t vexp_mask = vdupq_n_u32(0x7F800000u);
  const uint32x4_t vbias = vdupq_n_u32(0x7FFFu);
  const uint32x4_t vone = vdupq_n_u32(1u);
  const uint32x4_t vabs_mask = vdupq_n_u32(0x7FFFFFFFu);
  const uint32x4_t vquiet = vdupq_n_u32(0x00400000u);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const float32x4_t vf0 = vld1q_f32(input); input += 4;
    const float32x4_t vf1 = vld1q_f32(input); input += 4;
    const float32x4_t vf2 = vld1q_f32(input); input += 4;
    const float32x4_t vf3 = vld1q_f32(input); input += 4;
    const float32x4_t vf4 = vld1q_f32(input); input += 4;
    const float32x4_t vf5 = vld1q_f32(input); input += 4;

    uint32x4_t vi0 = vreinterpretq_u32_f32(vf0);
    uint32x4_t vi1 = vreinterpretq_u32_f32(vf1);
    uint32x4_t vi2 = vreinterpretq_u32_f32(vf2);
    uint32x4_t vi3 = vreinterpretq_u32_f32(vf3);
    uint32x4_t vi4 = vreinterpretq_u32_f32(vf4);
    uint32x4_t vi5 = vreinterpretq_u32_f32(vf5);

    const uint32x4_t vlsb0 = vandq_u32(vshrq_n_u32(vi0, 16), vone);
    const uint32x4_t vlsb1 = vandq_u32(vshrq_n_u32(vi1, 16), vone);
    const uint32x4_t vlsb2 = vandq_u32(vshrq_n_u32(vi2, 16), vone);
    const uint32x4_t vlsb3 = vandq_u32(vshrq_n_u32(vi3, 16), vone);
    const uint32x4_t vlsb4 = vandq_u32(vshrq_n_u32(vi4, 16), vone);
    const uint32x4_t vlsb5 = vandq_u32(vshrq_n_u32(vi5, 16), vone);

    const uint32x4_t vrounded0 = vaddq_u32(vaddq_u32(vi0, vbias), vlsb0);
    const uint32x4_t vrounded1 = vaddq_u32(vaddq_u32(vi1, vbias), vlsb1);
    const uint32x4_t vrounded2 = vaddq_u32(vaddq_u32(vi2, vbias), vlsb2);
    const uint32x4_t vrounded3 = vaddq_u32(vaddq_u32(vi3, vbias), vlsb3);
    const uint32x4_t vrounded4 = vaddq_u32(vaddq_u32(vi4, vbias), vlsb4);
    const uint32x4_t vrounded5 = vaddq_u32(vaddq_u32(vi5, vbias), vlsb5);

    const uint32x4_t vabsi0 = vandq_u32(vi0, vabs_mask);
    const uint32x4_t vabsi1 = vandq_u32(vi1, vabs_mask);
    const uint32x4_t vabsi2 = vandq_u32(vi2, vabs_mask);
    const uint32x4_t vabsi3 = vandq_u32(vi3, vabs_mask);
    const uint32x4_t vabsi4 = vandq_u32(vi4, vabs_mask);
    const uint32x4_t vabsi5 = vandq_u32(vi5, vabs_mask);

    const uint32x4_t vnanmask0 = vcgtq_u32(vabsi0, vexp_mask);
    const uint32x4_t vnanmask1 = vcgtq_u32(vabsi1, vexp_mask);
    const uint32x4_t vnanmask2 = vcgtq_u32(vabsi2, vexp_mask);
    const uint32x4_t vnanmask3 = vcgtq_u32(vabsi3, vexp_mask);
    const uint32x4_t vnanmask4 = vcgtq_u32(vabsi4, vexp_mask);
    const uint32x4_t vnanmask5 = vcgtq_u32(vabsi5, vexp_mask);

    vi0 = vbslq_u32(vnanmask0, vorrq_u32(vi0, vquiet), vrounded0);
    vi1 = vbslq_u32(vnanmask1, vorrq_u32(vi1, vquiet), vrounded1);
    vi2 = vbslq_u32(vnanmask2, vorrq_u32(vi2, vquiet), vrounded2);
    vi3 = vbslq_u32(vnanmask3, vorrq_u32(vi3, vquiet), vrounded3);
    vi4 = vbslq_u32(vnanmask4, vorrq_u32(vi4, vquiet), vrounded4);
    vi5 = vbslq_u32(vnanmask5, vorrq_u32(vi5, vquiet), vrounded5);

    const uint16x8_t vbf0 = vcombine_u16(vshrn_n_u32(vi0, 16), vshrn_n_u32(vi1, 16));
    const uint16x8_t vbf1 = vcombine_u16(vshrn_n_u32(vi2, 16), vshrn_n_u32(vi3, 16));
    const uint16x8_t vbf2 = vcombine_u16(vshrn_n_u32(vi4, 16), vshrn_n_u32(vi5, 16));

    vst1q_u16(o, vbf0); o += 8;
    vst1q_u16(o, vbf1); o += 8;
    vst1q_u16(o, vbf2); o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vf = vld1q_f32(input); input += 4;

    uint32x4_t vi = vreinterpretq_u32_f32(vf);
    const uint32x4_t vlsb = vandq_u32(vshrq_n_u32(vi, 16), vone);
    const uint32x4_t vrounded = vaddq_u32(vaddq_u32(vi, vbias), vlsb);
    const uint32x4_t vabsi = vandq_u32(vi, vabs_mask);
    const uint32x4_t vnanmask = vcgtq_u32(vabsi, vexp_mask);
    vi = vbslq_u32(vnanmask, vorrq_u32(vi, vquiet), vrounded);
    const uint16x4_t vbf = vshrn_n_u32(vi, 16);

    vst1_u16(o, vbf); o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch % sizeof(float) == 0);
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vf = vld1q_f32(input);

    uint32x4_t vi = vreinterpretq_u32_f32(vf);
    const uint32x4_t vlsb = vandq_u32(vshrq_n_u32(vi, 16), vone);
    const uint32x4_t vrounded = vaddq_u32(vaddq_u32(vi, vbias), vlsb);
    const uint32x4_t vabsi = vandq_u32(vi, vabs_mask);
    const uint32x4_t vnanmask = vcgtq_u32(vabsi, vexp_mask);
    vi = vbslq_u32(vnanmask, vorrq_u32(vi, vquiet), vrounded);
    uint16x4_t vbf = vshrn_n_u32(vi, 16);

    if (batch & (2 * sizeof(float))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vbf), 0); o += 2;
      vbf = vext_u16(vbf, vbf, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u16(o, vbf, 0);
    }
  }
}
