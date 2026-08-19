// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vunary/neon-f32acc.c.in
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
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


static inline float32x4_t load_bf16_as_f32(const uint16_t* input) {
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(input)), 16));
}

static inline uint16x4_t convert_f32_to_bf16(float32x4_t value) {
  const uint32x4_t vexp_mask = vdupq_n_u32(UINT32_C(0x7F800000));
  const uint32x4_t vbias = vdupq_n_u32(UINT32_C(0x00007FFF));
  const uint32x4_t vone = vdupq_n_u32(UINT32_C(1));
  const uint32x4_t vabs_mask = vdupq_n_u32(UINT32_C(0x7FFFFFFF));
  const uint32x4_t vquiet = vdupq_n_u32(UINT32_C(0x00400000));

  uint32x4_t vi = vreinterpretq_u32_f32(value);
  const uint32x4_t vlsb = vandq_u32(vshrq_n_u32(vi, 16), vone);
  const uint32x4_t vrounded = vaddq_u32(vaddq_u32(vi, vbias), vlsb);
  const uint32x4_t vnanmask =
      vcgtq_u32(vandq_u32(vi, vabs_mask), vexp_mask);
  vi = vbslq_u32(vnanmask, vorrq_u32(vi, vquiet), vrounded);
  return vshrn_n_u32(vi, 16);
}

static inline void store_f32_as_bf16(uint16_t* output, float32x4_t value) {
  vst1_u16(output, convert_f32_to_bf16(value));
}

static inline void store_tail_f32_as_bf16(
    uint16_t* output, float32x4_t value, size_t elements) {
  uint16x4_t vbf16 = convert_f32_to_bf16(value);
  if (elements & 2) {
    vst1_lane_u32((void*) output, vreinterpret_u32_u16(vbf16), 0);
    output += 2;
    vbf16 = vext_u16(vbf16, vbf16, 2);
  }
  if (elements & 1) {
    vst1_lane_u16(output, vbf16, 0);
  }
}

static inline float32x4_t sigmoid_f32(float32x4_t vx) {
  const float32x4_t vmagic_bias = vdupq_n_f32(0x1.8000FEp23f);
  const float32x4_t vminus_log2e = vdupq_n_f32(-0x1.715476p0f);
  const float32x4_t vln2_hi = vdupq_n_f32(0x1.62E400p-1f);
  const float32x4_t vln2_lo = vdupq_n_f32(0x1.7F7D1Cp-20f);
  const float32x4_t vc5 = vdupq_n_f32(-0x1.0F9F9Cp-7f);
  const float32x4_t vc4 = vdupq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc3 = vdupq_n_f32(-0x1.555A80p-3f);
  const float32x4_t vc2 = vdupq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc1 = vdupq_n_f32(-0x1.FFFFF6p-1f);
  const float32x4_t vone = vdupq_n_f32(1.0f);
  const float32x4_t vdenorm_cutoff = vdupq_n_f32(0x1.5D589Ep+6f);

  const float32x4_t vz = vabsq_f32(vx);
  float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);
  const float32x4_t vs =
      vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
  vn = vsubq_f32(vn, vmagic_bias);
  float32x4_t vt = vmlaq_f32(vz, vn, vln2_hi);
  vt = vmlaq_f32(vt, vn, vln2_lo);
  float32x4_t vp = vmlaq_f32(vc4, vc5, vt);
  vp = vmlaq_f32(vc3, vp, vt);
  vp = vmlaq_f32(vc2, vp, vt);
  vp = vmlaq_f32(vc1, vp, vt);
  vt = vmulq_f32(vt, vs);
  const float32x4_t ve = vmlaq_f32(vs, vp, vt);
  const float32x4_t vd = vaddq_f32(ve, vone);
  float32x4_t vr = vrecpeq_f32(vd);
  vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
  vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
  float32x4_t vf = vmulq_f32(ve, vr);
  vf = vreinterpretq_f32_u32(vbicq_u32(
      vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
  return vbslq_f32(vcltq_f32(vx, vdupq_n_f32(0.0f)), vf,
                    vsubq_f32(vone, vf));
}

static inline float32x4_t rsqrt_f32(float32x4_t vx) {
  const uint32x4_t vzero_mask = vceqq_f32(vx, vdupq_n_f32(0.0f));
  const uint32x4_t vnegative_mask = vcltq_f32(vx, vdupq_n_f32(0.0f));
  const uint32x4_t vpos_inf_mask = vceqq_u32(
      vreinterpretq_u32_f32(vx), vdupq_n_u32(UINT32_C(0x7F800000)));
  const uint32x4_t vspecial_mask =
      vorrq_u32(vorrq_u32(vzero_mask, vnegative_mask), vpos_inf_mask);
  const float32x4_t vsafe_x =
      vbslq_f32(vspecial_mask, vdupq_n_f32(1.0f), vx);

  float32x4_t vy = vrsqrteq_f32(vsafe_x);
  vy = vmulq_f32(vy, vrsqrtsq_f32(vmulq_f32(vsafe_x, vy), vy));
  vy = vmulq_f32(vy, vrsqrtsq_f32(vmulq_f32(vsafe_x, vy), vy));

  const uint32x4_t vsigned_inf = vorrq_u32(
      vandq_u32(vreinterpretq_u32_f32(vx),
                vdupq_n_u32(UINT32_C(0x80000000))),
      vdupq_n_u32(UINT32_C(0x7F800000)));
  vy = vbslq_f32(vzero_mask, vreinterpretq_f32_u32(vsigned_inf), vy);
  vy = vbslq_f32(vpos_inf_mask, vdupq_n_f32(0.0f), vy);
  // Match 1.0f / sqrtf(vx) without relying on estimate behavior for negatives.
  return vbslq_f32(vnegative_mask,
                   vreinterpretq_f32_u32(
                       vdupq_n_u32(UINT32_C(0x7FC00000))),
                   vy);
}

void xnn_bf16_vsqr_ukernel__neon_u8(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(xnn_bfloat16);
       batch -= 8 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx0 = load_bf16_as_f32(i + 0);
    const float32x4_t vx1 = load_bf16_as_f32(i + 4);
    i += 8;

    const float32x4_t vy0 = vmulq_f32(vx0, vx0);
    const float32x4_t vy1 = vmulq_f32(vx1, vx1);

    store_f32_as_bf16(o + 0, vy0);
    store_f32_as_bf16(o + 4, vy1);
    o += 8;
  }

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx = load_bf16_as_f32(i);
    i += 4;
    const float32x4_t vy = vmulq_f32(vx, vx);
    store_f32_as_bf16(o, vy);
    o += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = load_bf16_as_f32(i);
    const float32x4_t vy = vmulq_f32(vx, vx);
    store_tail_f32_as_bf16(o, vy, batch / sizeof(xnn_bfloat16));
  }
}
void xnn_bf16_vrsqrt_ukernel__neon_u8(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(xnn_bfloat16);
       batch -= 8 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx0 = load_bf16_as_f32(i + 0);
    const float32x4_t vx1 = load_bf16_as_f32(i + 4);
    i += 8;

    const float32x4_t vy0 = rsqrt_f32(vx0);
    const float32x4_t vy1 = rsqrt_f32(vx1);

    store_f32_as_bf16(o + 0, vy0);
    store_f32_as_bf16(o + 4, vy1);
    o += 8;
  }

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx = load_bf16_as_f32(i);
    i += 4;
    const float32x4_t vy = rsqrt_f32(vx);
    store_f32_as_bf16(o, vy);
    o += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = load_bf16_as_f32(i);
    const float32x4_t vy = rsqrt_f32(vx);
    store_tail_f32_as_bf16(o, vy, batch / sizeof(xnn_bfloat16));
  }
}
void xnn_bf16_vsigmoid_ukernel__neon_u8(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(xnn_bfloat16);
       batch -= 8 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx0 = load_bf16_as_f32(i + 0);
    const float32x4_t vx1 = load_bf16_as_f32(i + 4);
    i += 8;

    const float32x4_t vy0 = sigmoid_f32(vx0);
    const float32x4_t vy1 = sigmoid_f32(vx1);

    store_f32_as_bf16(o + 0, vy0);
    store_f32_as_bf16(o + 4, vy1);
    o += 8;
  }

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float32x4_t vx = load_bf16_as_f32(i);
    i += 4;
    const float32x4_t vy = sigmoid_f32(vx);
    store_f32_as_bf16(o, vy);
    o += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = load_bf16_as_f32(i);
    const float32x4_t vy = sigmoid_f32(vx);
    store_tail_f32_as_bf16(o, vy, batch / sizeof(xnn_bfloat16));
  }
}
