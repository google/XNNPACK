// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rsum2/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"


void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u8_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u8(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16_acc4(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc2);
  vacc1 = vaddq_f32(vacc1, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc0 = vaddq_f32(vacc0, vt2);
    vacc1 = vaddq_f32(vacc1, vt3);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u16(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
    vacc0 = vaddq_f32(vacc0, vt2);
    vacc0 = vaddq_f32(vacc0, vt3);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24_acc6(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  float32x4_t vacc4 = vmovq_n_f32(0.0f);
  float32x4_t vacc5 = vmovq_n_f32(0.0f);
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
    vacc4 = vaddq_f32(vacc4, vt4);
    vacc5 = vaddq_f32(vacc5, vt5);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc2 = vaddq_f32(vacc2, vt0);
    vacc3 = vaddq_f32(vacc3, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc3);
  vacc1 = vaddq_f32(vacc1, vacc4);
  vacc2 = vaddq_f32(vacc2, vacc5);
  vacc1 = vaddq_f32(vacc1, vacc2);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24_acc3(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc0 = vaddq_f32(vacc0, vt3);
    vacc1 = vaddq_f32(vacc1, vt4);
    vacc2 = vaddq_f32(vacc2, vt5);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc2 = vaddq_f32(vacc2, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  vacc1 = vaddq_f32(vacc1, vacc2);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u24(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
    vacc0 = vaddq_f32(vacc0, vt2);
    vacc0 = vaddq_f32(vacc0, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc0 = vaddq_f32(vacc0, vt5);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc8(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  float32x4_t vacc4 = vmovq_n_f32(0.0f);
  float32x4_t vacc5 = vmovq_n_f32(0.0f);
  float32x4_t vacc6 = vmovq_n_f32(0.0f);
  float32x4_t vacc7 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);
    vt6 = vmulq_f32(vt6, vt6);
    vt7 = vmulq_f32(vt7, vt7);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
    vacc4 = vaddq_f32(vacc4, vt4);
    vacc5 = vaddq_f32(vacc5, vt5);
    vacc6 = vaddq_f32(vacc6, vt6);
    vacc7 = vaddq_f32(vacc7, vt7);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc2 = vaddq_f32(vacc2, vt0);
    vacc3 = vaddq_f32(vacc3, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc4 = vaddq_f32(vacc4, vt0);
    vacc5 = vaddq_f32(vacc5, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc4);
  vacc1 = vaddq_f32(vacc1, vacc5);
  vacc2 = vaddq_f32(vacc2, vacc6);
  vacc3 = vaddq_f32(vacc3, vacc7);
  vacc0 = vaddq_f32(vacc0, vacc2);
  vacc1 = vaddq_f32(vacc1, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc4(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);
    vt6 = vmulq_f32(vt6, vt6);
    vt7 = vmulq_f32(vt7, vt7);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc1 = vaddq_f32(vacc1, vt5);
    vacc2 = vaddq_f32(vacc2, vt6);
    vacc3 = vaddq_f32(vacc3, vt7);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc2 = vaddq_f32(vacc2, vt0);
    vacc3 = vaddq_f32(vacc3, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc2);
  vacc1 = vaddq_f32(vacc1, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32_acc2(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);
    vt6 = vmulq_f32(vt6, vt6);
    vt7 = vmulq_f32(vt7, vt7);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc0 = vaddq_f32(vacc0, vt2);
    vacc1 = vaddq_f32(vacc1, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc1 = vaddq_f32(vacc1, vt5);
    vacc0 = vaddq_f32(vacc0, vt6);
    vacc1 = vaddq_f32(vacc1, vt7);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}

void xnn_f16_f32acc_rsum2_ukernel__neonfp16arith_u32(
    size_t batch, const xnn_float16* input, float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params) {
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);
    vt2 = vmulq_f32(vt2, vt2);
    vt3 = vmulq_f32(vt3, vt3);
    vt4 = vmulq_f32(vt4, vt4);
    vt5 = vmulq_f32(vt5, vt5);
    vt6 = vmulq_f32(vt6, vt6);
    vt7 = vmulq_f32(vt7, vt7);

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
    vacc0 = vaddq_f32(vacc0, vt2);
    vacc0 = vaddq_f32(vacc0, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc0 = vaddq_f32(vacc0, vt5);
    vacc0 = vaddq_f32(vacc0, vt6);
    vacc0 = vaddq_f32(vacc0, vt7);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  if (batch >= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i));
    i += 8;

    float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh));
    float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh));

    vt0 = vmulq_f32(vt0, vt0);
    vt1 = vmulq_f32(vt1, vt1);

    batch -= 8 * sizeof(uint16_t);
    vacc0 = vaddq_f32(vacc0, vt0);
    vacc0 = vaddq_f32(vacc0, vt1);
  }
  const float32x2_t vscale = vdup_n_f32(params->scalar.scale);
  if XNN_UNLIKELY(batch & (4 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16((const void*) i));
    i += 4;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i));
    i += 2;
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    float32x4_t vt = vcvt_f32_f16(vh);
    vt = vmulq_f32(vt, vt);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);

  float vout = vget_lane_f32(vacc, 0);
  *output += vout;
}
