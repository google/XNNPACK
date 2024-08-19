// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vunary.h"


void xnn_f32_qs8_vcvt_ukernel__neonv8_u32(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->scalar.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->scalar.output_max);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;
    float32x4_t vx89AB = vld1q_f32(input); input += 4;
    float32x4_t vxCDEF = vld1q_f32(input); input += 4;
    float32x4_t vxGHIJ = vld1q_f32(input); input += 4;
    float32x4_t vxKLMN = vld1q_f32(input); input += 4;
    float32x4_t vxOPQR = vld1q_f32(input); input += 4;
    float32x4_t vxSTUV = vld1q_f32(input); input += 4;

    vx0123 = vmulq_f32(vx0123, vscale);
    vx4567 = vmulq_f32(vx4567, vscale);
    vx89AB = vmulq_f32(vx89AB, vscale);
    vxCDEF = vmulq_f32(vxCDEF, vscale);
    vxGHIJ = vmulq_f32(vxGHIJ, vscale);
    vxKLMN = vmulq_f32(vxKLMN, vscale);
    vxOPQR = vmulq_f32(vxOPQR, vscale);
    vxSTUV = vmulq_f32(vxSTUV, vscale);

    const int32x4_t vacc0123 = vcvtnq_s32_f32(vx0123);
    const int32x4_t vacc4567 = vcvtnq_s32_f32(vx4567);
    const int32x4_t vacc89AB = vcvtnq_s32_f32(vx89AB);
    const int32x4_t vaccCDEF = vcvtnq_s32_f32(vxCDEF);
    const int32x4_t vaccGHIJ = vcvtnq_s32_f32(vxGHIJ);
    const int32x4_t vaccKLMN = vcvtnq_s32_f32(vxKLMN);
    const int32x4_t vaccOPQR = vcvtnq_s32_f32(vxOPQR);
    const int32x4_t vaccSTUV = vcvtnq_s32_f32(vxSTUV);

    int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
    int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
    int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));
    int16x8_t vaccOPQRSTUV = vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV));

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);
    vaccGHIJKLMN = vqaddq_s16(vaccGHIJKLMN, voutput_zero_point);
    vaccOPQRSTUV = vqaddq_s16(vaccOPQRSTUV, voutput_zero_point);

    int8x16_t vy0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
    int8x16_t vyGHIJKLMNOPQRSTUV = vcombine_s8(vqmovn_s16(vaccGHIJKLMN), vqmovn_s16(vaccOPQRSTUV));

    vy0123456789ABCDEF = vmaxq_s8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = vmaxq_s8(vyGHIJKLMNOPQRSTUV, voutput_min);

    vy0123456789ABCDEF = vminq_s8(vy0123456789ABCDEF, voutput_max);
    vyGHIJKLMNOPQRSTUV = vminq_s8(vyGHIJKLMNOPQRSTUV, voutput_max);

    vst1q_s8(output, vy0123456789ABCDEF); output += 16;
    vst1q_s8(output, vyGHIJKLMNOPQRSTUV); output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx_lo = vld1q_f32(input); input += 4;
    float32x4_t vx_hi = vld1q_f32(input); input += 4;

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    float32x4_t vx_lo = vld1q_f32(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    float32x4_t vx_hi = vld1q_f32(x_hi);

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));

    if (batch & (4 * sizeof(float))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(float))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}

void xnn_f32_qu8_vcvt_ukernel__neonv8_u32(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->scalar.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->scalar.output_max);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;
    float32x4_t vx89AB = vld1q_f32(input); input += 4;
    float32x4_t vxCDEF = vld1q_f32(input); input += 4;
    float32x4_t vxGHIJ = vld1q_f32(input); input += 4;
    float32x4_t vxKLMN = vld1q_f32(input); input += 4;
    float32x4_t vxOPQR = vld1q_f32(input); input += 4;
    float32x4_t vxSTUV = vld1q_f32(input); input += 4;

    vx0123 = vmulq_f32(vx0123, vscale);
    vx4567 = vmulq_f32(vx4567, vscale);
    vx89AB = vmulq_f32(vx89AB, vscale);
    vxCDEF = vmulq_f32(vxCDEF, vscale);
    vxGHIJ = vmulq_f32(vxGHIJ, vscale);
    vxKLMN = vmulq_f32(vxKLMN, vscale);
    vxOPQR = vmulq_f32(vxOPQR, vscale);
    vxSTUV = vmulq_f32(vxSTUV, vscale);

    const int32x4_t vacc0123 = vcvtnq_s32_f32(vx0123);
    const int32x4_t vacc4567 = vcvtnq_s32_f32(vx4567);
    const int32x4_t vacc89AB = vcvtnq_s32_f32(vx89AB);
    const int32x4_t vaccCDEF = vcvtnq_s32_f32(vxCDEF);
    const int32x4_t vaccGHIJ = vcvtnq_s32_f32(vxGHIJ);
    const int32x4_t vaccKLMN = vcvtnq_s32_f32(vxKLMN);
    const int32x4_t vaccOPQR = vcvtnq_s32_f32(vxOPQR);
    const int32x4_t vaccSTUV = vcvtnq_s32_f32(vxSTUV);

    int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
    int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
    int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));
    int16x8_t vaccOPQRSTUV = vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV));

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);
    vaccGHIJKLMN = vqaddq_s16(vaccGHIJKLMN, voutput_zero_point);
    vaccOPQRSTUV = vqaddq_s16(vaccOPQRSTUV, voutput_zero_point);

    uint8x16_t vy0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
    uint8x16_t vyGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vaccGHIJKLMN), vqmovun_s16(vaccOPQRSTUV));

    vy0123456789ABCDEF = vmaxq_u8(vy0123456789ABCDEF, voutput_min);
    vyGHIJKLMNOPQRSTUV = vmaxq_u8(vyGHIJKLMNOPQRSTUV, voutput_min);

    vy0123456789ABCDEF = vminq_u8(vy0123456789ABCDEF, voutput_max);
    vyGHIJKLMNOPQRSTUV = vminq_u8(vyGHIJKLMNOPQRSTUV, voutput_max);

    vst1q_u8(output, vy0123456789ABCDEF); output += 16;
    vst1q_u8(output, vyGHIJKLMNOPQRSTUV); output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx_lo = vld1q_f32(input); input += 4;
    float32x4_t vx_hi = vld1q_f32(input); input += 4;

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    uint8x8_t vy = vqmovun_s16(vacc);
    vy = vmax_u8(vy, vget_low_u8(voutput_min));
    vy = vmin_u8(vy, vget_low_u8(voutput_max));
    vst1_u8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    float32x4_t vx_lo = vld1q_f32(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    float32x4_t vx_hi = vld1q_f32(x_hi);

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    uint8x8_t vy = vqmovun_s16(vacc);
    vy = vmax_u8(vy, vget_low_u8(voutput_min));
    vy = vmin_u8(vy, vget_low_u8(voutput_max));

    if (batch & (4 * sizeof(float))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_u8(vy), 0); output += 4;
      vy = vext_u8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(float))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_u8(vy), 0); output += 2;
      vy = vext_u8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u8(output, vy, 0);
    }
  }
}

void xnn_f32_vrndd_ukernel__neonv8_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vy0123 = vrndmq_f32(vx0123);
    const float32x4_t vy4567 = vrndmq_f32(vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vrndmq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x4_t vy = vrndmq_f32(vx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_f32_vrndne_ukernel__neonv8_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vy0123 = vrndnq_f32(vx0123);
    const float32x4_t vy4567 = vrndnq_f32(vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vrndnq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x4_t vy = vrndnq_f32(vx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_f32_vrndu_ukernel__neonv8_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vy0123 = vrndpq_f32(vx0123);
    const float32x4_t vy4567 = vrndpq_f32(vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vrndpq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x4_t vy = vrndpq_f32(vx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_f32_vrndz_ukernel__neonv8_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vy0123 = vrndq_f32(vx0123);
    const float32x4_t vy4567 = vrndq_f32(vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const float32x4_t vy = vrndq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x4_t vy = vrndq_f32(vx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p16c__neonv8_mla8_ld64(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi0x89ABCDEF = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);
      int16x8_t vprod89ABCDEF = vmull_s8(vi0x89ABCDEF, vk0x89ABCDEF);

      const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi1x89ABCDEF = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi1x89ABCDEF, vk1x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi2x89ABCDEF = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);
      vprod89ABCDEF = vmull_s8(vi2x89ABCDEF, vk2x89ABCDEF);

      const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi3x89ABCDEF = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi3x89ABCDEF, vk3x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi4x89ABCDEF = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);
      vprod89ABCDEF = vmull_s8(vi4x89ABCDEF, vk4x89ABCDEF);

      const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi5x89ABCDEF = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi5x89ABCDEF, vk5x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi6x89ABCDEF = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);
      vprod89ABCDEF = vmull_s8(vi6x89ABCDEF, vk6x89ABCDEF);

      const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi7x89ABCDEF = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi7x89ABCDEF, vk7x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi8x89ABCDEF = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);
      vprod89ABCDEF = vmull_s8(vi8x89ABCDEF, vk8x89ABCDEF);

      const int8x8_t vi9x01234567 = vld1_s8(i9); i9 += 8;
      const int8x8_t vk9x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi9x89ABCDEF = vld1_s8(i9); i9 += 8;
      const int8x8_t vk9x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi9x01234567, vk9x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi9x89ABCDEF, vk9x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi10x01234567 = vld1_s8(i10); i10 += 8;
      const int8x8_t vk10x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi10x89ABCDEF = vld1_s8(i10); i10 += 8;
      const int8x8_t vk10x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi10x01234567, vk10x01234567);
      vprod89ABCDEF = vmull_s8(vi10x89ABCDEF, vk10x89ABCDEF);

      const int8x8_t vi11x01234567 = vld1_s8(i11); i11 += 8;
      const int8x8_t vk11x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi11x89ABCDEF = vld1_s8(i11); i11 += 8;
      const int8x8_t vk11x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi11x01234567, vk11x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi11x89ABCDEF, vk11x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi12x01234567 = vld1_s8(i12); i12 += 8;
      const int8x8_t vk12x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi12x89ABCDEF = vld1_s8(i12); i12 += 8;
      const int8x8_t vk12x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi12x01234567, vk12x01234567);
      vprod89ABCDEF = vmull_s8(vi12x89ABCDEF, vk12x89ABCDEF);

      const int8x8_t vi13x01234567 = vld1_s8(i13); i13 += 8;
      const int8x8_t vk13x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi13x89ABCDEF = vld1_s8(i13); i13 += 8;
      const int8x8_t vk13x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi13x01234567, vk13x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi13x89ABCDEF, vk13x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi14x01234567 = vld1_s8(i14); i14 += 8;
      const int8x8_t vk14x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi14x89ABCDEF = vld1_s8(i14); i14 += 8;
      const int8x8_t vk14x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi14x01234567, vk14x01234567);
      vprod89ABCDEF = vmull_s8(vi14x89ABCDEF, vk14x89ABCDEF);

      const int8x8_t vi15x01234567 = vld1_s8(i15); i15 += 8;
      const int8x8_t vk15x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi15x89ABCDEF = vld1_s8(i15); i15 += 8;
      const int8x8_t vk15x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi15x01234567, vk15x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi15x89ABCDEF, vk15x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi16x01234567 = vld1_s8(i16); i16 += 8;
      const int8x8_t vk16x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi16x89ABCDEF = vld1_s8(i16); i16 += 8;
      const int8x8_t vk16x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi16x01234567, vk16x01234567);
      vprod89ABCDEF = vmull_s8(vi16x89ABCDEF, vk16x89ABCDEF);

      const int8x8_t vi17x01234567 = vld1_s8(i17); i17 += 8;
      const int8x8_t vk17x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi17x89ABCDEF = vld1_s8(i17); i17 += 8;
      const int8x8_t vk17x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi17x01234567, vk17x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi17x89ABCDEF, vk17x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi18x01234567 = vld1_s8(i18); i18 += 8;
      const int8x8_t vk18x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi18x89ABCDEF = vld1_s8(i18); i18 += 8;
      const int8x8_t vk18x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi18x01234567, vk18x01234567);
      vprod89ABCDEF = vmull_s8(vi18x89ABCDEF, vk18x89ABCDEF);

      const int8x8_t vi19x01234567 = vld1_s8(i19); i19 += 8;
      const int8x8_t vk19x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi19x89ABCDEF = vld1_s8(i19); i19 += 8;
      const int8x8_t vk19x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi19x01234567, vk19x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi19x89ABCDEF, vk19x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi20x01234567 = vld1_s8(i20); i20 += 8;
      const int8x8_t vk20x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi20x89ABCDEF = vld1_s8(i20); i20 += 8;
      const int8x8_t vk20x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi20x01234567, vk20x01234567);
      vprod89ABCDEF = vmull_s8(vi20x89ABCDEF, vk20x89ABCDEF);

      const int8x8_t vi21x01234567 = vld1_s8(i21); i21 += 8;
      const int8x8_t vk21x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi21x89ABCDEF = vld1_s8(i21); i21 += 8;
      const int8x8_t vk21x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi21x01234567, vk21x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi21x89ABCDEF, vk21x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi22x01234567 = vld1_s8(i22); i22 += 8;
      const int8x8_t vk22x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi22x89ABCDEF = vld1_s8(i22); i22 += 8;
      const int8x8_t vk22x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi22x01234567, vk22x01234567);
      vprod89ABCDEF = vmull_s8(vi22x89ABCDEF, vk22x89ABCDEF);

      const int8x8_t vi23x01234567 = vld1_s8(i23); i23 += 8;
      const int8x8_t vk23x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi23x89ABCDEF = vld1_s8(i23); i23 += 8;
      const int8x8_t vk23x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi23x01234567, vk23x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi23x89ABCDEF, vk23x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi24x01234567 = vld1_s8(i24); i24 += 8;
      const int8x8_t vk24x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi24x89ABCDEF = vld1_s8(i24); i24 += 8;
      const int8x8_t vk24x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi24x01234567, vk24x01234567);
      vprod89ABCDEF = vmull_s8(vi24x89ABCDEF, vk24x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);
      vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
      vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

      vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);

      vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);

      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
        const int8x8_t vk0x01234567 = vld1_s8(k); k += 8;

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
        const int8x8_t vk1x01234567 = vld1_s8((const void*) (k + 8));

        vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
        const int8x8_t vk2x01234567 = vld1_s8((const void*) (k + 24));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
        const int8x8_t vk3x01234567 = vld1_s8((const void*) (k + 40));

        vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
        const int8x8_t vk4x01234567 = vld1_s8((const void*) (k + 56));

        vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

        const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
        const int8x8_t vk5x01234567 = vld1_s8((const void*) (k + 72));

        vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
        const int8x8_t vk6x01234567 = vld1_s8((const void*) (k + 88));

        vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

        const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
        const int8x8_t vk7x01234567 = vld1_s8((const void*) (k + 104));

        vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
        const int8x8_t vk8x01234567 = vld1_s8((const void*) (k + 120));

        vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

        const int8x8_t vi9x01234567 = vld1_s8(i9); i9 += 8;
        const int8x8_t vk9x01234567 = vld1_s8((const void*) (k + 136));

        vprod01234567 = vmlal_s8(vprod01234567, vi9x01234567, vk9x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi10x01234567 = vld1_s8(i10); i10 += 8;
        const int8x8_t vk10x01234567 = vld1_s8((const void*) (k + 152));

        vprod01234567 = vmull_s8(vi10x01234567, vk10x01234567);

        const int8x8_t vi11x01234567 = vld1_s8(i11); i11 += 8;
        const int8x8_t vk11x01234567 = vld1_s8((const void*) (k + 168));

        vprod01234567 = vmlal_s8(vprod01234567, vi11x01234567, vk11x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi12x01234567 = vld1_s8(i12); i12 += 8;
        const int8x8_t vk12x01234567 = vld1_s8((const void*) (k + 184));

        vprod01234567 = vmull_s8(vi12x01234567, vk12x01234567);

        const int8x8_t vi13x01234567 = vld1_s8(i13); i13 += 8;
        const int8x8_t vk13x01234567 = vld1_s8((const void*) (k + 200));

        vprod01234567 = vmlal_s8(vprod01234567, vi13x01234567, vk13x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi14x01234567 = vld1_s8(i14); i14 += 8;
        const int8x8_t vk14x01234567 = vld1_s8((const void*) (k + 216));

        vprod01234567 = vmull_s8(vi14x01234567, vk14x01234567);

        const int8x8_t vi15x01234567 = vld1_s8(i15); i15 += 8;
        const int8x8_t vk15x01234567 = vld1_s8((const void*) (k + 232));

        vprod01234567 = vmlal_s8(vprod01234567, vi15x01234567, vk15x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi16x01234567 = vld1_s8(i16); i16 += 8;
        const int8x8_t vk16x01234567 = vld1_s8((const void*) (k + 248));

        vprod01234567 = vmull_s8(vi16x01234567, vk16x01234567);

        const int8x8_t vi17x01234567 = vld1_s8(i17); i17 += 8;
        const int8x8_t vk17x01234567 = vld1_s8((const void*) (k + 264));

        vprod01234567 = vmlal_s8(vprod01234567, vi17x01234567, vk17x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi18x01234567 = vld1_s8(i18); i18 += 8;
        const int8x8_t vk18x01234567 = vld1_s8((const void*) (k + 280));

        vprod01234567 = vmull_s8(vi18x01234567, vk18x01234567);

        const int8x8_t vi19x01234567 = vld1_s8(i19); i19 += 8;
        const int8x8_t vk19x01234567 = vld1_s8((const void*) (k + 296));

        vprod01234567 = vmlal_s8(vprod01234567, vi19x01234567, vk19x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi20x01234567 = vld1_s8(i20); i20 += 8;
        const int8x8_t vk20x01234567 = vld1_s8((const void*) (k + 312));

        vprod01234567 = vmull_s8(vi20x01234567, vk20x01234567);

        const int8x8_t vi21x01234567 = vld1_s8(i21); i21 += 8;
        const int8x8_t vk21x01234567 = vld1_s8((const void*) (k + 328));

        vprod01234567 = vmlal_s8(vprod01234567, vi21x01234567, vk21x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi22x01234567 = vld1_s8(i22); i22 += 8;
        const int8x8_t vk22x01234567 = vld1_s8((const void*) (k + 344));

        vprod01234567 = vmull_s8(vi22x01234567, vk22x01234567);

        const int8x8_t vi23x01234567 = vld1_s8(i23); i23 += 8;
        const int8x8_t vk23x01234567 = vld1_s8((const void*) (k + 360));

        vprod01234567 = vmlal_s8(vprod01234567, vi23x01234567, vk23x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi24x01234567 = vld1_s8(i24); i24 += 8;
        const int8x8_t vk24x01234567 = vld1_s8((const void*) (k + 376));

        vprod01234567 = vmull_s8(vi24x01234567, vk24x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 400 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 400 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
        vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mla8_ld64(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
  const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

      const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

      const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

      const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

      const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

      const int8x8_t vi9x01234567 = vld1_s8(i9); i9 += 8;
      const int8x8_t vk9x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi9x01234567, vk9x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi10x01234567 = vld1_s8(i10); i10 += 8;
      const int8x8_t vk10x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi10x01234567, vk10x01234567);

      const int8x8_t vi11x01234567 = vld1_s8(i11); i11 += 8;
      const int8x8_t vk11x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi11x01234567, vk11x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi12x01234567 = vld1_s8(i12); i12 += 8;
      const int8x8_t vk12x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi12x01234567, vk12x01234567);

      const int8x8_t vi13x01234567 = vld1_s8(i13); i13 += 8;
      const int8x8_t vk13x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi13x01234567, vk13x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi14x01234567 = vld1_s8(i14); i14 += 8;
      const int8x8_t vk14x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi14x01234567, vk14x01234567);

      const int8x8_t vi15x01234567 = vld1_s8(i15); i15 += 8;
      const int8x8_t vk15x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi15x01234567, vk15x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi16x01234567 = vld1_s8(i16); i16 += 8;
      const int8x8_t vk16x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi16x01234567, vk16x01234567);

      const int8x8_t vi17x01234567 = vld1_s8(i17); i17 += 8;
      const int8x8_t vk17x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi17x01234567, vk17x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi18x01234567 = vld1_s8(i18); i18 += 8;
      const int8x8_t vk18x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi18x01234567, vk18x01234567);

      const int8x8_t vi19x01234567 = vld1_s8(i19); i19 += 8;
      const int8x8_t vk19x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi19x01234567, vk19x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi20x01234567 = vld1_s8(i20); i20 += 8;
      const int8x8_t vk20x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi20x01234567, vk20x01234567);

      const int8x8_t vi21x01234567 = vld1_s8(i21); i21 += 8;
      const int8x8_t vk21x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi21x01234567, vk21x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi22x01234567 = vld1_s8(i22); i22 += 8;
      const int8x8_t vk22x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi22x01234567, vk22x01234567);

      const int8x8_t vi23x01234567 = vld1_s8(i23); i23 += 8;
      const int8x8_t vk23x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi23x01234567, vk23x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      const int8x8_t vi24x01234567 = vld1_s8(i24); i24 += 8;
      const int8x8_t vk24x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi24x01234567, vk24x01234567);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#endif  // !XNN_ARCH_ARM64

      vout01234567 = vmax_s8(vout01234567, voutput_min);

      vout01234567 = vmin_s8(vout01234567, voutput_max);

      vst1_s8(output, vout01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0);
        const int8x8_t vk0x01234567 = vld1_s8(w);

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        const int8x8_t vi1x01234567 = vld1_s8(i1);
        const int8x8_t vk1x01234567 = vld1_s8((const void*) ((const int8_t*) w + 8));

        vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2);
        const int8x8_t vk2x01234567 = vld1_s8((const void*) ((const int8_t*) w + 16));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        const int8x8_t vi3x01234567 = vld1_s8(i3);
        const int8x8_t vk3x01234567 = vld1_s8((const void*) ((const int8_t*) w + 24));

        vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi4x01234567 = vld1_s8(i4);
        const int8x8_t vk4x01234567 = vld1_s8((const void*) ((const int8_t*) w + 32));

        vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

        const int8x8_t vi5x01234567 = vld1_s8(i5);
        const int8x8_t vk5x01234567 = vld1_s8((const void*) ((const int8_t*) w + 40));

        vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi6x01234567 = vld1_s8(i6);
        const int8x8_t vk6x01234567 = vld1_s8((const void*) ((const int8_t*) w + 48));

        vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

        const int8x8_t vi7x01234567 = vld1_s8(i7);
        const int8x8_t vk7x01234567 = vld1_s8((const void*) ((const int8_t*) w + 56));

        vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi8x01234567 = vld1_s8(i8);
        const int8x8_t vk8x01234567 = vld1_s8((const void*) ((const int8_t*) w + 64));

        vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

        const int8x8_t vi9x01234567 = vld1_s8(i9);
        const int8x8_t vk9x01234567 = vld1_s8((const void*) ((const int8_t*) w + 72));

        vprod01234567 = vmlal_s8(vprod01234567, vi9x01234567, vk9x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi10x01234567 = vld1_s8(i10);
        const int8x8_t vk10x01234567 = vld1_s8((const void*) ((const int8_t*) w + 80));

        vprod01234567 = vmull_s8(vi10x01234567, vk10x01234567);

        const int8x8_t vi11x01234567 = vld1_s8(i11);
        const int8x8_t vk11x01234567 = vld1_s8((const void*) ((const int8_t*) w + 88));

        vprod01234567 = vmlal_s8(vprod01234567, vi11x01234567, vk11x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi12x01234567 = vld1_s8(i12);
        const int8x8_t vk12x01234567 = vld1_s8((const void*) ((const int8_t*) w + 96));

        vprod01234567 = vmull_s8(vi12x01234567, vk12x01234567);

        const int8x8_t vi13x01234567 = vld1_s8(i13);
        const int8x8_t vk13x01234567 = vld1_s8((const void*) ((const int8_t*) w + 104));

        vprod01234567 = vmlal_s8(vprod01234567, vi13x01234567, vk13x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi14x01234567 = vld1_s8(i14);
        const int8x8_t vk14x01234567 = vld1_s8((const void*) ((const int8_t*) w + 112));

        vprod01234567 = vmull_s8(vi14x01234567, vk14x01234567);

        const int8x8_t vi15x01234567 = vld1_s8(i15);
        const int8x8_t vk15x01234567 = vld1_s8((const void*) ((const int8_t*) w + 120));

        vprod01234567 = vmlal_s8(vprod01234567, vi15x01234567, vk15x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi16x01234567 = vld1_s8(i16);
        const int8x8_t vk16x01234567 = vld1_s8((const void*) ((const int8_t*) w + 128));

        vprod01234567 = vmull_s8(vi16x01234567, vk16x01234567);

        const int8x8_t vi17x01234567 = vld1_s8(i17);
        const int8x8_t vk17x01234567 = vld1_s8((const void*) ((const int8_t*) w + 136));

        vprod01234567 = vmlal_s8(vprod01234567, vi17x01234567, vk17x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi18x01234567 = vld1_s8(i18);
        const int8x8_t vk18x01234567 = vld1_s8((const void*) ((const int8_t*) w + 144));

        vprod01234567 = vmull_s8(vi18x01234567, vk18x01234567);

        const int8x8_t vi19x01234567 = vld1_s8(i19);
        const int8x8_t vk19x01234567 = vld1_s8((const void*) ((const int8_t*) w + 152));

        vprod01234567 = vmlal_s8(vprod01234567, vi19x01234567, vk19x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi20x01234567 = vld1_s8(i20);
        const int8x8_t vk20x01234567 = vld1_s8((const void*) ((const int8_t*) w + 160));

        vprod01234567 = vmull_s8(vi20x01234567, vk20x01234567);

        const int8x8_t vi21x01234567 = vld1_s8(i21);
        const int8x8_t vk21x01234567 = vld1_s8((const void*) ((const int8_t*) w + 168));

        vprod01234567 = vmlal_s8(vprod01234567, vi21x01234567, vk21x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi22x01234567 = vld1_s8(i22);
        const int8x8_t vk22x01234567 = vld1_s8((const void*) ((const int8_t*) w + 176));

        vprod01234567 = vmull_s8(vi22x01234567, vk22x01234567);

        const int8x8_t vi23x01234567 = vld1_s8(i23);
        const int8x8_t vk23x01234567 = vld1_s8((const void*) ((const int8_t*) w + 184));

        vprod01234567 = vmlal_s8(vprod01234567, vi23x01234567, vk23x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi24x01234567 = vld1_s8(i24);
        const int8x8_t vk24x01234567 = vld1_s8((const void*) ((const int8_t*) w + 192));

        vprod01234567 = vmull_s8(vi24x01234567, vk24x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 0 * sizeof(int32_t) + 200 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 0 * sizeof(int32_t) + 200 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, voutput_min);
        vout01234567 = vmin_s8(vout01234567, voutput_max);

        if (c & 4) {
          vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
          vout01234567 = vext_s8(vout01234567, vout01234567, 4);
        }
        if (c & 2) {
          vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
          vout01234567 = vext_s8(vout01234567, vout01234567, 2);
        }
        if (c & 1) {
          vst1_lane_s8(output, vout01234567, 0); output += 1;
        }
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__neonv8_mla8_ld128(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x16_t vi0x0123456789ABCDEF = vld1q_s8(i0); i0 += 16;
      const int8x16_t vk0x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      int16x8_t vprod01234567 = vmull_s8(vget_low_s8(vi0x0123456789ABCDEF), vget_low_s8(vk0x0123456789ABCDEF));
      int16x8_t vprod89ABCDEF = vmull_s8(vget_high_s8(vi0x0123456789ABCDEF), vget_high_s8(vk0x0123456789ABCDEF));

      const int8x16_t vi1x0123456789ABCDEF = vld1q_s8(i1); i1 += 16;
      const int8x16_t vk1x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmlal_s8(vprod01234567, vget_low_s8(vi1x0123456789ABCDEF), vget_low_s8(vk1x0123456789ABCDEF));
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vget_high_s8(vi1x0123456789ABCDEF), vget_high_s8(vk1x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x16_t vi2x0123456789ABCDEF = vld1q_s8(i2); i2 += 16;
      const int8x16_t vk2x0123456789ABCDEF = vld1q_s8(w); w = (const int8_t*) w + 16;

      vprod01234567 = vmull_s8(vget_low_s8(vi2x0123456789ABCDEF), vget_low_s8(vk2x0123456789ABCDEF));
      vprod89ABCDEF = vmull_s8(vget_high_s8(vi2x0123456789ABCDEF), vget_high_s8(vk2x0123456789ABCDEF));

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);
      vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
      vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

      vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);

      vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);

      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
        const int8x8_t vk0x01234567 = vld1_s8(k); k += 8;

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
        const int8x8_t vk1x01234567 = vld1_s8((const void*) (k + 8));

        vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
        const int8x8_t vk2x01234567 = vld1_s8((const void*) (k + 24));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
        vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p16c__neonv8_mla8_ld64(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 16; c -= 16) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

      const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi0x89ABCDEF = vld1_s8(i0); i0 += 8;
      const int8x8_t vk0x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);
      int16x8_t vprod89ABCDEF = vmull_s8(vi0x89ABCDEF, vk0x89ABCDEF);

      const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi1x89ABCDEF = vld1_s8(i1); i1 += 8;
      const int8x8_t vk1x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi1x89ABCDEF, vk1x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi2x89ABCDEF = vld1_s8(i2); i2 += 8;
      const int8x8_t vk2x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);
      vprod89ABCDEF = vmull_s8(vi2x89ABCDEF, vk2x89ABCDEF);

      const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi3x89ABCDEF = vld1_s8(i3); i3 += 8;
      const int8x8_t vk3x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi3x89ABCDEF, vk3x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi4x89ABCDEF = vld1_s8(i4); i4 += 8;
      const int8x8_t vk4x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);
      vprod89ABCDEF = vmull_s8(vi4x89ABCDEF, vk4x89ABCDEF);

      const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi5x89ABCDEF = vld1_s8(i5); i5 += 8;
      const int8x8_t vk5x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi5x89ABCDEF, vk5x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi6x89ABCDEF = vld1_s8(i6); i6 += 8;
      const int8x8_t vk6x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);
      vprod89ABCDEF = vmull_s8(vi6x89ABCDEF, vk6x89ABCDEF);

      const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi7x89ABCDEF = vld1_s8(i7); i7 += 8;
      const int8x8_t vk7x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);
      vprod89ABCDEF = vmlal_s8(vprod89ABCDEF, vi7x89ABCDEF, vk7x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));
      const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x01234567 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vi8x89ABCDEF = vld1_s8(i8); i8 += 8;
      const int8x8_t vk8x89ABCDEF = vld1_s8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);
      vprod89ABCDEF = vmull_s8(vi8x89ABCDEF, vk8x89ABCDEF);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
      vacc89AB = vaddw_s16(vacc89AB, vget_low_s16(vprod89ABCDEF));
      vaccCDEF = vaddw_s16(vaccCDEF, vget_high_s16(vprod89ABCDEF));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);
      vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
      vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
      vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
#endif  // !XNN_ARCH_ARM64

      vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);

      vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);

      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 16);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
        const int8x8_t vk0x01234567 = vld1_s8(k); k += 8;

        int16x8_t vprod01234567 = vmull_s8(vi0x01234567, vk0x01234567);

        const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
        const int8x8_t vk1x01234567 = vld1_s8((const void*) (k + 8));

        vprod01234567 = vmlal_s8(vprod01234567, vi1x01234567, vk1x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
        const int8x8_t vk2x01234567 = vld1_s8((const void*) (k + 24));

        vprod01234567 = vmull_s8(vi2x01234567, vk2x01234567);

        const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
        const int8x8_t vk3x01234567 = vld1_s8((const void*) (k + 40));

        vprod01234567 = vmlal_s8(vprod01234567, vi3x01234567, vk3x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
        const int8x8_t vk4x01234567 = vld1_s8((const void*) (k + 56));

        vprod01234567 = vmull_s8(vi4x01234567, vk4x01234567);

        const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
        const int8x8_t vk5x01234567 = vld1_s8((const void*) (k + 72));

        vprod01234567 = vmlal_s8(vprod01234567, vi5x01234567, vk5x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
        const int8x8_t vk6x01234567 = vld1_s8((const void*) (k + 88));

        vprod01234567 = vmull_s8(vi6x01234567, vk6x01234567);

        const int8x8_t vi7x01234567 = vld1_s8(i7); i7 += 8;
        const int8x8_t vk7x01234567 = vld1_s8((const void*) (k + 104));

        vprod01234567 = vmlal_s8(vprod01234567, vi7x01234567, vk7x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));
        const int8x8_t vi8x01234567 = vld1_s8(i8); i8 += 8;
        const int8x8_t vk8x01234567 = vld1_s8((const void*) (k + 120));

        vprod01234567 = vmull_s8(vi8x01234567, vk8x01234567);

        vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vprod01234567));
        vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vprod01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 8 * sizeof(int32_t) + 144 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vacc0123 = vcvtnq_s32_f32(vfpacc0123);
        vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
#else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
#endif
        vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

        int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
        vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
        vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int16x8_t vxa0 = vmovl_s8(va0);

      const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c7 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc7 = vld1_s8(w); w = (const int8_t*) w + 8;

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
      const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
      const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
      const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
      const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
      const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
      const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
      const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
      const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
      const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
      const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
      const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
      const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
      const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
      const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);


      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vmovl_s8(va0);

      const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

      if (k >= 2 * sizeof(int8_t)) {
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
          const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

          if (k >= 4 * sizeof(int8_t)) {
            const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
            const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

            if (k > 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
              const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

              if (k >= 6 * sizeof(int8_t)) {
                const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                if (k > 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
                  const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                }
              }
            }
          }
        }
      }
    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    #endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #endif

    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
      vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
      vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
      vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
      vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
      vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
      vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va0x1 = vext_s8(va0x1, va0x1, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      va1x1 = vext_s8(va1x1, va1x1, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
      vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
      vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      k -= 16 * sizeof(int8_t);
    }
    if (k != 0) {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      va1x0 = vext_s8(va1x0, va1x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

    }

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);

    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    #endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);


        const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        const int8x8_t vb01234567c7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        const int8x8_t vb89ABCDEFc7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

        if (k >= 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                  if (k > 6 * sizeof(int8_t)) {
                    const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
                    const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
                    const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
                    const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

                    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                  }
                }
              }
            }
          }
        }
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    // Post-accumulation work
    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale89AB);
    const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscaleCDEF);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
    #endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

      int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
    #endif

    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
        int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
        vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
        vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
        vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
        vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
        vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
        vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
        vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
        vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      }

      p -= 2 * sizeof(void*);
    } while (p != 0);

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    #endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
