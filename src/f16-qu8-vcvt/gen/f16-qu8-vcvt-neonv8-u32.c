// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/neonv8.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_f16_qu8_vcvt_ukernel__neonv8_u32(
    size_t batch,
    const xnn_float16* input,
    uint8_t* output,
    const struct xnn_f16_qu8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_float16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vdupq_n_f32(xnn_float16_to_float(params->scalar.scale));
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 32 * sizeof(xnn_float16); batch -= 32 * sizeof(xnn_float16)) {
    const float16x8_t vfp01234567 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input)); input += 8;
    float32x4_t vx0123 = vcvt_f32_f16(vget_low_f16(vfp01234567));
    float32x4_t vx4567 = vcvt_f32_f16(vget_high_f16(vfp01234567));
    const float16x8_t vfp89ABCDEF = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input)); input += 8;
    float32x4_t vx89AB = vcvt_f32_f16(vget_low_f16(vfp89ABCDEF));
    float32x4_t vxCDEF = vcvt_f32_f16(vget_high_f16(vfp89ABCDEF));
    const float16x8_t vfpGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input)); input += 8;
    float32x4_t vxGHIJ = vcvt_f32_f16(vget_low_f16(vfpGHIJKLMN));
    float32x4_t vxKLMN = vcvt_f32_f16(vget_high_f16(vfpGHIJKLMN));
    const float16x8_t vfpOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input)); input += 8;
    float32x4_t vxOPQR = vcvt_f32_f16(vget_low_f16(vfpOPQRSTUV));
    float32x4_t vxSTUV = vcvt_f32_f16(vget_high_f16(vfpOPQRSTUV));

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

    vst1q_u8(output, vy0123456789ABCDEF); output += 16;
    vst1q_u8(output, vyGHIJKLMNOPQRSTUV); output += 16;
  }
  for (; batch >= 8 * sizeof(xnn_float16); batch -= 8 * sizeof(xnn_float16)) {
    const float16x8_t vfp = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input)); input += 8;
    float32x4_t vx_lo = vcvt_f32_f16(vget_low_f16(vfp));
    float32x4_t vx_hi = vcvt_f32_f16(vget_high_f16(vfp));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    uint8x8_t vy = vqmovun_s16(vacc);
    vst1_u8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_float16));
    assert(batch <= 7 * sizeof(xnn_float16));
    const float16x8_t vfp = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) input));
    float32x4_t vx_lo = vcvt_f32_f16(vget_low_f16(vfp));
    float32x4_t vx_hi = vcvt_f32_f16(vget_high_f16(vfp));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    uint8x8_t vy = vqmovun_s16(vacc);

    if (batch & (4 * sizeof(xnn_float16))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_u8(vy), 0); output += 4;
      vy = vext_u8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(xnn_float16))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_u8(vy), 0); output += 2;
      vy = vext_u8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(xnn_float16))) {
      vst1_lane_u8(output, vy, 0);
    }
  }
}
