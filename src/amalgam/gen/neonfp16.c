// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/vcvt.h>


void xnn_f16_f32_vcvt_ukernel__neonfp16_u16(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vh0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vf0 = vcvt_f32_f16(vget_low_f16(vh0));
    const float32x4_t vf1 = vcvt_f32_f16(vget_high_f16(vh0));
    const float32x4_t vf2 = vcvt_f32_f16(vget_low_f16(vh1));
    const float32x4_t vf3 = vcvt_f32_f16(vget_high_f16(vh1));

    vst1q_f32(output, vf0); output += 4;
    vst1q_f32(output, vf1); output += 4;
    vst1q_f32(output, vf2); output += 4;
    vst1q_f32(output, vf3); output += 4;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vf_lo = vcvt_f32_f16(vget_low_f16(vh));
    const float32x4_t vf_hi = vcvt_f32_f16(vget_high_f16(vh));

    vst1q_f32(output, vf_lo); output += 4;
    vst1q_f32(output, vf_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float32x4_t vf = vcvt_f32_f16(vget_low_f16(vh));
    if (batch & (4 * sizeof(uint16_t))) {
      vst1q_f32(output, vf); output += 4;
      vf = vcvt_f32_f16(vget_high_f16(vh));
    }
    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_f32(output, vf_lo); output += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_f32(output, vf_lo, 0);
    }
  }
}

void xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vh01 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh23 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh45 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vh67 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vt0 = vcvt_f32_f16(vget_low_f16(vh01));
    const float32x4_t vt1 = vcvt_f32_f16(vget_high_f16(vh01));
    const float32x4_t vt2 = vcvt_f32_f16(vget_low_f16(vh23));
    const float32x4_t vt3 = vcvt_f32_f16(vget_high_f16(vh23));
    const float32x4_t vt4 = vcvt_f32_f16(vget_low_f16(vh45));
    const float32x4_t vt5 = vcvt_f32_f16(vget_high_f16(vh45));
    const float32x4_t vt6 = vcvt_f32_f16(vget_low_f16(vh67));
    const float32x4_t vt7 = vcvt_f32_f16(vget_high_f16(vh67));

    vacc0 = vaddq_f32(vacc0, vt0);
    vacc1 = vaddq_f32(vacc1, vt1);
    vacc2 = vaddq_f32(vacc2, vt2);
    vacc3 = vaddq_f32(vacc3, vt3);
    vacc0 = vaddq_f32(vacc0, vt4);
    vacc1 = vaddq_f32(vacc1, vt5);
    vacc2 = vaddq_f32(vacc2, vt6);
    vacc3 = vaddq_f32(vacc3, vt7);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  vacc2 = vaddq_f32(vacc2, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc2);
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_u16(i)); i += 4;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc0 = vaddq_f32(vacc0, vt);
  }
  const float32x2_t vscale = vld1_dup_f32(&params->scalar.scale);
  float32x2_t vacc = vadd_f32(vget_low_f32(vacc0), vget_high_f32(vacc0));
  if XNN_UNLIKELY(batch & (2 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u32(vld1_dup_u32((const void*) i)); i += 2;
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vpadd_f32(vacc, vacc);
  if XNN_UNLIKELY(batch & (1 * sizeof(uint16_t))) {
    const float16x4_t vh = vreinterpret_f16_u16(vld1_dup_u16(i));
    const float32x4_t vt = vcvt_f32_f16(vh);
    vacc = vadd_f32(vacc, vget_low_f32(vt));
  }
  vacc = vmul_f32(vacc, vscale);
  const float16x4_t vout = vcvt_f16_f32(vcombine_f32(vacc, vacc));
  vst1_lane_u16(o, vreinterpret_u16_f16(vout), 0);
}

void xnn_f32_f16_vcvt_ukernel__neonfp16_u16(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vf0 = vld1q_f32(input); input += 4;
    const float32x4_t vf1 = vld1q_f32(input); input += 4;
    const float32x4_t vf2 = vld1q_f32(input); input += 4;
    const float32x4_t vf3 = vld1q_f32(input); input += 4;

    const uint16x8_t vh0 = vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vf0), vcvt_f16_f32(vf1)));
    const uint16x8_t vh1 = vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vf2), vcvt_f16_f32(vf3)));

    vst1q_u16(o, vh0); o += 8;
    vst1q_u16(o, vh1); o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vf = vld1q_f32(input); input += 4;

    const uint16x4_t vh = vreinterpret_u16_f16(vcvt_f16_f32(vf));

    vst1_u16(o, vh); o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch % sizeof(float) == 0);
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vf = vld1q_f32(input);

    uint16x4_t vh = vreinterpret_u16_f16(vcvt_f16_f32(vf));

    if (batch & (2 * sizeof(float))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vh), 0); o += 2;
      vh = vext_u16(vh, vh, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u16(o, vh, 0);
    }
  }
}
