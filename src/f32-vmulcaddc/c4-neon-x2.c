// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>


void xnn_f32_vmulcaddc_ukernel_c4__neon_x2(
    size_t m,
    size_t channels,
    const float*restrict x,
    size_t x_stride,
    const float*restrict weights,
    float*restrict y,
    size_t y_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(m != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t x_increment = x_stride * 2 - channels;
  const size_t y_increment = y_stride * 2 - channels;

  const float* x0 = x;
  float* y0 = y;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if XNN_UNPREDICTABLE(m < 2) {
    x1 = x0;
    y1 = y0;
  }

  const float32x4x2_t voutput_clamp = vld2q_dup_f32(&params->scalar.max);
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float32x4_t vscale0123 = vld1q_f32(w); w += 4;

      const float32x4_t vx0x0123 = vld1q_f32(x0); x0 += 4;
      const float32x4_t vx1x0123 = vld1q_f32(x1); x1 += 4;

      float32x4_t vacc0x0123 = vmulq_f32(vx0x0123, vscale0123);
      float32x4_t vacc1x0123 = vmulq_f32(vx1x0123, vscale0123);

      const float32x4_t vbias0123 = vld1q_f32(w); w += 4;

      vacc0x0123 = vaddq_f32(vacc0x0123, vbias0123);
      vacc1x0123 = vaddq_f32(vacc1x0123, vbias0123);

      vacc0x0123 = vmaxq_f32(vacc0x0123, voutput_clamp.val[1]);
      vacc1x0123 = vmaxq_f32(vacc1x0123, voutput_clamp.val[1]);

      vacc0x0123 = vminq_f32(vacc0x0123, voutput_clamp.val[0]);
      vacc1x0123 = vminq_f32(vacc1x0123, voutput_clamp.val[0]);

      vst1q_f32(y0, vacc0x0123); y0 += 4;
      vst1q_f32(y1, vacc1x0123); y1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const float32x4_t vscale0123 = vld1q_f32(w); w += 4;

      const float32x4_t vx0x0123 = vld1q_f32(x0); x0 = (const float*) ((uintptr_t) x0 + c);
      const float32x4_t vx1x0123 = vld1q_f32(x1); x1 = (const float*) ((uintptr_t) x1 + c);

      float32x4_t vacc0x0123 = vmulq_f32(vx0x0123, vscale0123);
      float32x4_t vacc1x0123 = vmulq_f32(vx1x0123, vscale0123);

      const float32x4_t vbias0123 = vld1q_f32(w); w += 4;

      vacc0x0123 = vaddq_f32(vacc0x0123, vbias0123);
      vacc1x0123 = vaddq_f32(vacc1x0123, vbias0123);

      vacc0x0123 = vmaxq_f32(vacc0x0123, voutput_clamp.val[1]);
      vacc1x0123 = vmaxq_f32(vacc1x0123, voutput_clamp.val[1]);

      vacc0x0123 = vminq_f32(vacc0x0123, voutput_clamp.val[0]);
      vacc1x0123 = vminq_f32(vacc1x0123, voutput_clamp.val[0]);

      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      float32x2_t vacc1x01 = vget_low_f32(vacc1x0123);
      if (c & (2 * sizeof(float))) {
        vst1_f32(y0, vacc0x01); y0 += 2;
        vst1_f32(y1, vacc1x01); y1 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
        vacc1x01 = vget_high_f32(vacc1x0123);
      }
      if (c & (1 * sizeof(float))) {
        vst1_lane_f32(y0, vacc0x01, 0); y0 += 1;
        vst1_lane_f32(y1, vacc1x01, 0); y1 += 1;
      }
    }
    x0 = (const float*) ((uintptr_t) x0 + x_increment);
    y0 = (float*) ((uintptr_t) y0 + y_increment);
    x1 = (const float*) ((uintptr_t) x1 + x_increment);
    y1 = (float*) ((uintptr_t) y1 + y_increment);
    if XNN_UNPREDICTABLE(m < 4) {
      x1 = x0;
      y1 = y0;
    }
    m = doz(m, 2);
  } while (m != 0);
}
