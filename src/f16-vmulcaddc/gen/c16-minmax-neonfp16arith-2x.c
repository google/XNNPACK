// Auto-generated file. Do not edit!
//   Template: src/f16-vmulcaddc/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>


void xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x(
    size_t rows,
    size_t channels,
    const void*restrict input,
    size_t input_stride,
    const void*restrict weights,
    void*restrict output,
    size_t output_stride,
    const struct xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(__fp16) == 0);

  const __fp16* i0 = (const __fp16*) input;
  __fp16* o0 = (__fp16*) output;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_stride);
  __fp16* o1 = (__fp16*) ((uintptr_t) o0 + output_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = i0;
    o1 = o0;
  }

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));
  do {
    const __fp16* w = (const __fp16*) weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(__fp16); c -= 16 * sizeof(__fp16)) {
      const float16x8_t vscale01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vscale89ABCDEF = vld1q_f16(w); w += 8;

      float16x8_t vacc0x01234567 = vld1q_f16(i0); i0 += 8;
      float16x8_t vacc0x89ABCDEF = vld1q_f16(i0); i0 += 8;
      float16x8_t vacc1x01234567 = vld1q_f16(i1); i1 += 8;
      float16x8_t vacc1x89ABCDEF = vld1q_f16(i1); i1 += 8;

      const float16x8_t vbias01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vbias89ABCDEF = vld1q_f16(w); w += 8;

      vacc0x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc0x01234567);
      vacc0x89ABCDEF = vfmaq_f16(vbias89ABCDEF, vscale89ABCDEF, vacc0x89ABCDEF);
      vacc1x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc1x01234567);
      vacc1x89ABCDEF = vfmaq_f16(vbias89ABCDEF, vscale89ABCDEF, vacc1x89ABCDEF);

      vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
      vacc0x89ABCDEF = vmaxq_f16(vacc0x89ABCDEF, vmin);
      vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);
      vacc1x89ABCDEF = vmaxq_f16(vacc1x89ABCDEF, vmin);

      vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
      vacc0x89ABCDEF = vminq_f16(vacc0x89ABCDEF, vmax);
      vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);
      vacc1x89ABCDEF = vminq_f16(vacc1x89ABCDEF, vmax);

      vst1q_f16(o0, vacc0x01234567); o0 += 8;
      vst1q_f16(o0, vacc0x89ABCDEF); o0 += 8;
      vst1q_f16(o1, vacc1x01234567); o1 += 8;
      vst1q_f16(o1, vacc1x89ABCDEF); o1 += 8;
    }
    for (; c >= 8 * sizeof(__fp16); c -= 8 * sizeof(__fp16)) {
      const float16x8_t vscale01234567 = vld1q_f16(w); w += 8;

      float16x8_t vacc0x01234567 = vld1q_f16(i0); i0 += 8;
      float16x8_t vacc1x01234567 = vld1q_f16(i1); i1 += 8;

      const float16x8_t vbias01234567 = vld1q_f16(w + 8);

      vacc0x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc0x01234567);
      vacc1x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc1x01234567);

      vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
      vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);

      vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
      vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);

      vst1q_f16(o0, vacc0x01234567); o0 += 8;
      vst1q_f16(o1, vacc1x01234567); o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const float16x8_t vscale01234567 = vld1q_f16(w);

      float16x8_t vacc0x01234567 = vld1q_f16(i0); i0 = (const __fp16*) ((uintptr_t) i0 + c);
      float16x8_t vacc1x01234567 = vld1q_f16(i1); i1 = (const __fp16*) ((uintptr_t) i1 + c);

      const float16x8_t vbias01234567 = vld1q_f16(w + 16);

      vacc0x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc0x01234567);
      vacc1x01234567 = vfmaq_f16(vbias01234567, vscale01234567, vacc1x01234567);

      vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
      vacc1x01234567 = vmaxq_f16(vacc1x01234567, vmin);

      vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
      vacc1x01234567 = vminq_f16(vacc1x01234567, vmax);

      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      if (c & (4 * sizeof(__fp16))) {
        vst1_f16(o0, vacc0x0123); o0 += 4;
        vst1_f16(o1, vacc1x0123); o1 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
      }
      if (c & (2 * sizeof(__fp16))) {
        vst1_lane_u32(__builtin_assume_aligned(o0, 1), vreinterpret_u32_f16(vacc0x0123), 0); o0 += 2;
        vst1_lane_u32(__builtin_assume_aligned(o1, 1), vreinterpret_u32_f16(vacc1x0123), 0); o1 += 2;

        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
      }
      if (c & (1 * sizeof(__fp16))) {
        vst1_lane_f16(o0, vacc0x0123, 0); o0 += 1;
        vst1_lane_f16(o1, vacc1x0123, 0); o1 += 1;
      }
    }
    i0 = (const __fp16*) ((uintptr_t) i0 + input_increment);
    o0 = (__fp16*) ((uintptr_t) o0 + output_increment);
    i1 = (const __fp16*) ((uintptr_t) i1 + input_increment);
    o1 = (__fp16*) ((uintptr_t) o1 + output_increment);
    if XNN_UNPREDICTABLE(rows < 4) {
      i1 = i0;
      o1 = o0;
    }
    rows = doz(rows, 2);
  } while (rows != 0);
}
