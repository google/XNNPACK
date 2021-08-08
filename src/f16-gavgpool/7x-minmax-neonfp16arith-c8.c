// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>


void xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output_ptr,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

   __fp16* output = (__fp16*) output_ptr;
  const __fp16* i0 = (const __fp16*) input;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = (const __fp16*) zero;
  }
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = (const __fp16*) zero;
  }
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = (const __fp16*) zero;
  }
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = (const __fp16*) zero;
  }
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = (const __fp16*) zero;
  }
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = (const __fp16*) zero;
  }
  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));

  while (channels >= 8) {
    const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
    const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
    const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;
    const float16x8_t vi4 = vld1q_f16(i4); i4 += 8;
    const float16x8_t vi5 = vld1q_f16(i5); i5 += 8;
    const float16x8_t vi6 = vld1q_f16(i6); i6 += 8;

    const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
    const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
    const float16x8_t vsum45 = vaddq_f16(vi4, vi5);

    const float16x8_t vsum016 = vaddq_f16(vsum01, vi6);
    const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);

    const float16x8_t vsum = vaddq_f16(vsum016, vsum2345);

    float16x8_t vout = vmulq_f16(vsum, vscale);
    vout = vmaxq_f16(vout, vmin);
    vout = vminq_f16(vout, vmax);

    vst1q_f16(output, vout); output += 8;

    channels -= 8;
  }
  if (channels != 0) {
    const float16x8_t vi0 = vld1q_f16(i0);
    const float16x8_t vi1 = vld1q_f16(i1);
    const float16x8_t vi2 = vld1q_f16(i2);
    const float16x8_t vi3 = vld1q_f16(i3);
    const float16x8_t vi4 = vld1q_f16(i4);
    const float16x8_t vi5 = vld1q_f16(i5);
    const float16x8_t vi6 = vld1q_f16(i6);

    const float16x8_t vsum01 = vaddq_f16(vi0, vi1);
    const float16x8_t vsum23 = vaddq_f16(vi2, vi3);
    const float16x8_t vsum45 = vaddq_f16(vi4, vi5);

    const float16x8_t vsum016 = vaddq_f16(vsum01, vi6);
    const float16x8_t vsum2345 = vaddq_f16(vsum23, vsum45);

    const float16x8_t vsum = vaddq_f16(vsum016, vsum2345);

    float16x8_t vout = vmulq_f16(vsum, vscale);
    vout = vmaxq_f16(vout, vmin);
    vout = vminq_f16(vout, vmax);

    float16x4_t vout_lo = vget_low_f16(vout);
    if (channels & 4) {
      vst1_f16(output, vout_lo); output += 4;
      vout_lo = vget_high_f16(vout);
    }
    if (channels & 2) {
      vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_f16(vout_lo), 0); output += 2;
      vout_lo = vext_f16(vout_lo, vout_lo, 2);
    }
    if (channels & 1) {
      vst1_lane_f16(output, vout_lo, 0);
    }
  }
}
