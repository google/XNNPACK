// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t packed_channels = round_up_po2(channels, 4);
  const size_t input_increment = 7 * input_stride - packed_channels * sizeof(float);

  float* b = buffer;
  for (size_t c = 0; c < channels; c += 4) {
    const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
    const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
    const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
    const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
    const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
    const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
    const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;

    const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
    const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
    const float32x4_t vsum45 = vaddq_f32(vi4, vi5);

    const float32x4_t vsum016 = vaddq_f32(vsum01, vi6);
    const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);

    const float32x4_t vsum = vaddq_f32(vsum016, vsum2345);

    vst1q_f32(b, vsum); b += 4;
  }
  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    for (size_t c = 0; c < channels; c += 4) {
      const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vacc = vld1q_f32(b);

      const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
      const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
      const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
      const float32x4_t vsum6a = vaddq_f32(vi6, vacc);

      const float32x4_t vsum0123 = vaddq_f32(vsum01, vsum23);
      const float32x4_t vsum456a = vaddq_f32(vsum45, vsum6a);

      const float32x4_t vsum = vaddq_f32(vsum0123, vsum456a);

      vst1q_f32(b, vsum); b += 4;
    }
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);

  b = buffer;
  while (channels >= 4) {
    const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
    const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
    const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
    const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
    const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
    const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
    const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
    const float32x4_t vacc = vld1q_f32(b); b += 4;

    const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
    const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
    const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
    const float32x4_t vsum6a = vaddq_f32(vi6, vacc);

    const float32x4_t vsum0123 = vaddq_f32(vsum01, vsum23);
    const float32x4_t vsum456a = vaddq_f32(vsum45, vsum6a);

    const float32x4_t vsum = vaddq_f32(vsum0123, vsum456a);

    float32x4_t vout = vmulq_f32(vsum, vscale);
    vout = vmaxq_f32(vout, vmin);
    vout = vminq_f32(vout, vmax);

    vst1q_f32(output, vout); output += 4;

    channels -= 4;
  }
  if (channels != 0) {
    const float32x4_t vi0 = vld1q_f32(i0);
    const float32x4_t vi1 = vld1q_f32(i1);
    const float32x4_t vi2 = vld1q_f32(i2);
    const float32x4_t vi3 = vld1q_f32(i3);
    const float32x4_t vi4 = vld1q_f32(i4);
    const float32x4_t vi5 = vld1q_f32(i5);
    const float32x4_t vi6 = vld1q_f32(i6);
    const float32x4_t vacc = vld1q_f32(b);

    const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
    const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
    const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
    const float32x4_t vsum6a = vaddq_f32(vi6, vacc);

    const float32x4_t vsum0123 = vaddq_f32(vsum01, vsum23);
    const float32x4_t vsum456a = vaddq_f32(vsum45, vsum6a);

    const float32x4_t vsum = vaddq_f32(vsum0123, vsum456a);

    float32x4_t vout = vmulq_f32(vsum, vscale);
    vout = vmaxq_f32(vout, vmin);
    vout = vminq_f32(vout, vmax);

    float32x2_t vout_lo = vget_low_f32(vout);
    if (channels & 2) {
      vst1_f32(output, vout_lo); output += 2;
      vout_lo = vget_high_f32(vout);
    }
    if (channels & 1) {
      vst1_lane_f32(output, vout_lo, 0);
    }
  }
}
