// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/avgpool.h>


void xnn_f32_avgpool_ukernel_up9__neon(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    const float* zero,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_avgpool_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(ks <= 9);
  assert(kc != 0);

  const float32x4_t vmultiplier = vld1q_dup_f32(&params->scalar.multiplier);
  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.output_min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.output_max);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    input = (const float**) ((uintptr_t) input + input_increment);
    if (ks < 2) {
      i1 = zero;
    }
    if (ks <= 2) {
      i2 = zero;
    }
    if (ks < 4) {
      i3 = zero;
    }
    if (ks <= 4) {
      i4 = zero;
    }
    if (ks < 6) {
      i5 = zero;
    }
    if (ks <= 6) {
      i6 = zero;
    }
    if (ks < 8) {
      i7 = zero;
    }
    if (ks <= 8) {
      i8 = zero;
    }

    size_t k = kc;
    while (k >= 4) {
      const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi8 = vld1q_f32(i8); i8 += 4;

      const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
      const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
      const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
      const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
      const float32x4_t vsum018 = vaddq_f32(vsum01, vi8);
      const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
      const float32x4_t vsum01678 = vaddq_f32(vsum018, vsum67);
      const float32x4_t vsum = vaddq_f32(vsum2345, vsum01678);

      float32x4_t vout = vmulq_f32(vsum, vmultiplier);
      vout = vmaxq_f32(vout, voutput_min);
      vout = vminq_f32(vout, voutput_max);

      vst1q_f32(output, vout); output += 4;

      k -= 4;
    }
    if (k != 0) {
      const float32x4_t vi0 = vld1q_f32(i0);
      const float32x4_t vi1 = vld1q_f32(i1);
      const float32x4_t vi2 = vld1q_f32(i2);
      const float32x4_t vi3 = vld1q_f32(i3);
      const float32x4_t vi4 = vld1q_f32(i4);
      const float32x4_t vi5 = vld1q_f32(i5);
      const float32x4_t vi6 = vld1q_f32(i6);
      const float32x4_t vi7 = vld1q_f32(i7);
      const float32x4_t vi8 = vld1q_f32(i8);

      const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
      const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
      const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
      const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
      const float32x4_t vsum018 = vaddq_f32(vsum01, vi8);
      const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
      const float32x4_t vsum01678 = vaddq_f32(vsum018, vsum67);
      const float32x4_t vsum = vaddq_f32(vsum2345, vsum01678);

      float32x4_t vout = vmulq_f32(vsum, vmultiplier);
      vout = vmaxq_f32(vout, voutput_min);
      vout = vminq_f32(vout, voutput_max);

      float32x2_t vout_lo = vget_low_f32(vout);
      if (k & 2) {
        vst1_f32(output, vout_lo); output += 2;
        vout_lo = vget_high_f32(vout);
      }
      if (k & 1) {
        vst1_lane_f32(output, vout_lo, 0); output += 1;
      }
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--n != 0);
}
