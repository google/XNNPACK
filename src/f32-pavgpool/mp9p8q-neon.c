// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/pavgpool.h>


void xnn_f32_pavgpool_ukernel_mp9p8q__neon(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks > 9);
  assert(kc != 0);

  const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);

  do {
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;

      float* b = buffer;
      for (size_t k = 0; k < kc; k += 4) {
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

        vst1q_f32(b, vsum); b += 4;
      }
    }

    size_t m = ks;
    for (m -= 9; m > 8; m -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;

      float* b = buffer;
      for (size_t k = 0; k < kc; k += 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vacc = vld1q_f32(b);

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

        vst1q_f32(b, vsum); b += 4;
      }
    }

    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (m < 2) {
        i1 = zero;
      }
      if (m <= 2) {
        i2 = zero;
      }
      if (m < 4) {
        i3 = zero;
      }
      if (m <= 4) {
        i4 = zero;
      }
      if (m < 6) {
        i5 = zero;
      }
      if (m <= 6) {
        i6 = zero;
      }
      if (m != 8) {
        i7 = zero;
      }

      const float32x4_t vmultiplier = vld1q_dup_f32(multiplier); multiplier += 1;

      size_t k = kc;
      float* b = buffer;
      while (k >= 4) {
        const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi4 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi5 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi6 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi7 = vld1q_f32(i7); i7 += 4;
        const float32x4_t vacc = vld1q_f32(b); b += 4;

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

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
        const float32x4_t vacc = vld1q_f32(b);

        const float32x4_t vsum01 = vaddq_f32(vi0, vi1);
        const float32x4_t vsum23 = vaddq_f32(vi2, vi3);
        const float32x4_t vsum45 = vaddq_f32(vi4, vi5);
        const float32x4_t vsum67 = vaddq_f32(vi6, vi7);
        const float32x4_t vsum01a = vaddq_f32(vsum01, vacc);
        const float32x4_t vsum2345 = vaddq_f32(vsum23, vsum45);
        const float32x4_t vsum0167a = vaddq_f32(vsum01a, vsum67);
        const float32x4_t vsum = vaddq_f32(vsum2345, vsum0167a);

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
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--n != 0);
}
