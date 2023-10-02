// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_cw_ukernel__neon_u4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const uint32x4_t vmask = vld1q_u32(params->neon.mask);
  const float32x2_t vmultiplier = vld1_dup_f32(&params->neon.multiplier);
  const float32x2_t voutput_min = vld1_dup_f32(&params->neon.output_min);
  const float32x2_t voutput_max = vld1_dup_f32(&params->neon.output_max);

  do {
    float32x4_t vsum0 = vmovq_n_f32(0.0f);
    size_t n = elements;
    if (n >= 16 * sizeof(float)) {
      float32x4_t vsum1 = vmovq_n_f32(0.0f);
      do {
        const float32x4_t vi0 = vld1q_f32(input);
        const float32x4_t vi1 = vld1q_f32(input + 4);
        const float32x4_t vi2 = vld1q_f32(input + 8);
        const float32x4_t vi3 = vld1q_f32(input + 12);
        input += 16;
        const float32x4_t acc0 = vaddq_f32(vi0, vi1);
        const float32x4_t acc1 = vaddq_f32(vi2, vi3);
        vsum0 = vaddq_f32(vsum0, acc0);
        vsum1 = vaddq_f32(vsum1, acc1);
        n -= 16 * sizeof(float);
      } while (n >= 32 * sizeof(float));
      vsum0 = vaddq_f32(vsum0, vsum1);
    }

    while (n >= 4 * sizeof(float)) {
      const float32x4_t vi0 = vld1q_f32(input);
      input += 4;
      vsum0 = vaddq_f32(vsum0, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      float32x4_t vi0 = vld1q_f32(input); input = (const float*) ((uintptr_t) input + n);

      vi0 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0)));

      vsum0 = vaddq_f32(vsum0, vi0);
    }

    const float32x2_t vout2 = vpadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
    const float32x2_t vout1 = vpadd_f32(vout2, vout2);

    float32x2_t vout = vmul_f32(vout1, vmultiplier);
    vout = vmax_f32(vout, voutput_min);
    vout = vmin_f32(vout, voutput_max);

    vst1_lane_f32(output, vout, 0); output += 1;
  } while (--channels != 0);
}
