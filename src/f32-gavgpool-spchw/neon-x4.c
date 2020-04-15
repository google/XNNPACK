// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_spchw_ukernel__neon_x4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + elements);
  const float* i2 = (const float*) ((uintptr_t) i1 + elements);
  const float* i3 = (const float*) ((uintptr_t) i2 + elements);

  const uint32x4_t vmask = vld1q_u32(params->neon.mask);
  const float32x4_t vmultiplier = vld1q_dup_f32(&params->neon.multiplier);
  const float32x4_t voutput_min = vld1q_dup_f32(&params->neon.output_min);
  const float32x4_t voutput_max = vld1q_dup_f32(&params->neon.output_max);

  while (channels >= 4) {
    float32x4_t vsum0 = vmovq_n_f32(0.0f);
    float32x4_t vsum1 = vmovq_n_f32(0.0f);
    float32x4_t vsum2 = vmovq_n_f32(0.0f);
    float32x4_t vsum3 = vmovq_n_f32(0.0f);
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi1 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi2 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi3 = vld1q_f32(i3); i3 += 4;

      vsum0 = vaddq_f32(vsum0, vi0);
      vsum1 = vaddq_f32(vsum1, vi1);
      vsum2 = vaddq_f32(vsum2, vi2);
      vsum3 = vaddq_f32(vsum3, vi3);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      float32x4_t vi0 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + n);
      float32x4_t vi1 = vld1q_f32(i1); i1 = (const float*) ((uintptr_t) i1 + n);
      float32x4_t vi2 = vld1q_f32(i2); i2 = (const float*) ((uintptr_t) i2 + n);
      float32x4_t vi3 = vld1q_f32(i3); i3 = (const float*) ((uintptr_t) i3 + n);

      vi0 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0)));
      vi1 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi1)));
      vi2 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi2)));
      vi3 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi3)));

      vsum0 = vaddq_f32(vsum0, vi0);
      vsum1 = vaddq_f32(vsum1, vi1);
      vsum2 = vaddq_f32(vsum2, vi2);
      vsum3 = vaddq_f32(vsum3, vi3);
    }

    // Having exaclty 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
#if XNN_ARCH_ARM64
    const float32x4_t vsum01 = vpaddq_f32(vsum0, vsum1);
    const float32x4_t vsum23 = vpaddq_f32(vsum2, vsum3);
    const float32x4_t vsum = vpaddq_f32(vsum01, vsum23);
#else
    const float32x4_t vsum01 = vcombine_f32(vadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0)),
                                            vadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1)));
    const float32x4_t vsum23 = vcombine_f32(vadd_f32(vget_low_f32(vsum2), vget_high_f32(vsum2)),
                                            vadd_f32(vget_low_f32(vsum3), vget_high_f32(vsum3)));
    const float32x4_t vsum = vcombine_f32(vpadd_f32(vget_low_f32(vsum01), vget_high_f32(vsum01)),
                                          vpadd_f32(vget_low_f32(vsum23), vget_high_f32(vsum23)));
#endif

    float32x4_t vout = vmulq_f32(vsum, vmultiplier);

    vout = vmaxq_f32(vout, voutput_min);
    vout = vminq_f32(vout, voutput_max);

    vst1q_f32(output, vout); output += 4;
    i0 = i3;
    i1 = (const float*) ((uintptr_t) i0 + elements);
    i2 = (const float*) ((uintptr_t) i1 + elements);
    i3 = (const float*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    float32x4_t vsum0 = vmovq_n_f32(0.0f);
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const float32x4_t vi0 = vld1q_f32(i0); i0 += 4;
      vsum0 = vaddq_f32(vsum0, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      float32x4_t vi0 = vld1q_f32(i0); i0 = (const float*) ((uintptr_t) i0 + n);
      vi0 = vreinterpretq_f32_u32(vandq_u32(vmask, vreinterpretq_u32_f32(vi0)));
      vsum0 = vaddq_f32(vsum0, vi0);
    }

    float32x2_t vsum = vadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
    vsum = vpadd_f32(vsum, vsum);

    float32x2_t vout = vmul_f32(vsum, vget_low_f32(vmultiplier));

    vout = vmax_f32(vout, vget_low_f32(voutput_min));
    vout = vmin_f32(vout, vget_low_f32(voutput_max));

    vst1_lane_f32(output, vout, 0); output += 1;
    channels -= 1;
  }
}
