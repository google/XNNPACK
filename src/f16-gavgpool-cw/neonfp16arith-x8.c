// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8(
    size_t elements,
    size_t channels,
    const void* input,
    void* output,
    const union xnn_f16_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(__fp16) == 0);
  assert(channels != 0);

  __fp16* o = (__fp16*) output;
  const __fp16* i0 = input;

  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith.mask);
  const float16x4_t vmultiplier = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.multiplier));
  const float16x4_t voutput_min = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_min));
  const float16x4_t voutput_max = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_max));

#if XNN_ARCH_ARM64
  while (channels >= 4) {
    const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + elements);
    const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + elements);
    const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + elements);
    float16x8_t vsum0 = vmovq_n_f16(0);
    float16x8_t vsum1 = vmovq_n_f16(0);
    float16x8_t vsum2 = vmovq_n_f16(0);
    float16x8_t vsum3 = vmovq_n_f16(0);
    size_t n = elements;
    while (n >= 8 * sizeof(__fp16)) {
      const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1 = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi2 = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi3 = vld1q_f16(i3); i3 += 8;

      vsum0 = vaddq_f16(vsum0, vi0);
      vsum1 = vaddq_f16(vsum1, vi1);
      vsum2 = vaddq_f16(vsum2, vi2);
      vsum3 = vaddq_f16(vsum3, vi3);
      n -= 8 * sizeof(__fp16);
    }

    if XNN_UNLIKELY(n != 0) {
      float16x8_t vi0 = vld1q_f16(i0); i0 = (const __fp16*) ((uintptr_t) i0 + n);
      float16x8_t vi1 = vld1q_f16(i1); i1 = (const __fp16*) ((uintptr_t) i1 + n);
      float16x8_t vi2 = vld1q_f16(i2); i2 = (const __fp16*) ((uintptr_t) i2 + n);
      float16x8_t vi3 = vld1q_f16(i3); i3 = (const __fp16*) ((uintptr_t) i3 + n);

      vi0 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0)));
      vi1 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi1)));
      vi2 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi2)));
      vi3 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi3)));

      vsum0 = vaddq_f16(vsum0, vi0);
      vsum1 = vaddq_f16(vsum1, vi1);
      vsum2 = vaddq_f16(vsum2, vi2);
      vsum3 = vaddq_f16(vsum3, vi3);
    }

    // Having exactly 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const float16x8_t vsum01 = vpaddq_f16(vsum0, vsum1);
    const float16x8_t vsum23 = vpaddq_f16(vsum2, vsum3);
    const float16x8_t vsum0123 = vpaddq_f16(vsum01, vsum23);
    const float16x4_t vsum = vpadd_f16(vget_low_f16(vsum0123), vget_high_f16(vsum0123));

    float16x4_t vout = vmul_f16(vsum, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_f16(o, vout); o += 4;

    i0 = i3;
    channels -= 4;
  }
#endif  // XNN_ARCH_ARM64

  while (channels != 0) {
    float16x8_t vsum0 = vmovq_n_f16(0);
    size_t n = elements;
    if (n >= 8 * sizeof(__fp16)) {
      do {
        const float16x8_t vi0 = vld1q_f16(i0); i0 += 8;
        vsum0 = vaddq_f16(vsum0, vi0);
        n -= 8 * sizeof(__fp16);
      } while (n >= 8 * sizeof(__fp16));
    }

    if XNN_UNLIKELY(n != 0) {
      float16x8_t vi0 = vld1q_f16(i0); i0 = (const __fp16*) ((uintptr_t) i0 + n);

      vi0 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0)));

      vsum0 = vaddq_f16(vsum0, vi0);
    }

    const float16x4_t vout4 = vpadd_f16(vget_low_f16(vsum0), vget_high_f16(vsum0));
    const float16x4_t vout2 = vpadd_f16(vout4, vout4);
    const float16x4_t vout1 = vpadd_f16(vout2, vout2);

    float16x4_t vout = vmul_f16(vout1, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_lane_f16(o, vout, 0); o += 1;
    channels -= 1;
  }
}
