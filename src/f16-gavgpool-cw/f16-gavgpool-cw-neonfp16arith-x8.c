// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8(
    size_t elements,
    size_t channels,
    const void* input,
    void* output,
    const union xnn_f16_gavgpool_params* params) XNN_OOB_READS
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

  do {
    float16x8_t vsum0 = vmovq_n_f16(0);
    size_t n = elements;
    if (n >= 32 * sizeof(__fp16)) {
      float16x8_t vsum1 = vmovq_n_f16(0);
      do {
        const float16x8_t vi0 = vld1q_f16(i0);
        const float16x8_t vi1 = vld1q_f16(i0 + 8);
        const float16x8_t vi2 = vld1q_f16(i0 + 16);
        const float16x8_t vi3 = vld1q_f16(i0 + 24);
        i0 += 32;
        const float16x8_t acc0 = vaddq_f16(vi0, vi1);
        const float16x8_t acc1 = vaddq_f16(vi2, vi3);
        vsum0 = vaddq_f16(vsum0, acc0);
        vsum1 = vaddq_f16(vsum1, acc1);
        n -= 32 * sizeof(__fp16);
      } while (n >= 32 * sizeof(__fp16));
      vsum0 = vaddq_f16(vsum0, vsum1);
    }

    while (n >= 8 * sizeof(__fp16)) {
      const float16x8_t vi0 = vld1q_f16(i0);
      i0 += 8;
      vsum0 = vaddq_f16(vsum0, vi0);
      n -= 8 * sizeof(__fp16);
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
  } while (--channels != 0);
}
