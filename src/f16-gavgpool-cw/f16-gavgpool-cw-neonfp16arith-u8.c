// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8(
    size_t elements,
    size_t channels,
    const void* input,
    void* output,
    const union xnn_f16_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(uint16_t) == 0);
  assert(channels != 0);

  const uint16x8_t vmask = vld1q_u16(params->neonfp16arith.mask);
  const float16x4_t vmultiplier = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.multiplier));
  const float16x4_t voutput_min = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_min));
  const float16x4_t voutput_max = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_max));

  uint16_t* o = (uint16_t*) output;
  const uint16_t* i = input;
  do {
    float16x8_t vsum0 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vsum1 = vreinterpretq_f16_u16(vmovq_n_u16(0));
    size_t n = elements;
    if (n >= 32 * sizeof(uint16_t)) {
      do {
        const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i));
        const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i + 8));
        const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i + 16));
        const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i + 24));
        i += 32;
        const float16x8_t acc0 = vaddq_f16(vi0, vi1);
        const float16x8_t acc1 = vaddq_f16(vi2, vi3);
        vsum0 = vaddq_f16(vsum0, acc0);
        vsum1 = vaddq_f16(vsum1, acc1);
        n -= 32 * sizeof(uint16_t);
      } while (n >= 32 * sizeof(uint16_t));
    }
    vsum0 = vaddq_f16(vsum0, vsum1);

    while (n >= 8 * sizeof(uint16_t)) {
      const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i));
      i += 8;
      vsum0 = vaddq_f16(vsum0, vi0);
      n -= 8 * sizeof(uint16_t);
    }

    if XNN_UNLIKELY(n != 0) {
      float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i = (const uint16_t*) ((uintptr_t) i + n);

      vi0 = vreinterpretq_f16_u16(vandq_u16(vmask, vreinterpretq_u16_f16(vi0)));

      vsum0 = vaddq_f16(vsum0, vi0);
    }

    const float16x4_t vout4 = vpadd_f16(vget_low_f16(vsum0), vget_high_f16(vsum0));
    const float16x4_t vout2 = vpadd_f16(vout4, vout4);
    const float16x4_t vout1 = vpadd_f16(vout2, vout2);

    float16x4_t vout = vmul_f16(vout1, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_lane_u16(o, vreinterpret_u16_f16(vout), 0); o += 1;
  } while (--channels != 0);
}
