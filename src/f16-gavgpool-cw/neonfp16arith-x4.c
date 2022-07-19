// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x4(
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
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + elements);
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + elements);
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + elements);

  const uint16x4_t vmask = vld1_u16(params->neonfp16arith.mask);
  const float16x4_t vmultiplier = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.multiplier));
  const float16x4_t voutput_min = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_min));
  const float16x4_t voutput_max = vreinterpret_f16_u16(vld1_dup_u16(&params->neonfp16arith.output_max));

  while (channels >= 4) {
    float16x4_t vsum0 = vmov_n_f16(0);
    float16x4_t vsum1 = vmov_n_f16(0);
    float16x4_t vsum2 = vmov_n_f16(0);
    float16x4_t vsum3 = vmov_n_f16(0);
    size_t n = elements;
    while (n >= 4 * sizeof(__fp16)) {
      const float16x4_t vi0 = vld1_f16(i0); i0 += 4;
      const float16x4_t vi1 = vld1_f16(i1); i1 += 4;
      const float16x4_t vi2 = vld1_f16(i2); i2 += 4;
      const float16x4_t vi3 = vld1_f16(i3); i3 += 4;

      vsum0 = vadd_f16(vsum0, vi0);
      vsum1 = vadd_f16(vsum1, vi1);
      vsum2 = vadd_f16(vsum2, vi2);
      vsum3 = vadd_f16(vsum3, vi3);
      n -= 4 * sizeof(__fp16);
    }

    if XNN_UNLIKELY(n != 0) {
      float16x4_t vi0 = vld1_f16(i0); i0 = (const __fp16*) ((uintptr_t) i0 + n);
      float16x4_t vi1 = vld1_f16(i1); i1 = (const __fp16*) ((uintptr_t) i1 + n);
      float16x4_t vi2 = vld1_f16(i2); i2 = (const __fp16*) ((uintptr_t) i2 + n);
      float16x4_t vi3 = vld1_f16(i3); i3 = (const __fp16*) ((uintptr_t) i3 + n);

      vi0 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi0)));
      vi1 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi1)));
      vi2 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi2)));
      vi3 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi3)));

      vsum0 = vadd_f16(vsum0, vi0);
      vsum1 = vadd_f16(vsum1, vi1);
      vsum2 = vadd_f16(vsum2, vi2);
      vsum3 = vadd_f16(vsum3, vi3);
    }

    // Having exactly 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const float16x4_t vsum01 = vpadd_f16(vsum0, vsum1);
    const float16x4_t vsum23 = vpadd_f16(vsum2, vsum3);
    const float16x4_t vsum = vpadd_f16(vsum01, vsum23);

    float16x4_t vout = vmul_f16(vsum, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_f16(o, vout); o += 4;

    i0 = i3;
    i1 = (const __fp16*) ((uintptr_t) i0 + elements);
    i2 = (const __fp16*) ((uintptr_t) i1 + elements);
    i3 = (const __fp16*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    float16x4_t vsum0 = vmov_n_f16(0);
    size_t n = elements;
    while (n >= 4 * sizeof(__fp16)) {
      const float16x4_t vi0 = vld1_f16(i0); i0 += 4;

      vsum0 = vadd_f16(vsum0, vi0);
      n -= 4 * sizeof(__fp16);
    }

    if XNN_UNLIKELY(n != 0) {
      float16x4_t vi0 = vld1_f16(i0); i0 = (const __fp16*) ((uintptr_t) i0 + n);

      vi0 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi0)));

      vsum0 = vadd_f16(vsum0, vi0);
    }

    const float16x4_t vsum01 = vpadd_f16(vsum0, vsum0);
    const float16x4_t vsum = vpadd_f16(vsum01, vsum01);

    float16x4_t vout = vmul_f16(vsum, vmultiplier);

    vout = vmax_f16(vout, voutput_min);
    vout = vmin_f16(vout, voutput_max);

    vst1_lane_f16(o, vout, 0); o += 1;
    channels -= 1;
  }
}
