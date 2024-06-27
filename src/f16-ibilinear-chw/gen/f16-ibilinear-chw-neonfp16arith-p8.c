// Auto-generated file. Do not edit!
//   Template: src/f16-ibilinear-chw/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/ibilinear.h"


void xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8(
    size_t output_pixels,
    size_t channels,
    const void** restrict input,
    size_t input_offset,
    const void* restrict weights,
    void* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(uint16_t) == 0);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t** i = (const uint16_t**)input;
    const uint16_t* w = weights;
    size_t p = output_pixels;

    for (; p >= 8; p -= 8) {
      const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
      const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
      const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
      const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
      const uint16_t* itl2 = (const uint16_t*) ((uintptr_t) i[4] + input_offset);
      const uint16_t* ibl2 = (const uint16_t*) ((uintptr_t) i[5] + input_offset);
      const uint16_t* itl3 = (const uint16_t*) ((uintptr_t) i[6] + input_offset);
      const uint16_t* ibl3 = (const uint16_t*) ((uintptr_t) i[7] + input_offset);
      const uint16_t* itl4 = (const uint16_t*) ((uintptr_t) i[8] + input_offset);
      const uint16_t* ibl4 = (const uint16_t*) ((uintptr_t) i[9] + input_offset);
      const uint16_t* itl5 = (const uint16_t*) ((uintptr_t) i[10] + input_offset);
      const uint16_t* ibl5 = (const uint16_t*) ((uintptr_t) i[11] + input_offset);
      const uint16_t* itl6 = (const uint16_t*) ((uintptr_t) i[12] + input_offset);
      const uint16_t* ibl6 = (const uint16_t*) ((uintptr_t) i[13] + input_offset);
      const uint16_t* itl7 = (const uint16_t*) ((uintptr_t) i[14] + input_offset);
      const uint16_t* ibl7 = (const uint16_t*) ((uintptr_t) i[15] + input_offset);
      i += 2 * 8;

      const uint16x4x2_t vw0123 = vld2_u16(w); w += 8;
      const uint16x4x2_t vw4567 = vld2_u16(w); w += 8;

      float16x8_t vtltr0123 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl0));
      float16x8_t vblbr0123 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl0));
      float16x8_t vtltr4567 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl4));
      float16x8_t vblbr4567 = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl4));

      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr0123), 1));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr0123), 1));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl5, vreinterpretq_u32_f16(vtltr4567), 1));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl5, vreinterpretq_u32_f16(vblbr4567), 1));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr0123), 2));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr0123), 2));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl6, vreinterpretq_u32_f16(vtltr4567), 2));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl6, vreinterpretq_u32_f16(vblbr4567), 2));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr0123), 3));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr0123), 3));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl7, vreinterpretq_u32_f16(vtltr4567), 3));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl7, vreinterpretq_u32_f16(vblbr4567), 3));

      const float16x8_t valphah01234567 = vreinterpretq_f16_u16(vcombine_u16(vw0123.val[0], vw4567.val[0]));
      const float16x8_t valphav01234567 = vreinterpretq_f16_u16(vcombine_u16(vw0123.val[1], vw4567.val[1]));

      const float16x8_t vldrd0123 = vsubq_f16(vblbr0123, vtltr0123);
      const float16x8_t vldrd4567 = vsubq_f16(vblbr4567, vtltr4567);

      const float16x8x2_t vld_t01234567 = vuzpq_f16(vldrd0123, vldrd4567);
      const float16x8_t vld01234567 = vld_t01234567.val[0];
      const float16x8_t vrd01234567 = vld_t01234567.val[1];

      const float16x8x2_t vtl_t01234567 = vuzpq_f16(vtltr0123, vtltr4567);
      const float16x8_t vtl01234567 = vtl_t01234567.val[0];
      const float16x8_t vtr01234567 = vtl_t01234567.val[1];

      const float16x8_t vl01234567 = vfmaq_f16(vtl01234567, vld01234567, valphav01234567);
      const float16x8_t vr01234567 = vfmaq_f16(vtr01234567, vrd01234567, valphav01234567);

      const float16x8_t vd01234567 = vsubq_f16(vr01234567, vl01234567);
      const float16x8_t vo01234567 = vfmaq_f16(vl01234567, vd01234567, valphah01234567);

      vst1q_u16(o, vreinterpretq_u16_f16(vo01234567)); o += 8;
    }
    for (; p >= 4; p -= 4) {
      const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
      const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
      const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
      const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
      const uint16_t* itl2 = (const uint16_t*) ((uintptr_t) i[4] + input_offset);
      const uint16_t* ibl2 = (const uint16_t*) ((uintptr_t) i[5] + input_offset);
      const uint16_t* itl3 = (const uint16_t*) ((uintptr_t) i[6] + input_offset);
      const uint16_t* ibl3 = (const uint16_t*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const uint16x4x2_t vw = vld2_u16(w); w += 8;

      float16x8_t vtltr = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) itl0));
      float16x8_t vblbr = vreinterpretq_f16_u32(vld1q_dup_u32((const void*) ibl0));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr), 1));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr), 1));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr), 2));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr), 2));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr), 3));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr), 3));

      const float16x4_t valphah = vreinterpret_f16_u16(vw.val[0]);
      const float16x4_t valphav = vreinterpret_f16_u16(vw.val[1]);

      const float16x8_t vldrd = vsubq_f16(vblbr, vtltr);

      const float16x4x2_t vld_t = vuzp_f16(vget_low_f16(vldrd), vget_high_f16(vldrd));
      const float16x4_t vld = vld_t.val[0];
      const float16x4_t vrd = vld_t.val[1];

      const float16x4x2_t vtl_t = vuzp_f16(vget_low_f16(vtltr), vget_high_f16(vtltr));
      const float16x4_t vtl = vtl_t.val[0];
      const float16x4_t vtr = vtl_t.val[1];

      const float16x4_t vl = vfma_f16(vtl, vld, valphav);
      const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

      const float16x4_t vd = vsub_f16(vr, vl);
      const float16x4_t vo = vfma_f16(vl, vd, valphah);

      vst1_u16(o, vreinterpret_u16_f16(vo)); o += 4;
    }
    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const uint16_t* itl0 = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
        const uint16_t* ibl0 = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
        const uint16_t* itl1 = (const uint16_t*) ((uintptr_t) i[2] + input_offset);
        const uint16_t* ibl1 = (const uint16_t*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float16x4_t vw = vreinterpret_f16_u16(vld1_u16(w)); w += 4;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        float16x4_t vtltr = vreinterpret_f16_u32(vld1_dup_u32((const void*) itl0));
        float16x4_t vblbr = vreinterpret_f16_u32(vld1_dup_u32((const void*) ibl0));

        vtltr = vreinterpret_f16_u32(vld1_lane_u32((const void*) itl1, vreinterpret_u32_f16(vtltr), 1));
        vblbr = vreinterpret_f16_u32(vld1_lane_u32((const void*) ibl1, vreinterpret_u32_f16(vblbr), 1));

        const float16x4_t vldrd = vsub_f16(vblbr, vtltr);

        const float16x4x2_t vld_t = vuzp_f16(vldrd, vldrd);
        const float16x4_t vld = vld_t.val[0];
        const float16x4_t vrd = vld_t.val[1];

        const float16x4x2_t vtl_t = vuzp_f16(vtltr, vtltr);
        const float16x4_t vtl = vtl_t.val[0];
        const float16x4_t vtr = vtl_t.val[1];

        const float16x4_t vl = vfma_f16(vtl, vld, valphav);
        const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

        const float16x4_t vd = vsub_f16(vr, vl);
        const float16x4_t vo = vfma_f16(vl, vd, valphah);

        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vo), 0); o += 2;
      }
      if (p & 1) {
        // We are computing the following formula:
        //   result = (1 - alpha_h) * (1 - alpha_v) * top_left +
        //                 alpha_h  * (1 - alpha_v) * top_right +
        //            (1 - alpha_h) *      alpha_v  * bottom_left +
        //                 alpha_h  *      alpha_v  * bottom_right.
        //
        // Rearranging gives
        //   result =    left + alpha_h * (right        - left),
        // where
        //   left =  top_left + alpha_v * (bottom_left  - top_left),
        //  right = top_right + alpha_v * (bottom_right - top_right).

        const uint16_t* itl = (const uint16_t*) ((uintptr_t) i[0] + input_offset);
        const uint16_t* ibl = (const uint16_t*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const float16x4_t vw = vreinterpret_f16_u32(vld1_dup_u32((const void*) w)); w += 2;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        const float16x4_t vtltr = vreinterpret_f16_u32(vld1_dup_u32((const void*) itl));
        const float16x4_t vblbr = vreinterpret_f16_u32(vld1_dup_u32((const void*) ibl));

        const float16x4_t vldrd = vsub_f16(vblbr, vtltr);

        const float16x4x2_t vld_t = vuzp_f16(vldrd, vldrd);
        const float16x4_t vld = vld_t.val[0];
        const float16x4_t vrd = vld_t.val[1];

        const float16x4x2_t vtl_t = vuzp_f16(vtltr, vtltr);
        const float16x4_t vtl = vtl_t.val[0];
        const float16x4_t vtr = vtl_t.val[1];

        const float16x4_t vl = vfma_f16(vtl, vld, valphav);
        const float16x4_t vr = vfma_f16(vtr, vrd, valphav);

        const float16x4_t vd = vsub_f16(vr, vl);
        const float16x4_t vo = vfma_f16(vl, vd, valphah);

        vst1_lane_u16(o, vreinterpret_u16_f16(vo), 0); o += 1;
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
