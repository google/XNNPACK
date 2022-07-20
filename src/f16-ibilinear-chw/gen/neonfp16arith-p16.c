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

#include <xnnpack/ibilinear.h>


void xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16(
    size_t output_pixels,
    size_t channels,
    const void**restrict input,
    size_t input_offset,
    const void*restrict weights,
    void*restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(__fp16) == 0);

  __fp16* o = (__fp16*) output;
  do {
    const __fp16** i = (const __fp16**)input;
    const __fp16* w = weights;
    size_t p = output_pixels;

    for (; p >= 16; p -= 16) {
      const __fp16* itl0 = (const __fp16*) ((uintptr_t) i[0] + input_offset);
      const __fp16* ibl0 = (const __fp16*) ((uintptr_t) i[1] + input_offset);
      const __fp16* itl1 = (const __fp16*) ((uintptr_t) i[2] + input_offset);
      const __fp16* ibl1 = (const __fp16*) ((uintptr_t) i[3] + input_offset);
      const __fp16* itl2 = (const __fp16*) ((uintptr_t) i[4] + input_offset);
      const __fp16* ibl2 = (const __fp16*) ((uintptr_t) i[5] + input_offset);
      const __fp16* itl3 = (const __fp16*) ((uintptr_t) i[6] + input_offset);
      const __fp16* ibl3 = (const __fp16*) ((uintptr_t) i[7] + input_offset);
      const __fp16* itl4 = (const __fp16*) ((uintptr_t) i[8] + input_offset);
      const __fp16* ibl4 = (const __fp16*) ((uintptr_t) i[9] + input_offset);
      const __fp16* itl5 = (const __fp16*) ((uintptr_t) i[10] + input_offset);
      const __fp16* ibl5 = (const __fp16*) ((uintptr_t) i[11] + input_offset);
      const __fp16* itl6 = (const __fp16*) ((uintptr_t) i[12] + input_offset);
      const __fp16* ibl6 = (const __fp16*) ((uintptr_t) i[13] + input_offset);
      const __fp16* itl7 = (const __fp16*) ((uintptr_t) i[14] + input_offset);
      const __fp16* ibl7 = (const __fp16*) ((uintptr_t) i[15] + input_offset);
      const __fp16* itl8 = (const __fp16*) ((uintptr_t) i[16] + input_offset);
      const __fp16* ibl8 = (const __fp16*) ((uintptr_t) i[17] + input_offset);
      const __fp16* itl9 = (const __fp16*) ((uintptr_t) i[18] + input_offset);
      const __fp16* ibl9 = (const __fp16*) ((uintptr_t) i[19] + input_offset);
      const __fp16* itlA = (const __fp16*) ((uintptr_t) i[20] + input_offset);
      const __fp16* iblA = (const __fp16*) ((uintptr_t) i[21] + input_offset);
      const __fp16* itlB = (const __fp16*) ((uintptr_t) i[22] + input_offset);
      const __fp16* iblB = (const __fp16*) ((uintptr_t) i[23] + input_offset);
      const __fp16* itlC = (const __fp16*) ((uintptr_t) i[24] + input_offset);
      const __fp16* iblC = (const __fp16*) ((uintptr_t) i[25] + input_offset);
      const __fp16* itlD = (const __fp16*) ((uintptr_t) i[26] + input_offset);
      const __fp16* iblD = (const __fp16*) ((uintptr_t) i[27] + input_offset);
      const __fp16* itlE = (const __fp16*) ((uintptr_t) i[28] + input_offset);
      const __fp16* iblE = (const __fp16*) ((uintptr_t) i[29] + input_offset);
      const __fp16* itlF = (const __fp16*) ((uintptr_t) i[30] + input_offset);
      const __fp16* iblF = (const __fp16*) ((uintptr_t) i[31] + input_offset);
      i += 2 * 16;

      const float16x4x2_t vw0123 = vld2_f16(w + 0);
      const float16x4x2_t vw4567 = vld2_f16(w + 8);
      const float16x4x2_t vw89AB = vld2_f16(w + 16);
      const float16x4x2_t vwCDEF = vld2_f16(w + 24);
      w += 2 * 16;

      float16x8_t vtltr0123 = vmovq_n_f16(0);  // vmov for uninitialized var warning
      float16x8_t vblbr0123 = vmovq_n_f16(0);
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl0, vreinterpretq_u32_f16(vtltr0123), 0));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl0, vreinterpretq_u32_f16(vblbr0123), 0));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr0123), 1));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr0123), 1));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr0123), 2));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr0123), 2));
      vtltr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr0123), 3));
      vblbr0123 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr0123), 3));
      float16x8_t vtltr4567 = vmovq_n_f16(0);  // vmov for uninitialized var warning
      float16x8_t vblbr4567 = vmovq_n_f16(0);
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl4, vreinterpretq_u32_f16(vtltr4567), 0));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl4, vreinterpretq_u32_f16(vblbr4567), 0));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl5, vreinterpretq_u32_f16(vtltr4567), 1));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl5, vreinterpretq_u32_f16(vblbr4567), 1));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl6, vreinterpretq_u32_f16(vtltr4567), 2));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl6, vreinterpretq_u32_f16(vblbr4567), 2));
      vtltr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl7, vreinterpretq_u32_f16(vtltr4567), 3));
      vblbr4567 = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl7, vreinterpretq_u32_f16(vblbr4567), 3));
      float16x8_t vtltr89AB = vmovq_n_f16(0);  // vmov for uninitialized var warning
      float16x8_t vblbr89AB = vmovq_n_f16(0);
      vtltr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl8, vreinterpretq_u32_f16(vtltr89AB), 0));
      vblbr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl8, vreinterpretq_u32_f16(vblbr89AB), 0));
      vtltr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl9, vreinterpretq_u32_f16(vtltr89AB), 1));
      vblbr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl9, vreinterpretq_u32_f16(vblbr89AB), 1));
      vtltr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlA, vreinterpretq_u32_f16(vtltr89AB), 2));
      vblbr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblA, vreinterpretq_u32_f16(vblbr89AB), 2));
      vtltr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlB, vreinterpretq_u32_f16(vtltr89AB), 3));
      vblbr89AB = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblB, vreinterpretq_u32_f16(vblbr89AB), 3));
      float16x8_t vtltrCDEF = vmovq_n_f16(0);  // vmov for uninitialized var warning
      float16x8_t vblbrCDEF = vmovq_n_f16(0);
      vtltrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlC, vreinterpretq_u32_f16(vtltrCDEF), 0));
      vblbrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblC, vreinterpretq_u32_f16(vblbrCDEF), 0));
      vtltrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlD, vreinterpretq_u32_f16(vtltrCDEF), 1));
      vblbrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblD, vreinterpretq_u32_f16(vblbrCDEF), 1));
      vtltrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlE, vreinterpretq_u32_f16(vtltrCDEF), 2));
      vblbrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblE, vreinterpretq_u32_f16(vblbrCDEF), 2));
      vtltrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itlF, vreinterpretq_u32_f16(vtltrCDEF), 3));
      vblbrCDEF = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) iblF, vreinterpretq_u32_f16(vblbrCDEF), 3));

      const float16x8_t valphah01234567 = vcombine_f16(vw0123.val[0], vw4567.val[0]);
      const float16x8_t valphav01234567 = vcombine_f16(vw0123.val[1], vw4567.val[1]);
      const float16x8_t valphah89ABCDEF = vcombine_f16(vw89AB.val[0], vwCDEF.val[0]);
      const float16x8_t valphav89ABCDEF = vcombine_f16(vw89AB.val[1], vwCDEF.val[1]);

      const float16x8_t vldrd0123 = vsubq_f16(vblbr0123, vtltr0123);
      const float16x8_t vldrd4567 = vsubq_f16(vblbr4567, vtltr4567);
      const float16x8_t vldrd89AB = vsubq_f16(vblbr89AB, vtltr89AB);
      const float16x8_t vldrdCDEF = vsubq_f16(vblbrCDEF, vtltrCDEF);

      const float16x8x2_t vld_t01234567 = vuzpq_f16(vldrd0123, vldrd4567);
      const float16x8_t vld01234567 = vld_t01234567.val[0];
      const float16x8_t vrd01234567 = vld_t01234567.val[1];
      const float16x8x2_t vld_t89ABCDEF = vuzpq_f16(vldrd89AB, vldrdCDEF);
      const float16x8_t vld89ABCDEF = vld_t89ABCDEF.val[0];
      const float16x8_t vrd89ABCDEF = vld_t89ABCDEF.val[1];

      const float16x8x2_t vtl_t01234567 = vuzpq_f16(vtltr0123, vtltr4567);
      const float16x8_t vtl01234567 = vtl_t01234567.val[0];
      const float16x8_t vtr01234567 = vtl_t01234567.val[1];
      const float16x8x2_t vtl_t89ABCDEF = vuzpq_f16(vtltr89AB, vtltrCDEF);
      const float16x8_t vtl89ABCDEF = vtl_t89ABCDEF.val[0];
      const float16x8_t vtr89ABCDEF = vtl_t89ABCDEF.val[1];

      const float16x8_t vl01234567 = vfmaq_f16(vtl01234567, vld01234567, valphav01234567);
      const float16x8_t vr01234567 = vfmaq_f16(vtr01234567, vrd01234567, valphav01234567);
      const float16x8_t vl89ABCDEF = vfmaq_f16(vtl89ABCDEF, vld89ABCDEF, valphav89ABCDEF);
      const float16x8_t vr89ABCDEF = vfmaq_f16(vtr89ABCDEF, vrd89ABCDEF, valphav89ABCDEF);

      const float16x8_t vd01234567 = vsubq_f16(vr01234567, vl01234567);
      const float16x8_t vd89ABCDEF = vsubq_f16(vr89ABCDEF, vl89ABCDEF);
      const float16x8_t vo01234567 = vfmaq_f16(vl01234567, vd01234567, valphah01234567);
      const float16x8_t vo89ABCDEF = vfmaq_f16(vl89ABCDEF, vd89ABCDEF, valphah89ABCDEF);

      vst1q_f16(o + 0, vo01234567);
      vst1q_f16(o + 8, vo89ABCDEF);
      o += 16;
    }

    for (; p >= 4; p -= 4) {
      const __fp16* itl0 = (const __fp16*) ((uintptr_t) i[0] + input_offset);
      const __fp16* ibl0 = (const __fp16*) ((uintptr_t) i[1] + input_offset);
      const __fp16* itl1 = (const __fp16*) ((uintptr_t) i[2] + input_offset);
      const __fp16* ibl1 = (const __fp16*) ((uintptr_t) i[3] + input_offset);
      const __fp16* itl2 = (const __fp16*) ((uintptr_t) i[4] + input_offset);
      const __fp16* ibl2 = (const __fp16*) ((uintptr_t) i[5] + input_offset);
      const __fp16* itl3 = (const __fp16*) ((uintptr_t) i[6] + input_offset);
      const __fp16* ibl3 = (const __fp16*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const float16x4x2_t vw = vld2_f16(w);
      w += 8;

      float16x8_t vtltr = vmovq_n_f16(0);  // vmov for uninitialized var warning
      float16x8_t vblbr = vmovq_n_f16(0);
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl0, vreinterpretq_u32_f16(vtltr), 0));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl0, vreinterpretq_u32_f16(vblbr), 0));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl1, vreinterpretq_u32_f16(vtltr), 1));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl1, vreinterpretq_u32_f16(vblbr), 1));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl2, vreinterpretq_u32_f16(vtltr), 2));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl2, vreinterpretq_u32_f16(vblbr), 2));
      vtltr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) itl3, vreinterpretq_u32_f16(vtltr), 3));
      vblbr = vreinterpretq_f16_u32(vld1q_lane_u32((const void*) ibl3, vreinterpretq_u32_f16(vblbr), 3));

      const float16x4_t valphah = vw.val[0];
      const float16x4_t valphav = vw.val[1];

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

      vst1_f16(o, vo);
      o += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const __fp16* itl0 = (const __fp16*) ((uintptr_t) i[0] + input_offset);
        const __fp16* ibl0 = (const __fp16*) ((uintptr_t) i[1] + input_offset);
        const __fp16* itl1 = (const __fp16*) ((uintptr_t) i[2] + input_offset);
        const __fp16* ibl1 = (const __fp16*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float16x4_t vw = vld1_f16(w);
        w += 4;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        float16x4_t vtltr = vmov_n_f16(0);  // vmov for uninitialized var warning
        float16x4_t vblbr = vmov_n_f16(0);

        vtltr = vreinterpret_f16_u32(vld1_lane_u32((const void*) itl0, vreinterpret_u32_f16(vtltr), 0));
        vblbr = vreinterpret_f16_u32(vld1_lane_u32((const void*) ibl0, vreinterpret_u32_f16(vblbr), 0));
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

        vst1_lane_u32((void*) o, vreinterpret_u32_f16(vo), 0);
        o += 2;
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

        const __fp16* itl = (const __fp16*) ((uintptr_t) i[0] + input_offset);
        const __fp16* ibl = (const __fp16*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        float16x4_t vw = vmov_n_f16(0);
        vw = vreinterpret_f16_u32(vld1_lane_u32((const void*) w, vreinterpret_u32_f16(vw), 0));
        w += 2;

        const float16x4x2_t vwhv = vuzp_f16(vw, vw);
        const float16x4_t valphah = vwhv.val[0];
        const float16x4_t valphav = vwhv.val[1];

        float16x4_t vtltr = vmov_n_f16(0);  // vmov for uninitialized var warning
        float16x4_t vblbr = vmov_n_f16(0);

        vtltr = vreinterpret_f16_u32(vld1_lane_u32((const void*) itl, vreinterpret_u32_f16(vtltr), 0));
        vblbr = vreinterpret_f16_u32(vld1_lane_u32((const void*) ibl, vreinterpret_u32_f16(vblbr), 0));

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

        vst1_lane_f16(o, vo, 0);
        o += 1;
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
