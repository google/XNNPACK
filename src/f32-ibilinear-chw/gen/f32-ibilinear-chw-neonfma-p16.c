// Auto-generated file. Do not edit!
//   Template: src/f32-ibilinear-chw/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/ibilinear.h"


void xnn_f32_ibilinear_chw_ukernel__neonfma_p16(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t input_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(input_increment % sizeof(float) == 0);

  do {
    const float** i = input;
    const float* w = weights;
    size_t p = output_pixels;
    for (; p >= 16; p -= 16) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      const float* itl4 = (const float*) ((uintptr_t) i[8] + input_offset);
      const float* ibl4 = (const float*) ((uintptr_t) i[9] + input_offset);
      const float* itl5 = (const float*) ((uintptr_t) i[10] + input_offset);
      const float* ibl5 = (const float*) ((uintptr_t) i[11] + input_offset);
      const float* itl6 = (const float*) ((uintptr_t) i[12] + input_offset);
      const float* ibl6 = (const float*) ((uintptr_t) i[13] + input_offset);
      const float* itl7 = (const float*) ((uintptr_t) i[14] + input_offset);
      const float* ibl7 = (const float*) ((uintptr_t) i[15] + input_offset);
      const float* itl8 = (const float*) ((uintptr_t) i[16] + input_offset);
      const float* ibl8 = (const float*) ((uintptr_t) i[17] + input_offset);
      const float* itl9 = (const float*) ((uintptr_t) i[18] + input_offset);
      const float* ibl9 = (const float*) ((uintptr_t) i[19] + input_offset);
      const float* itlA = (const float*) ((uintptr_t) i[20] + input_offset);
      const float* iblA = (const float*) ((uintptr_t) i[21] + input_offset);
      const float* itlB = (const float*) ((uintptr_t) i[22] + input_offset);
      const float* iblB = (const float*) ((uintptr_t) i[23] + input_offset);
      const float* itlC = (const float*) ((uintptr_t) i[24] + input_offset);
      const float* iblC = (const float*) ((uintptr_t) i[25] + input_offset);
      const float* itlD = (const float*) ((uintptr_t) i[26] + input_offset);
      const float* iblD = (const float*) ((uintptr_t) i[27] + input_offset);
      const float* itlE = (const float*) ((uintptr_t) i[28] + input_offset);
      const float* iblE = (const float*) ((uintptr_t) i[29] + input_offset);
      const float* itlF = (const float*) ((uintptr_t) i[30] + input_offset);
      const float* iblF = (const float*) ((uintptr_t) i[31] + input_offset);
      i += 2 * 16;

      const float32x4x2_t vw0123 = vld2q_f32(w + 0);
      const float32x4x2_t vw4567 = vld2q_f32(w + 8);
      const float32x4x2_t vw89AB = vld2q_f32(w + 16);
      const float32x4x2_t vwCDEF = vld2q_f32(w + 24);
      w += 2 * 16;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);
      const float32x2_t vtltr4 = vld1_f32(itl4);
      const float32x2_t vblbr4 = vld1_f32(ibl4);
      const float32x2_t vtltr5 = vld1_f32(itl5);
      const float32x2_t vblbr5 = vld1_f32(ibl5);
      const float32x2_t vtltr6 = vld1_f32(itl6);
      const float32x2_t vblbr6 = vld1_f32(ibl6);
      const float32x2_t vtltr7 = vld1_f32(itl7);
      const float32x2_t vblbr7 = vld1_f32(ibl7);
      const float32x2_t vtltr8 = vld1_f32(itl8);
      const float32x2_t vblbr8 = vld1_f32(ibl8);
      const float32x2_t vtltr9 = vld1_f32(itl9);
      const float32x2_t vblbr9 = vld1_f32(ibl9);
      const float32x2_t vtltrA = vld1_f32(itlA);
      const float32x2_t vblbrA = vld1_f32(iblA);
      const float32x2_t vtltrB = vld1_f32(itlB);
      const float32x2_t vblbrB = vld1_f32(iblB);
      const float32x2_t vtltrC = vld1_f32(itlC);
      const float32x2_t vblbrC = vld1_f32(iblC);
      const float32x2_t vtltrD = vld1_f32(itlD);
      const float32x2_t vblbrD = vld1_f32(iblD);
      const float32x2_t vtltrE = vld1_f32(itlE);
      const float32x2_t vblbrE = vld1_f32(iblE);
      const float32x2_t vtltrF = vld1_f32(itlF);
      const float32x2_t vblbrF = vld1_f32(iblF);

      const float32x4_t valphah0123 = vw0123.val[0];
      const float32x4_t valphav0123 = vw0123.val[1];
      const float32x4_t valphah4567 = vw4567.val[0];
      const float32x4_t valphav4567 = vw4567.val[1];
      const float32x4_t valphah89AB = vw89AB.val[0];
      const float32x4_t valphav89AB = vw89AB.val[1];
      const float32x4_t valphahCDEF = vwCDEF.val[0];
      const float32x4_t valphavCDEF = vwCDEF.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);
      const float32x4_t vtltr45 = vcombine_f32(vtltr4, vtltr5);
      const float32x4_t vblbr45 = vcombine_f32(vblbr4, vblbr5);
      const float32x4_t vtltr67 = vcombine_f32(vtltr6, vtltr7);
      const float32x4_t vblbr67 = vcombine_f32(vblbr6, vblbr7);
      const float32x4_t vtltr89 = vcombine_f32(vtltr8, vtltr9);
      const float32x4_t vblbr89 = vcombine_f32(vblbr8, vblbr9);
      const float32x4_t vtltrAB = vcombine_f32(vtltrA, vtltrB);
      const float32x4_t vblbrAB = vcombine_f32(vblbrA, vblbrB);
      const float32x4_t vtltrCD = vcombine_f32(vtltrC, vtltrD);
      const float32x4_t vblbrCD = vcombine_f32(vblbrC, vblbrD);
      const float32x4_t vtltrEF = vcombine_f32(vtltrE, vtltrF);
      const float32x4_t vblbrEF = vcombine_f32(vblbrE, vblbrF);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);
      const float32x4_t vldrd45 = vsubq_f32(vblbr45, vtltr45);
      const float32x4_t vldrd67 = vsubq_f32(vblbr67, vtltr67);
      const float32x4_t vldrd89 = vsubq_f32(vblbr89, vtltr89);
      const float32x4_t vldrdAB = vsubq_f32(vblbrAB, vtltrAB);
      const float32x4_t vldrdCD = vsubq_f32(vblbrCD, vtltrCD);
      const float32x4_t vldrdEF = vsubq_f32(vblbrEF, vtltrEF);

      const float32x4x2_t vld_t0123 = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld0123 = vld_t0123.val[0];
      const float32x4_t vrd0123 = vld_t0123.val[1];
      const float32x4x2_t vld_t4567 = vuzpq_f32(vldrd45, vldrd67);
      const float32x4_t vld4567 = vld_t4567.val[0];
      const float32x4_t vrd4567 = vld_t4567.val[1];
      const float32x4x2_t vld_t89AB = vuzpq_f32(vldrd89, vldrdAB);
      const float32x4_t vld89AB = vld_t89AB.val[0];
      const float32x4_t vrd89AB = vld_t89AB.val[1];
      const float32x4x2_t vld_tCDEF = vuzpq_f32(vldrdCD, vldrdEF);
      const float32x4_t vldCDEF = vld_tCDEF.val[0];
      const float32x4_t vrdCDEF = vld_tCDEF.val[1];

      const float32x4x2_t vtl_t0123 = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl0123 = vtl_t0123.val[0];
      const float32x4_t vtr0123 = vtl_t0123.val[1];
      const float32x4x2_t vtl_t4567 = vuzpq_f32(vtltr45, vtltr67);
      const float32x4_t vtl4567 = vtl_t4567.val[0];
      const float32x4_t vtr4567 = vtl_t4567.val[1];
      const float32x4x2_t vtl_t89AB = vuzpq_f32(vtltr89, vtltrAB);
      const float32x4_t vtl89AB = vtl_t89AB.val[0];
      const float32x4_t vtr89AB = vtl_t89AB.val[1];
      const float32x4x2_t vtl_tCDEF = vuzpq_f32(vtltrCD, vtltrEF);
      const float32x4_t vtlCDEF = vtl_tCDEF.val[0];
      const float32x4_t vtrCDEF = vtl_tCDEF.val[1];

      const float32x4_t vl0123 = vfmaq_f32(vtl0123, vld0123, valphav0123);
      const float32x4_t vr0123 = vfmaq_f32(vtr0123, vrd0123, valphav0123);
      const float32x4_t vl4567 = vfmaq_f32(vtl4567, vld4567, valphav4567);
      const float32x4_t vr4567 = vfmaq_f32(vtr4567, vrd4567, valphav4567);
      const float32x4_t vl89AB = vfmaq_f32(vtl89AB, vld89AB, valphav89AB);
      const float32x4_t vr89AB = vfmaq_f32(vtr89AB, vrd89AB, valphav89AB);
      const float32x4_t vlCDEF = vfmaq_f32(vtlCDEF, vldCDEF, valphavCDEF);
      const float32x4_t vrCDEF = vfmaq_f32(vtrCDEF, vrdCDEF, valphavCDEF);

      const float32x4_t vd0123 = vsubq_f32(vr0123, vl0123);
      const float32x4_t vd4567 = vsubq_f32(vr4567, vl4567);
      const float32x4_t vd89AB = vsubq_f32(vr89AB, vl89AB);
      const float32x4_t vdCDEF = vsubq_f32(vrCDEF, vlCDEF);

      const float32x4_t vo0123 = vfmaq_f32(vl0123, vd0123, valphah0123);
      const float32x4_t vo4567 = vfmaq_f32(vl4567, vd4567, valphah4567);
      const float32x4_t vo89AB = vfmaq_f32(vl89AB, vd89AB, valphah89AB);
      const float32x4_t voCDEF = vfmaq_f32(vlCDEF, vdCDEF, valphahCDEF);

      vst1q_f32(output + 0, vo0123);
      vst1q_f32(output + 4, vo4567);
      vst1q_f32(output + 8, vo89AB);
      vst1q_f32(output + 12, voCDEF);
      output += 16;
    }

    for (; p >= 4; p -= 4) {
      const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
      const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
      const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
      const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
      const float* itl2 = (const float*) ((uintptr_t) i[4] + input_offset);
      const float* ibl2 = (const float*) ((uintptr_t) i[5] + input_offset);
      const float* itl3 = (const float*) ((uintptr_t) i[6] + input_offset);
      const float* ibl3 = (const float*) ((uintptr_t) i[7] + input_offset);
      i += 8;

      const float32x4x2_t vw = vld2q_f32(w);
      w += 8;

      const float32x2_t vtltr0 = vld1_f32(itl0);
      const float32x2_t vblbr0 = vld1_f32(ibl0);
      const float32x2_t vtltr1 = vld1_f32(itl1);
      const float32x2_t vblbr1 = vld1_f32(ibl1);
      const float32x2_t vtltr2 = vld1_f32(itl2);
      const float32x2_t vblbr2 = vld1_f32(ibl2);
      const float32x2_t vtltr3 = vld1_f32(itl3);
      const float32x2_t vblbr3 = vld1_f32(ibl3);

      const float32x4_t valphah = vw.val[0];
      const float32x4_t valphav = vw.val[1];

      const float32x4_t vtltr01 = vcombine_f32(vtltr0, vtltr1);
      const float32x4_t vblbr01 = vcombine_f32(vblbr0, vblbr1);
      const float32x4_t vtltr23 = vcombine_f32(vtltr2, vtltr3);
      const float32x4_t vblbr23 = vcombine_f32(vblbr2, vblbr3);

      const float32x4_t vldrd01 = vsubq_f32(vblbr01, vtltr01);
      const float32x4_t vldrd23 = vsubq_f32(vblbr23, vtltr23);

      const float32x4x2_t vld_t = vuzpq_f32(vldrd01, vldrd23);
      const float32x4_t vld = vld_t.val[0];
      const float32x4_t vrd = vld_t.val[1];

      const float32x4x2_t vtl_t = vuzpq_f32(vtltr01, vtltr23);
      const float32x4_t vtl = vtl_t.val[0];
      const float32x4_t vtr = vtl_t.val[1];

      const float32x4_t vl = vfmaq_f32(vtl, vld, valphav);
      const float32x4_t vr = vfmaq_f32(vtr, vrd, valphav);

      const float32x4_t vd = vsubq_f32(vr, vl);
      const float32x4_t vo = vfmaq_f32(vl, vd, valphah);

      vst1q_f32(output, vo);
      output += 4;
    }

    if XNN_UNLIKELY(p != 0) {
      if (p & 2) {
        const float32x2x2_t vw = vld2_f32(w);
        w += 4;

        const float32x2_t valphah = vw.val[0];
        const float32x2_t valphav = vw.val[1];

        const float* itl0 = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl0 = (const float*) ((uintptr_t) i[1] + input_offset);
        const float* itl1 = (const float*) ((uintptr_t) i[2] + input_offset);
        const float* ibl1 = (const float*) ((uintptr_t) i[3] + input_offset);
        i += 4;

        const float32x2_t vtltr0 = vld1_f32(itl0);
        const float32x2_t vblbr0 = vld1_f32(ibl0);
        const float32x2_t vtltr1 = vld1_f32(itl1);
        const float32x2_t vblbr1 = vld1_f32(ibl1);

        const float32x2_t vldrd0 = vsub_f32(vblbr0, vtltr0);
        const float32x2_t vldrd1 = vsub_f32(vblbr1, vtltr1);

        const float32x2x2_t vld_t = vuzp_f32(vldrd0, vldrd1);
        const float32x2_t vld = vld_t.val[0];
        const float32x2_t vrd = vld_t.val[1];

        const float32x2x2_t vtl_t = vuzp_f32(vtltr0, vtltr1);
        const float32x2_t vtl = vtl_t.val[0];
        const float32x2_t vtr = vtl_t.val[1];

        const float32x2_t vl = vfma_f32(vtl, vld, valphav);
        const float32x2_t vr = vfma_f32(vtr, vrd, valphav);

        const float32x2_t vd = vsub_f32(vr, vl);
        const float32x2_t vo = vfma_f32(vl, vd, valphah);

        vst1_f32(output, vo);
        output += 2;
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

        const float alphah = *w;
        const float32x2_t valphav = vld1_dup_f32(w + 1);
        w += 2;

        const float* itl = (const float*) ((uintptr_t) i[0] + input_offset);
        const float* ibl = (const float*) ((uintptr_t) i[1] + input_offset);
        i += 2;

        const float32x2_t vtltr = vld1_f32(itl);
        const float32x2_t vblbr = vld1_f32(ibl);

        // Compute at once
        //    left_diff = bottom_left  - top_left
        //   right_diff = bottom_right - top_right
        const float32x2_t vldrd = vsub_f32(vblbr, vtltr);
        const float32x2_t vlr = vfma_f32(vtltr, vldrd, valphav);

        // Extract them and compute the result.
        const float l = vget_lane_f32(vlr, 0);
        const float r = vget_lane_f32(vlr, 1);

        *output++ = l + alphah * (r - l);
      }
    }

    input_offset += input_increment;
  } while (--channels != 0);
}
