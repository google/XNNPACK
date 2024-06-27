// Auto-generated file. Do not edit!
//   Template: src/s8-ibilinear/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"


void xnn_u8_ibilinear_ukernel__neon_c16(
    size_t output_pixels,
    size_t channels,
    const uint8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    uint8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    #if XNN_ARCH_ARM64
      const int16x8_t valphah = vld1q_dup_s16(weights); weights += 1;
    #else
      const int16x4_t valphah = vld1_dup_s16(weights); weights += 1;
    #endif
    const int32x4_t valphav = vmovl_s16(vld1_dup_s16(weights)); weights += 1;

    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const uint8x8_t vtl01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vtr01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vbl01234567 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vbr01234567 = vld1_u8(i3); i3 += 8;
      const uint8x8_t vtl89ABCDEF = vld1_u8(i0); i0 += 8;
      const uint8x8_t vtr89ABCDEF = vld1_u8(i1); i1 += 8;
      const uint8x8_t vbl89ABCDEF = vld1_u8(i2); i2 += 8;
      const uint8x8_t vbr89ABCDEF = vld1_u8(i3); i3 += 8;

      const int16x8_t vtd01234567 = vreinterpretq_s16_u16(vsubl_u8(vtr01234567, vtl01234567));
      const int16x8_t vbd01234567 = vreinterpretq_s16_u16(vsubl_u8(vbr01234567, vbl01234567));
      const int16x8_t vdl01234567 = vreinterpretq_s16_u16(vsubl_u8(vbl01234567, vtl01234567));
      const int16x8_t vxtl01234567 = vreinterpretq_s16_u16(vmovl_u8(vtl01234567));
      const int16x8_t vtd89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vtr89ABCDEF, vtl89ABCDEF));
      const int16x8_t vbd89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vbr89ABCDEF, vbl89ABCDEF));
      const int16x8_t vdl89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(vbl89ABCDEF, vtl89ABCDEF));
      const int16x8_t vxtl89ABCDEF = vreinterpretq_s16_u16(vmovl_u8(vtl89ABCDEF));

      const int16x8_t vdd01234567 = vsubq_s16(vbd01234567, vtd01234567);
      const int16x8_t vdd89ABCDEF = vsubq_s16(vbd89ABCDEF, vtd89ABCDEF);

      #if XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), vget_low_s16(valphah));
        const int32x4_t vt4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vtd01234567, valphah);
        const int32x4_t vt89AB = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl89ABCDEF), 11), vget_low_s16(vtd89ABCDEF), vget_low_s16(valphah));
        const int32x4_t vtCDEF = vmlal_high_s16(vshll_n_s16(vget_high_s16(vxtl89ABCDEF), 11), vtd89ABCDEF, valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), vget_low_s16(valphah));
        const int32x4_t vd4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vdd01234567, valphah);
        const int32x4_t vd89AB = vmlal_s16(vshll_n_s16(vget_low_s16(vdl89ABCDEF), 11), vget_low_s16(vdd89ABCDEF), vget_low_s16(valphah));
        const int32x4_t vdCDEF = vmlal_high_s16(vshll_n_s16(vget_high_s16(vdl89ABCDEF), 11), vdd89ABCDEF, valphah);
      #else  // !XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), valphah);
        const int32x4_t vt4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vget_high_s16(vtd01234567), valphah);
        const int32x4_t vt89AB = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl89ABCDEF), 11), vget_low_s16(vtd89ABCDEF), valphah);
        const int32x4_t vtCDEF = vmlal_s16(vshll_n_s16(vget_high_s16(vxtl89ABCDEF), 11), vget_high_s16(vtd89ABCDEF), valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), valphah);
        const int32x4_t vd4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vget_high_s16(vdd01234567), valphah);
        const int32x4_t vd89AB = vmlal_s16(vshll_n_s16(vget_low_s16(vdl89ABCDEF), 11), vget_low_s16(vdd89ABCDEF), valphah);
        const int32x4_t vdCDEF = vmlal_s16(vshll_n_s16(vget_high_s16(vdl89ABCDEF), 11), vget_high_s16(vdd89ABCDEF), valphah);
      #endif  // !XNN_ARCH_ARM64

      const int32x4_t vacc0123 = vmlaq_s32(vshlq_n_s32(vt0123, 11), vd0123, valphav);
      const int32x4_t vacc4567 = vmlaq_s32(vshlq_n_s32(vt4567, 11), vd4567, valphav);
      const int32x4_t vacc89AB = vmlaq_s32(vshlq_n_s32(vt89AB, 11), vd89AB, valphav);
      const int32x4_t vaccCDEF = vmlaq_s32(vshlq_n_s32(vtCDEF, 11), vdCDEF, valphav);

      #if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vuzp2q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
        const int16x8_t vacc89ABCDEF = vuzp2q_s16(vreinterpretq_s16_s32(vacc89AB), vreinterpretq_s16_s32(vaccCDEF));
      #else  // !XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vcombine_s16(vshrn_n_s32(vacc0123, 16), vshrn_n_s32(vacc4567, 16));
        const int16x8_t vacc89ABCDEF = vcombine_s16(vshrn_n_s32(vacc89AB, 16), vshrn_n_s32(vaccCDEF, 16));
      #endif  // !XNN_ARCH_ARM64

      const uint8x8_t vo01234567 = vrshrn_n_u16(vreinterpretq_u16_s16(vacc01234567), 6);
      const uint8x8_t vo89ABCDEF = vrshrn_n_u16(vreinterpretq_u16_s16(vacc89ABCDEF), 6);

      vst1_u8(output, vo01234567); output += 8;
      vst1_u8(output, vo89ABCDEF); output += 8;
    }
    for (; c >= 8 * sizeof(uint8_t); c -= 8 * sizeof(uint8_t)) {
      const uint8x8_t vtl01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vtr01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vbl01234567 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vbr01234567 = vld1_u8(i3); i3 += 8;

      const int16x8_t vtd01234567 = vreinterpretq_s16_u16(vsubl_u8(vtr01234567, vtl01234567));
      const int16x8_t vbd01234567 = vreinterpretq_s16_u16(vsubl_u8(vbr01234567, vbl01234567));
      const int16x8_t vdl01234567 = vreinterpretq_s16_u16(vsubl_u8(vbl01234567, vtl01234567));
      const int16x8_t vxtl01234567 = vreinterpretq_s16_u16(vmovl_u8(vtl01234567));

      const int16x8_t vdd01234567 = vsubq_s16(vbd01234567, vtd01234567);

      #if XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), vget_low_s16(valphah));
        const int32x4_t vt4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vtd01234567, valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), vget_low_s16(valphah));
        const int32x4_t vd4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vdd01234567, valphah);
      #else  // !XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), valphah);
        const int32x4_t vt4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vget_high_s16(vtd01234567), valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), valphah);
        const int32x4_t vd4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vget_high_s16(vdd01234567), valphah);
      #endif  // !XNN_ARCH_ARM64

      const int32x4_t vacc0123 = vmlaq_s32(vshlq_n_s32(vt0123, 11), vd0123, valphav);
      const int32x4_t vacc4567 = vmlaq_s32(vshlq_n_s32(vt4567, 11), vd4567, valphav);

      #if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vuzp2q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
      #else  // !XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vcombine_s16(vshrn_n_s32(vacc0123, 16), vshrn_n_s32(vacc4567, 16));
      #endif  // !XNN_ARCH_ARM64

      const uint8x8_t vo01234567 = vrshrn_n_u16(vreinterpretq_u16_s16(vacc01234567), 6);

      vst1_u8(output, vo01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8x8_t vtl01234567 = vld1_u8(i0);
      const uint8x8_t vtr01234567 = vld1_u8(i1);
      const uint8x8_t vbl01234567 = vld1_u8(i2);
      const uint8x8_t vbr01234567 = vld1_u8(i3);

      const int16x8_t vtd01234567 = vreinterpretq_s16_u16(vsubl_u8(vtr01234567, vtl01234567));
      const int16x8_t vbd01234567 = vreinterpretq_s16_u16(vsubl_u8(vbr01234567, vbl01234567));
      const int16x8_t vdl01234567 = vreinterpretq_s16_u16(vsubl_u8(vbl01234567, vtl01234567));
      const int16x8_t vxtl01234567 = vreinterpretq_s16_u16(vmovl_u8(vtl01234567));

      const int16x8_t vdd01234567 = vsubq_s16(vbd01234567, vtd01234567);

      #if XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), vget_low_s16(valphah));
        const int32x4_t vt4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vtd01234567, valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), vget_low_s16(valphah));
        const int32x4_t vd4567 = vmlal_high_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vdd01234567, valphah);
      #else  // !XNN_ARCH_ARM64
        const int32x4_t vt0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vxtl01234567), 11), vget_low_s16(vtd01234567), valphah);
        const int32x4_t vt4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vxtl01234567), 11), vget_high_s16(vtd01234567), valphah);

        const int32x4_t vd0123 = vmlal_s16(vshll_n_s16(vget_low_s16(vdl01234567), 11), vget_low_s16(vdd01234567), valphah);
        const int32x4_t vd4567 = vmlal_s16(vshll_n_s16(vget_high_s16(vdl01234567), 11), vget_high_s16(vdd01234567), valphah);
      #endif  // !XNN_ARCH_ARM64

      const int32x4_t vacc0123 = vmlaq_s32(vshlq_n_s32(vt0123, 11), vd0123, valphav);
      const int32x4_t vacc4567 = vmlaq_s32(vshlq_n_s32(vt4567, 11), vd4567, valphav);

      #if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vuzp2q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
      #else  // !XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vcombine_s16(vshrn_n_s32(vacc0123, 16), vshrn_n_s32(vacc4567, 16));
      #endif  // !XNN_ARCH_ARM64

      uint8x8_t vo01234567 = vrshrn_n_u16(vreinterpretq_u16_s16(vacc01234567), 6);

      if (c & (4 * sizeof(uint8_t))) {
        vst1_lane_u32((void*) output, vreinterpret_u32_u8(vo01234567), 0); output += 4;
        vo01234567 = vext_u8(vo01234567, vo01234567, 4);
      }
      if (c & (2 * sizeof(uint8_t))) {
        vst1_lane_u16((void*) output, vreinterpret_u16_u8(vo01234567), 0); output += 2;
        vo01234567 = vext_u8(vo01234567, vo01234567, 2);
      }
      if (c & (1 * sizeof(uint8_t))) {
        vst1_lane_u8(output, vo01234567, 0); output += 1;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
