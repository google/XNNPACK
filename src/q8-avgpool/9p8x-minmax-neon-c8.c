// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/avgpool.h>
#include <xnnpack/common.h>


void xnn_q8_avgpool_minmax_ukernel_9p8x__neon_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_q8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const int32x4_t vbias = vld1q_dup_s32(&params->neon.bias);
#if XNN_ARCH_ARM64
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
#else
  const int32x2_t vmultiplier = vld1_dup_s32(&params->neon.multiplier);
#endif
  const int64x2_t vleft_shift = vld1q_dup_s64(&params->neon.left_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->neon.output_max);

  do {
    {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
        const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
        const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
        const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
        const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
        const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
        const uint8x8_t vi8 = vld1_u8(i8); i8 += 8;

        const uint16x8_t vsum018 = vaddw_u8(vaddl_u8(vi0, vi1), vi8);
        const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
        const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
        const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

        const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);
        const uint16x8_t vsum01678 = vaddq_u16(vsum018, vsum67);
        const uint16x8_t vsum = vaddq_u16(vsum2345, vsum01678);

        const int32x4_t vacc_lo = vaddw_s16(vbias, vreinterpret_s16_u16(vget_low_u16(vsum)));
        const int32x4_t vacc_hi = vaddw_s16(vbias, vreinterpret_s16_u16(vget_high_u16(vsum)));

        vst1q_s32(b, vacc_lo); b += 4;
        vst1q_s32(b, vacc_hi); b += 4;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
        const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
        const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
        const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
        const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
        const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
        int32x4_t vacc_lo = vld1q_s32(b);
        int32x4_t vacc_hi = vld1q_s32(b + 4);

        const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);
        const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
        const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);
        const uint16x8_t vsum67 = vaddl_u8(vi6, vi7);

        const uint16x8_t vsum0123 = vaddq_u16(vsum01, vsum23);
        const uint16x8_t vsum4567 = vaddq_u16(vsum45, vsum67);
        const uint16x8_t vsum = vaddq_u16(vsum0123, vsum4567);

        vacc_lo = vaddw_s16(vacc_lo, vreinterpret_s16_u16(vget_low_u16(vsum)));
        vacc_hi = vaddw_s16(vacc_hi, vreinterpret_s16_u16(vget_high_u16(vsum)));

        vst1q_s32(b, vacc_lo); b += 4;
        vst1q_s32(b, vacc_hi); b += 4;
      }
    }

    {
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      int32_t* b = buffer;
      while (c >= 8) {
        const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
        const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
        const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
        const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
        const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
        const uint8x8_t vi7 = vld1_u8(i7); i7 += 8;
        int32x4_t vacc_lo = vld1q_s32(b); b += 4;
        int32x4_t vacc_hi = vld1q_s32(b); b += 4;

        const int16x8_t vsum01 = vreinterpretq_s16_u16(vaddl_u8(vi0, vi1));
        const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
        const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
        const int16x8_t vsum67 = vreinterpretq_s16_u16(vaddl_u8(vi6, vi7));

        const int16x8_t vsum0123 = vaddq_s16(vsum01, vsum23);
        const int16x8_t vsum4567 = vaddq_s16(vsum45, vsum67);
        const int16x8_t vsum = vaddq_s16(vsum0123, vsum4567);

        vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
        vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));

        const int32x4_t vneg_mask_lo = vreinterpretq_s32_u32(vcltq_s32(vacc_lo, vmovq_n_s32(0)));
        const int32x4_t vneg_mask_hi = vreinterpretq_s32_u32(vcltq_s32(vacc_hi, vmovq_n_s32(0)));

#if XNN_ARCH_ARM64
        const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vget_low_s32(vmultiplier));
        const int64x2_t vproduct23 = vmull_high_s32(vacc_lo, vmultiplier);
        const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vget_low_s32(vmultiplier));
        const int64x2_t vproduct67 = vmull_high_s32(vacc_hi, vmultiplier);

        const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product23 = vaddw_high_s32(vproduct23, vneg_mask_lo);
        const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
        const int64x2_t vadjusted_product67 = vaddw_high_s32(vproduct67, vneg_mask_hi);
#else
        const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vmultiplier);
        const int64x2_t vproduct23 = vmull_s32(vget_high_s32(vacc_lo), vmultiplier);
        const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vmultiplier);
        const int64x2_t vproduct67 = vmull_s32(vget_high_s32(vacc_hi), vmultiplier);

        const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product23 = vaddw_s32(vproduct23, vget_high_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
        const int64x2_t vadjusted_product67 = vaddw_s32(vproduct67, vget_high_s32(vneg_mask_hi));
#endif

        const int64x2_t vscaled_acc01 = vrshlq_s64(vadjusted_product01, vleft_shift);
        const int64x2_t vscaled_acc23 = vrshlq_s64(vadjusted_product23, vleft_shift);
        const int64x2_t vscaled_acc45 = vrshlq_s64(vadjusted_product45, vleft_shift);
        const int64x2_t vscaled_acc67 = vrshlq_s64(vadjusted_product67, vleft_shift);

#if XNN_ARCH_ARM64
        vacc_lo = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc01), vreinterpretq_s32_s64(vscaled_acc23));
        vacc_hi = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc45), vreinterpretq_s32_s64(vscaled_acc67));

        const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
        vacc_lo = vcombine_s32(vmovn_s64(vscaled_acc01), vmovn_s64(vscaled_acc23));
        vacc_hi = vcombine_s32(vmovn_s64(vscaled_acc45), vmovn_s64(vscaled_acc67));

        const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif

        uint8x8_t vout = vqmovun_s16(vacc);
        vout = vmax_u8(vout, voutput_min);
        vout = vmin_u8(vout, voutput_max);

        vst1_u8(output, vout); output += 8;

        c -= 8;
      }
      if (c != 0) {
        const uint8x8_t vi0 = vld1_u8(i0);
        const uint8x8_t vi1 = vld1_u8(i1);
        const uint8x8_t vi2 = vld1_u8(i2);
        const uint8x8_t vi3 = vld1_u8(i3);
        const uint8x8_t vi4 = vld1_u8(i4);
        const uint8x8_t vi5 = vld1_u8(i5);
        const uint8x8_t vi6 = vld1_u8(i6);
        const uint8x8_t vi7 = vld1_u8(i7);
        int32x4_t vacc_lo = vld1q_s32(b); b += 4;
        int32x4_t vacc_hi = vld1q_s32(b);

        const int16x8_t vsum01 = vreinterpretq_s16_u16(vaddl_u8(vi0, vi1));
        const int16x8_t vsum23 = vreinterpretq_s16_u16(vaddl_u8(vi2, vi3));
        const int16x8_t vsum45 = vreinterpretq_s16_u16(vaddl_u8(vi4, vi5));
        const int16x8_t vsum67 = vreinterpretq_s16_u16(vaddl_u8(vi6, vi7));

        const int16x8_t vsum0123 = vaddq_s16(vsum01, vsum23);
        const int16x8_t vsum4567 = vaddq_s16(vsum45, vsum67);
        const int16x8_t vsum = vaddq_s16(vsum0123, vsum4567);

        vacc_lo = vaddw_s16(vacc_lo, vget_low_s16(vsum));
        vacc_hi = vaddw_s16(vacc_hi, vget_high_s16(vsum));

        const int32x4_t vneg_mask_lo = vreinterpretq_s32_u32(vcltq_s32(vacc_lo, vmovq_n_s32(0)));
        const int32x4_t vneg_mask_hi = vreinterpretq_s32_u32(vcltq_s32(vacc_hi, vmovq_n_s32(0)));

#if XNN_ARCH_ARM64
        const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vget_low_s32(vmultiplier));
        const int64x2_t vproduct23 = vmull_high_s32(vacc_lo, vmultiplier);
        const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vget_low_s32(vmultiplier));
        const int64x2_t vproduct67 = vmull_high_s32(vacc_hi, vmultiplier);

        const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product23 = vaddw_high_s32(vproduct23, vneg_mask_lo);
        const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
        const int64x2_t vadjusted_product67 = vaddw_high_s32(vproduct67, vneg_mask_hi);
#else
        const int64x2_t vproduct01 = vmull_s32(vget_low_s32(vacc_lo), vmultiplier);
        const int64x2_t vproduct23 = vmull_s32(vget_high_s32(vacc_lo), vmultiplier);
        const int64x2_t vproduct45 = vmull_s32(vget_low_s32(vacc_hi), vmultiplier);
        const int64x2_t vproduct67 = vmull_s32(vget_high_s32(vacc_hi), vmultiplier);

        const int64x2_t vadjusted_product01 = vaddw_s32(vproduct01, vget_low_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product23 = vaddw_s32(vproduct23, vget_high_s32(vneg_mask_lo));
        const int64x2_t vadjusted_product45 = vaddw_s32(vproduct45, vget_low_s32(vneg_mask_hi));
        const int64x2_t vadjusted_product67 = vaddw_s32(vproduct67, vget_high_s32(vneg_mask_hi));
#endif

        const int64x2_t vscaled_acc01 = vrshlq_s64(vadjusted_product01, vleft_shift);
        const int64x2_t vscaled_acc23 = vrshlq_s64(vadjusted_product23, vleft_shift);
        const int64x2_t vscaled_acc45 = vrshlq_s64(vadjusted_product45, vleft_shift);
        const int64x2_t vscaled_acc67 = vrshlq_s64(vadjusted_product67, vleft_shift);

#if XNN_ARCH_ARM64
        vacc_lo = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc01), vreinterpretq_s32_s64(vscaled_acc23));
        vacc_hi = vuzp1q_s32(vreinterpretq_s32_s64(vscaled_acc45), vreinterpretq_s32_s64(vscaled_acc67));

        const int16x8_t vacc = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_lo), vacc_hi), voutput_zero_point);
#else
        vacc_lo = vcombine_s32(vmovn_s64(vscaled_acc01), vmovn_s64(vscaled_acc23));
        vacc_hi = vcombine_s32(vmovn_s64(vscaled_acc45), vmovn_s64(vscaled_acc67));

        const int16x8_t vacc = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi)), voutput_zero_point);
#endif

        uint8x8_t vout = vqmovun_s16(vacc);
        vout = vmax_u8(vout, voutput_min);
        vout = vmin_u8(vout, voutput_max);

        if (c & 4) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0); output += 4;
          vout = vext_u8(vout, vout, 4);
        }
        if (c & 2) {
          vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0); output += 2;
          vout = vext_u8(vout, vout, 2);
        }
        if (c & 1) {
          vst1_lane_u8(output, vout, 0); output += 1;
        }
      }
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
