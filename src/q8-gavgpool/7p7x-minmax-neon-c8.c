// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_q8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t packed_channels = round_up_po2(channels, 8);
  const size_t input_increment = 7 * input_stride - packed_channels;
  const int32x4_t vbias = vld1q_dup_s32(&params->neon.bias);

  int32_t* acc = buffer;
  for (size_t c = 0; c < channels; c += 8) {
    const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
    const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
    const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
    const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
    const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;

    const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);
    const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
    const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);

    const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);
    const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);

    const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));

    const int32x4_t vacc_lo = vaddw_s16(vbias, vget_low_s16(vsum));
    const int32x4_t vacc_hi = vaddw_s16(vbias, vget_high_s16(vsum));

    vst1q_s32(acc, vacc_lo); acc += 4;
    vst1q_s32(acc, vacc_hi); acc += 4;
  }
  for (rows -= 7; rows > 7; rows -= 7) {
    acc = buffer;

    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    for (size_t c = 0; c < channels; c += 8) {
      const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
      const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
      const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
      const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
      const int32x4_t vacc_lo = vld1q_s32(acc);
      const int32x4_t vacc_hi = vld1q_s32(acc + 4);

      const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);
      const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
      const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);

      const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);
      const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);

      const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));

      vst1q_s32(acc, vaddw_s16(vacc_lo, vget_low_s16(vsum))); acc += 4;
      vst1q_s32(acc, vaddw_s16(vacc_hi, vget_high_s16(vsum))); acc += 4;
    }
  }

#if XNN_ARCH_ARM64
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
#else
  const int32x2_t vmultiplier = vld1_dup_s32(&params->neon.multiplier);
#endif
  const int64x2_t vleft_shift = vld1q_dup_s64(&params->neon.left_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->neon.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->neon.output_max);

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }

  acc = buffer;
  while (channels >= 8) {
    const uint8x8_t vi0 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi2 = vld1_u8(i2); i2 += 8;
    const uint8x8_t vi3 = vld1_u8(i3); i3 += 8;
    const uint8x8_t vi4 = vld1_u8(i4); i4 += 8;
    const uint8x8_t vi5 = vld1_u8(i5); i5 += 8;
    const uint8x8_t vi6 = vld1_u8(i6); i6 += 8;
    int32x4_t vacc_lo = vld1q_s32(acc); acc += 4;
    int32x4_t vacc_hi = vld1q_s32(acc); acc += 4;

    const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);
    const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
    const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);

    const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);
    const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);

    const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));
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

    channels -= 8;
  }
  if (channels != 0) {
    const uint8x8_t vi0 = vld1_u8(i0);
    const uint8x8_t vi1 = vld1_u8(i1);
    const uint8x8_t vi2 = vld1_u8(i2);
    const uint8x8_t vi3 = vld1_u8(i3);
    const uint8x8_t vi4 = vld1_u8(i4);
    const uint8x8_t vi5 = vld1_u8(i5);
    const uint8x8_t vi6 = vld1_u8(i6);
    int32x4_t vacc_lo = vld1q_s32(acc); acc += 4;
    int32x4_t vacc_hi = vld1q_s32(acc);

    const uint16x8_t vsum01 = vaddl_u8(vi0, vi1);
    const uint16x8_t vsum23 = vaddl_u8(vi2, vi3);
    const uint16x8_t vsum45 = vaddl_u8(vi4, vi5);

    const uint16x8_t vsum016 = vaddw_u8(vsum01, vi6);
    const uint16x8_t vsum2345 = vaddq_u16(vsum23, vsum45);

    const int16x8_t vsum = vreinterpretq_s16_u16(vaddq_u16(vsum016, vsum2345));
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

    if (channels & 4) {
      vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout), 0); output += 4;
      vout = vext_u8(vout, vout, 4);
    }
    if (channels & 2) {
      vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout), 0); output += 2;
      vout = vext_u8(vout, vout, 2);
    }
    if (channels & 1) {
      vst1_lane_u8(output, vout, 0);
    }
  }
}
