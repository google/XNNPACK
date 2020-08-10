// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_ukernel_7p7x__neon_c8_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8);

  const int32x4_t vbias = vld1q_dup_s32(&params->neon.bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
    const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
    const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
    const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
    const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
    const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
    const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;

    int16x8_t vacc0x01234567 = vaddl_s8(vi0x01234567, vi1x01234567);
    int16x8_t vacc1x01234567 = vaddl_s8(vi2x01234567, vi3x01234567);

    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi4x01234567);
    vacc1x01234567 = vaddw_s8(vacc1x01234567, vi5x01234567);
    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);

    const int32x4_t vacc0123 = vaddw_s16(vbias, vget_low_s16(vacc0x01234567));
    const int32x4_t vacc4567 = vaddw_s16(vbias, vget_high_s16(vacc0x01234567));

    vst1q_s32(b, vacc0123); b += 4;
    vst1q_s32(b, vacc4567); b += 4;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
      const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
      const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
      const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
      const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
      const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
      const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;

      int16x8_t vacc0x01234567 = vaddl_s8(vi0x01234567, vi1x01234567);
      int16x8_t vacc1x01234567 = vaddl_s8(vi2x01234567, vi3x01234567);

      vacc0x01234567 = vaddw_s8(vacc0x01234567, vi4x01234567);
      vacc1x01234567 = vaddw_s8(vacc1x01234567, vi5x01234567);
      vacc0x01234567 = vaddw_s8(vacc0x01234567, vi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);

      int32x4_t vacc0123 = vld1q_s32(b);
      int32x4_t vacc4567 = vld1q_s32(b + 4);

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vacc0x01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vacc0x01234567));

      vst1q_s32(b, vacc0123); b += 4;
      vst1q_s32(b, vacc4567); b += 4;
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

#if XNN_ARCH_ARM64
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
#else
  const int32x2_t vmultiplier = vld1_dup_s32(&params->neon.multiplier);
#endif
  const int64x2_t vleft_shift = vld1q_dup_s64(&params->neon.left_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const int8x8_t voutput_min = vld1_dup_s8(&params->neon.output_min);
  const int8x8_t voutput_max = vld1_dup_s8(&params->neon.output_max);
  while (channels >= 8) {
    const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
    const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
    const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
    const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
    const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
    const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
    const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;

    int16x8_t vacc0x01234567 = vaddl_s8(vi0x01234567, vi1x01234567);
    int16x8_t vacc1x01234567 = vaddl_s8(vi2x01234567, vi3x01234567);

    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi4x01234567);
    vacc1x01234567 = vaddw_s8(vacc1x01234567, vi5x01234567);
    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);

    int32x4_t vacc0123 = vld1q_s32(buffer); buffer += 4;
    int32x4_t vacc4567 = vld1q_s32(buffer); buffer += 4;

    vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vacc0x01234567));
    vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vacc0x01234567));

    const int32x4_t vsgnacc0123 = vreinterpretq_s32_u32(vcltq_s32(vacc0123, vmovq_n_s32(0)));
    const int32x4_t vsgnacc4567 = vreinterpretq_s32_u32(vcltq_s32(vacc4567, vmovq_n_s32(0)));

#if XNN_ARCH_ARM64
    const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vget_low_s32(vmultiplier));
    const int64x2_t vprod23 = vmull_high_s32(vacc0123, vmultiplier);
    const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vget_low_s32(vmultiplier));
    const int64x2_t vprod67 = vmull_high_s32(vacc4567, vmultiplier);

    const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
    const int64x2_t vadjprod23 = vaddw_high_s32(vprod23, vsgnacc0123);
    const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
    const int64x2_t vadjprod67 = vaddw_high_s32(vprod67, vsgnacc4567);
#else
    const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vmultiplier);
    const int64x2_t vprod23 = vmull_s32(vget_high_s32(vacc0123), vmultiplier);
    const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vmultiplier);
    const int64x2_t vprod67 = vmull_s32(vget_high_s32(vacc4567), vmultiplier);

    const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
    const int64x2_t vadjprod23 = vaddw_s32(vprod23, vget_high_s32(vsgnacc0123));
    const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
    const int64x2_t vadjprod67 = vaddw_s32(vprod67, vget_high_s32(vsgnacc4567));
#endif

    const int64x2_t vacc01 = vrshlq_s64(vadjprod01, vleft_shift);
    const int64x2_t vacc23 = vrshlq_s64(vadjprod23, vleft_shift);
    const int64x2_t vacc45 = vrshlq_s64(vadjprod45, vleft_shift);
    const int64x2_t vacc67 = vrshlq_s64(vadjprod67, vleft_shift);

#if XNN_ARCH_ARM64
    vacc0123 = vuzp1q_s32(vreinterpretq_s32_s64(vacc01), vreinterpretq_s32_s64(vacc23));
    vacc4567 = vuzp1q_s32(vreinterpretq_s32_s64(vacc45), vreinterpretq_s32_s64(vacc67));

    const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);

    int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#else
    vacc0123 = vcombine_s32(vmovn_s64(vacc01), vmovn_s64(vacc23));
    vacc4567 = vcombine_s32(vmovn_s64(vacc45), vmovn_s64(vacc67));

    const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

    int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#endif

    vout01234567 = vmax_s8(vout01234567, voutput_min);

    vout01234567 = vmin_s8(vout01234567, voutput_max);

    vst1_s8(output, vout01234567); output += 8;

    channels -= 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const int8x8_t vi0x01234567 = vld1_s8(i0);
      const int8x8_t vi1x01234567 = vld1_s8(i1);
      const int8x8_t vi2x01234567 = vld1_s8(i2);
      const int8x8_t vi3x01234567 = vld1_s8(i3);
      const int8x8_t vi4x01234567 = vld1_s8(i4);
      const int8x8_t vi5x01234567 = vld1_s8(i5);
      const int8x8_t vi6x01234567 = vld1_s8(i6);

      int16x8_t vacc0x01234567 = vaddl_s8(vi0x01234567, vi1x01234567);
      int16x8_t vacc1x01234567 = vaddl_s8(vi2x01234567, vi3x01234567);

      vacc0x01234567 = vaddw_s8(vacc0x01234567, vi4x01234567);
      vacc1x01234567 = vaddw_s8(vacc1x01234567, vi5x01234567);
      vacc0x01234567 = vaddw_s8(vacc0x01234567, vi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);

      int32x4_t vacc0123 = vld1q_s32(buffer); buffer += 4;
      int32x4_t vacc4567 = vld1q_s32(buffer); buffer += 4;

      vacc0123 = vaddw_s16(vacc0123, vget_low_s16(vacc0x01234567));
      vacc4567 = vaddw_s16(vacc4567, vget_high_s16(vacc0x01234567));

      const int32x4_t vsgnacc0123 = vreinterpretq_s32_u32(vcltq_s32(vacc0123, vmovq_n_s32(0)));
      const int32x4_t vsgnacc4567 = vreinterpretq_s32_u32(vcltq_s32(vacc4567, vmovq_n_s32(0)));

#if XNN_ARCH_ARM64
      const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vget_low_s32(vmultiplier));
      const int64x2_t vprod23 = vmull_high_s32(vacc0123, vmultiplier);
      const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vget_low_s32(vmultiplier));
      const int64x2_t vprod67 = vmull_high_s32(vacc4567, vmultiplier);

      const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
      const int64x2_t vadjprod23 = vaddw_high_s32(vprod23, vsgnacc0123);
      const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
      const int64x2_t vadjprod67 = vaddw_high_s32(vprod67, vsgnacc4567);
#else
      const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vmultiplier);
      const int64x2_t vprod23 = vmull_s32(vget_high_s32(vacc0123), vmultiplier);
      const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vmultiplier);
      const int64x2_t vprod67 = vmull_s32(vget_high_s32(vacc4567), vmultiplier);

      const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
      const int64x2_t vadjprod23 = vaddw_s32(vprod23, vget_high_s32(vsgnacc0123));
      const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
      const int64x2_t vadjprod67 = vaddw_s32(vprod67, vget_high_s32(vsgnacc4567));
#endif

      const int64x2_t vacc01 = vrshlq_s64(vadjprod01, vleft_shift);
      const int64x2_t vacc23 = vrshlq_s64(vadjprod23, vleft_shift);
      const int64x2_t vacc45 = vrshlq_s64(vadjprod45, vleft_shift);
      const int64x2_t vacc67 = vrshlq_s64(vadjprod67, vleft_shift);

#if XNN_ARCH_ARM64
      vacc0123 = vuzp1q_s32(vreinterpretq_s32_s64(vacc01), vreinterpretq_s32_s64(vacc23));
      vacc4567 = vuzp1q_s32(vreinterpretq_s32_s64(vacc45), vreinterpretq_s32_s64(vacc67));

      const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#else
      vacc0123 = vcombine_s32(vmovn_s64(vacc01), vmovn_s64(vacc23));
      vacc4567 = vcombine_s32(vmovn_s64(vacc45), vmovn_s64(vacc67));

      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

      int8x8_t vout01234567 = vqmovn_s16(vacc01234567);
#endif

      vout01234567 = vmax_s8(vout01234567, voutput_min);
      vout01234567 = vmin_s8(vout01234567, voutput_max);

      if (channels & 4) {
        vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_s8(vout01234567), 0); output += 4;
        vout01234567 = vext_s8(vout01234567, vout01234567, 4);
      }
      if (channels & 2) {
        vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_s8(vout01234567), 0); output += 2;
        vout01234567 = vext_s8(vout01234567, vout01234567, 2);
      }
      if (channels & 1) {
        vst1_lane_s8(output, vout01234567, 0);
      }
    }
  }
}
