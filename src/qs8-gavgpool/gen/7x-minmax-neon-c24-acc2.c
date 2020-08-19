// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>


void xnn_qs8_gavgpool_minmax_ukernel_7x__neon_c24_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const int32x4_t vbias = vld1q_dup_s32(&params->neon.bias);
#if XNN_ARCH_ARM64
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
#else
  const int32x2_t vmultiplier = vld1_dup_s32(&params->neon.multiplier);
#endif
  const int64x2_t vleft_shift = vld1q_dup_s64(&params->neon.left_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);
  while (channels >= 24) {
    const int8x8_t vi0x01234567 = vld1_s8(i0); i0 += 8;
    const int8x8_t vi0x89ABCDEF = vld1_s8(i0); i0 += 8;
    const int8x8_t vi0xGHIJKLMN = vld1_s8(i0); i0 += 8;
    const int8x8_t vi1x01234567 = vld1_s8(i1); i1 += 8;
    const int8x8_t vi1x89ABCDEF = vld1_s8(i1); i1 += 8;
    const int8x8_t vi1xGHIJKLMN = vld1_s8(i1); i1 += 8;
    const int8x8_t vi2x01234567 = vld1_s8(i2); i2 += 8;
    const int8x8_t vi2x89ABCDEF = vld1_s8(i2); i2 += 8;
    const int8x8_t vi2xGHIJKLMN = vld1_s8(i2); i2 += 8;
    const int8x8_t vi3x01234567 = vld1_s8(i3); i3 += 8;
    const int8x8_t vi3x89ABCDEF = vld1_s8(i3); i3 += 8;
    const int8x8_t vi3xGHIJKLMN = vld1_s8(i3); i3 += 8;
    const int8x8_t vi4x01234567 = vld1_s8(i4); i4 += 8;
    const int8x8_t vi4x89ABCDEF = vld1_s8(i4); i4 += 8;
    const int8x8_t vi4xGHIJKLMN = vld1_s8(i4); i4 += 8;
    const int8x8_t vi5x01234567 = vld1_s8(i5); i5 += 8;
    const int8x8_t vi5x89ABCDEF = vld1_s8(i5); i5 += 8;
    const int8x8_t vi5xGHIJKLMN = vld1_s8(i5); i5 += 8;
    const int8x8_t vi6x01234567 = vld1_s8(i6); i6 += 8;
    const int8x8_t vi6x89ABCDEF = vld1_s8(i6); i6 += 8;
    const int8x8_t vi6xGHIJKLMN = vld1_s8(i6); i6 += 8;

    int16x8_t vacc0x01234567 = vaddl_s8(vi0x01234567, vi1x01234567);
    int16x8_t vacc0x89ABCDEF = vaddl_s8(vi0x89ABCDEF, vi1x89ABCDEF);
    int16x8_t vacc0xGHIJKLMN = vaddl_s8(vi0xGHIJKLMN, vi1xGHIJKLMN);
    int16x8_t vacc1x01234567 = vaddl_s8(vi2x01234567, vi3x01234567);
    int16x8_t vacc1x89ABCDEF = vaddl_s8(vi2x89ABCDEF, vi3x89ABCDEF);
    int16x8_t vacc1xGHIJKLMN = vaddl_s8(vi2xGHIJKLMN, vi3xGHIJKLMN);

    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi4x01234567);
    vacc0x89ABCDEF = vaddw_s8(vacc0x89ABCDEF, vi4x89ABCDEF);
    vacc0xGHIJKLMN = vaddw_s8(vacc0xGHIJKLMN, vi4xGHIJKLMN);
    vacc1x01234567 = vaddw_s8(vacc1x01234567, vi5x01234567);
    vacc1x89ABCDEF = vaddw_s8(vacc1x89ABCDEF, vi5x89ABCDEF);
    vacc1xGHIJKLMN = vaddw_s8(vacc1xGHIJKLMN, vi5xGHIJKLMN);
    vacc0x01234567 = vaddw_s8(vacc0x01234567, vi6x01234567);
    vacc0x89ABCDEF = vaddw_s8(vacc0x89ABCDEF, vi6x89ABCDEF);
    vacc0xGHIJKLMN = vaddw_s8(vacc0xGHIJKLMN, vi6xGHIJKLMN);

    // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
    vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);
    vacc0x89ABCDEF = vaddq_s16(vacc0x89ABCDEF, vacc1x89ABCDEF);
    vacc0xGHIJKLMN = vaddq_s16(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

    int32x4_t vacc0123 = vaddw_s16(vbias, vget_low_s16(vacc0x01234567));
    int32x4_t vacc4567 = vaddw_s16(vbias, vget_high_s16(vacc0x01234567));
    int32x4_t vacc89AB = vaddw_s16(vbias, vget_low_s16(vacc0x89ABCDEF));
    int32x4_t vaccCDEF = vaddw_s16(vbias, vget_high_s16(vacc0x89ABCDEF));
    int32x4_t vaccGHIJ = vaddw_s16(vbias, vget_low_s16(vacc0xGHIJKLMN));
    int32x4_t vaccKLMN = vaddw_s16(vbias, vget_high_s16(vacc0xGHIJKLMN));

    const int32x4_t vsgnacc0123 = vreinterpretq_s32_u32(vcltq_s32(vacc0123, vmovq_n_s32(0)));
    const int32x4_t vsgnacc4567 = vreinterpretq_s32_u32(vcltq_s32(vacc4567, vmovq_n_s32(0)));
    const int32x4_t vsgnacc89AB = vreinterpretq_s32_u32(vcltq_s32(vacc89AB, vmovq_n_s32(0)));
    const int32x4_t vsgnaccCDEF = vreinterpretq_s32_u32(vcltq_s32(vaccCDEF, vmovq_n_s32(0)));
    const int32x4_t vsgnaccGHIJ = vreinterpretq_s32_u32(vcltq_s32(vaccGHIJ, vmovq_n_s32(0)));
    const int32x4_t vsgnaccKLMN = vreinterpretq_s32_u32(vcltq_s32(vaccKLMN, vmovq_n_s32(0)));

#if XNN_ARCH_ARM64
    const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vget_low_s32(vmultiplier));
    const int64x2_t vprod23 = vmull_high_s32(vacc0123, vmultiplier);
    const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vget_low_s32(vmultiplier));
    const int64x2_t vprod67 = vmull_high_s32(vacc4567, vmultiplier);
    const int64x2_t vprod89 = vmull_s32(vget_low_s32(vacc89AB), vget_low_s32(vmultiplier));
    const int64x2_t vprodAB = vmull_high_s32(vacc89AB, vmultiplier);
    const int64x2_t vprodCD = vmull_s32(vget_low_s32(vaccCDEF), vget_low_s32(vmultiplier));
    const int64x2_t vprodEF = vmull_high_s32(vaccCDEF, vmultiplier);
    const int64x2_t vprodGH = vmull_s32(vget_low_s32(vaccGHIJ), vget_low_s32(vmultiplier));
    const int64x2_t vprodIJ = vmull_high_s32(vaccGHIJ, vmultiplier);
    const int64x2_t vprodKL = vmull_s32(vget_low_s32(vaccKLMN), vget_low_s32(vmultiplier));
    const int64x2_t vprodMN = vmull_high_s32(vaccKLMN, vmultiplier);

    const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
    const int64x2_t vadjprod23 = vaddw_high_s32(vprod23, vsgnacc0123);
    const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
    const int64x2_t vadjprod67 = vaddw_high_s32(vprod67, vsgnacc4567);
    const int64x2_t vadjprod89 = vaddw_s32(vprod89, vget_low_s32(vsgnacc89AB));
    const int64x2_t vadjprodAB = vaddw_high_s32(vprodAB, vsgnacc89AB);
    const int64x2_t vadjprodCD = vaddw_s32(vprodCD, vget_low_s32(vsgnaccCDEF));
    const int64x2_t vadjprodEF = vaddw_high_s32(vprodEF, vsgnaccCDEF);
    const int64x2_t vadjprodGH = vaddw_s32(vprodGH, vget_low_s32(vsgnaccGHIJ));
    const int64x2_t vadjprodIJ = vaddw_high_s32(vprodIJ, vsgnaccGHIJ);
    const int64x2_t vadjprodKL = vaddw_s32(vprodKL, vget_low_s32(vsgnaccKLMN));
    const int64x2_t vadjprodMN = vaddw_high_s32(vprodMN, vsgnaccKLMN);
#else
    const int64x2_t vprod01 = vmull_s32(vget_low_s32(vacc0123), vmultiplier);
    const int64x2_t vprod23 = vmull_s32(vget_high_s32(vacc0123), vmultiplier);
    const int64x2_t vprod45 = vmull_s32(vget_low_s32(vacc4567), vmultiplier);
    const int64x2_t vprod67 = vmull_s32(vget_high_s32(vacc4567), vmultiplier);
    const int64x2_t vprod89 = vmull_s32(vget_low_s32(vacc89AB), vmultiplier);
    const int64x2_t vprodAB = vmull_s32(vget_high_s32(vacc89AB), vmultiplier);
    const int64x2_t vprodCD = vmull_s32(vget_low_s32(vaccCDEF), vmultiplier);
    const int64x2_t vprodEF = vmull_s32(vget_high_s32(vaccCDEF), vmultiplier);
    const int64x2_t vprodGH = vmull_s32(vget_low_s32(vaccGHIJ), vmultiplier);
    const int64x2_t vprodIJ = vmull_s32(vget_high_s32(vaccGHIJ), vmultiplier);
    const int64x2_t vprodKL = vmull_s32(vget_low_s32(vaccKLMN), vmultiplier);
    const int64x2_t vprodMN = vmull_s32(vget_high_s32(vaccKLMN), vmultiplier);

    const int64x2_t vadjprod01 = vaddw_s32(vprod01, vget_low_s32(vsgnacc0123));
    const int64x2_t vadjprod23 = vaddw_s32(vprod23, vget_high_s32(vsgnacc0123));
    const int64x2_t vadjprod45 = vaddw_s32(vprod45, vget_low_s32(vsgnacc4567));
    const int64x2_t vadjprod67 = vaddw_s32(vprod67, vget_high_s32(vsgnacc4567));
    const int64x2_t vadjprod89 = vaddw_s32(vprod89, vget_low_s32(vsgnacc89AB));
    const int64x2_t vadjprodAB = vaddw_s32(vprodAB, vget_high_s32(vsgnacc89AB));
    const int64x2_t vadjprodCD = vaddw_s32(vprodCD, vget_low_s32(vsgnaccCDEF));
    const int64x2_t vadjprodEF = vaddw_s32(vprodEF, vget_high_s32(vsgnaccCDEF));
    const int64x2_t vadjprodGH = vaddw_s32(vprodGH, vget_low_s32(vsgnaccGHIJ));
    const int64x2_t vadjprodIJ = vaddw_s32(vprodIJ, vget_high_s32(vsgnaccGHIJ));
    const int64x2_t vadjprodKL = vaddw_s32(vprodKL, vget_low_s32(vsgnaccKLMN));
    const int64x2_t vadjprodMN = vaddw_s32(vprodMN, vget_high_s32(vsgnaccKLMN));
#endif

    const int64x2_t vacc01 = vrshlq_s64(vadjprod01, vleft_shift);
    const int64x2_t vacc23 = vrshlq_s64(vadjprod23, vleft_shift);
    const int64x2_t vacc45 = vrshlq_s64(vadjprod45, vleft_shift);
    const int64x2_t vacc67 = vrshlq_s64(vadjprod67, vleft_shift);
    const int64x2_t vacc89 = vrshlq_s64(vadjprod89, vleft_shift);
    const int64x2_t vaccAB = vrshlq_s64(vadjprodAB, vleft_shift);
    const int64x2_t vaccCD = vrshlq_s64(vadjprodCD, vleft_shift);
    const int64x2_t vaccEF = vrshlq_s64(vadjprodEF, vleft_shift);
    const int64x2_t vaccGH = vrshlq_s64(vadjprodGH, vleft_shift);
    const int64x2_t vaccIJ = vrshlq_s64(vadjprodIJ, vleft_shift);
    const int64x2_t vaccKL = vrshlq_s64(vadjprodKL, vleft_shift);
    const int64x2_t vaccMN = vrshlq_s64(vadjprodMN, vleft_shift);

#if XNN_ARCH_ARM64
    vacc0123 = vuzp1q_s32(vreinterpretq_s32_s64(vacc01), vreinterpretq_s32_s64(vacc23));
    vacc4567 = vuzp1q_s32(vreinterpretq_s32_s64(vacc45), vreinterpretq_s32_s64(vacc67));
    vacc89AB = vuzp1q_s32(vreinterpretq_s32_s64(vacc89), vreinterpretq_s32_s64(vaccAB));
    vaccCDEF = vuzp1q_s32(vreinterpretq_s32_s64(vaccCD), vreinterpretq_s32_s64(vaccEF));
    vaccGHIJ = vuzp1q_s32(vreinterpretq_s32_s64(vaccGH), vreinterpretq_s32_s64(vaccIJ));
    vaccKLMN = vuzp1q_s32(vreinterpretq_s32_s64(vaccKL), vreinterpretq_s32_s64(vaccMN));

    const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
    const int16x8_t vacc89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF), voutput_zero_point);
    const int16x8_t vaccGHIJKLMN = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vaccGHIJ), vaccKLMN), voutput_zero_point);

    int8x16_t vout0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc01234567), vacc89ABCDEF);
    int8x8_t voutGHIJKLMN = vqmovn_s16(vaccGHIJKLMN);
#else
    vacc0123 = vcombine_s32(vmovn_s64(vacc01), vmovn_s64(vacc23));
    vacc4567 = vcombine_s32(vmovn_s64(vacc45), vmovn_s64(vacc67));
    vacc89AB = vcombine_s32(vmovn_s64(vacc89), vmovn_s64(vaccAB));
    vaccCDEF = vcombine_s32(vmovn_s64(vaccCD), vmovn_s64(vaccEF));
    vaccGHIJ = vcombine_s32(vmovn_s64(vaccGH), vmovn_s64(vaccIJ));
    vaccKLMN = vcombine_s32(vmovn_s64(vaccKL), vmovn_s64(vaccMN));

    const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
    const int16x8_t vacc89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF)), voutput_zero_point);
    const int16x8_t vaccGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN)), voutput_zero_point);

    int8x16_t vout0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
    int8x8_t voutGHIJKLMN = vqmovn_s16(vaccGHIJKLMN);
#endif

    vout0123456789ABCDEF = vmaxq_s8(vout0123456789ABCDEF, voutput_min);
    voutGHIJKLMN = vmax_s8(voutGHIJKLMN, vget_low_s8(voutput_min));

    vout0123456789ABCDEF = vminq_s8(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMN = vmin_s8(voutGHIJKLMN, vget_low_s8(voutput_max));

    vst1q_s8(output, vout0123456789ABCDEF); output += 16;
    vst1_s8(output, voutGHIJKLMN); output += 8;

    channels -= 24;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
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

      // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
      vacc0x01234567 = vaddq_s16(vacc0x01234567, vacc1x01234567);

      int32x4_t vacc0123 = vaddw_s16(vbias, vget_low_s16(vacc0x01234567));
      int32x4_t vacc4567 = vaddw_s16(vbias, vget_high_s16(vacc0x01234567));

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

      vout01234567 = vmax_s8(vout01234567, vget_low_s8(voutput_min));
      vout01234567 = vmin_s8(vout01234567, vget_low_s8(voutput_max));

      if XNN_LIKELY(channels >= 8) {
        vst1_s8(output, vout01234567); output += 8;
        channels -= 8;
      } else {
        if (channels & 4) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_s8(vout01234567), 0); output += 4;
          vout01234567 = vext_s8(vout01234567, vout01234567, 4);
        }
        if (channels & 2) {
          vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_s8(vout01234567), 0); output += 2;
          vout01234567 = vext_s8(vout01234567, vout01234567, 2);
        }
        if (channels & 1) {
          vst1_lane_s8(output, vout01234567, 0); output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
