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
#include <xnnpack/gemm.h>


void xnn_q8_gemm_minmax_ukernel_8x8__neon(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_q8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint8_t* a4 = (const uint8_t*) ((uintptr_t) a3 + a_stride);
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint8_t* a5 = (const uint8_t*) ((uintptr_t) a4 + a_stride);
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint8_t* a6 = (const uint8_t*) ((uintptr_t) a5 + a_stride);
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const uint8_t* a7 = (const uint8_t*) ((uintptr_t) a6 + a_stride);
  uint8_t* c7 = (uint8_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  const uint8x8_t vb_zero_point = vld1_dup_u8((const uint8_t*) &params->neon.kernel_zero_point);

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 16);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc4x0123 = vacc0x0123;
    int32x4_t vacc4x4567 = vacc0x4567;
    int32x4_t vacc5x0123 = vacc0x0123;
    int32x4_t vacc5x4567 = vacc0x4567;
    int32x4_t vacc6x0123 = vacc0x0123;
    int32x4_t vacc6x4567 = vacc0x4567;
    int32x4_t vacc7x0123 = vacc0x0123;
    int32x4_t vacc7x4567 = vacc0x4567;

    size_t k = kc;
    while (k >= 8 * sizeof(uint8_t)) {
      const uint8x8_t va0 = vld1_u8(a0);
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0)); a0 += 8;
      const uint8x8_t va1 = vld1_u8(a1);
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1)); a1 += 8;
      const uint8x8_t va2 = vld1_u8(a2);
      const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2)); a2 += 8;
      const uint8x8_t va3 = vld1_u8(a3);
      const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3)); a3 += 8;
      const uint8x8_t va4 = vld1_u8(a4);
      const int16x8_t vxa4 = vreinterpretq_s16_u16(vmovl_u8(va4)); a4 += 8;
      const uint8x8_t va5 = vld1_u8(a5);
      const int16x8_t vxa5 = vreinterpretq_s16_u16(vmovl_u8(va5)); a5 += 8;
      const uint8x8_t va6 = vld1_u8(a6);
      const int16x8_t vxa6 = vreinterpretq_s16_u16(vmovl_u8(va6)); a6 += 8;
      const uint8x8_t va7 = vld1_u8(a7);
      const int16x8_t vxa7 = vreinterpretq_s16_u16(vmovl_u8(va7)); a7 += 8;

      const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa7), 0);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa7), 0);

      const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa7), 1);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa7), 1);

      const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa7), 2);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa7), 2);

      const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa7), 3);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa7), 3);

      const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa7), 0);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa7), 0);

      const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa7), 1);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa7), 1);

      const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa7), 2);

      const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa6), 3);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa6), 3);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa7), 3);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa7), 3);

      k -= 8 * sizeof(uint8_t);
    }
    if (k != 0) {
      const uint8x8_t va0 = vld1_u8(a0); a0 = (const uint8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));
      const uint8x8_t va1 = vld1_u8(a1); a1 = (const uint8_t*) ((uintptr_t) a1 + k);
      const int16x8_t vxa1 = vreinterpretq_s16_u16(vmovl_u8(va1));
      const uint8x8_t va2 = vld1_u8(a2); a2 = (const uint8_t*) ((uintptr_t) a2 + k);
      const int16x8_t vxa2 = vreinterpretq_s16_u16(vmovl_u8(va2));
      const uint8x8_t va3 = vld1_u8(a3); a3 = (const uint8_t*) ((uintptr_t) a3 + k);
      const int16x8_t vxa3 = vreinterpretq_s16_u16(vmovl_u8(va3));
      const uint8x8_t va4 = vld1_u8(a4); a4 = (const uint8_t*) ((uintptr_t) a4 + k);
      const int16x8_t vxa4 = vreinterpretq_s16_u16(vmovl_u8(va4));
      const uint8x8_t va5 = vld1_u8(a5); a5 = (const uint8_t*) ((uintptr_t) a5 + k);
      const int16x8_t vxa5 = vreinterpretq_s16_u16(vmovl_u8(va5));
      const uint8x8_t va6 = vld1_u8(a6); a6 = (const uint8_t*) ((uintptr_t) a6 + k);
      const int16x8_t vxa6 = vreinterpretq_s16_u16(vmovl_u8(va6));
      const uint8x8_t va7 = vld1_u8(a7); a7 = (const uint8_t*) ((uintptr_t) a7 + k);
      const int16x8_t vxa7 = vreinterpretq_s16_u16(vmovl_u8(va7));

      const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
      const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
      vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa6), 0);
      vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa7), 0);
      vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa7), 0);

      if (k >= 2 * sizeof(uint8_t)) {
        const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
        const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
        vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa6), 1);
        vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa7), 1);
        vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa7), 1);

        if (k > 2 * sizeof(uint8_t)) {
          const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
          const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
          vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa6), 2);
          vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa7), 2);
          vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa7), 2);

          if (k >= 4 * sizeof(uint8_t)) {
            const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
            const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
            vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa6), 3);
            vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa7), 3);
            vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa7), 3);

            if (k > 4 * sizeof(uint8_t)) {
              const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
              const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
              vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa6), 0);
              vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa7), 0);
              vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa7), 0);

              if (k >= 6 * sizeof(uint8_t)) {
                const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
                const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
                vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa6), 1);
                vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa7), 1);
                vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa7), 1);

                if (k > 6 * sizeof(uint8_t)) {
                  const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const void*) ((uintptr_t) w + 8);
                  const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                  vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                  vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                  vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                  vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                  vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                  vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                  vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
                  vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
                  vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
                  vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
                  vacc6x0123 = vmlal_lane_s16(vacc6x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
                  vacc6x4567 = vmlal_lane_s16(vacc6x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa6), 2);
                  vacc7x0123 = vmlal_lane_s16(vacc7x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
                  vacc7x4567 = vmlal_lane_s16(vacc7x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa7), 2);
                }
              }
            }
          }
        }
      }
    }

    const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);
    vacc4x0123 = vqrdmulhq_s32(vacc4x0123, vmultiplier);
    vacc4x4567 = vqrdmulhq_s32(vacc4x4567, vmultiplier);
    vacc5x0123 = vqrdmulhq_s32(vacc5x0123, vmultiplier);
    vacc5x4567 = vqrdmulhq_s32(vacc5x4567, vmultiplier);
    vacc6x0123 = vqrdmulhq_s32(vacc6x0123, vmultiplier);
    vacc6x4567 = vqrdmulhq_s32(vacc6x4567, vmultiplier);
    vacc7x0123 = vqrdmulhq_s32(vacc7x0123, vmultiplier);
    vacc7x4567 = vqrdmulhq_s32(vacc7x4567, vmultiplier);

    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
    vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
    vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
    vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
    vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
    vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);
    vacc4x0123 = vsraq_n_s32(vacc4x0123, vbicq_s32(vacc4x0123, vzero_shift_mask), 31);
    vacc4x4567 = vsraq_n_s32(vacc4x4567, vbicq_s32(vacc4x4567, vzero_shift_mask), 31);
    vacc5x0123 = vsraq_n_s32(vacc5x0123, vbicq_s32(vacc5x0123, vzero_shift_mask), 31);
    vacc5x4567 = vsraq_n_s32(vacc5x4567, vbicq_s32(vacc5x4567, vzero_shift_mask), 31);
    vacc6x0123 = vsraq_n_s32(vacc6x0123, vbicq_s32(vacc6x0123, vzero_shift_mask), 31);
    vacc6x4567 = vsraq_n_s32(vacc6x4567, vbicq_s32(vacc6x4567, vzero_shift_mask), 31);
    vacc7x0123 = vsraq_n_s32(vacc7x0123, vbicq_s32(vacc7x0123, vzero_shift_mask), 31);
    vacc7x4567 = vsraq_n_s32(vacc7x4567, vbicq_s32(vacc7x4567, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);
    vacc4x0123 = vrshlq_s32(vacc4x0123, vright_shift);
    vacc4x4567 = vrshlq_s32(vacc4x4567, vright_shift);
    vacc5x0123 = vrshlq_s32(vacc5x0123, vright_shift);
    vacc5x4567 = vrshlq_s32(vacc5x4567, vright_shift);
    vacc6x0123 = vrshlq_s32(vacc6x0123, vright_shift);
    vacc6x4567 = vrshlq_s32(vacc6x4567, vright_shift);
    vacc7x0123 = vrshlq_s32(vacc7x0123, vright_shift);
    vacc7x4567 = vrshlq_s32(vacc7x4567, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);
    const int16x8_t vacc4x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567), voutput_zero_point);
    const int16x8_t vacc5x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567), voutput_zero_point);
    const int16x8_t vacc6x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc6x0123), vacc6x4567), voutput_zero_point);
    const int16x8_t vacc7x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc7x0123), vacc7x4567), voutput_zero_point);

    uint8x16_t vout0x01234567_1x01234567 = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc1x01234567);
    uint8x16_t vout2x01234567_3x01234567 = vqmovun_high_s16(vqmovun_s16(vacc2x01234567), vacc3x01234567);
    uint8x16_t vout4x01234567_5x01234567 = vqmovun_high_s16(vqmovun_s16(vacc4x01234567), vacc5x01234567);
    uint8x16_t vout6x01234567_7x01234567 = vqmovun_high_s16(vqmovun_s16(vacc6x01234567), vacc7x01234567);
#else
    const int16x8_t vacc0x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);
    const int16x8_t vacc4x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567)), voutput_zero_point);
    const int16x8_t vacc5x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567)), voutput_zero_point);
    const int16x8_t vacc6x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc6x0123), vqmovn_s32(vacc6x4567)), voutput_zero_point);
    const int16x8_t vacc7x01234567 =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vacc7x0123), vqmovn_s32(vacc7x4567)), voutput_zero_point);

    uint8x16_t vout0x01234567_1x01234567 = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc1x01234567));
    uint8x16_t vout2x01234567_3x01234567 = vcombine_u8(vqmovun_s16(vacc2x01234567), vqmovun_s16(vacc3x01234567));
    uint8x16_t vout4x01234567_5x01234567 = vcombine_u8(vqmovun_s16(vacc4x01234567), vqmovun_s16(vacc5x01234567));
    uint8x16_t vout6x01234567_7x01234567 = vcombine_u8(vqmovun_s16(vacc6x01234567), vqmovun_s16(vacc7x01234567));
#endif
    const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
    const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_u8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_u8(vout2x01234567_3x01234567, voutput_min);
    vout4x01234567_5x01234567 = vmaxq_u8(vout4x01234567_5x01234567, voutput_min);
    vout6x01234567_7x01234567 = vmaxq_u8(vout6x01234567_7x01234567, voutput_min);
    vout0x01234567_1x01234567 = vminq_u8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_u8(vout2x01234567_3x01234567, voutput_max);
    vout4x01234567_5x01234567 = vminq_u8(vout4x01234567_5x01234567, voutput_max);
    vout6x01234567_7x01234567 = vminq_u8(vout6x01234567_7x01234567, voutput_max);

    if (nc >= 8) {
      vst1_u8(c0, vget_low_u8(vout0x01234567_1x01234567)); c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      vst1_u8(c1, vget_high_u8(vout0x01234567_1x01234567)); c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      vst1_u8(c2, vget_low_u8(vout2x01234567_3x01234567)); c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      vst1_u8(c3, vget_high_u8(vout2x01234567_3x01234567)); c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      vst1_u8(c4, vget_low_u8(vout4x01234567_5x01234567)); c4 = (uint8_t*) ((uintptr_t) c4 + cn_stride);
      vst1_u8(c5, vget_high_u8(vout4x01234567_5x01234567)); c5 = (uint8_t*) ((uintptr_t) c5 + cn_stride);
      vst1_u8(c6, vget_low_u8(vout6x01234567_7x01234567)); c6 = (uint8_t*) ((uintptr_t) c6 + cn_stride);
      vst1_u8(c7, vget_high_u8(vout6x01234567_7x01234567)); c7 = (uint8_t*) ((uintptr_t) c7 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint8_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint8_t*) ((uintptr_t) a5 - kc);
      a6 = (const uint8_t*) ((uintptr_t) a6 - kc);
      a7 = (const uint8_t*) ((uintptr_t) a7 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_u8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_u8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c4, 1), vreinterpretq_u32_u8(vout4x01234567_5x01234567), 0); c4 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c5, 1), vreinterpretq_u32_u8(vout4x01234567_5x01234567), 2); c5 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c6, 1), vreinterpretq_u32_u8(vout6x01234567_7x01234567), 0); c6 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c7, 1), vreinterpretq_u32_u8(vout6x01234567_7x01234567), 2); c7 += 4;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
        vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_u8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_u8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c4, 1), vreinterpretq_u16_u8(vout4x01234567_5x01234567), 0); c4 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c5, 1), vreinterpretq_u16_u8(vout4x01234567_5x01234567), 4); c5 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c6, 1), vreinterpretq_u16_u8(vout6x01234567_7x01234567), 0); c6 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c7, 1), vreinterpretq_u16_u8(vout6x01234567_7x01234567), 4); c7 += 2;
        vout0x01234567_1x01234567 = vextq_u8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_u8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout4x01234567_5x01234567 = vextq_u8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
        vout6x01234567_7x01234567 = vextq_u8(vout6x01234567_7x01234567, vout6x01234567_7x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_u8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_u8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_u8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_u8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_u8(c4, vout4x01234567_5x01234567, 0);
        vst1q_lane_u8(c5, vout4x01234567_5x01234567, 8);
        vst1q_lane_u8(c6, vout6x01234567_7x01234567, 0);
        vst1q_lane_u8(c7, vout6x01234567_7x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
