// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/igemm.h"
#include "xnnpack/prefetch.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
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

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 += 8;
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 += 8;
        const int16x8_t vxa3 = vmovl_s8(va3);
        const int8x8_t va4 = vld1_s8(a4); a4 += 8;
        const int16x8_t vxa4 = vmovl_s8(va4);
        const int8x8_t va5 = vld1_s8(a5); a5 += 8;
        const int16x8_t vxa5 = vmovl_s8(va5);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);

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
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);

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
        const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);

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
        const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);

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

        xnn_prefetch_to_l1((const int8_t*) w + 448);

        const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);

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
        const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);

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
        const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);

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
        const int8x8_t vb01234567c7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);

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

        k -= 8 * sizeof(int8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vmovl_s8(va0);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const int16x8_t vxa1 = vmovl_s8(va1);
        const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
        const int16x8_t vxa2 = vmovl_s8(va2);
        const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);
        const int16x8_t vxa3 = vmovl_s8(va3);
        const int8x8_t va4 = vld1_s8(a4); a4 = (const int8_t*) ((uintptr_t) a4 + k);
        const int16x8_t vxa4 = vmovl_s8(va4);
        const int8x8_t va5 = vld1_s8(a5); a5 = (const int8_t*) ((uintptr_t) a5 + k);
        const int16x8_t vxa5 = vmovl_s8(va5);

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);

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

        if (k >= 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);

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

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);

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

            if (k >= 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);

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

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);

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

                if (k >= 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);

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

                  if (k > 6 * sizeof(int8_t)) {
                    const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
                    const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);

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
                  }
                }
              }
            }
          }
        }
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    // Post-accumulation work
    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vfpacc2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vfpacc2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vfpacc3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vfpacc3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vfpacc4x0123 = vcvtq_f32_s32(vacc4x0123);
    float32x4_t vfpacc4x4567 = vcvtq_f32_s32(vacc4x4567);
    float32x4_t vfpacc5x0123 = vcvtq_f32_s32(vacc5x0123);
    float32x4_t vfpacc5x4567 = vcvtq_f32_s32(vacc5x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    vfpacc2x0123 = vmulq_f32(vfpacc2x0123, vscale0123);
    vfpacc3x0123 = vmulq_f32(vfpacc3x0123, vscale0123);
    vfpacc4x0123 = vmulq_f32(vfpacc4x0123, vscale0123);
    vfpacc5x0123 = vmulq_f32(vfpacc5x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);
    vfpacc2x4567 = vmulq_f32(vfpacc2x4567, vscale4567);
    vfpacc3x4567 = vmulq_f32(vfpacc3x4567, vscale4567);
    vfpacc4x4567 = vmulq_f32(vfpacc4x4567, vscale4567);
    vfpacc5x4567 = vmulq_f32(vfpacc5x4567, vscale4567);

    const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
    vacc0x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x0123, vmagic_bias));
    vacc0x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x4567, vmagic_bias));
    vacc1x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc1x0123, vmagic_bias));
    vacc1x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc1x4567, vmagic_bias));
    vacc2x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc2x0123, vmagic_bias));
    vacc2x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc2x4567, vmagic_bias));
    vacc3x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc3x0123, vmagic_bias));
    vacc3x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc3x4567, vmagic_bias));
    vacc4x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4x0123, vmagic_bias));
    vacc4x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4x4567, vmagic_bias));
    vacc5x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc5x0123, vmagic_bias));
    vacc5x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc5x4567, vmagic_bias));

    const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
    vacc0x0123 = vqsubq_s32(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = vqsubq_s32(vacc0x4567, vmagic_bias_less_output_zero_point);
    vacc1x0123 = vqsubq_s32(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc1x4567 = vqsubq_s32(vacc1x4567, vmagic_bias_less_output_zero_point);
    vacc2x0123 = vqsubq_s32(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc2x4567 = vqsubq_s32(vacc2x4567, vmagic_bias_less_output_zero_point);
    vacc3x0123 = vqsubq_s32(vacc3x0123, vmagic_bias_less_output_zero_point);
    vacc3x4567 = vqsubq_s32(vacc3x4567, vmagic_bias_less_output_zero_point);
    vacc4x0123 = vqsubq_s32(vacc4x0123, vmagic_bias_less_output_zero_point);
    vacc4x4567 = vqsubq_s32(vacc4x4567, vmagic_bias_less_output_zero_point);
    vacc5x0123 = vqsubq_s32(vacc5x0123, vmagic_bias_less_output_zero_point);
    vacc5x4567 = vqsubq_s32(vacc5x4567, vmagic_bias_less_output_zero_point);

    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);
      int16x8_t vacc2x01234567 = vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567);
      int16x8_t vacc3x01234567 = vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567);
      int16x8_t vacc4x01234567 = vqmovn_high_s32(vqmovn_s32(vacc4x0123), vacc4x4567);
      int16x8_t vacc5x01234567 = vqmovn_high_s32(vqmovn_s32(vacc5x0123), vacc5x4567);


      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
      int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
      int8x16_t vout4x01234567_5x01234567 = vqmovn_high_s16(vqmovn_s16(vacc4x01234567), vacc5x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));
      int16x8_t vacc2x01234567 = vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567));
      int16x8_t vacc3x01234567 = vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567));
      int16x8_t vacc4x01234567 = vcombine_s16(vqmovn_s32(vacc4x0123), vqmovn_s32(vacc4x4567));
      int16x8_t vacc5x01234567 = vcombine_s16(vqmovn_s32(vacc5x0123), vqmovn_s32(vacc5x4567));


      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
      int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
      int8x16_t vout4x01234567_5x01234567 = vcombine_s8(vqmovn_s16(vacc4x01234567), vqmovn_s16(vacc5x01234567));
    #endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neon.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);
    vout4x01234567_5x01234567 = vmaxq_s8(vout4x01234567_5x01234567, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neon.output_max);
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);
    vout4x01234567_5x01234567 = vminq_s8(vout4x01234567_5x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c5 + 0, vget_high_s8(vout4x01234567_5x01234567));
      vst1_s8(c4 + 0, vget_low_s8(vout4x01234567_5x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));

      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((void*) c5, vreinterpretq_u32_s8(vout4x01234567_5x01234567), 2); c5 += 4;
        vst1q_lane_u32((void*) c4, vreinterpretq_u32_s8(vout4x01234567_5x01234567), 0); c4 += 4;
        vst1q_lane_u32((void*) c3, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vst1q_lane_u32((void*) c2, vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c5, vreinterpretq_u16_s8(vout4x01234567_5x01234567), 4); c5 += 2;
        vst1q_lane_u16((void*) c4, vreinterpretq_u16_s8(vout4x01234567_5x01234567), 0); c4 += 2;
        vst1q_lane_u16((void*) c3, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vst1q_lane_u16((void*) c2, vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout4x01234567_5x01234567 = vextq_s8(vout4x01234567_5x01234567, vout4x01234567_5x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c5, vout4x01234567_5x01234567, 8);
        vst1q_lane_s8(c4, vout4x01234567_5x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
