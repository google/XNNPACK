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


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32x4_t vksum0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksum89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vksumCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
    const int32x4_t vinput_zero_point = vld1q_dup_s32(&quantization_params->zero_point);
    int32x4_t vacc0x0123 = vmulq_s32(vksum0123, vinput_zero_point);
    int32x4_t vacc0x4567 = vmulq_s32(vksum4567, vinput_zero_point);
    int32x4_t vacc0x89AB = vmulq_s32(vksum89AB, vinput_zero_point);
    int32x4_t vacc0xCDEF = vmulq_s32(vksumCDEF, vinput_zero_point);
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc1x89AB = vacc0x89AB;
    int32x4_t vacc1xCDEF = vacc0xCDEF;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc2x89AB = vacc0x89AB;
    int32x4_t vacc2xCDEF = vacc0xCDEF;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;
    int32x4_t vacc3x89AB = vacc0x89AB;
    int32x4_t vacc3xCDEF = vacc0xCDEF;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

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
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
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
        const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
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
        const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
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
        const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);

        xnn_prefetch_to_l1((const int8_t*) w + 448);
        xnn_prefetch_to_l1((const int8_t*) w + 512);

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
        const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
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
        const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
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
        const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
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
        const int8x8_t vb89ABCDEFc7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);

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

        const int8x8_t vb01234567c0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
        const int8x8_t vb89ABCDEFc0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);

        if (k >= 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c1 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
          const int8x8_t vb89ABCDEFc1 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
          vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
          vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
          vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
          vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);

          if (k > 2 * sizeof(int8_t)) {
            const int8x8_t vb01234567c2 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
            const int8x8_t vb89ABCDEFc2 = vld1_s8(w); w = (const int8_t*) w + 8;
            const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
            vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
            vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
            vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
            vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);

            if (k >= 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c3 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
              const int8x8_t vb89ABCDEFc3 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
              vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
              vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
              vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
              vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);

              if (k > 4 * sizeof(int8_t)) {
                const int8x8_t vb01234567c4 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
                const int8x8_t vb89ABCDEFc4 = vld1_s8(w); w = (const int8_t*) w + 8;
                const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
                vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
                vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
                vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
                vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);

                if (k >= 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c5 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                  const int8x8_t vb89ABCDEFc5 = vld1_s8(w); w = (const int8_t*) w + 8;
                  const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                  vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                  vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                  vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                  vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                  vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                  vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);

                  if (k > 6 * sizeof(int8_t)) {
                    const int8x8_t vb01234567c6 = vld1_s8(w); w = (const int8_t*) w + 8;
                    const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
                    const int8x8_t vb89ABCDEFc6 = vld1_s8(w); w = (const int8_t*) w + 8;
                    const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);

                    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                    vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
                    vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
                    vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
                    vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                    vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
                    vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
                    vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
                    vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                    vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
                    vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
                    vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
                  }
                }
              }
            }
          }
        }
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float32x4_t vout0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vout0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vout0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vout0xCDEF = vcvtq_f32_s32(vacc0xCDEF);
    float32x4_t vout1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vout1x4567 = vcvtq_f32_s32(vacc1x4567);
    float32x4_t vout1x89AB = vcvtq_f32_s32(vacc1x89AB);
    float32x4_t vout1xCDEF = vcvtq_f32_s32(vacc1xCDEF);
    float32x4_t vout2x0123 = vcvtq_f32_s32(vacc2x0123);
    float32x4_t vout2x4567 = vcvtq_f32_s32(vacc2x4567);
    float32x4_t vout2x89AB = vcvtq_f32_s32(vacc2x89AB);
    float32x4_t vout2xCDEF = vcvtq_f32_s32(vacc2xCDEF);
    float32x4_t vout3x0123 = vcvtq_f32_s32(vacc3x0123);
    float32x4_t vout3x4567 = vcvtq_f32_s32(vacc3x4567);
    float32x4_t vout3x89AB = vcvtq_f32_s32(vacc3x89AB);
    float32x4_t vout3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    const float32x4_t vinput_scale = vld1q_dup_f32(&quantization_params->inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale);
    vout0x89AB = vmulq_f32(vout0x89AB, vinput_scale);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinput_scale);
    vout1x0123 = vmulq_f32(vout1x0123, vinput_scale);
    vout1x4567 = vmulq_f32(vout1x4567, vinput_scale);
    vout1x89AB = vmulq_f32(vout1x89AB, vinput_scale);
    vout1xCDEF = vmulq_f32(vout1xCDEF, vinput_scale);
    vout2x0123 = vmulq_f32(vout2x0123, vinput_scale);
    vout2x4567 = vmulq_f32(vout2x4567, vinput_scale);
    vout2x89AB = vmulq_f32(vout2x89AB, vinput_scale);
    vout2xCDEF = vmulq_f32(vout2xCDEF, vinput_scale);
    vout3x0123 = vmulq_f32(vout3x0123, vinput_scale);
    vout3x4567 = vmulq_f32(vout3x4567, vinput_scale);
    vout3x89AB = vmulq_f32(vout3x89AB, vinput_scale);
    vout3xCDEF = vmulq_f32(vout3xCDEF, vinput_scale);

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vfmaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vfmaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vfmaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #else
      vout0x0123 = vmlaq_f32(vbias0123, vout0x0123, vfilter_output_scale0123);
      vout1x0123 = vmlaq_f32(vbias0123, vout1x0123, vfilter_output_scale0123);
      vout2x0123 = vmlaq_f32(vbias0123, vout2x0123, vfilter_output_scale0123);
      vout3x0123 = vmlaq_f32(vbias0123, vout3x0123, vfilter_output_scale0123);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vfmaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vfmaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vfmaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #else
      vout0x4567 = vmlaq_f32(vbias4567, vout0x4567, vfilter_output_scale4567);
      vout1x4567 = vmlaq_f32(vbias4567, vout1x4567, vfilter_output_scale4567);
      vout2x4567 = vmlaq_f32(vbias4567, vout2x4567, vfilter_output_scale4567);
      vout3x4567 = vmlaq_f32(vbias4567, vout3x4567, vfilter_output_scale4567);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vfmaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vfmaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vfmaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #else
      vout0x89AB = vmlaq_f32(vbias89AB, vout0x89AB, vfilter_output_scale89AB);
      vout1x89AB = vmlaq_f32(vbias89AB, vout1x89AB, vfilter_output_scale89AB);
      vout2x89AB = vmlaq_f32(vbias89AB, vout2x89AB, vfilter_output_scale89AB);
      vout3x89AB = vmlaq_f32(vbias89AB, vout3x89AB, vfilter_output_scale89AB);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;
    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vfmaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vfmaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vfmaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #else
      vout0xCDEF = vmlaq_f32(vbiasCDEF, vout0xCDEF, vfilter_output_scaleCDEF);
      vout1xCDEF = vmlaq_f32(vbiasCDEF, vout1xCDEF, vfilter_output_scaleCDEF);
      vout2xCDEF = vmlaq_f32(vbiasCDEF, vout2xCDEF, vfilter_output_scaleCDEF);
      vout3xCDEF = vmlaq_f32(vbiasCDEF, vout3xCDEF, vfilter_output_scaleCDEF);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(&c3[0], vout3x0123);
      vst1q_f32(&c3[4], vout3x4567);
      vst1q_f32(&c3[8], vout3x89AB);
      vst1q_f32(&c3[12], vout3xCDEF);
      vst1q_f32(&c2[0], vout2x0123);
      vst1q_f32(&c2[4], vout2x4567);
      vst1q_f32(&c2[8], vout2x89AB);
      vst1q_f32(&c2[12], vout2xCDEF);
      vst1q_f32(&c1[0], vout1x0123);
      vst1q_f32(&c1[4], vout1x4567);
      vst1q_f32(&c1[8], vout1x89AB);
      vst1q_f32(&c1[12], vout1xCDEF);
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);
      vst1q_f32(&c0[8], vout0x89AB);
      vst1q_f32(&c0[12], vout0xCDEF);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c3, vout3x0123); c3 += 4;
        vout3x0123 = vout3x89AB;
        vst1q_f32(c2, vout2x0123); c2 += 4;
        vout2x0123 = vout2x89AB;
        vst1q_f32(c1, vout1x0123); c1 += 4;
        vout1x0123 = vout1x89AB;
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x89AB;
        vst1q_f32(c3, vout3x4567); c3 += 4;
        vout3x4567 = vout3xCDEF;
        vst1q_f32(c2, vout2x4567); c2 += 4;
        vout2x4567 = vout2xCDEF;
        vst1q_f32(c1, vout1x4567); c1 += 4;
        vout1x4567 = vout1xCDEF;
        vst1q_f32(c0, vout0x4567); c0 += 4;
        vout0x4567 = vout0xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c3, vout3x0123); c3 += 4;
        vout3x0123 = vout3x4567;
        vst1q_f32(c2, vout2x0123); c2 += 4;
        vout2x0123 = vout2x4567;
        vst1q_f32(c1, vout1x0123); c1 += 4;
        vout1x0123 = vout1x4567;
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x4567;
      }
      float32x2_t vout3x01 = vget_low_f32(vout3x0123);
      float32x2_t vout2x01 = vget_low_f32(vout2x0123);
      float32x2_t vout1x01 = vget_low_f32(vout1x0123);
      float32x2_t vout0x01 = vget_low_f32(vout0x0123);
      if (nc & 2) {
        vst1_f32(c3, vout3x01); c3 += 2;
        vst1_f32(c2, vout2x01); c2 += 2;
        vst1_f32(c1, vout1x01); c1 += 2;
        vst1_f32(c0, vout0x01); c0 += 2;
        vout3x01 = vget_high_f32(vout3x0123);
        vout2x01 = vget_high_f32(vout2x0123);
        vout1x01 = vget_high_f32(vout1x0123);
        vout0x01 = vget_high_f32(vout0x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c3, vout3x01, 0);
        vst1_lane_f32(c2, vout2x01, 0);
        vst1_lane_f32(c1, vout1x01, 0);
        vst1_lane_f32(c0, vout0x01, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
