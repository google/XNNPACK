// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f16_vmul_minmax_ukernel__neonfp16arith_x16(
    size_t n,
    const void* restrict a_ptr,
    const void* restrict b_ptr,
    void* restrict y_ptr,
    const struct xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(__fp16) == 0);
  assert(a_ptr != NULL);
  assert(b_ptr != NULL);
  assert(y_ptr != NULL);

  const __fp16* a = (const __fp16*) a_ptr;
  const __fp16* b = (const __fp16*) b_ptr;
  __fp16* y = (__fp16*) y_ptr;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));

  for (; n >= 16 * sizeof(__fp16); n -= 16 * sizeof(__fp16)) {
    const float16x8_t va01234567 = vld1q_f16(a); a += 8;
    const float16x8_t vb01234567 = vld1q_f16(b); b += 8;
    const float16x8_t va456789AB = vld1q_f16(a); a += 8;
    const float16x8_t vb456789AB = vld1q_f16(b); b += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vmulq_f16(va456789AB, vb456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_f16(y, vy01234567); y += 8;
    vst1q_f16(y, vy456789AB); y += 8;
  }
  for (; n >= 8 * sizeof(__fp16); n -= 8 * sizeof(__fp16)) {
    const float16x8_t va01234567 = vld1q_f16(a); a += 8;
    const float16x8_t vb01234567 = vld1q_f16(b); b += 8;

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_f16(y, vy01234567); y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    const float16x8_t va01234567 = vld1q_f16(a);
    const float16x8_t vb01234567 = vld1q_f16(b);

    float16x8_t vy01234567 = vmulq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (n & (4 * sizeof(__fp16))) {
      vst1_f16(y, vy0123); y += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (n & (2 * sizeof(__fp16))) {
      vst1_lane_u32(__builtin_assume_aligned(y, 1), vreinterpret_u32_f16(vy0123), 0); y += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (n & (1 * sizeof(__fp16))) {
      vst1_lane_f16(y, vy0123, 0);
    }
  }
}
