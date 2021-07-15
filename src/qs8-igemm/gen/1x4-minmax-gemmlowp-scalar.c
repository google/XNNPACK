// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/gemm.h>


void xnn_qs8_igemm_minmax_gemmlowp_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        w = (const void*) ((const int8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const int32_t vmultiplier = params->gemmlowp_scalar.multiplier;
    const int64_t vproduct0x0 = (int64_t) vacc0x0 * (int64_t) vmultiplier;
    const int64_t vproduct0x1 = (int64_t) vacc0x1 * (int64_t) vmultiplier;
    const int64_t vproduct0x2 = (int64_t) vacc0x2 * (int64_t) vmultiplier;
    const int64_t vproduct0x3 = (int64_t) vacc0x3 * (int64_t) vmultiplier;

    const int64_t vq31rounding = INT64_C(0x40000000);
    const int32_t vq31product0x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x0 + vq31rounding) >> 31);
    const int32_t vq31product0x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x1 + vq31rounding) >> 31);
    const int32_t vq31product0x2 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x2 + vq31rounding) >> 31);
    const int32_t vq31product0x3 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x3 + vq31rounding) >> 31);

    const int32_t vremainder_mask = params->gemmlowp_scalar.remainder_mask;
    const int32_t vremainder0x0 = (vq31product0x0 & vremainder_mask) - (int32_t) (vq31product0x0 < 0);
    const int32_t vremainder0x1 = (vq31product0x1 & vremainder_mask) - (int32_t) (vq31product0x1 < 0);
    const int32_t vremainder0x2 = (vq31product0x2 & vremainder_mask) - (int32_t) (vq31product0x2 < 0);
    const int32_t vremainder0x3 = (vq31product0x3 & vremainder_mask) - (int32_t) (vq31product0x3 < 0);

    const uint32_t vshift = params->gemmlowp_scalar.shift;
    const int32_t vremainder_threshold = params->gemmlowp_scalar.remainder_threshold;
    int32_t vout0x0 = asr_s32(vq31product0x0, vshift) + (int32_t) (vremainder0x0 > vremainder_threshold);
    int32_t vout0x1 = asr_s32(vq31product0x1, vshift) + (int32_t) (vremainder0x1 > vremainder_threshold);
    int32_t vout0x2 = asr_s32(vq31product0x2, vshift) + (int32_t) (vremainder0x2 > vremainder_threshold);
    int32_t vout0x3 = asr_s32(vq31product0x3, vshift) + (int32_t) (vremainder0x3 > vremainder_threshold);

    const int32_t voutput_min_less_zero_point = params->gemmlowp_scalar.output_min_less_zero_point;
    vout0x0 = math_max_s32(vout0x0, voutput_min_less_zero_point);
    vout0x1 = math_max_s32(vout0x1, voutput_min_less_zero_point);
    vout0x2 = math_max_s32(vout0x2, voutput_min_less_zero_point);
    vout0x3 = math_max_s32(vout0x3, voutput_min_less_zero_point);

    const int32_t voutput_max_less_zero_point = params->gemmlowp_scalar.output_max_less_zero_point;
    vout0x0 = math_min_s32(vout0x0, voutput_max_less_zero_point);
    vout0x1 = math_min_s32(vout0x1, voutput_max_less_zero_point);
    vout0x2 = math_min_s32(vout0x2, voutput_max_less_zero_point);
    vout0x3 = math_min_s32(vout0x3, voutput_max_less_zero_point);

    const int32_t voutput_zero_point = params->gemmlowp_scalar.output_zero_point;
    vout0x0 += voutput_zero_point;
    vout0x1 += voutput_zero_point;
    vout0x2 += voutput_zero_point;
    vout0x3 += voutput_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
