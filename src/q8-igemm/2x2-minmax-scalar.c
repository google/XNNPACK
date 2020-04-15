// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/scalar-utils.h>
#include <xnnpack/igemm.h>


void xnn_q8_igemm_minmax_ukernel_2x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_q8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if (mr != 2) {
    c1 = c0;
  }

  const int32_t vb_zero_point = params->scalar.kernel_zero_point;

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;
        const int32_t va1 = (int32_t) (uint32_t) *a1++;

        const uint32_t vb0 = ((const uint8_t*) w)[0];
        const uint32_t vb1 = ((const uint8_t*) w)[1];
        w = (const void*) ((uintptr_t) w + 2 * sizeof(uint8_t));

        const int32_t vxb0 = (int32_t) vb0 - vb_zero_point;
        const int32_t vxb1 = (int32_t) vb1 - vb_zero_point;

        vacc0x0 += va0 * vxb0;
        vacc0x1 += va0 * vxb1;
        vacc1x0 += va1 * vxb0;
        vacc1x1 += va1 * vxb1;

      } while (--k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const int32_t vmultiplier = params->scalar.multiplier;
    const int64_t vproduct0x0 = (int64_t) vacc0x0 * (int64_t) vmultiplier;
    const int64_t vproduct0x1 = (int64_t) vacc0x1 * (int64_t) vmultiplier;
    const int64_t vproduct1x0 = (int64_t) vacc1x0 * (int64_t) vmultiplier;
    const int64_t vproduct1x1 = (int64_t) vacc1x1 * (int64_t) vmultiplier;

    const int64_t vq31rounding = INT64_C(0x40000000);
    const int32_t vq31product0x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x0 + vq31rounding) >> 31);
    const int32_t vq31product0x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct0x1 + vq31rounding) >> 31);
    const int32_t vq31product1x0 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x0 + vq31rounding) >> 31);
    const int32_t vq31product1x1 = (int32_t) (uint32_t) ((uint64_t) (vproduct1x1 + vq31rounding) >> 31);

    const int32_t vremainder_mask = params->scalar.remainder_mask;
    const int32_t vremainder0x0 = (vq31product0x0 & vremainder_mask) - (int32_t) (vq31product0x0 < 0);
    const int32_t vremainder0x1 = (vq31product0x1 & vremainder_mask) - (int32_t) (vq31product0x1 < 0);
    const int32_t vremainder1x0 = (vq31product1x0 & vremainder_mask) - (int32_t) (vq31product1x0 < 0);
    const int32_t vremainder1x1 = (vq31product1x1 & vremainder_mask) - (int32_t) (vq31product1x1 < 0);

    const uint32_t vshift = params->scalar.shift;
    const int32_t vremainder_threshold = params->scalar.remainder_threshold;
    int32_t vout0x0 = asr_s32(vq31product0x0, vshift) + (int32_t) (vremainder0x0 > vremainder_threshold);
    int32_t vout0x1 = asr_s32(vq31product0x1, vshift) + (int32_t) (vremainder0x1 > vremainder_threshold);
    int32_t vout1x0 = asr_s32(vq31product1x0, vshift) + (int32_t) (vremainder1x0 > vremainder_threshold);
    int32_t vout1x1 = asr_s32(vq31product1x1, vshift) + (int32_t) (vremainder1x1 > vremainder_threshold);

    const int32_t vout_min = params->scalar.output_min_less_zero_point;
    vout0x0 = vout0x0 < vout_min ? vout_min : vout0x0;
    vout0x1 = vout0x1 < vout_min ? vout_min : vout0x1;
    vout1x0 = vout1x0 < vout_min ? vout_min : vout1x0;
    vout1x1 = vout1x1 < vout_min ? vout_min : vout1x1;

    const int32_t vout_max = params->scalar.output_max_less_zero_point;
    vout0x0 = vout0x0 > vout_max ? vout_max : vout0x0;
    vout0x1 = vout0x1 > vout_max ? vout_max : vout0x1;
    vout1x0 = vout1x0 > vout_max ? vout_max : vout1x0;
    vout1x1 = vout1x1 > vout_max ? vout_max : vout1x1;

    const int32_t voutput_zero_point = params->scalar.output_zero_point;
    vout0x0 += voutput_zero_point;
    vout0x1 += voutput_zero_point;
    vout1x0 += voutput_zero_point;
    vout1x1 += voutput_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;

      c1 += cn_stride;
      c0 += cn_stride;

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 2;
    } else {
      c1[0] = (uint8_t) vout1x0;
      c0[0] = (uint8_t) vout0x0;

      nc = 0;
    }
  } while (nc != 0);
}
