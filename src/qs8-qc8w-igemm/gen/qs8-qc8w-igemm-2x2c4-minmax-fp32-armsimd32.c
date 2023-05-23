// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-armsimd32.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_acle.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/gemm.h>
#include <xnnpack/unaligned.h>


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;
        const int8x4_t va1 = (int8x4_t) unaligned_load_s32(a1); a1 += 4;

        const int16x2_t va0c02 = __sxtb16(va0);
        const int16x2_t va0c13 = __sxtb16(__ror(va0, 8));
        const int16x2_t va1c02 = __sxtb16(va1);
        const int16x2_t va1c13 = __sxtb16(__ror(va1, 8));

        const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb0c02 = __sxtb16(vb0);

        vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);
        vacc1x0 = __smlad(va1c02, vb0c02, vacc1x0);

        const int16x2_t vb0c13 = __sxtb16(__ror(vb0, 8));
        vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
        vacc1x0 = __smlad(va1c13, vb0c13, vacc1x0);
        const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb1c02 = __sxtb16(vb1);

        vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);
        vacc1x1 = __smlad(va1c02, vb1c02, vacc1x1);

        const int16x2_t vb1c13 = __sxtb16(__ror(vb1, 8));
        vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);
        vacc1x1 = __smlad(va1c13, vb1c13, vacc1x1);

        k -= 4 * sizeof(int8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout0x1 = __qsub(vout0x1, vmagic_bias_less_zero_point);
    vout1x0 = __qsub(vout1x0, vmagic_bias_less_zero_point);
    vout1x1 = __qsub(vout1x1, vmagic_bias_less_zero_point);

    vout0x0 = __ssat(vout0x0, 8);
    vout0x1 = __ssat(vout0x1, 8);
    vout1x0 = __ssat(vout1x0, 8);
    vout1x1 = __ssat(vout1x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);
    const uint32_t vout1 = (uint32_t) (uint8_t) vout1x0 | ((uint32_t) vout1x1 << 8);

    uint32_t vout = (uint32_t) (uint16_t) vout1 | (vout0 << 16);

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __ssub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __ssub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c1, (uint16_t) vout);
      vout >>= 16;
      unaligned_store_u16(c0, (uint16_t) vout);

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      *c1 = (int8_t) vout;
      vout >>= 16;
      *c0 = (int8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}
