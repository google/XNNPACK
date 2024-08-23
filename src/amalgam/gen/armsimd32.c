// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <arm_acle.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vlrelu.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    w = (const void*) ((const int32_t*) w + 2);

    size_t k = kc;
    do {
      const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;

      const int16x2_t va0c02 = __sxtb16(va0);
      const int16x2_t va0c13 = __sxtb16(__ror(va0, 8));

      const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb0c02 = __sxtb16(vb0);

      vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);

      const int16x2_t vb0c13 = __sxtb16(__ror(vb0, 8));
      vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
      const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb1c02 = __sxtb16(vb1);

      vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);

      const int16x2_t vb1c13 = __sxtb16(__ror(vb1, 8));
      vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);

      k -= 4 * sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout0x1 = __qsub(vout0x1, vmagic_bias_less_zero_point);

    vout0x0 = __ssat(vout0x0, 8);
    vout0x1 = __ssat(vout0x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);

    uint32_t vout = vout0;

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __ssub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __ssub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      *c0 = (int8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const void*) ((const int32_t*) w + 2);

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

    uint32_t vout = (uint32_t) (uint16_t) vout0 | (vout1 << 16);

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __ssub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __ssub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);
      vout >>= 16;
      unaligned_store_u16(c1, (uint16_t) vout);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      *c0 = (int8_t) vout;
      vout >>= 16;
      *c1 = (int8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;

  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    w = (const void*) ((const int32_t*) w + 2);

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
        const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;

        const int16x2_t va0c02 = __sxtb16(va0);
        const int16x2_t va0c13 = __sxtb16(__ror(va0, 8));

        const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb0c02 = __sxtb16(vb0);

        vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);

        const int16x2_t vb0c13 = __sxtb16(__ror(vb0, 8));
        vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
        const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb1c02 = __sxtb16(vb1);

        vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);

        const int16x2_t vb1c13 = __sxtb16(__ror(vb1, 8));
        vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);

        k -= 4 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout0x1 = __qsub(vout0x1, vmagic_bias_less_zero_point);

    vout0x0 = __ssat(vout0x0, 8);
    vout0x1 = __ssat(vout0x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);

    uint32_t vout = vout0;

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __ssub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __ssub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      *c0 = (int8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

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

void xnn_qs8_vcvt_ukernel__armsimd32_u8(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x2_t vminus_input_zero_point = (int16x2_t) broadcast2x_uint16(-params->scalar.input_zero_point);
  const int32_t vbias = ((int32_t) params->scalar.output_zero_point << 1) + INT32_C(1);
  const int32_t vmultiplier = (int32_t) params->scalar.multiplier << 9;
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_indexed_load_u32(input, 0);
    const int8x4_t vx4567 = (int8x4_t) unaligned_indexed_load_u32(input, 1);
    input += 8;

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));
    const int16x2_t vx46 = __sxtab16(vminus_input_zero_point, vx4567);
    const int16x2_t vx57 = __sxtab16(vminus_input_zero_point, __ror(vx4567, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);
    int32_t vacc4 = __smlawb(vmultiplier, vx46, vbias);
    int32_t vacc5 = __smlawb(vmultiplier, vx57, vbias);
    int32_t vacc6 = __smlawt(vmultiplier, vx46, vbias);
    int32_t vacc7 = __smlawt(vmultiplier, vx57, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 1), 8);
    vacc4 = __ssat(math_asr_s32(vacc4, 1), 8);
    vacc5 = __ssat(math_asr_s32(vacc5, 1), 8);
    vacc6 = __ssat(math_asr_s32(vacc6, 1), 8);
    vacc7 = __ssat(math_asr_s32(vacc7, 1), 8);

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output[4] = (int8_t) vacc4;
    output[5] = (int8_t) vacc5;
    output[6] = (int8_t) vacc6;
    output[7] = (int8_t) vacc7;
    output += 8;
  }
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);
    input += 4;

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 1), 8);

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);

    const int16x2_t vx02 = __sxtab16(vminus_input_zero_point, vx0123);
    const int16x2_t vx13 = __sxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    const int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 1), 8);

    if (batch & (2 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
      output[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 1), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
    }
  }
}

void xnn_qs8_vlrelu_ukernel__armsimd32_u4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int16x2_t vinput_zero_point = (int16x2_t) broadcast2x_uint16(params->scalar.input_zero_point);
  const int16x2_t vpositive_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.positive_multiplier);
  const int16x2_t vnegative_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.negative_multiplier);
  const int32_t vbias = (params->scalar.output_zero_point << 8) + 0x80;
  for (; batch >= 4 * sizeof(int8_t); batch -= 4 * sizeof(int8_t)) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);
    input += 4;

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __ssat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __ssat(math_asr_s32(vacc3, 8), 8);

    output[0] = (int8_t) vacc0;
    output[1] = (int8_t) vacc1;
    output[2] = (int8_t) vacc2;
    output[3] = (int8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int8x4_t vx0123 = (int8x4_t) unaligned_load_u32(input);

    int16x2_t vx02 = __sxtb16(vx0123);
    int16x2_t vx13 = __sxtb16(__ror(vx0123, 8));

    vx02 = __ssub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __ssub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    const int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);

    vacc0 = __ssat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __ssat(math_asr_s32(vacc1, 8), 8);

    if (batch & (2 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
      output[1] = (int8_t) vacc1;
      vacc0 = __ssat(math_asr_s32(vacc2, 8), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      output[0] = (int8_t) vacc0;
    }
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;

  const int16x2_t vb_minus_zero_point = (int16x2_t) params->fp32_armsimd32.minus_kernel_zero_point;
  const float vscale = params->fp32_armsimd32.scale;
  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    w = (const void*) ((const int32_t*) w + 2);

    size_t k = kc;
    do {
      const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;

      const int16x2_t va0c02 = __uxtb16(va0);
      const int16x2_t va0c13 = __uxtb16(__ror(va0, 8));

      const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb0c02 = __uxtab16(vb_minus_zero_point, vb0);

      vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);

      const int16x2_t vb0c13 = __uxtab16(vb_minus_zero_point, __ror(vb0, 8));
      vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
      const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb1c02 = __uxtab16(vb_minus_zero_point, vb1);

      vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);

      const int16x2_t vb1c13 = __uxtab16(vb_minus_zero_point, __ror(vb1, 8));
      vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);

      k -= 4 * sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout0x1 = __qsub(vout0x1, vmagic_bias_less_zero_point);

    vout0x0 = __usat(vout0x0, 8);
    vout0x1 = __usat(vout0x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);

    uint32_t vout = vout0;

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __usub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __usub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      *c0 = (uint8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const int16x2_t vb_minus_zero_point = (int16x2_t) params->fp32_armsimd32.minus_kernel_zero_point;
  const float vscale = params->fp32_armsimd32.scale;
  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const void*) ((const int32_t*) w + 2);

    size_t k = kc;
    do {
      const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;
      const int8x4_t va1 = (int8x4_t) unaligned_load_s32(a1); a1 += 4;

      const int16x2_t va0c02 = __uxtb16(va0);
      const int16x2_t va0c13 = __uxtb16(__ror(va0, 8));
      const int16x2_t va1c02 = __uxtb16(va1);
      const int16x2_t va1c13 = __uxtb16(__ror(va1, 8));

      const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb0c02 = __uxtab16(vb_minus_zero_point, vb0);

      vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);
      vacc1x0 = __smlad(va1c02, vb0c02, vacc1x0);

      const int16x2_t vb0c13 = __uxtab16(vb_minus_zero_point, __ror(vb0, 8));
      vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
      vacc1x0 = __smlad(va1c13, vb0c13, vacc1x0);
      const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
      const int16x2_t vb1c02 = __uxtab16(vb_minus_zero_point, vb1);

      vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);
      vacc1x1 = __smlad(va1c02, vb1c02, vacc1x1);

      const int16x2_t vb1c13 = __uxtab16(vb_minus_zero_point, __ror(vb1, 8));
      vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);
      vacc1x1 = __smlad(va1c13, vb1c13, vacc1x1);

      k -= 4 * sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;

    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;

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

    vout0x0 = __usat(vout0x0, 8);
    vout0x1 = __usat(vout0x1, 8);
    vout1x0 = __usat(vout1x0, 8);
    vout1x1 = __usat(vout1x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);
    const uint32_t vout1 = (uint32_t) (uint8_t) vout1x0 | ((uint32_t) vout1x1 << 8);

    uint32_t vout = (uint32_t) (uint16_t) vout0 | (vout1 << 16);

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __usub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __usub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);
      vout >>= 16;
      unaligned_store_u16(c1, (uint16_t) vout);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      *c0 = (uint8_t) vout;
      vout >>= 16;
      *c1 = (uint8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x2c4__armsimd32(
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
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  uint8_t* c0 = c;

  const int16x2_t vb_minus_zero_point = (int16x2_t) params->fp32_armsimd32.minus_kernel_zero_point;
  const float vscale = params->fp32_armsimd32.scale;
  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;

        const int16x2_t va0c02 = __uxtb16(va0);
        const int16x2_t va0c13 = __uxtb16(__ror(va0, 8));

        const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb0c02 = __uxtab16(vb_minus_zero_point, vb0);

        vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);

        const int16x2_t vb0c13 = __uxtab16(vb_minus_zero_point, __ror(vb0, 8));
        vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
        const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb1c02 = __uxtab16(vb_minus_zero_point, vb1);

        vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);

        const int16x2_t vb1c13 = __uxtab16(vb_minus_zero_point, __ror(vb1, 8));
        vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);

        k -= 4 * sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout0x1 = __qsub(vout0x1, vmagic_bias_less_zero_point);

    vout0x0 = __usat(vout0x0, 8);
    vout0x1 = __usat(vout0x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);

    uint32_t vout = vout0;

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __usub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __usub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c0, (uint16_t) vout);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      *c0 = (uint8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_2x2c4__armsimd32(
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
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  kc = round_up_po2(kc, 4 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  const int16x2_t vb_minus_zero_point = (int16x2_t) params->fp32_armsimd32.minus_kernel_zero_point;
  const float vscale = params->fp32_armsimd32.scale;
  const float vmagic_bias = params->fp32_armsimd32.magic_bias;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const int8x4_t va0 = (int8x4_t) unaligned_load_s32(a0); a0 += 4;
        const int8x4_t va1 = (int8x4_t) unaligned_load_s32(a1); a1 += 4;

        const int16x2_t va0c02 = __uxtb16(va0);
        const int16x2_t va0c13 = __uxtb16(__ror(va0, 8));
        const int16x2_t va1c02 = __uxtb16(va1);
        const int16x2_t va1c13 = __uxtb16(__ror(va1, 8));

        const int8x4_t vb0 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb0c02 = __uxtab16(vb_minus_zero_point, vb0);

        vacc0x0 = __smlad(va0c02, vb0c02, vacc0x0);
        vacc1x0 = __smlad(va1c02, vb0c02, vacc1x0);

        const int16x2_t vb0c13 = __uxtab16(vb_minus_zero_point, __ror(vb0, 8));
        vacc0x0 = __smlad(va0c13, vb0c13, vacc0x0);
        vacc1x0 = __smlad(va1c13, vb0c13, vacc1x0);
        const int8x4_t vb1 = *((const int8x4_t*) w); w = (const int8_t*) w + 4;
        const int16x2_t vb1c02 = __uxtab16(vb_minus_zero_point, vb1);

        vacc0x1 = __smlad(va0c02, vb1c02, vacc0x1);
        vacc1x1 = __smlad(va1c02, vb1c02, vacc1x1);

        const int16x2_t vb1c13 = __uxtab16(vb_minus_zero_point, __ror(vb1, 8));
        vacc0x1 = __smlad(va0c13, vb1c13, vacc0x1);
        vacc1x1 = __smlad(va1c13, vb1c13, vacc1x1);

        k -= 4 * sizeof(uint8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;

    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;

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

    vout0x0 = __usat(vout0x0, 8);
    vout0x1 = __usat(vout0x1, 8);
    vout1x0 = __usat(vout1x0, 8);
    vout1x1 = __usat(vout1x1, 8);

    const uint32_t vout0 = (uint32_t) (uint8_t) vout0x0 | ((uint32_t) vout0x1 << 8);
    const uint32_t vout1 = (uint32_t) (uint8_t) vout1x0 | ((uint32_t) vout1x1 << 8);

    uint32_t vout = (uint32_t) (uint16_t) vout1 | (vout0 << 16);

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __usub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __usub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    if XNN_LIKELY(nc >= 2) {
      unaligned_store_u16(c1, (uint16_t) vout);
      vout >>= 16;
      unaligned_store_u16(c0, (uint16_t) vout);

      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      *c1 = (uint8_t) vout;
      vout >>= 16;
      *c0 = (uint8_t) vout;

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vcvt_ukernel__armsimd32_u8(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x2_t vminus_input_zero_point = (uint16x2_t) broadcast2x_uint16(-params->scalar.input_zero_point);
  const int32_t vbias = ((int32_t) params->scalar.output_zero_point << 1) + INT32_C(1);
  const int32_t vmultiplier = (int32_t) params->scalar.multiplier << 9;
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_indexed_load_u32(input, 0);
    const uint8x4_t vx4567 = (uint8x4_t) unaligned_indexed_load_u32(input, 1);
    input += 8;

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));
    const uint16x2_t vx46 = __uxtab16(vminus_input_zero_point, vx4567);
    const uint16x2_t vx57 = __uxtab16(vminus_input_zero_point, __ror(vx4567, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);
    int32_t vacc4 = __smlawb(vmultiplier, vx46, vbias);
    int32_t vacc5 = __smlawb(vmultiplier, vx57, vbias);
    int32_t vacc6 = __smlawt(vmultiplier, vx46, vbias);
    int32_t vacc7 = __smlawt(vmultiplier, vx57, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 1), 8);
    vacc4 = __usat(math_asr_s32(vacc4, 1), 8);
    vacc5 = __usat(math_asr_s32(vacc5, 1), 8);
    vacc6 = __usat(math_asr_s32(vacc6, 1), 8);
    vacc7 = __usat(math_asr_s32(vacc7, 1), 8);

    output[0] = (uint8_t) vacc0;
    output[1] = (uint8_t) vacc1;
    output[2] = (uint8_t) vacc2;
    output[3] = (uint8_t) vacc3;
    output[4] = (uint8_t) vacc4;
    output[5] = (uint8_t) vacc5;
    output[6] = (uint8_t) vacc6;
    output[7] = (uint8_t) vacc7;
    output += 8;
  }
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);
    input += 4;

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);
    int32_t vacc3 = __smlawt(vmultiplier, vx13, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 1), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 1), 8);

    output[0] = (uint8_t) vacc0;
    output[1] = (uint8_t) vacc1;
    output[2] = (uint8_t) vacc2;
    output[3] = (uint8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);

    const uint16x2_t vx02 = __uxtab16(vminus_input_zero_point, vx0123);
    const uint16x2_t vx13 = __uxtab16(vminus_input_zero_point, __ror(vx0123, 8));

    int32_t vacc0 = __smlawb(vmultiplier, vx02, vbias);
    int32_t vacc1 = __smlawb(vmultiplier, vx13, vbias);
    const int32_t vacc2 = __smlawt(vmultiplier, vx02, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 1), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 1), 8);

    if (batch & (2 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
      output[1] = (uint8_t) vacc1;
      vacc0 = __usat(math_asr_s32(vacc2, 1), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
    }
  }
}

void xnn_qu8_vlrelu_ukernel__armsimd32_u4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16x2_t vinput_zero_point = (uint16x2_t) broadcast2x_uint16(params->scalar.input_zero_point);
  const int16x2_t vpositive_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.positive_multiplier);
  const int16x2_t vnegative_multiplier = (int16x2_t) broadcast2x_uint16(-params->scalar.negative_multiplier);
  const int32_t vbias = (params->scalar.output_zero_point << 8) + 0x80;
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);
    input += 4;

    uint16x2_t vx02 = __uxtb16(vx0123);
    uint16x2_t vx13 = __uxtb16(__ror(vx0123, 8));

    vx02 = __usub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __usub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);
    int32_t vacc3 = __smlatt(vmultiplier13, vx13, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 8), 8);
    vacc2 = __usat(math_asr_s32(vacc2, 8), 8);
    vacc3 = __usat(math_asr_s32(vacc3, 8), 8);

    output[0] = (uint8_t) vacc0;
    output[1] = (uint8_t) vacc1;
    output[2] = (uint8_t) vacc2;
    output[3] = (uint8_t) vacc3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const uint8x4_t vx0123 = (uint8x4_t) unaligned_load_u32(input);

    uint16x2_t vx02 = __uxtb16(vx0123);
    uint16x2_t vx13 = __uxtb16(__ror(vx0123, 8));

    vx02 = __usub16(vinput_zero_point, vx02);
    const int16x2_t vmultiplier02 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);
    vx13 = __usub16(vinput_zero_point, vx13);
    const int16x2_t vmultiplier13 = (int16x2_t) __sel((uint8x4_t) vnegative_multiplier, (uint8x4_t) vpositive_multiplier);

    int32_t vacc0 = __smlabb(vmultiplier02, vx02, vbias);
    int32_t vacc1 = __smlabb(vmultiplier13, vx13, vbias);
    const int32_t vacc2 = __smlatt(vmultiplier02, vx02, vbias);

    vacc0 = __usat(math_asr_s32(vacc0, 8), 8);
    vacc1 = __usat(math_asr_s32(vacc1, 8), 8);

    if (batch & (2 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
      output[1] = (uint8_t) vacc1;
      vacc0 = __usat(math_asr_s32(vacc2, 8), 8);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      output[0] = (uint8_t) vacc0;
    }
  }
}
