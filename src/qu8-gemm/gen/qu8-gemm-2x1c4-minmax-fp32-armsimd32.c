// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-armsimd32.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_acle.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/gemm.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32(
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
    int32_t vacc1x0 = vacc0x0;
    w = (const void*) ((const int32_t*) w + 1);

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

      k -= 4 * sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc1x0 = (float) vacc1x0;

    vfpacc0x0 *= vscale;
    vfpacc1x0 *= vscale;

    vfpacc0x0 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);

    const int32_t vmagic_bias_less_zero_point = params->fp32_armsimd32.magic_bias_less_zero_point;
    vout0x0 = __qsub(vout0x0, vmagic_bias_less_zero_point);
    vout1x0 = __qsub(vout1x0, vmagic_bias_less_zero_point);

    vout0x0 = __usat(vout0x0, 8);
    vout1x0 = __usat(vout1x0, 8);

    const uint32_t vout0 = (uint32_t) vout0x0;
    const uint32_t vout1 = (uint32_t) vout1x0;

    uint32_t vout = (uint32_t) (uint16_t) vout0 | (vout1 << 16);

    const int8x4_t voutput_min = (int8x4_t) params->fp32_armsimd32.output_min;
    __usub8((int8x4_t) vout, voutput_min);
    vout = (uint32_t) __sel((uint8x4_t) vout, (uint8x4_t) voutput_min);

    const int8x4_t voutput_max = (int8x4_t) params->fp32_armsimd32.output_max;
    __usub8((int8x4_t) vout, voutput_max);
    vout = (uint32_t) __sel((uint8x4_t) voutput_max, (uint8x4_t) vout);

    *c0 = (uint8_t) vout;
    vout >>= 16;
    *c1 = (uint8_t) vout;

    a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
    a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

    c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
    c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

    nc -= 1;
  } while (nc != 0);
}
