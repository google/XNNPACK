// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#include "src/xnnpack/unaligned.h"


void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_qb4w_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params)
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  size_t bl = params->scalar.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);

  const int8_t* a0 = a;
  xnn_float16* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  xnn_float16* c1 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }


  kc = round_up_po2(kc, 2);
  do {
    const float vksum0 = unaligned_indexed_load_f32(w, 0);
    const float vksum1 = unaligned_indexed_load_f32(w, 1);
    const float vinput_zero_point0 = (const float) quantization_params[0].zero_point;
    float vout0x0 = vksum0 * vinput_zero_point0;
    float vout0x1 = vksum1 * vinput_zero_point0;
    const float vinput_zero_point1 = (const float) quantization_params[1].zero_point;
    float vout1x0 = vksum0 * vinput_zero_point1;
    float vout1x1 = vksum1 * vinput_zero_point1;
    w = (const float*) w + 2;

    for (size_t kb=0; kb < kc; kb += bl) {
      int32_t vacc0x0 = 0;
      int32_t vacc0x1 = 0;
      int32_t vacc1x0 = 0;
      int32_t vacc1x1 = 0;
      size_t k = bl;
      for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
        const int32_t va0c0 = (int32_t) a0[0];
        const int32_t va0c1 = (int32_t) a0[1];
        a0 += 2;
        const int32_t va1c0 = (int32_t) a1[0];
        const int32_t va1c1 = (int32_t) a1[1];
        a1 += 2;

        const uint8_t vbi0 = ((const uint8_t*) w)[0];
        const uint8_t vbi1 = ((const uint8_t*) w)[1];
        w = (const uint8_t*) w + 2;
        const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
        const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
        const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
        const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);

        vacc0x0 += va0c0 * vb0c0;
        vacc0x1 += va0c0 * vb1c0;
        vacc1x0 += va1c0 * vb0c0;
        vacc1x1 += va1c0 * vb1c0;
        vacc0x0 += va0c1 * vb0c1;
        vacc0x1 += va0c1 * vb1c1;
        vacc1x0 += va1c1 * vb0c1;
        vacc1x1 += va1c1 * vb1c1;
    }
    // accumulate in float
      float vf0x0 = vacc0x0;
      const float vfilter_output_scale0 = math_cvt_fp32_bf16(unaligned_indexed_load_u16(w, 0));
      float vf0x1 = vacc0x1;
      const float vfilter_output_scale1 = math_cvt_fp32_bf16(unaligned_indexed_load_u16(w, 1));
      float vf1x0 = vacc1x0;
      float vf1x1 = vacc1x1;

      vf0x0 *= vfilter_output_scale0;
      vout0x0 += vf0x0;
      vf0x1 *= vfilter_output_scale1;
      vout0x1 += vf0x1;
      vf1x0 *= vfilter_output_scale0;
      vout1x0 += vf1x0;
      vf1x1 *= vfilter_output_scale1;
      vout1x1 += vf1x1;
      w = (const uint16_t*) w + 2;
    }


    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    vout1x0 *= vinput_scale1;
    vout1x1 *= vinput_scale1;


    const float vbias0 = unaligned_indexed_load_f32(w, 0);
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    const float vbias1 = unaligned_indexed_load_f32(w, 1);
    vout0x1 += vbias1;
    vout1x1 += vbias1;

    w = (const float*) w + 2;

    const float voutput_min = xnn_float16_to_float(params->scalar.min);
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout1x0 = math_max_f32(vout1x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);
    vout1x1 = math_max_f32(vout1x1, voutput_min);

    const float voutput_max = xnn_float16_to_float(params->scalar.max);
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout1x0 = math_min_f32(vout1x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);
    vout1x1 = math_min_f32(vout1x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = xnn_float16_from_float(vout0x0);
      c0[1] = xnn_float16_from_float(vout0x1);
      c1[0] = xnn_float16_from_float(vout1x0);
      c1[1] = xnn_float16_from_float(vout1x1);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);
      c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = xnn_float16_from_float(vout0x0);
        c1[0] = xnn_float16_from_float(vout1x0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
