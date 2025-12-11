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


void xnn_qd8_f32_qc2w_gemm_minmax_ukernel_1x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params,
    const float* row_sum,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_DISABLE_UBSAN
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;


  kc = round_up_po2(kc, 4);
  do {
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const float vkernel_zero_point0 = unaligned_indexed_load_f32(w, 2);
    const float vkernel_zero_point1 = unaligned_indexed_load_f32(w, 3);
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const int32_t va0c0 = (int32_t) (uint32_t) a0[0];
      const int32_t va0c1 = (int32_t) (uint32_t) a0[1];
      const int32_t va0c2 = (int32_t) (uint32_t) a0[2];
      const int32_t va0c3 = (int32_t) (uint32_t) a0[3];
      a0 += 4;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      w = (const uint8_t*) w + 2;
      const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 & 3);
      const int32_t vb0c1 = (int32_t) (int8_t) ((vbi0 >> 2) & 3);
      const int32_t vb0c2 = (int32_t) (int8_t) ((vbi0 >> 4) & 3);
      const int32_t vb0c3 = (int32_t) (int8_t) ((vbi0 >> 6) & 3);
      const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 & 3);
      const int32_t vb1c1 = (int32_t) (int8_t) ((vbi1 >> 2) & 3);
      const int32_t vb1c2 = (int32_t) (int8_t) ((vbi1 >> 4) & 3);
      const int32_t vb1c3 = (int32_t) (int8_t) ((vbi1 >> 6) & 3);

      vacc0x0 += va0c0 * vb0c0;
      vacc0x1 += va0c0 * vb1c0;
      vacc0x0 += va0c1 * vb0c1;
      vacc0x1 += va0c1 * vb1c1;
      vacc0x0 += va0c2 * vb0c2;
      vacc0x1 += va0c2 * vb1c2;
      vacc0x0 += va0c3 * vb0c3;
      vacc0x1 += va0c3 * vb1c3;
    }

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    vout0x0 -= (vkernel_zero_point0 + 2.0f) * row_sum[0];
    vout0x0 += vkernel_zero_point0 * (float) kc * (float) vinput_zero_point0;
    vout0x1 -= (vkernel_zero_point1 + 2.0f) * row_sum[0];
    vout0x1 += vkernel_zero_point1 * (float) kc * (float) vinput_zero_point0;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;

    const float vfilter_output_scale0 = unaligned_indexed_load_f32(w, 0);
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = unaligned_indexed_load_f32(w, 1);
    vout0x1 *= vfilter_output_scale1;

    const float vbias0 = unaligned_indexed_load_f32(w, 2);
    vout0x0 += vbias0;
    const float vbias1 = unaligned_indexed_load_f32(w, 3);
    vout0x1 += vbias1;

    w = (const float*) w + 4;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
