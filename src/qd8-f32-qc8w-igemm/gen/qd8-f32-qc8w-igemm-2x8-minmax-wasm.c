// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/gemm.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_2x8__wasm(
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
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
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

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vksum4 = ((const int32_t*) w)[4];
    const int32_t vksum5 = ((const int32_t*) w)[5];
    const int32_t vksum6 = ((const int32_t*) w)[6];
    const int32_t vksum7 = ((const int32_t*) w)[7];
    const int32_t vinput_zero_point = quantization_params->zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point;
    int32_t vacc0x1 = vksum1 * vinput_zero_point;
    int32_t vacc0x2 = vksum2 * vinput_zero_point;
    int32_t vacc0x3 = vksum3 * vinput_zero_point;
    int32_t vacc0x4 = vksum4 * vinput_zero_point;
    int32_t vacc0x5 = vksum5 * vinput_zero_point;
    int32_t vacc0x6 = vksum6 * vinput_zero_point;
    int32_t vacc0x7 = vksum7 * vinput_zero_point;
    int32_t vacc1x0 = vksum0 * vinput_zero_point;
    int32_t vacc1x1 = vksum1 * vinput_zero_point;
    int32_t vacc1x2 = vksum2 * vinput_zero_point;
    int32_t vacc1x3 = vksum3 * vinput_zero_point;
    int32_t vacc1x4 = vksum4 * vinput_zero_point;
    int32_t vacc1x5 = vksum5 * vinput_zero_point;
    int32_t vacc1x6 = vksum6 * vinput_zero_point;
    int32_t vacc1x7 = vksum7 * vinput_zero_point;
    w = (const void*) ((const int32_t*) w + 8);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      a += 2;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
        const int32_t vb4 = (int32_t) ((const int8_t*) w)[4];
        const int32_t vb5 = (int32_t) ((const int8_t*) w)[5];
        const int32_t vb6 = (int32_t) ((const int8_t*) w)[6];
        const int32_t vb7 = (int32_t) ((const int8_t*) w)[7];
        w = (const void*) ((const int8_t*) w + 8);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc0x4 += va0 * vb4;
        vacc0x5 += va0 * vb5;
        vacc0x6 += va0 * vb6;
        vacc0x7 += va0 * vb7;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;
        vacc1x4 += va1 * vb4;
        vacc1x5 += va1 * vb5;
        vacc1x6 += va1 * vb6;
        vacc1x7 += va1 * vb7;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    float vout0x4 = (float) vacc0x4;
    float vout0x5 = (float) vacc0x5;
    float vout0x6 = (float) vacc0x6;
    float vout0x7 = (float) vacc0x7;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    float vout1x2 = (float) vacc1x2;
    float vout1x3 = (float) vacc1x3;
    float vout1x4 = (float) vacc1x4;
    float vout1x5 = (float) vacc1x5;
    float vout1x6 = (float) vacc1x6;
    float vout1x7 = (float) vacc1x7;

    const float vinput_scale = quantization_params->inv_scale;
    vout0x0 *= vinput_scale;
    vout0x1 *= vinput_scale;
    vout0x2 *= vinput_scale;
    vout0x3 *= vinput_scale;
    vout0x4 *= vinput_scale;
    vout0x5 *= vinput_scale;
    vout0x6 *= vinput_scale;
    vout0x7 *= vinput_scale;
    vout1x0 *= vinput_scale;
    vout1x1 *= vinput_scale;
    vout1x2 *= vinput_scale;
    vout1x3 *= vinput_scale;
    vout1x4 *= vinput_scale;
    vout1x5 *= vinput_scale;
    vout1x6 *= vinput_scale;
    vout1x7 *= vinput_scale;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    vout1x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    vout1x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    vout1x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;
    vout1x3 *= vfilter_output_scale3;
    const float vfilter_output_scale4 = ((const float*) w)[4];
    vout0x4 *= vfilter_output_scale4;
    vout1x4 *= vfilter_output_scale4;
    const float vfilter_output_scale5 = ((const float*) w)[5];
    vout0x5 *= vfilter_output_scale5;
    vout1x5 *= vfilter_output_scale5;
    const float vfilter_output_scale6 = ((const float*) w)[6];
    vout0x6 *= vfilter_output_scale6;
    vout1x6 *= vfilter_output_scale6;
    const float vfilter_output_scale7 = ((const float*) w)[7];
    vout0x7 *= vfilter_output_scale7;
    vout1x7 *= vfilter_output_scale7;

    const float vbias0 = ((const float*) w)[8];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    const float vbias1 = ((const float*) w)[9];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    const float vbias2 = ((const float*) w)[10];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    const float vbias3 = ((const float*) w)[11];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    const float vbias4 = ((const float*) w)[12];
    vout0x4 += vbias4;
    vout1x4 += vbias4;
    const float vbias5 = ((const float*) w)[13];
    vout0x5 += vbias5;
    vout1x5 += vbias5;
    const float vbias6 = ((const float*) w)[14];
    vout0x6 += vbias6;
    vout1x6 += vbias6;
    const float vbias7 = ((const float*) w)[15];
    vout0x7 += vbias7;
    vout1x7 += vbias7;

    w = (const float*) w + 16;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout1x0 = __builtin_wasm_max_f32(vout1x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout1x1 = __builtin_wasm_max_f32(vout1x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout1x2 = __builtin_wasm_max_f32(vout1x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);
    vout1x3 = __builtin_wasm_max_f32(vout1x3, voutput_min);
    vout0x4 = __builtin_wasm_max_f32(vout0x4, voutput_min);
    vout1x4 = __builtin_wasm_max_f32(vout1x4, voutput_min);
    vout0x5 = __builtin_wasm_max_f32(vout0x5, voutput_min);
    vout1x5 = __builtin_wasm_max_f32(vout1x5, voutput_min);
    vout0x6 = __builtin_wasm_max_f32(vout0x6, voutput_min);
    vout1x6 = __builtin_wasm_max_f32(vout1x6, voutput_min);
    vout0x7 = __builtin_wasm_max_f32(vout0x7, voutput_min);
    vout1x7 = __builtin_wasm_max_f32(vout1x7, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout1x0 = __builtin_wasm_min_f32(vout1x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout1x1 = __builtin_wasm_min_f32(vout1x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout1x2 = __builtin_wasm_min_f32(vout1x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);
    vout1x3 = __builtin_wasm_min_f32(vout1x3, voutput_max);
    vout0x4 = __builtin_wasm_min_f32(vout0x4, voutput_max);
    vout1x4 = __builtin_wasm_min_f32(vout1x4, voutput_max);
    vout0x5 = __builtin_wasm_min_f32(vout0x5, voutput_max);
    vout1x5 = __builtin_wasm_min_f32(vout1x5, voutput_max);
    vout0x6 = __builtin_wasm_min_f32(vout0x6, voutput_max);
    vout1x6 = __builtin_wasm_min_f32(vout1x6, voutput_max);
    vout0x7 = __builtin_wasm_min_f32(vout0x7, voutput_max);
    vout1x7 = __builtin_wasm_min_f32(vout1x7, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c1[4] = vout1x4;
      c1[5] = vout1x5;
      c1[6] = vout1x6;
      c1[7] = vout1x7;
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;
      c0[4] = vout0x4;
      c0[5] = vout0x5;
      c0[6] = vout0x6;
      c0[7] = vout0x7;

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        c1[2] = vout1x2;
        c1[3] = vout1x3;
        vout1x0 = vout1x4;
        vout1x1 = vout1x5;
        vout1x2 = vout1x6;
        c1 += 4;
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        c0[2] = vout0x2;
        c0[3] = vout0x3;
        vout0x0 = vout0x4;
        vout0x1 = vout0x5;
        vout0x2 = vout0x6;
        c0 += 4;
      }
      if (nc & 2) {
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        vout1x1 = vout1x3;
        vout1x2 = vout1x4;
        vout1x3 = vout1x5;
        vout1x4 = vout1x6;
        c1 += 2;
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        vout0x1 = vout0x3;
        vout0x2 = vout0x4;
        vout0x3 = vout0x5;
        vout0x4 = vout0x6;
        c0 += 2;
      }
      if (nc & 1) {
        c1[0] = vout1x0;
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
