// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic(
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

  const int8_t* a0 = a;
  int8_t* c0 = c;


  do {
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    w = (const int32_t*) w + 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const int32_t va00 = (int32_t) a0[0];
      const int32_t va01 = (int32_t) a0[1];
      const int32_t va02 = (int32_t) a0[2];
      const int32_t va03 = (int32_t) a0[3];
      a0 += 4;

      const int32_t vb00 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb10 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb01 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb11 = (int32_t) ((const int8_t*) w)[3];
      const int32_t vb02 = (int32_t) ((const int8_t*) w)[4];
      const int32_t vb12 = (int32_t) ((const int8_t*) w)[5];
      const int32_t vb03 = (int32_t) ((const int8_t*) w)[6];
      const int32_t vb13 = (int32_t) ((const int8_t*) w)[7];
      w = (const int8_t*) w + 8;

      vacc0x0 += va00 * vb00;
      vacc0x1 += va00 * vb10;
      vacc0x0 += va01 * vb01;
      vacc0x1 += va01 * vb11;
      vacc0x0 += va02 * vb02;
      vacc0x1 += va02 * vb12;
      vacc0x0 += va03 * vb03;
      vacc0x1 += va03 * vb13;
    }
    if XNN_UNLIKELY(k != 0) {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
    }

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    const float vscale0 = unaligned_indexed_load_f32(w, 0);
    vfpacc0x0 *= vscale0;
    const float vscale1 = unaligned_indexed_load_f32(w, 1);
    vfpacc0x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
