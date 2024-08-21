// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(
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
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);
  const v128_t vinput_zero_point = wasm_v128_load32_splat(&quantization_params->zero_point);
    const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
    const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
    XNN_FORCE_REALIZATION(vmin);
    XNN_FORCE_REALIZATION(vmax);
  do {
    const v128_t vksum0123 = wasm_v128_load(w);
    v128_t vacc0x0123 = wasm_i32x4_mul(vksum0123, vinput_zero_point);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      a += 1;

      size_t k = kc;
      do {
        v128_t vxa0 = wasm_i16x8_load8x8(a0);
        a0 += 8;

        const v128_t vxb0 = wasm_i16x8_load8x8(w);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb0));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb1 = wasm_i16x8_load8x8((const int8_t*) w + 8);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb1));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb2 = wasm_i16x8_load8x8((const int8_t*) w + 16);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb2));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb3 = wasm_i16x8_load8x8((const int8_t*) w + 24);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
