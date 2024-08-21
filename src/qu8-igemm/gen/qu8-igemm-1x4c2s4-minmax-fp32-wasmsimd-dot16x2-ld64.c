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


void xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(
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
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      const v128_t vb_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.kernel_zero_point);
      size_t k = kc;
      do {
        v128_t vxa0 = wasm_u16x8_load8x8(a0);
        a0 += 8;

        const v128_t vxb0 = wasm_i16x8_sub(wasm_u16x8_load8x8(w), vb_zero_point);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb0));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb1 = wasm_i16x8_sub(wasm_u16x8_load8x8((const uint8_t*) w + 8), vb_zero_point);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb1));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb2 = wasm_i16x8_sub(wasm_u16x8_load8x8((const uint8_t*) w + 16), vb_zero_point);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb2));
        vxa0 = wasm_v32x4_shuffle(vxa0, vxa0, 1, 2, 3, 4);
        const v128_t vxb3 = wasm_i16x8_sub(wasm_u16x8_load8x8((const uint8_t*) w + 24), vb_zero_point);

        vacc0x0123 = wasm_i32x4_add(vacc0x0123, wasm_i32x4_dot_i16x8(vxa0, vxb3));

        w = (const uint8_t*) w + 32;
        k -= 8 * sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale = wasm_v128_load64_splat(params->fp32_wasmsimd.scale);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vout = wasm_u8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_u8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c0, vout, 0);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vout, 0);
        c0 += 2;

        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
