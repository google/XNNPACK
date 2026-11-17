// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qs8_avgpool_minmax_fp32_ukernel_9p__wasmsimd_u16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const int8_t* zero,
    const float* multiplier,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qs8_avgpool_minmax_params* restrict params) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const v128_t vinit_bias = wasm_i32x4_splat(params->scalar.init_bias);
  v128_t vscale = wasm_f32x4_splat(params->scalar.scale);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(12582912.0f);
  const int32_t output_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const v128_t vmagic_min = wasm_i32x4_splat((int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point));
  const v128_t vmagic_bias_less_output_zero_point =
      wasm_i32x4_splat(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  const v128_t voutput_max = wasm_i8x16_splat(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vinit_bias);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vmagic_min);
  XNN_FORCE_REALIZATION(vmagic_bias_less_output_zero_point);
  XNN_FORCE_REALIZATION(voutput_max);

  do {
    if (multiplier != NULL) {
      vscale = wasm_f32x4_splat(*multiplier++);
    }

    int8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 16; c -= 16) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      v128_t vacc0123 = wasm_i32x4_const_splat(0);
      v128_t vacc4567 = wasm_i32x4_const_splat(0);
      v128_t vacc89AB = wasm_i32x4_const_splat(0);
      v128_t vaccCDEF = wasm_i32x4_const_splat(0);
      do {
        vacc0123 = wasm_i32x4_add(vacc0123, vinit_bias);
        vacc4567 = wasm_i32x4_add(vacc4567, vinit_bias);
        vacc89AB = wasm_i32x4_add(vacc89AB, vinit_bias);
        vaccCDEF = wasm_i32x4_add(vaccCDEF, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const v128_t vi0_lo = wasm_i16x8_load8x8(i0);
        const v128_t vi0_hi = wasm_i16x8_load8x8(i0 + 8);
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const v128_t vi1_lo = wasm_i16x8_load8x8(i1);
        const v128_t vi1_hi = wasm_i16x8_load8x8(i1 + 8);
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const v128_t vi2_lo = wasm_i16x8_load8x8(i2);
        const v128_t vi2_hi = wasm_i16x8_load8x8(i2 + 8);
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const v128_t vi3_lo = wasm_i16x8_load8x8(i3);
        const v128_t vi3_hi = wasm_i16x8_load8x8(i3 + 8);
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const v128_t vi4_lo = wasm_i16x8_load8x8(i4);
        const v128_t vi4_hi = wasm_i16x8_load8x8(i4 + 8);
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const v128_t vi5_lo = wasm_i16x8_load8x8(i5);
        const v128_t vi5_hi = wasm_i16x8_load8x8(i5 + 8);
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const v128_t vi6_lo = wasm_i16x8_load8x8(i6);
        const v128_t vi6_hi = wasm_i16x8_load8x8(i6 + 8);
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const v128_t vi7_lo = wasm_i16x8_load8x8(i7);
        const v128_t vi7_hi = wasm_i16x8_load8x8(i7 + 8);
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const v128_t vi8_lo = wasm_i16x8_load8x8(i8);
        const v128_t vi8_hi = wasm_i16x8_load8x8(i8 + 8);

        const v128_t vsum018_lo = wasm_i16x8_add(wasm_i16x8_add(vi0_lo, vi1_lo), vi8_lo);
        const v128_t vsum23_lo = wasm_i16x8_add(vi2_lo, vi3_lo);
        const v128_t vsum45_lo = wasm_i16x8_add(vi4_lo, vi5_lo);
        const v128_t vsum67_lo = wasm_i16x8_add(vi6_lo, vi7_lo);
        const v128_t vsum_lo = wasm_i16x8_add(wasm_i16x8_add(vsum23_lo, vsum45_lo), wasm_i16x8_add(vsum018_lo, vsum67_lo));

        const v128_t vsum018_hi = wasm_i16x8_add(wasm_i16x8_add(vi0_hi, vi1_hi), vi8_hi);
        const v128_t vsum23_hi = wasm_i16x8_add(vi2_hi, vi3_hi);
        const v128_t vsum45_hi = wasm_i16x8_add(vi4_hi, vi5_hi);
        const v128_t vsum67_hi = wasm_i16x8_add(vi6_hi, vi7_hi);
        const v128_t vsum_hi = wasm_i16x8_add(wasm_i16x8_add(vsum23_hi, vsum45_hi), wasm_i16x8_add(vsum018_hi, vsum67_hi));

        vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vsum_lo));
        vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vsum_lo));
        vacc89AB = wasm_i32x4_add(vacc89AB, wasm_i32x4_extend_low_i16x8(vsum_hi));
        vaccCDEF = wasm_i32x4_add(vaccCDEF, wasm_i32x4_extend_high_i16x8(vsum_hi));

        k = doz(k, 9);
      } while (k != 0);

      vacc0123 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc0123), vscale);
      vacc4567 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc4567), vscale);
      vacc89AB = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc89AB), vscale);
      vaccCDEF = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vaccCDEF), vscale);

      vacc0123 = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vacc0123, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vacc4567, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);
      vacc89AB = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vacc89AB, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);
      vaccCDEF = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vaccCDEF, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);

      const v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
      const v128_t vout89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

      v128_t vout = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);
      vout = wasm_i8x16_min(vout, voutput_max);

      wasm_v128_store(o, vout);
      o += 16;
      c_offset += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const int8_t** i = input;
        size_t k = kernel_elements;
        v128_t vacc0123 = wasm_i32x4_const_splat(0);
        v128_t vacc4567 = wasm_i32x4_const_splat(0);
        do {
          vacc0123 = wasm_i32x4_add(vacc0123, vinit_bias);
          vacc4567 = wasm_i32x4_add(vacc4567, vinit_bias);

          const int8_t* i0 = 0 < k ? *i++ : zero;
          assert(i0 != NULL);
          if XNN_UNPREDICTABLE(i0 != zero) {
            i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
          }
          i0 += c_offset;
          const v128_t vi0 = wasm_i16x8_load8x8(i0);
          const int8_t* i1 = 1 < k ? *i++ : zero;
          assert(i1 != NULL);
          if XNN_UNPREDICTABLE(i1 != zero) {
            i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
          }
          i1 += c_offset;
          const v128_t vi1 = wasm_i16x8_load8x8(i1);
          const int8_t* i2 = 2 < k ? *i++ : zero;
          assert(i2 != NULL);
          if XNN_UNPREDICTABLE(i2 != zero) {
            i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
          }
          i2 += c_offset;
          const v128_t vi2 = wasm_i16x8_load8x8(i2);
          const int8_t* i3 = 3 < k ? *i++ : zero;
          assert(i3 != NULL);
          if XNN_UNPREDICTABLE(i3 != zero) {
            i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
          }
          i3 += c_offset;
          const v128_t vi3 = wasm_i16x8_load8x8(i3);
          const int8_t* i4 = 4 < k ? *i++ : zero;
          assert(i4 != NULL);
          if XNN_UNPREDICTABLE(i4 != zero) {
            i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
          }
          i4 += c_offset;
          const v128_t vi4 = wasm_i16x8_load8x8(i4);
          const int8_t* i5 = 5 < k ? *i++ : zero;
          assert(i5 != NULL);
          if XNN_UNPREDICTABLE(i5 != zero) {
            i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
          }
          i5 += c_offset;
          const v128_t vi5 = wasm_i16x8_load8x8(i5);
          const int8_t* i6 = 6 < k ? *i++ : zero;
          assert(i6 != NULL);
          if XNN_UNPREDICTABLE(i6 != zero) {
            i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
          }
          i6 += c_offset;
          const v128_t vi6 = wasm_i16x8_load8x8(i6);
          const int8_t* i7 = 7 < k ? *i++ : zero;
          assert(i7 != NULL);
          if XNN_UNPREDICTABLE(i7 != zero) {
            i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
          }
          i7 += c_offset;
          const v128_t vi7 = wasm_i16x8_load8x8(i7);
          const int8_t* i8 = 8 < k ? *i++ : zero;
          assert(i8 != NULL);
          if XNN_UNPREDICTABLE(i8 != zero) {
            i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
          }
          i8 += c_offset;
          const v128_t vi8 = wasm_i16x8_load8x8(i8);

          const v128_t vsum018 = wasm_i16x8_add(wasm_i16x8_add(vi0, vi1), vi8);
          const v128_t vsum23 = wasm_i16x8_add(vi2, vi3);
          const v128_t vsum45 = wasm_i16x8_add(vi4, vi5);
          const v128_t vsum67 = wasm_i16x8_add(vi6, vi7);
          const v128_t vsum = wasm_i16x8_add(wasm_i16x8_add(vsum23, vsum45), wasm_i16x8_add(vsum018, vsum67));

          vacc0123 = wasm_i32x4_add(vacc0123, wasm_i32x4_extend_low_i16x8(vsum));
          vacc4567 = wasm_i32x4_add(vacc4567, wasm_i32x4_extend_high_i16x8(vsum));

          k = doz(k, 9);
        } while (k != 0);

        vacc0123 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc0123), vscale);
        vacc4567 = wasm_f32x4_mul(wasm_f32x4_convert_i32x4(vacc4567), vscale);

        vacc0123 = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vacc0123, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);
        vacc4567 = wasm_i32x4_sub(wasm_i32x4_max(wasm_f32x4_add(vacc4567, vmagic_bias), vmagic_min), vmagic_bias_less_output_zero_point);

        const v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
        v128_t vout = wasm_i8x16_narrow_i16x8(vout01234567, vout01234567);
        vout = wasm_i8x16_min(vout, voutput_max);

        if XNN_LIKELY(c >= 8) {
          wasm_v128_store64_lane(o, vout, 0);
          o += 8;
          c_offset += 8;
          c -= 8;
        } else {
          if (c & 4) {
            wasm_v128_store32_lane(o, vout, 0);
            vout = wasm_u64x2_shr(vout, 32);
            o += 4;
          }
          if (c & 2) {
            wasm_v128_store16_lane(o, vout, 0);
            vout = wasm_u32x4_shr(vout, 16);
            o += 2;
          }
          if (c & 1) {
            wasm_v128_store8_lane(o, vout, 0);
          }
          c = 0;
        }
      } while (c != 0);
    }

    input = (const int8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
