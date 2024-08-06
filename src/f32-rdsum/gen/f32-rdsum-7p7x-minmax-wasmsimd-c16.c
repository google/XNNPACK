// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/wasm-simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC //
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"

void xnn_f32_rdsum_ukernel_7p7x__wasmsimd_c16(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vscale = wasm_v128_load32_splat(&params->scalar.scale);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 16; channels -= 16) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    v128_t vacc0 = wasm_i32x4_const_splat(0.f);
    v128_t vacc1 = wasm_i32x4_const_splat(0.f);
    v128_t vacc2 = wasm_i32x4_const_splat(0.f);
    v128_t vacc3 = wasm_i32x4_const_splat(0.f);

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      v128_t vin0;
      v128_t vin1;
      v128_t vin2;
      v128_t vin3;
      vin0 = wasm_v128_load(&i0[0]);
      vin1 = wasm_v128_load(&i0[4]);
      vin2 = wasm_v128_load(&i0[8]);
      vin3 = wasm_v128_load(&i0[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i1[0]);
      vin1 = wasm_v128_load(&i1[4]);
      vin2 = wasm_v128_load(&i1[8]);
      vin3 = wasm_v128_load(&i1[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i2[0]);
      vin1 = wasm_v128_load(&i2[4]);
      vin2 = wasm_v128_load(&i2[8]);
      vin3 = wasm_v128_load(&i2[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i3[0]);
      vin1 = wasm_v128_load(&i3[4]);
      vin2 = wasm_v128_load(&i3[8]);
      vin3 = wasm_v128_load(&i3[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i4[0]);
      vin1 = wasm_v128_load(&i4[4]);
      vin2 = wasm_v128_load(&i4[8]);
      vin3 = wasm_v128_load(&i4[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i5[0]);
      vin1 = wasm_v128_load(&i5[4]);
      vin2 = wasm_v128_load(&i5[8]);
      vin3 = wasm_v128_load(&i5[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      vin0 = wasm_v128_load(&i6[0]);
      vin1 = wasm_v128_load(&i6[4]);
      vin2 = wasm_v128_load(&i6[8]);
      vin3 = wasm_v128_load(&i6[12]);
      vacc0 = wasm_f32x4_add(vin0, vacc0);
      vacc1 = wasm_f32x4_add(vin1, vacc1);
      vacc2 = wasm_f32x4_add(vin2, vacc2);
      vacc3 = wasm_f32x4_add(vin3, vacc3);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = wasm_f32x4_mul(vacc0, vscale);
    vacc0 = wasm_f32x4_max(vacc0, vmin);
    vacc0 = wasm_f32x4_min(vacc0, vmax);
    vacc1 = wasm_f32x4_mul(vacc1, vscale);
    vacc1 = wasm_f32x4_max(vacc1, vmin);
    vacc1 = wasm_f32x4_min(vacc1, vmax);
    vacc2 = wasm_f32x4_mul(vacc2, vscale);
    vacc2 = wasm_f32x4_max(vacc2, vmin);
    vacc2 = wasm_f32x4_min(vacc2, vmax);
    vacc3 = wasm_f32x4_mul(vacc3, vscale);
    vacc3 = wasm_f32x4_max(vacc3, vmin);
    vacc3 = wasm_f32x4_min(vacc3, vmax);

    const float* o = output;
    v128_t vo0 = wasm_v128_load(o); o += 4;
    v128_t vo1 = wasm_v128_load(o); o += 4;
    v128_t vo2 = wasm_v128_load(o); o += 4;
    v128_t vo3 = wasm_v128_load(o); o += 4;
    vacc0 = wasm_f32x4_add(vo0, vacc0);
    vacc1 = wasm_f32x4_add(vo1, vacc1);
    vacc2 = wasm_f32x4_add(vo2, vacc2);
    vacc3 = wasm_f32x4_add(vo3, vacc3);
    wasm_v128_store(output, vacc0); output += 4;
    wasm_v128_store(output, vacc1); output += 4;
    wasm_v128_store(output, vacc2); output += 4;
    wasm_v128_store(output, vacc3); output += 4;

    input = (const float*) ((uintptr_t) input + 16 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    v128_t vacc[4];
    vacc[0] = wasm_i32x4_const_splat(0.f);
    vacc[1] = wasm_i32x4_const_splat(0.f);
    vacc[2] = wasm_i32x4_const_splat(0.f);
    vacc[3] = wasm_i32x4_const_splat(0.f);

    const size_t num_chunks = round_up_po2(channels, 4) >> 2;
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_chunks; ++i) {
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i0[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i1[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i2[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i3[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i4[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i5[i*4]), vacc[i]);
        vacc[i] = wasm_f32x4_add(wasm_v128_load(&i6[i*4]), vacc[i]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < num_chunks; ++i) {
      vacc[i] = wasm_f32x4_mul(vacc[i], vscale);
      vacc[i] = wasm_f32x4_max(vacc[i], vmin);
      vacc[i] = wasm_f32x4_min(vacc[i], vmax);
    }

    v128_t vo[4];
    const float* o = output;
    for (int i = 0; i < channels >> 2; ++i) {
      vo[i] = wasm_v128_load(o); o += 4;
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vacc[i] = wasm_f32x4_add(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 2; ++i) {
      wasm_v128_store(output, vacc[i]); output += 4;
    }
    const size_t pos = channels / 4;
    v128_t vout = vacc[pos];
    if (channels & 2) {
      v128_t vo = wasm_f32x4_make(output[0], output[1], 0.f, 0.f);
      wasm_v128_store64_lane(output, wasm_f32x4_add(vo, vout), 0);
      vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
      output += 2;
    }
    if (channels & 1) {
      *output += wasm_f32x4_extract_lane(vout, 0);
    }
  }
}
