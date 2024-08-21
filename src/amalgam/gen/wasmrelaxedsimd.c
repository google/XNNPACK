// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/lut.h"
#include "xnnpack/math.h"
#include "xnnpack/prelu.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/spmm.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_u16(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsign_mask = wasm_u16x8_const_splat(UINT16_C(0x8000));
  const v128_t vexp_offset = wasm_u16x8_const_splat(UINT16_C(0x7000));
  const v128_t vexp_scale = wasm_f32x4_const_splat(0x1.0p-112f);
  const v128_t vmagic_mask = wasm_u16x8_const_splat(UINT16_C(0x3F00));
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0.5f);
  const v128_t vdenorm_cutoff = wasm_u16x8_const_splat(UINT16_C(0x0400));

  XNN_FORCE_REALIZATION(vsign_mask);
  XNN_FORCE_REALIZATION(vexp_offset);
  XNN_FORCE_REALIZATION(vexp_scale);
  XNN_FORCE_REALIZATION(vmagic_mask);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const v128_t vh0 = wasm_v128_load(i);
    const v128_t vh1 = wasm_v128_load(i + 8);
    i += 16;

    const v128_t vsign0 = wasm_v128_and(vh0, vsign_mask);
    const v128_t vsign1 = wasm_v128_and(vh1, vsign_mask);

    const v128_t vnonsign0 = wasm_v128_xor(vh0, vsign0);
    const v128_t vnonsign1 = wasm_v128_xor(vh1, vsign1);

    const v128_t vprenorm0 = wasm_i16x8_shl(vnonsign0, 13);
    const v128_t vprenorm1 = wasm_i16x8_add(wasm_u16x8_shr(vnonsign0, 3), vexp_offset);
    const v128_t vprenorm2 = wasm_i16x8_shl(vnonsign1, 13);
    const v128_t vprenorm3 = wasm_i16x8_add(wasm_u16x8_shr(vnonsign1, 3), vexp_offset);

    const v128_t vnorm0 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm0, vprenorm1, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm1 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm0, vprenorm1, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);
    const v128_t vnorm2 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm2, vprenorm3, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm3 = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm2, vprenorm3, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm0 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign0, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm1 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign0, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);
    const v128_t vdenorm2 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign1, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm3 = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign1, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask0 = wasm_i16x8_gt(vnonsign0, vdenorm_cutoff);
    const v128_t vmask1 = wasm_i16x8_gt(vnonsign1, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask0 = wasm_i32x4_extend_low_i16x8(vmask0);
    const v128_t vxmask1 = wasm_i32x4_extend_high_i16x8(vmask0);
    const v128_t vxmask2 = wasm_i32x4_extend_low_i16x8(vmask1);
    const v128_t vxmask3 = wasm_i32x4_extend_high_i16x8(vmask1);

    const v128_t vabsf0 = wasm_i32x4_relaxed_laneselect(vnorm0, vdenorm0, vxmask0);
    const v128_t vsignf0 = wasm_v16x8_shuffle(vzero, vsign0, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf1 = wasm_i32x4_relaxed_laneselect(vnorm1, vdenorm1, vxmask1);
    const v128_t vsignf1 = wasm_v16x8_shuffle(vzero, vsign0, 4, 12, 5, 13, 6, 14, 7, 15);
    const v128_t vabsf2 = wasm_i32x4_relaxed_laneselect(vnorm2, vdenorm2, vxmask2);
    const v128_t vsignf2 = wasm_v16x8_shuffle(vzero, vsign1, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf3 = wasm_i32x4_relaxed_laneselect(vnorm3, vdenorm3, vxmask3);
    const v128_t vsignf3 = wasm_v16x8_shuffle(vzero, vsign1, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vf0 = wasm_v128_or(vsignf0, vabsf0);
    const v128_t vf1 = wasm_v128_or(vsignf1, vabsf1);
    const v128_t vf2 = wasm_v128_or(vsignf2, vabsf2);
    const v128_t vf3 = wasm_v128_or(vsignf3, vabsf3);

    wasm_v128_store(output, vf0);
    wasm_v128_store(output + 4, vf1);
    wasm_v128_store(output + 8, vf2);
    wasm_v128_store(output + 12, vf3);
    output += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const v128_t vh = wasm_v128_load(i);
    i += 8;

    const v128_t vsign = wasm_v128_and(vh, vsign_mask);

    const v128_t vnonsign = wasm_v128_xor(vh, vsign);

    const v128_t vprenorm_lo = wasm_i16x8_shl(vnonsign, 13);
    const v128_t vprenorm_hi = wasm_i16x8_add(wasm_u16x8_shr(vnonsign, 3), vexp_offset);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask = wasm_i16x8_gt(vnonsign, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask_lo = wasm_i32x4_extend_low_i16x8(vmask);
    const v128_t vxmask_hi = wasm_i32x4_extend_high_i16x8(vmask);

    const v128_t vabsf_lo = wasm_i32x4_relaxed_laneselect(vnorm_lo, vdenorm_lo, vxmask_lo);
    const v128_t vsignf_lo = wasm_v16x8_shuffle(vzero, vsign, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf_hi = wasm_i32x4_relaxed_laneselect(vnorm_hi, vdenorm_hi, vxmask_hi);
    const v128_t vsignf_hi = wasm_v16x8_shuffle(vzero, vsign, 4, 12, 5, 13, 6, 14, 7, 15);

    const v128_t vf_lo = wasm_v128_or(vsignf_lo, vabsf_lo);
    const v128_t vf_hi = wasm_v128_or(vsignf_hi, vabsf_hi);

    wasm_v128_store(output, vf_lo);
    wasm_v128_store(output + 4, vf_hi);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const v128_t vh = wasm_v128_load(i);

    const v128_t vsign = wasm_v128_and(vh, vsign_mask);

    const v128_t vnonsign = wasm_v128_xor(vh, vsign);

    const v128_t vprenorm_lo = wasm_i16x8_shl(vnonsign, 13);
    const v128_t vprenorm_hi = wasm_i16x8_add(wasm_u16x8_shr(vnonsign, 3), vexp_offset);

    const v128_t vnorm_lo = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 0,  8, 1,  9, 2, 10, 3, 11), vexp_scale);
    const v128_t vnorm_hi = wasm_f32x4_mul(wasm_v16x8_shuffle(vprenorm_lo, vprenorm_hi, 4, 12, 5, 13, 6, 14, 7, 15), vexp_scale);

    const v128_t vdenorm_lo = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 0,  8, 1,  9, 2, 10, 3, 11), vmagic_bias);
    const v128_t vdenorm_hi = wasm_f32x4_sub(wasm_v16x8_shuffle(vnonsign, vmagic_mask, 4, 12, 5, 13, 6, 14, 7, 15), vmagic_bias);

    const v128_t vmask = wasm_i16x8_gt(vnonsign, vdenorm_cutoff);
    const v128_t vzero = wasm_i16x8_const_splat(0);

    const v128_t vxmask_lo = wasm_i32x4_extend_low_i16x8(vmask);
    const v128_t vxmask_hi = wasm_i32x4_extend_high_i16x8(vmask);

    const v128_t vabsf_lo = wasm_i32x4_relaxed_laneselect(vnorm_lo, vdenorm_lo, vxmask_lo);
    const v128_t vsignf_lo = wasm_v16x8_shuffle(vzero, vsign, 0,  8, 1,  9, 2, 10, 3, 11);
    const v128_t vabsf_hi = wasm_i32x4_relaxed_laneselect(vnorm_hi, vdenorm_hi, vxmask_hi);
    const v128_t vsignf_hi = wasm_v16x8_shuffle(vzero, vsign, 4, 12, 5, 13, 6, 14, 7, 15);

    v128_t vf = wasm_v128_or(vsignf_lo, vabsf_lo);

    if (batch & (4 * sizeof(uint16_t))) {
      wasm_v128_store(output, vf);
      output += 4;

      vf = wasm_v128_or(vsignf_hi, vabsf_hi);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      wasm_v128_store64_lane(output, vf, 0);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      wasm_v128_store32_lane(output, vf, 0);
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi4x4567, vk4x4567, vacc4567p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi5x4567, vk5x4567, vacc4567p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi6x4567, vk6x4567, vacc4567p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi7x4567, vk7x4567, vacc4567p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi8x4567, vk8x4567, vacc4567p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vi9x4567 = wasm_v128_load(i9 + 4);
      i9 += 8;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      const v128_t vk9x4567 = wasm_v128_load(w + 84);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi9x4567, vk9x4567, vacc4567p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vi10x4567 = wasm_v128_load(i10 + 4);
      i10 += 8;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      const v128_t vk10x4567 = wasm_v128_load(w + 92);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi10x4567, vk10x4567, vacc4567p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vi11x4567 = wasm_v128_load(i11 + 4);
      i11 += 8;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      const v128_t vk11x4567 = wasm_v128_load(w + 100);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi11x4567, vk11x4567, vacc4567p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vi12x4567 = wasm_v128_load(i12 + 4);
      i12 += 8;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      const v128_t vk12x4567 = wasm_v128_load(w + 108);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi12x4567, vk12x4567, vacc4567p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vi13x4567 = wasm_v128_load(i13 + 4);
      i13 += 8;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      const v128_t vk13x4567 = wasm_v128_load(w + 116);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi13x4567, vk13x4567, vacc4567p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vi14x4567 = wasm_v128_load(i14 + 4);
      i14 += 8;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      const v128_t vk14x4567 = wasm_v128_load(w + 124);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi14x4567, vk14x4567, vacc4567p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vi15x4567 = wasm_v128_load(i15 + 4);
      i15 += 8;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      const v128_t vk15x4567 = wasm_v128_load(w + 132);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi15x4567, vk15x4567, vacc4567p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vi16x4567 = wasm_v128_load(i16 + 4);
      i16 += 8;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      const v128_t vk16x4567 = wasm_v128_load(w + 140);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi16x4567, vk16x4567, vacc4567p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vi17x4567 = wasm_v128_load(i17 + 4);
      i17 += 8;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      const v128_t vk17x4567 = wasm_v128_load(w + 148);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi17x4567, vk17x4567, vacc4567p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vi18x4567 = wasm_v128_load(i18 + 4);
      i18 += 8;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      const v128_t vk18x4567 = wasm_v128_load(w + 156);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi18x4567, vk18x4567, vacc4567p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vi19x4567 = wasm_v128_load(i19 + 4);
      i19 += 8;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      const v128_t vk19x4567 = wasm_v128_load(w + 164);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi19x4567, vk19x4567, vacc4567p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vi20x4567 = wasm_v128_load(i20 + 4);
      i20 += 8;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      const v128_t vk20x4567 = wasm_v128_load(w + 172);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi20x4567, vk20x4567, vacc4567p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vi21x4567 = wasm_v128_load(i21 + 4);
      i21 += 8;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      const v128_t vk21x4567 = wasm_v128_load(w + 180);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi21x4567, vk21x4567, vacc4567p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vi22x4567 = wasm_v128_load(i22 + 4);
      i22 += 8;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      const v128_t vk22x4567 = wasm_v128_load(w + 188);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi22x4567, vk22x4567, vacc4567p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vi23x4567 = wasm_v128_load(i23 + 4);
      i23 += 8;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      const v128_t vk23x4567 = wasm_v128_load(w + 196);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi23x4567, vk23x4567, vacc4567p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vi24x4567 = wasm_v128_load(i24 + 4);
      i24 += 8;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      const v128_t vk24x4567 = wasm_v128_load(w + 204);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi24x4567, vk24x4567, vacc4567p0);

      w += 208;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_relaxed_max(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);
      vacc4567 = wasm_f32x4_relaxed_min(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      i9 += 4;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      i10 += 4;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      i11 += 4;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      i12 += 4;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      i13 += 4;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      i14 += 4;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      i15 += 4;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      i16 += 4;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      i17 += 4;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      i18 += 4;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      i19 += 4;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      i20 += 4;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      i21 += 4;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      i22 += 4;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      i23 += 4;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      i24 += 4;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);

      w += 4;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi4x4567, vk4x4567, vacc4567p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi5x4567, vk5x4567, vacc4567p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi6x4567, vk6x4567, vacc4567p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi7x4567, vk7x4567, vacc4567p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi8x4567, vk8x4567, vacc4567p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vi9x4567 = wasm_v128_load(i9 + 4);
      i9 += 8;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      const v128_t vk9x4567 = wasm_v128_load(w + 84);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi9x4567, vk9x4567, vacc4567p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vi10x4567 = wasm_v128_load(i10 + 4);
      i10 += 8;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      const v128_t vk10x4567 = wasm_v128_load(w + 92);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi10x4567, vk10x4567, vacc4567p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vi11x4567 = wasm_v128_load(i11 + 4);
      i11 += 8;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      const v128_t vk11x4567 = wasm_v128_load(w + 100);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi11x4567, vk11x4567, vacc4567p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vi12x4567 = wasm_v128_load(i12 + 4);
      i12 += 8;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      const v128_t vk12x4567 = wasm_v128_load(w + 108);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi12x4567, vk12x4567, vacc4567p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vi13x4567 = wasm_v128_load(i13 + 4);
      i13 += 8;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      const v128_t vk13x4567 = wasm_v128_load(w + 116);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi13x4567, vk13x4567, vacc4567p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vi14x4567 = wasm_v128_load(i14 + 4);
      i14 += 8;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      const v128_t vk14x4567 = wasm_v128_load(w + 124);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi14x4567, vk14x4567, vacc4567p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vi15x4567 = wasm_v128_load(i15 + 4);
      i15 += 8;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      const v128_t vk15x4567 = wasm_v128_load(w + 132);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi15x4567, vk15x4567, vacc4567p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vi16x4567 = wasm_v128_load(i16 + 4);
      i16 += 8;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      const v128_t vk16x4567 = wasm_v128_load(w + 140);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi16x4567, vk16x4567, vacc4567p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vi17x4567 = wasm_v128_load(i17 + 4);
      i17 += 8;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      const v128_t vk17x4567 = wasm_v128_load(w + 148);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi17x4567, vk17x4567, vacc4567p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vi18x4567 = wasm_v128_load(i18 + 4);
      i18 += 8;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      const v128_t vk18x4567 = wasm_v128_load(w + 156);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi18x4567, vk18x4567, vacc4567p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vi19x4567 = wasm_v128_load(i19 + 4);
      i19 += 8;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      const v128_t vk19x4567 = wasm_v128_load(w + 164);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi19x4567, vk19x4567, vacc4567p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vi20x4567 = wasm_v128_load(i20 + 4);
      i20 += 8;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      const v128_t vk20x4567 = wasm_v128_load(w + 172);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi20x4567, vk20x4567, vacc4567p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vi21x4567 = wasm_v128_load(i21 + 4);
      i21 += 8;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      const v128_t vk21x4567 = wasm_v128_load(w + 180);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi21x4567, vk21x4567, vacc4567p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vi22x4567 = wasm_v128_load(i22 + 4);
      i22 += 8;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      const v128_t vk22x4567 = wasm_v128_load(w + 188);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi22x4567, vk22x4567, vacc4567p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vi23x4567 = wasm_v128_load(i23 + 4);
      i23 += 8;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      const v128_t vk23x4567 = wasm_v128_load(w + 196);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi23x4567, vk23x4567, vacc4567p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vi24x4567 = wasm_v128_load(i24 + 4);
      i24 += 8;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      const v128_t vk24x4567 = wasm_v128_load(w + 204);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi24x4567, vk24x4567, vacc4567p0);

      w += 208;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      i9 += 4;

      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      i10 += 4;

      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      i11 += 4;

      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      i12 += 4;

      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      i13 += 4;

      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      i14 += 4;

      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      i15 += 4;

      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      i16 += 4;

      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      i17 += 4;

      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      i18 += 4;

      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      i19 += 4;

      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      i20 += 4;

      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      i21 += 4;

      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      i22 += 4;

      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      i23 += 4;

      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      i24 += 4;

      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vk9x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vk10x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vk11x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vk12x0123 = wasm_v128_load(w + 104);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vk13x0123 = wasm_v128_load(w + 112);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vk14x0123 = wasm_v128_load(w + 120);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vk15x0123 = wasm_v128_load(w + 128);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vk16x0123 = wasm_v128_load(w + 136);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vk17x0123 = wasm_v128_load(w + 144);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vk18x0123 = wasm_v128_load(w + 152);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vk19x0123 = wasm_v128_load(w + 160);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vk20x0123 = wasm_v128_load(w + 168);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vk21x0123 = wasm_v128_load(w + 176);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vk22x0123 = wasm_v128_load(w + 184);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vk23x0123 = wasm_v128_load(w + 192);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vk24x0123 = wasm_v128_load(w + 200);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      w += 32;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_relaxed_max(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);
      vacc4567 = wasm_f32x4_relaxed_min(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      w += 4;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      w += 32;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      w += 40;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_relaxed_max(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);
      vacc4567 = wasm_f32x4_relaxed_min(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      w += 4;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      w += 40;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      size_t c = 0;
      for (; c < channels; c += 4) {
        v128_t vacc0p0 = wasm_v128_load(w);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        const v128_t vk3x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        const v128_t vk4x0123 = wasm_v128_load(w + 20);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 24;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = 0;
      for (; c < channels; c += 4) {
        v128_t vacc0p0 = wasm_v128_load(b);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        const v128_t vk3x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        const v128_t vk4x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 20;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;


      for (; c >= 4; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(b);
        b += 4;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        v128_t vk0x0123 = wasm_v128_load(w);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        v128_t vk1x0123 = wasm_v128_load(w + 4);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        v128_t vk2x0123 = wasm_v128_load(w + 8);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        v128_t vk3x0123 = wasm_v128_load(w + 12);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        v128_t vk4x0123 = wasm_v128_load(w + 16);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 20;



        v128_t vacc0 = wasm_f32x4_relaxed_max(vacc0p0, vmin);

        vacc0 = wasm_f32x4_relaxed_min(vacc0, vmax);

        wasm_v128_store(output, vacc0);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        v128_t vacc0p0 = wasm_v128_load(b);

        const v128_t vi0x0123 = wasm_v128_load(i0);
        v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        v128_t vk3x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        v128_t vk4x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);


        v128_t vacc0 = wasm_f32x4_relaxed_max(vacc0p0, vmin);

        vacc0 = wasm_f32x4_relaxed_min(vacc0, vmax);

        if (c & 2) {
          wasm_v128_store64_lane(output, vacc0, 0);
          vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
          output += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(output, vacc0, 0);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      size_t c = 0;
      for (; c < channels; c += 4) {
        v128_t vacc0p0 = wasm_v128_load(w);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        const v128_t vk3x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        const v128_t vk4x0123 = wasm_v128_load(w + 20);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 24;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = 0;
      for (; c < channels; c += 4) {
        v128_t vacc0p0 = wasm_v128_load(b);


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        const v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        const v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        const v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        const v128_t vk3x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        const v128_t vk4x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 20;


        wasm_v128_store(b, vacc0p0);
        b += 4;
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;


      for (; c >= 4; c -= 4) {
        v128_t vacc0p0 = wasm_v128_load(b);
        b += 4;


        const v128_t vi0x0123 = wasm_v128_load(i0);
        i0 += 4;

        v128_t vk0x0123 = wasm_v128_load(w);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        i1 += 4;

        v128_t vk1x0123 = wasm_v128_load(w + 4);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        i2 += 4;

        v128_t vk2x0123 = wasm_v128_load(w + 8);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        i3 += 4;

        v128_t vk3x0123 = wasm_v128_load(w + 12);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        i4 += 4;

        v128_t vk4x0123 = wasm_v128_load(w + 16);

        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);

        w += 20;



        const v128_t vacc0 = vacc0p0;

        wasm_v128_store(output, vacc0);
        output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        v128_t vacc0p0 = wasm_v128_load(b);

        const v128_t vi0x0123 = wasm_v128_load(i0);
        v128_t vk0x0123 = wasm_v128_load(w);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0p0);

        const v128_t vi1x0123 = wasm_v128_load(i1);
        v128_t vk1x0123 = wasm_v128_load(w + 4);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0p0);

        const v128_t vi2x0123 = wasm_v128_load(i2);
        v128_t vk2x0123 = wasm_v128_load(w + 8);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0p0);

        const v128_t vi3x0123 = wasm_v128_load(i3);
        v128_t vk3x0123 = wasm_v128_load(w + 12);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0p0);

        const v128_t vi4x0123 = wasm_v128_load(i4);
        v128_t vk4x0123 = wasm_v128_load(w + 16);
        vacc0p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0p0);


        v128_t vacc0 = vacc0p0;

        if (c & 2) {
          wasm_v128_store64_lane(output, vacc0, 0);
          vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
          output += 2;
        }
        if (c & 1) {
          wasm_v128_store32_lane(output, vacc0, 0);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi4x4567, vk4x4567, vacc4567p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi5x4567, vk5x4567, vacc4567p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi6x4567, vk6x4567, vacc4567p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi7x4567, vk7x4567, vacc4567p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi8x4567, vk8x4567, vacc4567p0);

      w += 80;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      v128_t vacc4567 = wasm_f32x4_relaxed_max(vmin, vacc4567p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);
      vacc4567 = wasm_f32x4_relaxed_min(vmax, vacc4567);

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      w += 4;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      v128_t vacc0123p0 = wasm_v128_load(w);
      v128_t vacc4567p0 = wasm_v128_load(w + 4);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vi0x4567 = wasm_v128_load(i0 + 4);
      i0 += 8;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      const v128_t vk0x4567 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi0x4567, vk0x4567, vacc4567p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vi1x4567 = wasm_v128_load(i1 + 4);
      i1 += 8;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      const v128_t vk1x4567 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi1x4567, vk1x4567, vacc4567p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vi2x4567 = wasm_v128_load(i2 + 4);
      i2 += 8;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      const v128_t vk2x4567 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi2x4567, vk2x4567, vacc4567p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vi3x4567 = wasm_v128_load(i3 + 4);
      i3 += 8;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      const v128_t vk3x4567 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi3x4567, vk3x4567, vacc4567p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vi4x4567 = wasm_v128_load(i4 + 4);
      i4 += 8;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      const v128_t vk4x4567 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi4x4567, vk4x4567, vacc4567p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vi5x4567 = wasm_v128_load(i5 + 4);
      i5 += 8;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      const v128_t vk5x4567 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi5x4567, vk5x4567, vacc4567p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vi6x4567 = wasm_v128_load(i6 + 4);
      i6 += 8;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      const v128_t vk6x4567 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi6x4567, vk6x4567, vacc4567p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vi7x4567 = wasm_v128_load(i7 + 4);
      i7 += 8;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      const v128_t vk7x4567 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi7x4567, vk7x4567, vacc4567p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vi8x4567 = wasm_v128_load(i8 + 4);
      i8 += 8;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      const v128_t vk8x4567 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);
      vacc4567p0 = wasm_f32x4_relaxed_madd(vi8x4567, vk8x4567, vacc4567p0);

      w += 80;


      const v128_t vacc0123 = vacc0123p0;
      const v128_t vacc4567 = vacc4567p0;

      wasm_v128_store(output, vacc0123);
      wasm_v128_store(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      w += 4;


      const v128_t vacc0123 = vacc0123p0;

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);


      v128_t vacc0123 = vacc0123p0;

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24(
    size_t batch,
    const float* input,
    void* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vexp_bias = wasm_u32x4_const_splat(UINT32_C(0x07800000));
  const v128_t vscale_to_inf = wasm_f32x4_const_splat(0x1.0p+112f);
  const v128_t vexpw_max = wasm_u32x4_const_splat(UINT32_C(0x7F800000));
  const v128_t vscale_to_zero = wasm_f32x4_const_splat(0x1.0p-110f);
  const v128_t vbias_min = wasm_u32x4_const_splat(UINT32_C(0x40008000));
  const v128_t vmanth_mask = wasm_u32x4_const_splat(UINT32_C(0x00000FFF));
  const v128_t vexph_mask = wasm_u32x4_const_splat(UINT32_C(0x00007C00));
  const v128_t vnanh = wasm_u16x8_const_splat(UINT16_C(0x7E00));

  XNN_FORCE_REALIZATION(vexp_bias);
  XNN_FORCE_REALIZATION(vscale_to_inf);
  XNN_FORCE_REALIZATION(vexpw_max);
  XNN_FORCE_REALIZATION(vscale_to_zero);
  XNN_FORCE_REALIZATION(vbias_min);
  XNN_FORCE_REALIZATION(vmanth_mask);
  XNN_FORCE_REALIZATION(vexph_mask);
  XNN_FORCE_REALIZATION(vnanh);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const v128_t vx0 = wasm_v128_load(input);
    const v128_t vx1 = wasm_v128_load(input + 4);
    const v128_t vx2 = wasm_v128_load(input + 8);
    const v128_t vx3 = wasm_v128_load(input + 12);
    const v128_t vx4 = wasm_v128_load(input + 16);
    const v128_t vx5 = wasm_v128_load(input + 20);
    input += 24;

    const v128_t vabsx0 = wasm_f32x4_abs(vx0);
    const v128_t vabsx1 = wasm_f32x4_abs(vx1);
    const v128_t vabsx2 = wasm_f32x4_abs(vx2);
    const v128_t vabsx3 = wasm_f32x4_abs(vx3);
    const v128_t vabsx4 = wasm_f32x4_abs(vx4);
    const v128_t vabsx5 = wasm_f32x4_abs(vx5);

    const v128_t vsignx0 = wasm_v128_xor(vx0, vabsx0);
    const v128_t vsignx1 = wasm_v128_xor(vx1, vabsx1);
    const v128_t vsignx2 = wasm_v128_xor(vx2, vabsx2);
    const v128_t vsignx3 = wasm_v128_xor(vx3, vabsx3);
    const v128_t vsignx4 = wasm_v128_xor(vx4, vabsx4);
    const v128_t vsignx5 = wasm_v128_xor(vx5, vabsx5);

    v128_t vbias0 = wasm_i32x4_add(vabsx0, vexp_bias);
    v128_t vbias1 = wasm_i32x4_add(vabsx1, vexp_bias);
    v128_t vbias2 = wasm_i32x4_add(vabsx2, vexp_bias);
    v128_t vbias3 = wasm_i32x4_add(vabsx3, vexp_bias);
    v128_t vbias4 = wasm_i32x4_add(vabsx4, vexp_bias);
    v128_t vbias5 = wasm_i32x4_add(vabsx5, vexp_bias);

    v128_t vf0 = wasm_f32x4_mul(vabsx0, vscale_to_inf);
    v128_t vf1 = wasm_f32x4_mul(vabsx1, vscale_to_inf);
    v128_t vf2 = wasm_f32x4_mul(vabsx2, vscale_to_inf);
    v128_t vf3 = wasm_f32x4_mul(vabsx3, vscale_to_inf);
    v128_t vf4 = wasm_f32x4_mul(vabsx4, vscale_to_inf);
    v128_t vf5 = wasm_f32x4_mul(vabsx5, vscale_to_inf);

    const v128_t vnanmaskw0 = wasm_i32x4_gt(vabsx0, vexpw_max);
    const v128_t vnanmaskw1 = wasm_i32x4_gt(vabsx1, vexpw_max);
    const v128_t vnanmaskw2 = wasm_i32x4_gt(vabsx2, vexpw_max);
    const v128_t vnanmaskw3 = wasm_i32x4_gt(vabsx3, vexpw_max);
    const v128_t vnanmaskw4 = wasm_i32x4_gt(vabsx4, vexpw_max);
    const v128_t vnanmaskw5 = wasm_i32x4_gt(vabsx5, vexpw_max);

    vbias0 = wasm_v128_and(vbias0, vexpw_max);
    vbias1 = wasm_v128_and(vbias1, vexpw_max);
    vbias2 = wasm_v128_and(vbias2, vexpw_max);
    vbias3 = wasm_v128_and(vbias3, vexpw_max);
    vbias4 = wasm_v128_and(vbias4, vexpw_max);
    vbias5 = wasm_v128_and(vbias5, vexpw_max);

    vf0 = wasm_f32x4_mul(vf0, vscale_to_zero);
    vf1 = wasm_f32x4_mul(vf1, vscale_to_zero);
    vf2 = wasm_f32x4_mul(vf2, vscale_to_zero);
    vf3 = wasm_f32x4_mul(vf3, vscale_to_zero);
    vf4 = wasm_f32x4_mul(vf4, vscale_to_zero);
    vf5 = wasm_f32x4_mul(vf5, vscale_to_zero);

    const v128_t vnanmaskh0 = wasm_i16x8_narrow_i32x4(vnanmaskw0, vnanmaskw1);
    const v128_t vnanmaskh1 = wasm_i16x8_narrow_i32x4(vnanmaskw2, vnanmaskw3);
    const v128_t vnanmaskh2 = wasm_i16x8_narrow_i32x4(vnanmaskw4, vnanmaskw5);

    const v128_t vsignh0 = wasm_i16x8_narrow_i32x4(vsignx0, vsignx1);
    const v128_t vsignh1 = wasm_i16x8_narrow_i32x4(vsignx2, vsignx3);
    const v128_t vsignh2 = wasm_i16x8_narrow_i32x4(vsignx4, vsignx5);

    vbias0 = wasm_i16x8_max(vbias0, vbias_min);
    vbias1 = wasm_i16x8_max(vbias1, vbias_min);
    vbias2 = wasm_i16x8_max(vbias2, vbias_min);
    vbias3 = wasm_i16x8_max(vbias3, vbias_min);
    vbias4 = wasm_i16x8_max(vbias4, vbias_min);
    vbias5 = wasm_i16x8_max(vbias5, vbias_min);

    vf0 = wasm_f32x4_add(vf0, vbias0);
    vf1 = wasm_f32x4_add(vf1, vbias1);
    vf2 = wasm_f32x4_add(vf2, vbias2);
    vf3 = wasm_f32x4_add(vf3, vbias3);
    vf4 = wasm_f32x4_add(vf4, vbias4);
    vf5 = wasm_f32x4_add(vf5, vbias5);

    v128_t vexpw0 = wasm_i32x4_shr(vf0, 13);
    v128_t vexpw1 = wasm_i32x4_shr(vf1, 13);
    v128_t vexpw2 = wasm_i32x4_shr(vf2, 13);
    v128_t vexpw3 = wasm_i32x4_shr(vf3, 13);
    v128_t vexpw4 = wasm_i32x4_shr(vf4, 13);
    v128_t vexpw5 = wasm_i32x4_shr(vf5, 13);

    const v128_t vmantw0 = wasm_v128_and(vf0, vmanth_mask);
    const v128_t vmantw1 = wasm_v128_and(vf1, vmanth_mask);
    const v128_t vmantw2 = wasm_v128_and(vf2, vmanth_mask);
    const v128_t vmantw3 = wasm_v128_and(vf3, vmanth_mask);
    const v128_t vmantw4 = wasm_v128_and(vf4, vmanth_mask);
    const v128_t vmantw5 = wasm_v128_and(vf5, vmanth_mask);

    vexpw0 = wasm_v128_and(vexpw0, vexph_mask);
    vexpw1 = wasm_v128_and(vexpw1, vexph_mask);
    vexpw2 = wasm_v128_and(vexpw2, vexph_mask);
    vexpw3 = wasm_v128_and(vexpw3, vexph_mask);
    vexpw4 = wasm_v128_and(vexpw4, vexph_mask);
    vexpw5 = wasm_v128_and(vexpw5, vexph_mask);

    const v128_t vnonsignw0 = wasm_i32x4_add(vmantw0, vexpw0);
    const v128_t vnonsignw1 = wasm_i32x4_add(vmantw1, vexpw1);
    const v128_t vnonsignw2 = wasm_i32x4_add(vmantw2, vexpw2);
    const v128_t vnonsignw3 = wasm_i32x4_add(vmantw3, vexpw3);
    const v128_t vnonsignw4 = wasm_i32x4_add(vmantw4, vexpw4);
    const v128_t vnonsignw5 = wasm_i32x4_add(vmantw5, vexpw5);

    const v128_t vnonsignh0 = wasm_i16x8_narrow_i32x4(vnonsignw0, vnonsignw1);
    const v128_t vnonsignh1 = wasm_i16x8_narrow_i32x4(vnonsignw2, vnonsignw3);
    const v128_t vnonsignh2 = wasm_i16x8_narrow_i32x4(vnonsignw4, vnonsignw5);

    const v128_t vabsh0 = wasm_i16x8_relaxed_laneselect(vnanh, vnonsignh0, vnanmaskh0);
    const v128_t vabsh1 = wasm_i16x8_relaxed_laneselect(vnanh, vnonsignh1, vnanmaskh1);
    const v128_t vabsh2 = wasm_i16x8_relaxed_laneselect(vnanh, vnonsignh2, vnanmaskh2);

    const v128_t vh0 = wasm_v128_or(vabsh0, vsignh0);
    const v128_t vh1 = wasm_v128_or(vabsh1, vsignh1);
    const v128_t vh2 = wasm_v128_or(vabsh2, vsignh2);

    wasm_v128_store(o, vh0);
    wasm_v128_store(o + 8, vh1);
    wasm_v128_store(o + 16, vh2);
    o += 24;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t vx_lo = wasm_v128_load(input);
    const v128_t vx_hi = wasm_v128_load(input + 4);
    input += 8;

    const v128_t vabsx_lo = wasm_f32x4_abs(vx_lo);
    const v128_t vabsx_hi = wasm_f32x4_abs(vx_hi);

    const v128_t vsignx_lo = wasm_v128_xor(vx_lo, vabsx_lo);
    const v128_t vsignx_hi = wasm_v128_xor(vx_hi, vabsx_hi);
    v128_t vbias_lo = wasm_i32x4_add(vabsx_lo, vexp_bias);
    v128_t vbias_hi = wasm_i32x4_add(vabsx_hi, vexp_bias);
    v128_t vf_lo = wasm_f32x4_mul(vabsx_lo, vscale_to_inf);
    v128_t vf_hi = wasm_f32x4_mul(vabsx_hi, vscale_to_inf);
    const v128_t vnanmaskw_lo = wasm_i32x4_gt(vabsx_lo, vexpw_max);
    const v128_t vnanmaskw_hi = wasm_i32x4_gt(vabsx_hi, vexpw_max);

    vbias_lo = wasm_v128_and(vbias_lo, vexpw_max);
    vbias_hi = wasm_v128_and(vbias_hi, vexpw_max);
    vf_lo = wasm_f32x4_mul(vf_lo, vscale_to_zero);
    vf_hi = wasm_f32x4_mul(vf_hi, vscale_to_zero);
    const v128_t vnanmaskh = wasm_i16x8_narrow_i32x4(vnanmaskw_lo, vnanmaskw_hi);
    const v128_t vsignh = wasm_i16x8_narrow_i32x4(vsignx_lo, vsignx_hi);

    vbias_lo = wasm_i16x8_max(vbias_lo, vbias_min);
    vbias_hi = wasm_i16x8_max(vbias_hi, vbias_min);

    vf_lo = wasm_f32x4_add(vf_lo, vbias_lo);
    vf_hi = wasm_f32x4_add(vf_hi, vbias_hi);

    v128_t vexpw_lo = wasm_i32x4_shr(vf_lo, 13);
    v128_t vexpw_hi = wasm_i32x4_shr(vf_hi, 13);
    const v128_t vmantw_lo = wasm_v128_and(vf_lo, vmanth_mask);
    const v128_t vmantw_hi = wasm_v128_and(vf_hi, vmanth_mask);

    vexpw_lo = wasm_v128_and(vexpw_lo, vexph_mask);
    vexpw_hi = wasm_v128_and(vexpw_hi, vexph_mask);

    const v128_t vnonsignw_lo = wasm_i32x4_add(vmantw_lo, vexpw_lo);
    const v128_t vnonsignw_hi = wasm_i32x4_add(vmantw_hi, vexpw_hi);

    const v128_t vnonsignh = wasm_i16x8_narrow_i32x4(vnonsignw_lo, vnonsignw_hi);

    const v128_t vabsh = wasm_i16x8_relaxed_laneselect(vnanh, vnonsignh, vnanmaskh);

    const v128_t vh = wasm_v128_or(vabsh, vsignh);

    wasm_v128_store(o, vh);
    o += 8;
  }
  if XNN_UNPREDICTABLE(batch != 0) {
    const v128_t vx_lo = wasm_v128_load(input);
    const float* input_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    const v128_t vx_hi = wasm_v128_load(input_hi);

    const v128_t vabsx_lo = wasm_f32x4_abs(vx_lo);
    const v128_t vabsx_hi = wasm_f32x4_abs(vx_hi);

    const v128_t vsignx_lo = wasm_v128_xor(vx_lo, vabsx_lo);
    const v128_t vsignx_hi = wasm_v128_xor(vx_hi, vabsx_hi);
    v128_t vbias_lo = wasm_i32x4_add(vabsx_lo, vexp_bias);
    v128_t vbias_hi = wasm_i32x4_add(vabsx_hi, vexp_bias);
    v128_t vf_lo = wasm_f32x4_mul(vabsx_lo, vscale_to_inf);
    v128_t vf_hi = wasm_f32x4_mul(vabsx_hi, vscale_to_inf);
    const v128_t vnanmaskw_lo = wasm_i32x4_gt(vabsx_lo, vexpw_max);
    const v128_t vnanmaskw_hi = wasm_i32x4_gt(vabsx_hi, vexpw_max);

    vbias_lo = wasm_v128_and(vbias_lo, vexpw_max);
    vbias_hi = wasm_v128_and(vbias_hi, vexpw_max);
    vf_lo = wasm_f32x4_mul(vf_lo, vscale_to_zero);
    vf_hi = wasm_f32x4_mul(vf_hi, vscale_to_zero);
    const v128_t vnanmaskh = wasm_i16x8_narrow_i32x4(vnanmaskw_lo, vnanmaskw_hi);
    const v128_t vsignh = wasm_i16x8_narrow_i32x4(vsignx_lo, vsignx_hi);

    vbias_lo = wasm_i16x8_max(vbias_lo, vbias_min);
    vbias_hi = wasm_i16x8_max(vbias_hi, vbias_min);

    vf_lo = wasm_f32x4_add(vf_lo, vbias_lo);
    vf_hi = wasm_f32x4_add(vf_hi, vbias_hi);

    v128_t vexpw_lo = wasm_i32x4_shr(vf_lo, 13);
    v128_t vexpw_hi = wasm_i32x4_shr(vf_hi, 13);
    const v128_t vmantw_lo = wasm_v128_and(vf_lo, vmanth_mask);
    const v128_t vmantw_hi = wasm_v128_and(vf_hi, vmanth_mask);

    vexpw_lo = wasm_v128_and(vexpw_lo, vexph_mask);
    vexpw_hi = wasm_v128_and(vexpw_hi, vexph_mask);

    const v128_t vnonsignw_lo = wasm_i32x4_add(vmantw_lo, vexpw_lo);
    const v128_t vnonsignw_hi = wasm_i32x4_add(vmantw_hi, vexpw_hi);

    const v128_t vnonsignh = wasm_i16x8_narrow_i32x4(vnonsignw_lo, vnonsignw_hi);

    const v128_t vabsh = wasm_i16x8_relaxed_laneselect(vnanh, vnonsignh, vnanmaskh);

    v128_t vh = wasm_v128_or(vabsh, vsignh);

    if (batch & (4 * sizeof(float))) {
      wasm_v128_store64_lane(o, vh, 0);
      vh = wasm_v64x2_shuffle(vh, vh, 1, 1);
      o += 4;
    }
    if (batch & (2 * sizeof(float))) {
      wasm_v128_store32_lane(o, vh, 0);
      vh = wasm_i64x2_shr(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store16_lane(o, vh, 0);
    }
  }
}

void xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0c4 = wasm_f32x4_relaxed_madd(va0, vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(va0, vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(va1, vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(va1, vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(va2, vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(va2, vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(va3, vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(va3, vb1, vacc3x1c4);
    }
    if XNN_UNLIKELY(k != 0) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const v128_t va1 = wasm_v128_load(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const v128_t va2 = wasm_v128_load(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const v128_t va3 = wasm_v128_load(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
      const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

      vacc0x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask0), vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask1), vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask0), vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask1), vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask0), vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask1), vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask0), vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask1), vb1, vacc3x1c4);
    }

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));

    vacc01x01 = wasm_f32x4_relaxed_max(vmin, vacc01x01);
    vacc23x01 = wasm_f32x4_relaxed_max(vmin, vacc23x01);

    vacc01x01 = wasm_f32x4_relaxed_min(vmax, vacc01x01);
    vacc23x01 = wasm_f32x4_relaxed_min(vmax, vacc23x01);

    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c0, vacc01x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c3, vacc23x01, 2);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0c4 = wasm_f32x4_relaxed_madd(va0, vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(va0, vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(va1, vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(va1, vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(va2, vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(va2, vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(va3, vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(va3, vb1, vacc3x1c4);
    }
    if XNN_UNLIKELY(k != 0) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      const v128_t va1 = wasm_v128_load(a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      const v128_t va2 = wasm_v128_load(a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      const v128_t va3 = wasm_v128_load(a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const v128_t vb0 = wasm_v128_load(w);
      const v128_t vb1 = wasm_v128_load(w + 4);
      w += 8;

      const v128_t vzero = wasm_f32x4_const_splat(0.0f);
      const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
      const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

      vacc0x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask0), vb0, vacc0x0c4);
      vacc0x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask1), vb1, vacc0x1c4);
      vacc1x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask0), vb0, vacc1x0c4);
      vacc1x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask1), vb1, vacc1x1c4);
      vacc2x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask0), vb0, vacc2x0c4);
      vacc2x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask1), vb1, vacc2x1c4);
      vacc3x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask0), vb0, vacc3x0c4);
      vacc3x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask1), vb1, vacc3x1c4);
    }

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));


    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c0, vacc01x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c3, vacc23x01, 2);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb0123 = wasm_v128_load(w);
      const v128_t vb4567 = wasm_v128_load(w + 4);
      w += 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_max(vmin, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_max(vmin, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_max(vmin, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_max(vmin, vacc5x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_min(vmax, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_min(vmax, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_min(vmax, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_min(vmax, vacc5x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc4x0123 = wasm_i32x4_max(vacc4x0123, vzero);
    vacc5x0123 = wasm_i32x4_max(vacc5x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);
    vacc4x4567 = wasm_i32x4_max(vacc4x4567, vzero);
    vacc5x4567 = wasm_i32x4_max(vacc5x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w + 0);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb0123c0 = wasm_v128_load(w + 0);
      const v128_t vb4567c0 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb0123c1 = wasm_v128_load(w + 8);
      const v128_t vb4567c1 = wasm_v128_load(w + 12);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb0123c2 = wasm_v128_load(w + 16);
      const v128_t vb4567c2 = wasm_v128_load(w + 20);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb0123c3 = wasm_v128_load(w + 24);
      const v128_t vb4567c3 = wasm_v128_load(w + 28);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w += 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_ibilinear_ukernel__wasmrelaxedsimd_c8(
    size_t output_pixels,
    size_t channels,
    const float** restrict input,
    size_t input_offset,
    const float* restrict weights,
    float* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  do {
    const float* i0 = (const float*) ((uintptr_t) input[0] + input_offset);
    const float* i1 = (const float*) ((uintptr_t) input[1] + input_offset);
    const float* i2 = (const float*) ((uintptr_t) input[2] + input_offset);
    const float* i3 = (const float*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const v128_t valphah = wasm_v128_load32_splat(weights);
    const v128_t valphav = wasm_v128_load32_splat(weights + 1);
    weights += 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const v128_t vtl0123 = wasm_v128_load(i0);
      const v128_t vtr0123 = wasm_v128_load(i1);
      const v128_t vbl0123 = wasm_v128_load(i2);
      const v128_t vbr0123 = wasm_v128_load(i3);
      const v128_t vtl4567 = wasm_v128_load(i0 + 4);
      const v128_t vtr4567 = wasm_v128_load(i1 + 4);
      const v128_t vbl4567 = wasm_v128_load(i2 + 4);
      const v128_t vbr4567 = wasm_v128_load(i3 + 4);
      i0 += 8;
      i1 += 8;
      i2 += 8;
      i3 += 8;

      const v128_t vtd0123 = wasm_f32x4_sub(vtr0123, vtl0123);
      const v128_t vbd0123 = wasm_f32x4_sub(vbr0123, vbl0123);
      const v128_t vtd4567 = wasm_f32x4_sub(vtr4567, vtl4567);
      const v128_t vbd4567 = wasm_f32x4_sub(vbr4567, vbl4567);

      const v128_t vt0123 = wasm_f32x4_relaxed_madd(vtd0123, valphah, vtl0123);
      const v128_t vb0123 = wasm_f32x4_relaxed_madd(vbd0123, valphah, vbl0123);
      const v128_t vt4567 = wasm_f32x4_relaxed_madd(vtd4567, valphah, vtl4567);
      const v128_t vb4567 = wasm_f32x4_relaxed_madd(vbd4567, valphah, vbl4567);

      const v128_t vd0123 = wasm_f32x4_sub(vb0123, vt0123);
      const v128_t vd4567 = wasm_f32x4_sub(vb4567, vt4567);

      const v128_t vo0123 = wasm_f32x4_relaxed_madd(vd0123, valphav, vt0123);
      const v128_t vo4567 = wasm_f32x4_relaxed_madd(vd4567, valphav, vt4567);

      wasm_v128_store(output, vo0123);
      wasm_v128_store(output + 4, vo4567);
      output += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vtl = wasm_v128_load(i0);
      const v128_t vtr = wasm_v128_load(i1);
      const v128_t vbl = wasm_v128_load(i2);
      const v128_t vbr = wasm_v128_load(i3);
      i0 += 4;
      i1 += 4;
      i2 += 4;
      i3 += 4;

      const v128_t vtd = wasm_f32x4_sub(vtr, vtl);
      const v128_t vbd = wasm_f32x4_sub(vbr, vbl);
      const v128_t vt = wasm_f32x4_relaxed_madd(vtd, valphah, vtl);
      const v128_t vb = wasm_f32x4_relaxed_madd(vbd, valphah, vbl);
      const v128_t vd = wasm_f32x4_sub(vb, vt);
      const v128_t vo = wasm_f32x4_relaxed_madd(vd, valphav, vt);

      wasm_v128_store(output, vo);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vtl = wasm_v128_load(i0);
      const v128_t vtr = wasm_v128_load(i1);
      const v128_t vbl = wasm_v128_load(i2);
      const v128_t vbr = wasm_v128_load(i3);

      const v128_t vtd = wasm_f32x4_sub(vtr, vtl);
      const v128_t vbd = wasm_f32x4_sub(vbr, vbl);
      const v128_t vt = wasm_f32x4_relaxed_madd(vtd, valphah, vtl);
      const v128_t vb = wasm_f32x4_relaxed_madd(vbd, valphah, vbl);
      const v128_t vd = wasm_f32x4_sub(vb, vt);
      v128_t vo = wasm_f32x4_relaxed_madd(vd, valphav, vt);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(output, vo, 0);
        vo = wasm_v64x2_shuffle(vo, vo, 1, 1);
        output += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(output, vo, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_igemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0c4 = wasm_f32x4_relaxed_madd(va0, vb0, vacc0x0c4);
        vacc0x1c4 = wasm_f32x4_relaxed_madd(va0, vb1, vacc0x1c4);
        vacc1x0c4 = wasm_f32x4_relaxed_madd(va1, vb0, vacc1x0c4);
        vacc1x1c4 = wasm_f32x4_relaxed_madd(va1, vb1, vacc1x1c4);
        vacc2x0c4 = wasm_f32x4_relaxed_madd(va2, vb0, vacc2x0c4);
        vacc2x1c4 = wasm_f32x4_relaxed_madd(va2, vb1, vacc2x1c4);
        vacc3x0c4 = wasm_f32x4_relaxed_madd(va3, vb0, vacc3x0c4);
        vacc3x1c4 = wasm_f32x4_relaxed_madd(va3, vb1, vacc3x1c4);
      }
      if XNN_UNLIKELY(k != 0) {
        const v128_t va0 = wasm_v128_load(a0);
        const v128_t va1 = wasm_v128_load(a1);
        const v128_t va2 = wasm_v128_load(a2);
        const v128_t va3 = wasm_v128_load(a3);

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t vzero = wasm_f32x4_const_splat(0.0f);
        const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
        const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

        vacc0x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask0), vb0, vacc0x0c4);
        vacc0x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask1), vb1, vacc0x1c4);
        vacc1x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask0), vb0, vacc1x0c4);
        vacc1x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask1), vb1, vacc1x1c4);
        vacc2x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask0), vb0, vacc2x0c4);
        vacc2x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask1), vb1, vacc2x1c4);
        vacc3x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask0), vb0, vacc3x0c4);
        vacc3x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask1), vb1, vacc3x1c4);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));

    vacc01x01 = wasm_f32x4_relaxed_max(vmin, vacc01x01);
    vacc23x01 = wasm_f32x4_relaxed_max(vmin, vacc23x01);

    vacc01x01 = wasm_f32x4_relaxed_min(vmax, vacc01x01);
    vacc23x01 = wasm_f32x4_relaxed_min(vmax, vacc23x01);

    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x2c4__wasmrelaxedsimd_fma(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0c4 = wasm_v128_load32_zero(w);
    v128_t vacc0x1c4 = wasm_v128_load32_zero(w + 1);
    v128_t vacc1x0c4 = vacc0x0c4;
    v128_t vacc1x1c4 = vacc0x1c4;
    v128_t vacc2x0c4 = vacc0x0c4;
    v128_t vacc2x1c4 = vacc0x1c4;
    v128_t vacc3x0c4 = vacc0x0c4;
    v128_t vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        vacc0x0c4 = wasm_f32x4_relaxed_madd(va0, vb0, vacc0x0c4);
        vacc0x1c4 = wasm_f32x4_relaxed_madd(va0, vb1, vacc0x1c4);
        vacc1x0c4 = wasm_f32x4_relaxed_madd(va1, vb0, vacc1x0c4);
        vacc1x1c4 = wasm_f32x4_relaxed_madd(va1, vb1, vacc1x1c4);
        vacc2x0c4 = wasm_f32x4_relaxed_madd(va2, vb0, vacc2x0c4);
        vacc2x1c4 = wasm_f32x4_relaxed_madd(va2, vb1, vacc2x1c4);
        vacc3x0c4 = wasm_f32x4_relaxed_madd(va3, vb0, vacc3x0c4);
        vacc3x1c4 = wasm_f32x4_relaxed_madd(va3, vb1, vacc3x1c4);
      }
      if XNN_UNLIKELY(k != 0) {
        const v128_t va0 = wasm_v128_load(a0);
        const v128_t va1 = wasm_v128_load(a1);
        const v128_t va2 = wasm_v128_load(a2);
        const v128_t va3 = wasm_v128_load(a3);

        const v128_t vb0 = wasm_v128_load(w);
        const v128_t vb1 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t vzero = wasm_f32x4_const_splat(0.0f);
        const v128_t vmask0 = wasm_f32x4_eq(vb0, vzero);
        const v128_t vmask1 = wasm_f32x4_eq(vb1, vzero);

        vacc0x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask0), vb0, vacc0x0c4);
        vacc0x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va0, vmask1), vb1, vacc0x1c4);
        vacc1x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask0), vb0, vacc1x0c4);
        vacc1x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va1, vmask1), vb1, vacc1x1c4);
        vacc2x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask0), vb0, vacc2x0c4);
        vacc2x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va2, vmask1), vb1, vacc2x1c4);
        vacc3x0c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask0), vb0, vacc3x0c4);
        vacc3x1c4 = wasm_f32x4_relaxed_madd(wasm_v128_andnot(va3, vmask1), vb1, vacc3x1c4);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc0x0c4, vacc0x1c4, 2, 6, 3, 7));
    const v128_t vacc1x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc1x0c4, vacc1x1c4, 2, 6, 3, 7));
    const v128_t vacc2x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc2x0c4, vacc2x1c4, 2, 6, 3, 7));
    const v128_t vacc3x01c2 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 0, 4, 1, 5),
      wasm_v32x4_shuffle(vacc3x0c4, vacc3x1c4, 2, 6, 3, 7));

    v128_t vacc01x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc0x01c2, vacc1x01c2, 2, 3, 6, 7));
    v128_t vacc23x01 = wasm_f32x4_add(
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 0, 1, 4, 5),
      wasm_v32x4_shuffle(vacc2x01c2, vacc3x01c2, 2, 3, 6, 7));


    if XNN_LIKELY(nc >= 2) {
      wasm_v128_store64_lane(c3, vacc23x01, 1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store64_lane(c2, vacc23x01, 0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store64_lane(c1, vacc01x01, 1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store64_lane(c0, vacc01x01, 0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      wasm_v128_store32_lane(c3, vacc23x01, 2);
      wasm_v128_store32_lane(c2, vacc23x01, 0);
      wasm_v128_store32_lane(c1, vacc01x01, 2);
      wasm_v128_store32_lane(c0, vacc01x01, 0);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t vb0123 = wasm_v128_load(w);
        const v128_t vb4567 = wasm_v128_load(w + 4);
        w += 8;

        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
          vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
          vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
          vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
          vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
          vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
          vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
          vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
          vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
          vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_max(vmin, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_max(vmin, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_max(vmin, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_max(vmin, vacc5x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_min(vmax, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_min(vmax, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_min(vmax, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_min(vmax, vacc5x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
          vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
          vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
          vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
          vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
          vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
          vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
          vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
          vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
          vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);

    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc4x0123 = wasm_i32x4_max(vacc4x0123, vzero);
    vacc5x0123 = wasm_i32x4_max(vacc5x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);
    vacc4x4567 = wasm_i32x4_max(vacc4x4567, vzero);
    vacc5x4567 = wasm_i32x4_max(vacc5x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load(w);
    v128_t vacc0x4567 = wasm_v128_load(w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 4;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 4;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 4;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 4;
        const v128_t va4 = wasm_v128_load(a4);
        a4 += 4;
        const v128_t va5 = wasm_v128_load(a5);
        a5 += 4;

        const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
        const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
        const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
        const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
        const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
        const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

        const v128_t vb0123c0 = wasm_v128_load(w + 0);
        const v128_t vb4567c0 = wasm_v128_load(w + 4);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
        const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
        const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
        const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
        const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
        const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
        const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

        const v128_t vb0123c1 = wasm_v128_load(w + 8);
        const v128_t vb4567c1 = wasm_v128_load(w + 12);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
        const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
        const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
        const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
        const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
        const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
        const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

        const v128_t vb0123c2 = wasm_v128_load(w + 16);
        const v128_t vb4567c2 = wasm_v128_load(w + 20);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
        const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
        const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
        const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
        const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
        const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
        const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

        const v128_t vb0123c3 = wasm_v128_load(w + 24);
        const v128_t vb4567c3 = wasm_v128_load(w + 28);

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const v128_t vb0123 = wasm_v128_load(w);
          const v128_t vb4567 = wasm_v128_load(w + 4);
          w += 8;

          const v128_t va0 = wasm_v128_load32_splat(a0);
          a0 += 1;
          const v128_t va1 = wasm_v128_load32_splat(a1);
          a1 += 1;
          const v128_t va2 = wasm_v128_load32_splat(a2);
          a2 += 1;
          const v128_t va3 = wasm_v128_load32_splat(a3);
          a3 += 1;
          const v128_t va4 = wasm_v128_load32_splat(a4);
          a4 += 1;
          const v128_t va5 = wasm_v128_load32_splat(a5);
          a5 += 1;

          vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
          vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
          vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
          vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
          vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
          vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
          vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
          vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
          vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
          vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
          vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
          vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 6 * sizeof(void*);
    } while (p != 0);


    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c5, vacc5x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c0, vacc0x0123);

        vacc5x0123 = vacc5x4567;
        vacc4x0123 = vacc4x4567;
        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c5, vacc5x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c0, vacc0x0123, 0);

        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);

        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const v128_t vzero = wasm_i32x4_const_splat(0);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      w += 4;

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;
      v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc1x0123 = wasm_i32x4_max(vi1x0123, vzero);
      vi1x0123 = wasm_i32x4_min(vi1x0123, vzero);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vi0x0123, vw0123, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vi1x0123, vw0123, vacc1x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
      wasm_v128_store(o1, vacc1x0123);
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vw0123 = wasm_v128_load(w);
      w = (const float*) ((uintptr_t) w + c);

      v128_t vi0x0123 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      v128_t vi1x0123 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      v128_t vacc0x0123 = wasm_i32x4_max(vi0x0123, vzero);
      vi0x0123 = wasm_i32x4_min(vi0x0123, vzero);
      v128_t vacc1x0123 = wasm_i32x4_max(vi1x0123, vzero);
      vi1x0123 = wasm_i32x4_min(vi1x0123, vzero);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vi0x0123, vw0123, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vi1x0123, vw0123, vacc1x0123);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0x0123, 0);
        wasm_v128_store64_lane(o1, vacc1x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0x0123, 0);
        wasm_v128_store32_lane(o1, vacc1x0123, 0);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vw0123 = wasm_v128_load(w);
      w += 4;

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;
      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      v128_t vacc0x0123 = wasm_f32x4_mul(vi0x0123, vw0123);
      const v128_t vmask0x0123 = wasm_i32x4_shr(vi0x0123, 31);
      v128_t vacc1x0123 = wasm_f32x4_mul(vi1x0123, vw0123);
      const v128_t vmask1x0123 = wasm_i32x4_shr(vi1x0123, 31);

      vacc0x0123 = wasm_i32x4_relaxed_laneselect(vacc0x0123, vi0x0123, vmask0x0123);
      vacc1x0123 = wasm_i32x4_relaxed_laneselect(vacc1x0123, vi1x0123, vmask1x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
      wasm_v128_store(o1, vacc1x0123);
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vw0123 = wasm_v128_load(w);
      w = (const float*) ((uintptr_t) w + c);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      v128_t vacc0x0123 = wasm_f32x4_mul(vi0x0123, vw0123);
      const v128_t vmask0x0123 = wasm_i32x4_shr(vi0x0123, 31);
      v128_t vacc1x0123 = wasm_f32x4_mul(vi1x0123, vw0123);
      const v128_t vmask1x0123 = wasm_i32x4_shr(vi1x0123, 31);

      vacc0x0123 = wasm_i32x4_relaxed_laneselect(vacc0x0123, vi0x0123, vmask0x0123);
      vacc1x0123 = wasm_i32x4_relaxed_laneselect(vacc1x0123, vi1x0123, vmask1x0123);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0x0123, 0);
        wasm_v128_store64_lane(o1, vacc1x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0x0123, 0);
        wasm_v128_store32_lane(o1, vacc1x0123, 0);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    w = (const float*) w + 8;

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
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

void xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    w = (const float*) w + 8;
    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load32_splat(a0);
      a0 += 1;
      const v128_t va1 = wasm_v128_load32_splat(a1);
      a1 += 1;
      const v128_t va2 = wasm_v128_load32_splat(a2);
      a2 += 1;
      const v128_t va3 = wasm_v128_load32_splat(a3);
      a3 += 1;

      const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
      const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
      const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
      const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
      w = (const int8_t*) w + 8;

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);

      k -= sizeof(float);
    } while (k != 0);

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    w = (const float*) w + 8;

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    vacc4x0123 = wasm_f32x4_mul(vacc4x0123, vscale0123);
    vacc5x0123 = wasm_f32x4_mul(vacc5x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    vacc4x4567 = wasm_f32x4_mul(vacc4x4567, vscale4567);
    vacc5x4567 = wasm_f32x4_mul(vacc5x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_max(vmin, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_max(vmin, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_max(vmin, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_max(vmin, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_max(vmin, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_max(vmin, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_max(vmin, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_max(vmin, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_max(vmin, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_max(vmin, vacc5x4567);

    vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
    vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);
    vacc2x0123 = wasm_f32x4_relaxed_min(vmax, vacc2x0123);
    vacc3x0123 = wasm_f32x4_relaxed_min(vmax, vacc3x0123);
    vacc4x0123 = wasm_f32x4_relaxed_min(vmax, vacc4x0123);
    vacc5x0123 = wasm_f32x4_relaxed_min(vmax, vacc5x0123);
    vacc0x4567 = wasm_f32x4_relaxed_min(vmax, vacc0x4567);
    vacc1x4567 = wasm_f32x4_relaxed_min(vmax, vacc1x4567);
    vacc2x4567 = wasm_f32x4_relaxed_min(vmax, vacc2x4567);
    vacc3x4567 = wasm_f32x4_relaxed_min(vmax, vacc3x4567);
    vacc4x4567 = wasm_f32x4_relaxed_min(vmax, vacc4x4567);
    vacc5x4567 = wasm_f32x4_relaxed_min(vmax, vacc5x4567);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    vacc4x0123 = wasm_f32x4_mul(vacc4x0123, vscale0123);
    vacc5x0123 = wasm_f32x4_mul(vacc5x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    vacc4x4567 = wasm_f32x4_mul(vacc4x4567, vscale4567);
    vacc5x4567 = wasm_f32x4_mul(vacc5x4567, vscale4567);
    w = (const float*) w + 8;
    const v128_t vzero = wasm_i32x4_const_splat(0);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vzero);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vzero);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vzero);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vzero);
    vacc4x0123 = wasm_i32x4_max(vacc4x0123, vzero);
    vacc5x0123 = wasm_i32x4_max(vacc5x0123, vzero);
    vacc0x4567 = wasm_i32x4_max(vacc0x4567, vzero);
    vacc1x4567 = wasm_i32x4_max(vacc1x4567, vzero);
    vacc2x4567 = wasm_i32x4_max(vacc2x4567, vzero);
    vacc3x4567 = wasm_i32x4_max(vacc3x4567, vzero);
    vacc4x4567 = wasm_i32x4_max(vacc4x4567, vzero);
    vacc5x4567 = wasm_i32x4_max(vacc5x4567, vzero);

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    v128_t vacc0x0123 = wasm_v128_load((const float*) w + 0);
    v128_t vacc0x4567 = wasm_v128_load((const float*) w + 4);
    v128_t vacc1x0123 = vacc0x0123;
    v128_t vacc1x4567 = vacc0x4567;
    v128_t vacc2x0123 = vacc0x0123;
    v128_t vacc2x4567 = vacc0x4567;
    v128_t vacc3x0123 = vacc0x0123;
    v128_t vacc3x4567 = vacc0x4567;
    v128_t vacc4x0123 = vacc0x0123;
    v128_t vacc4x4567 = vacc0x4567;
    v128_t vacc5x0123 = vacc0x0123;
    v128_t vacc5x4567 = vacc0x4567;
    w = (const float*) w + 8;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 4;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 4;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 4;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 4;
      const v128_t va4 = wasm_v128_load(a4);
      a4 += 4;
      const v128_t va5 = wasm_v128_load(a5);
      a5 += 4;

      const v128_t va0c0 = wasm_v32x4_shuffle(va0, va0, 0, 0, 0, 0);
      const v128_t va1c0 = wasm_v32x4_shuffle(va1, va1, 0, 0, 0, 0);
      const v128_t va2c0 = wasm_v32x4_shuffle(va2, va2, 0, 0, 0, 0);
      const v128_t va3c0 = wasm_v32x4_shuffle(va3, va3, 0, 0, 0, 0);
      const v128_t va4c0 = wasm_v32x4_shuffle(va4, va4, 0, 0, 0, 0);
      const v128_t va5c0 = wasm_v32x4_shuffle(va5, va5, 0, 0, 0, 0);

      const v128_t vb01234567c0 = wasm_i16x8_load8x8((const int8_t*) w + 0);
      const v128_t vbi0123c0 = wasm_i32x4_extend_low_i16x8(vb01234567c0);
      const v128_t vbi4567c0 = wasm_i32x4_extend_high_i16x8(vb01234567c0);
      const v128_t vb0123c0 = wasm_f32x4_convert_i32x4(vbi0123c0);
      const v128_t vb4567c0 = wasm_f32x4_convert_i32x4(vbi4567c0);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c0, vb0123c0, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c0, vb0123c0, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c0, vb0123c0, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c0, vb0123c0, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c0, vb0123c0, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c0, vb0123c0, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c0, vb4567c0, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c0, vb4567c0, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c0, vb4567c0, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c0, vb4567c0, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c0, vb4567c0, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c0, vb4567c0, vacc5x4567);
      const v128_t va0c1 = wasm_v32x4_shuffle(va0, va0, 1, 1, 1, 1);
      const v128_t va1c1 = wasm_v32x4_shuffle(va1, va1, 1, 1, 1, 1);
      const v128_t va2c1 = wasm_v32x4_shuffle(va2, va2, 1, 1, 1, 1);
      const v128_t va3c1 = wasm_v32x4_shuffle(va3, va3, 1, 1, 1, 1);
      const v128_t va4c1 = wasm_v32x4_shuffle(va4, va4, 1, 1, 1, 1);
      const v128_t va5c1 = wasm_v32x4_shuffle(va5, va5, 1, 1, 1, 1);

      const v128_t vb01234567c1 = wasm_i16x8_load8x8((const int8_t*) w + 8);
      const v128_t vbi0123c1 = wasm_i32x4_extend_low_i16x8(vb01234567c1);
      const v128_t vbi4567c1 = wasm_i32x4_extend_high_i16x8(vb01234567c1);
      const v128_t vb0123c1 = wasm_f32x4_convert_i32x4(vbi0123c1);
      const v128_t vb4567c1 = wasm_f32x4_convert_i32x4(vbi4567c1);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c1, vb0123c1, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c1, vb0123c1, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c1, vb0123c1, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c1, vb0123c1, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c1, vb0123c1, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c1, vb0123c1, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c1, vb4567c1, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c1, vb4567c1, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c1, vb4567c1, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c1, vb4567c1, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c1, vb4567c1, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c1, vb4567c1, vacc5x4567);
      const v128_t va0c2 = wasm_v32x4_shuffle(va0, va0, 2, 2, 2, 2);
      const v128_t va1c2 = wasm_v32x4_shuffle(va1, va1, 2, 2, 2, 2);
      const v128_t va2c2 = wasm_v32x4_shuffle(va2, va2, 2, 2, 2, 2);
      const v128_t va3c2 = wasm_v32x4_shuffle(va3, va3, 2, 2, 2, 2);
      const v128_t va4c2 = wasm_v32x4_shuffle(va4, va4, 2, 2, 2, 2);
      const v128_t va5c2 = wasm_v32x4_shuffle(va5, va5, 2, 2, 2, 2);

      const v128_t vb01234567c2 = wasm_i16x8_load8x8((const int8_t*) w + 16);
      const v128_t vbi0123c2 = wasm_i32x4_extend_low_i16x8(vb01234567c2);
      const v128_t vbi4567c2 = wasm_i32x4_extend_high_i16x8(vb01234567c2);
      const v128_t vb0123c2 = wasm_f32x4_convert_i32x4(vbi0123c2);
      const v128_t vb4567c2 = wasm_f32x4_convert_i32x4(vbi4567c2);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c2, vb0123c2, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c2, vb0123c2, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c2, vb0123c2, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c2, vb0123c2, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c2, vb0123c2, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c2, vb0123c2, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c2, vb4567c2, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c2, vb4567c2, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c2, vb4567c2, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c2, vb4567c2, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c2, vb4567c2, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c2, vb4567c2, vacc5x4567);
      const v128_t va0c3 = wasm_v32x4_shuffle(va0, va0, 3, 3, 3, 3);
      const v128_t va1c3 = wasm_v32x4_shuffle(va1, va1, 3, 3, 3, 3);
      const v128_t va2c3 = wasm_v32x4_shuffle(va2, va2, 3, 3, 3, 3);
      const v128_t va3c3 = wasm_v32x4_shuffle(va3, va3, 3, 3, 3, 3);
      const v128_t va4c3 = wasm_v32x4_shuffle(va4, va4, 3, 3, 3, 3);
      const v128_t va5c3 = wasm_v32x4_shuffle(va5, va5, 3, 3, 3, 3);

      const v128_t vb01234567c3 = wasm_i16x8_load8x8((const int8_t*) w + 24);
      const v128_t vbi0123c3 = wasm_i32x4_extend_low_i16x8(vb01234567c3);
      const v128_t vbi4567c3 = wasm_i32x4_extend_high_i16x8(vb01234567c3);
      const v128_t vb0123c3 = wasm_f32x4_convert_i32x4(vbi0123c3);
      const v128_t vb4567c3 = wasm_f32x4_convert_i32x4(vbi4567c3);

      vacc0x0123 = wasm_f32x4_relaxed_madd(va0c3, vb0123c3, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(va1c3, vb0123c3, vacc1x0123);
      vacc2x0123 = wasm_f32x4_relaxed_madd(va2c3, vb0123c3, vacc2x0123);
      vacc3x0123 = wasm_f32x4_relaxed_madd(va3c3, vb0123c3, vacc3x0123);
      vacc4x0123 = wasm_f32x4_relaxed_madd(va4c3, vb0123c3, vacc4x0123);
      vacc5x0123 = wasm_f32x4_relaxed_madd(va5c3, vb0123c3, vacc5x0123);
      vacc0x4567 = wasm_f32x4_relaxed_madd(va0c3, vb4567c3, vacc0x4567);
      vacc1x4567 = wasm_f32x4_relaxed_madd(va1c3, vb4567c3, vacc1x4567);
      vacc2x4567 = wasm_f32x4_relaxed_madd(va2c3, vb4567c3, vacc2x4567);
      vacc3x4567 = wasm_f32x4_relaxed_madd(va3c3, vb4567c3, vacc3x4567);
      vacc4x4567 = wasm_f32x4_relaxed_madd(va4c3, vb4567c3, vacc4x4567);
      vacc5x4567 = wasm_f32x4_relaxed_madd(va5c3, vb4567c3, vacc5x4567);
      w = (const int8_t*) w + 32;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const v128_t va0 = wasm_v128_load32_splat(a0);
        a0 += 1;
        const v128_t va1 = wasm_v128_load32_splat(a1);
        a1 += 1;
        const v128_t va2 = wasm_v128_load32_splat(a2);
        a2 += 1;
        const v128_t va3 = wasm_v128_load32_splat(a3);
        a3 += 1;
        const v128_t va4 = wasm_v128_load32_splat(a4);
        a4 += 1;
        const v128_t va5 = wasm_v128_load32_splat(a5);
        a5 += 1;

        const v128_t vb01234567 = wasm_i16x8_load8x8((const int8_t*) w + 0);
        const v128_t vbi0123 = wasm_i32x4_extend_low_i16x8(vb01234567);
        const v128_t vbi4567 = wasm_i32x4_extend_high_i16x8(vb01234567);
        const v128_t vb0123 = wasm_f32x4_convert_i32x4(vbi0123);
        const v128_t vb4567 = wasm_f32x4_convert_i32x4(vbi4567);
        w = (const int8_t*) w + 8;

        vacc0x0123 = wasm_f32x4_relaxed_madd(va0, vb0123, vacc0x0123);
        vacc1x0123 = wasm_f32x4_relaxed_madd(va1, vb0123, vacc1x0123);
        vacc2x0123 = wasm_f32x4_relaxed_madd(va2, vb0123, vacc2x0123);
        vacc3x0123 = wasm_f32x4_relaxed_madd(va3, vb0123, vacc3x0123);
        vacc4x0123 = wasm_f32x4_relaxed_madd(va4, vb0123, vacc4x0123);
        vacc5x0123 = wasm_f32x4_relaxed_madd(va5, vb0123, vacc5x0123);
        vacc0x4567 = wasm_f32x4_relaxed_madd(va0, vb4567, vacc0x4567);
        vacc1x4567 = wasm_f32x4_relaxed_madd(va1, vb4567, vacc1x4567);
        vacc2x4567 = wasm_f32x4_relaxed_madd(va2, vb4567, vacc2x4567);
        vacc3x4567 = wasm_f32x4_relaxed_madd(va3, vb4567, vacc3x4567);
        vacc4x4567 = wasm_f32x4_relaxed_madd(va4, vb4567, vacc4x4567);
        vacc5x4567 = wasm_f32x4_relaxed_madd(va5, vb4567, vacc5x4567);

        k -= sizeof(float);
      } while (k != 0);
    }

    const v128_t vscale0123 = wasm_v128_load((const float*) w + 0);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);
    vacc4x0123 = wasm_f32x4_mul(vacc4x0123, vscale0123);
    vacc5x0123 = wasm_f32x4_mul(vacc5x0123, vscale0123);
    const v128_t vscale4567 = wasm_v128_load((const float*) w + 4);
    vacc0x4567 = wasm_f32x4_mul(vacc0x4567, vscale4567);
    vacc1x4567 = wasm_f32x4_mul(vacc1x4567, vscale4567);
    vacc2x4567 = wasm_f32x4_mul(vacc2x4567, vscale4567);
    vacc3x4567 = wasm_f32x4_mul(vacc3x4567, vscale4567);
    vacc4x4567 = wasm_f32x4_mul(vacc4x4567, vscale4567);
    vacc5x4567 = wasm_f32x4_mul(vacc5x4567, vscale4567);
    w = (const float*) w + 8;

    if XNN_LIKELY(nc >= 8) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      wasm_v128_store(c4, vacc4x0123);
      wasm_v128_store(c4 + 4, vacc4x4567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      wasm_v128_store(c5, vacc5x0123);
      wasm_v128_store(c5 + 4, vacc5x4567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        wasm_v128_store(c0, vacc0x0123);
        wasm_v128_store(c1, vacc1x0123);
        wasm_v128_store(c2, vacc2x0123);
        wasm_v128_store(c3, vacc3x0123);
        wasm_v128_store(c4, vacc4x0123);
        wasm_v128_store(c5, vacc5x0123);

        vacc0x0123 = vacc0x4567;
        vacc1x0123 = vacc1x4567;
        vacc2x0123 = vacc2x4567;
        vacc3x0123 = vacc3x4567;
        vacc4x0123 = vacc4x4567;
        vacc5x0123 = vacc5x4567;

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
      }
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        wasm_v128_store64_lane(c4, vacc4x0123, 0);
        wasm_v128_store64_lane(c5, vacc5x0123, 0);

        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        vacc4x0123 = wasm_v64x2_shuffle(vacc4x0123, vacc4x0123, 1, 1);
        vacc5x0123 = wasm_v64x2_shuffle(vacc5x0123, vacc5x0123, 1, 1);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c4, vacc4x0123, 0);
        wasm_v128_store32_lane(c5, vacc5x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E400p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(-0x1.7F7D1Cp-20f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_const_splat(0x1.FFFFF6p-1f);
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const v128_t vi_max = wasm_v128_load32_splat(max);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
  v128_t vacc1 = vacc0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    // Load 16 (4x4) inputs at a time.
    const v128_t vi0123 = wasm_v128_load(input);
    const v128_t vi4567 = wasm_v128_load(input + 4);
    const v128_t vi89AB = wasm_v128_load(input + 8);
    const v128_t viCDEF = wasm_v128_load(input + 12);
    input += 16;

    const v128_t vx0123 = wasm_f32x4_sub(vi0123, vi_max);
    const v128_t vx4567 = wasm_f32x4_sub(vi4567, vi_max);
    const v128_t vx89AB = wasm_f32x4_sub(vi89AB, vi_max);
    const v128_t vxCDEF = wasm_f32x4_sub(viCDEF, vi_max);

    v128_t vn0123 = wasm_f32x4_relaxed_madd(vx0123, vlog2e, vmagic_bias);
    v128_t vn4567 = wasm_f32x4_relaxed_madd(vx4567, vlog2e, vmagic_bias);
    v128_t vn89AB = wasm_f32x4_relaxed_madd(vx89AB, vlog2e, vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_relaxed_madd(vxCDEF, vlog2e, vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_hi, vx0123);
    v128_t vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_hi, vx4567);
    v128_t vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_hi, vx89AB);
    v128_t vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_hi, vxCDEF);

    vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_lo, vt0123);
    vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_lo, vt4567);
    vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_lo, vt89AB);
    vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_lo, vtCDEF);

    v128_t vp0123 = wasm_f32x4_relaxed_madd(vc5, vt0123, vc4);
    v128_t vp4567 = wasm_f32x4_relaxed_madd(vc5, vt4567, vc4);
    v128_t vp89AB = wasm_f32x4_relaxed_madd(vc5, vt89AB, vc4);
    v128_t vpCDEF = wasm_f32x4_relaxed_madd(vc5, vtCDEF, vc4);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc3);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc3);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc3);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc3);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc2);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc2);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc2);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc2);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc1);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc1);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc1);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc1);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);

    v128_t vf0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vs0123);
    v128_t vf4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vs4567);
    v128_t vf89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vs89AB);
    v128_t vfCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vsCDEF);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_lt(vx0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_lt(vx4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_lt(vx89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_lt(vxCDEF, vdenorm_cutoff));

    wasm_v128_store(output, vf0123);
    wasm_v128_store(output + 4, vf4567);
    wasm_v128_store(output + 8, vf89AB);
    wasm_v128_store(output + 12, vfCDEF);
    output += 16;

    vacc0 = wasm_f32x4_add(vacc0, vf0123);
    vacc0 = wasm_f32x4_add(vacc0, vf4567);
    vacc0 = wasm_f32x4_add(vacc0, vf89AB);
    vacc0 = wasm_f32x4_add(vacc0, vfCDEF);
  }
  // Add up all accumulators to vacc0
  vacc0 = wasm_f32x4_add(vacc0, vacc1);

  v128_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vi = wasm_v128_load(input);
    input += 4;

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = wasm_f32x4_relaxed_madd(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vx);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc5, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_relaxed_madd(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    wasm_v128_store(output, vf);
    output += 4;

    vacc = wasm_f32x4_add(vacc, vf);
  }
  vacc = wasm_f32x4_add(vacc, wasm_v64x2_shuffle(vacc, vacc, 1, 1));
  float vsum = wasm_f32x4_extract_lane(vacc, 0) + wasm_f32x4_extract_lane(vacc, 1);
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));

    const v128_t vi = wasm_v128_load(input);

    const v128_t vx = wasm_f32x4_sub(vi, vi_max);

    v128_t vn = wasm_f32x4_relaxed_madd(vx, vlog2e, vmagic_bias);

    const v128_t vs = wasm_i32x4_shl(vn, 23);

    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vx);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc5, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    v128_t vf = wasm_f32x4_relaxed_madd(vt, vp, vs);

    vf = wasm_v128_andnot(vf, wasm_f32x4_lt(vx, vdenorm_cutoff));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vf, 0);
      output += 2;

      vsum += wasm_f32x4_extract_lane(vf, 0) + wasm_f32x4_extract_lane(vf, 1);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vf, 0);
      vsum += wasm_f32x4_extract_lane(vf, 0);
    }
  }
  *sum = vsum;
}

void xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
      v128_t vacc4567 = vacc0123;
      v128_t vacc89AB = vacc0123;
      v128_t vaccCDEF = vacc0123;
      v128_t vaccGHIJ = vacc0123;
      v128_t vaccKLMN = vacc0123;
      v128_t vaccOPQR = vacc0123;
      v128_t vaccSTUV = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const v128_t vi0123 = wasm_v128_load(input);
          const v128_t vi4567 = wasm_v128_load(input + 4);
          const v128_t vi89AB = wasm_v128_load(input + 8);
          const v128_t viCDEF = wasm_v128_load(input + 12);
          const v128_t viGHIJ = wasm_v128_load(input + 16);
          const v128_t viKLMN = wasm_v128_load(input + 20);
          const v128_t viOPQR = wasm_v128_load(input + 24);
          const v128_t viSTUV = wasm_v128_load(input + 28);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const v128_t vw = wasm_v128_load32_splat(w); w += 1;
          vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
          vacc89AB = wasm_f32x4_relaxed_madd(vi89AB, vw, vacc89AB);
          vaccCDEF = wasm_f32x4_relaxed_madd(viCDEF, vw, vaccCDEF);
          vaccGHIJ = wasm_f32x4_relaxed_madd(viGHIJ, vw, vaccGHIJ);
          vaccKLMN = wasm_f32x4_relaxed_madd(viKLMN, vw, vaccKLMN);
          vaccOPQR = wasm_f32x4_relaxed_madd(viOPQR, vw, vaccOPQR);
          vaccSTUV = wasm_f32x4_relaxed_madd(viSTUV, vw, vaccSTUV);
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
      v128_t vout4567 = wasm_f32x4_min(vmax, vacc4567);
      v128_t vout89AB = wasm_f32x4_min(vmax, vacc89AB);
      v128_t voutCDEF = wasm_f32x4_min(vmax, vaccCDEF);
      v128_t voutGHIJ = wasm_f32x4_min(vmax, vaccGHIJ);
      v128_t voutKLMN = wasm_f32x4_min(vmax, vaccKLMN);
      v128_t voutOPQR = wasm_f32x4_min(vmax, vaccOPQR);
      v128_t voutSTUV = wasm_f32x4_min(vmax, vaccSTUV);
      vout0123 = wasm_f32x4_max(vmin, vout0123);
      vout4567 = wasm_f32x4_max(vmin, vout4567);
      vout89AB = wasm_f32x4_max(vmin, vout89AB);
      voutCDEF = wasm_f32x4_max(vmin, voutCDEF);
      voutGHIJ = wasm_f32x4_max(vmin, voutGHIJ);
      voutKLMN = wasm_f32x4_max(vmin, voutKLMN);
      voutOPQR = wasm_f32x4_max(vmin, voutOPQR);
      voutSTUV = wasm_f32x4_max(vmin, voutSTUV);
      wasm_v128_store(output, vout0123);
      wasm_v128_store(output + 4, vout4567);
      wasm_v128_store(output + 8, vout89AB);
      wasm_v128_store(output + 12, voutCDEF);
      wasm_v128_store(output + 16, voutGHIJ);
      wasm_v128_store(output + 20, voutKLMN);
      wasm_v128_store(output + 24, voutOPQR);
      wasm_v128_store(output + 28, voutSTUV);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        v128_t vacc89AB = vacc0123;
        v128_t vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            const v128_t vi89AB = wasm_v128_load(input + 8);
            const v128_t viCDEF = wasm_v128_load(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
            vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
            vacc89AB = wasm_f32x4_relaxed_madd(vi89AB, vw, vacc89AB);
            vaccCDEF = wasm_f32x4_relaxed_madd(viCDEF, vw, vaccCDEF);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_min(vmax, vacc4567);
        v128_t vout89AB = wasm_f32x4_min(vmax, vacc89AB);
        v128_t voutCDEF = wasm_f32x4_min(vmax, vaccCDEF);
        vout0123 = wasm_f32x4_max(vmin, vout0123);
        vout4567 = wasm_f32x4_max(vmin, vout4567);
        vout89AB = wasm_f32x4_max(vmin, vout89AB);
        voutCDEF = wasm_f32x4_max(vmin, voutCDEF);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        wasm_v128_store(output + 8, vout89AB);
        wasm_v128_store(output + 12, voutCDEF);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
            vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_min(vmax, vacc4567);
        vout0123 = wasm_f32x4_max(vmin, vout0123);
        vout4567 = wasm_f32x4_max(vmin, vout4567);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_min(vmax, vacc0123);
        vout0123 = wasm_f32x4_max(vmin, vout0123);
        wasm_v128_store(output, vout0123);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc01 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi01 = wasm_v128_load64_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc01 = wasm_f32x4_relaxed_madd(vi01, vw, vacc01);
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_f32x4_min(vmax, vacc01);
        vout01 = wasm_f32x4_max(vmin, vout01);
        wasm_v128_store64_lane(output, vout01, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0 = wasm_v128_load32_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0 = wasm_f32x4_relaxed_madd(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_f32x4_min(vmax, vacc0);
        vout0 = wasm_f32x4_max(vmin, vout0);
        wasm_v128_store32_lane(output, vout0, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
      v128_t vacc4567 = vacc0123;
      v128_t vacc89AB = vacc0123;
      v128_t vaccCDEF = vacc0123;
      v128_t vaccGHIJ = vacc0123;
      v128_t vaccKLMN = vacc0123;
      v128_t vaccOPQR = vacc0123;
      v128_t vaccSTUV = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const v128_t vi0123 = wasm_v128_load(input);
          const v128_t vi4567 = wasm_v128_load(input + 4);
          const v128_t vi89AB = wasm_v128_load(input + 8);
          const v128_t viCDEF = wasm_v128_load(input + 12);
          const v128_t viGHIJ = wasm_v128_load(input + 16);
          const v128_t viKLMN = wasm_v128_load(input + 20);
          const v128_t viOPQR = wasm_v128_load(input + 24);
          const v128_t viSTUV = wasm_v128_load(input + 28);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const v128_t vw = wasm_v128_load32_splat(w); w += 1;
          vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
          vacc89AB = wasm_f32x4_relaxed_madd(vi89AB, vw, vacc89AB);
          vaccCDEF = wasm_f32x4_relaxed_madd(viCDEF, vw, vaccCDEF);
          vaccGHIJ = wasm_f32x4_relaxed_madd(viGHIJ, vw, vaccGHIJ);
          vaccKLMN = wasm_f32x4_relaxed_madd(viKLMN, vw, vaccKLMN);
          vaccOPQR = wasm_f32x4_relaxed_madd(viOPQR, vw, vaccOPQR);
          vaccSTUV = wasm_f32x4_relaxed_madd(viSTUV, vw, vaccSTUV);
        } while (--nnz != 0);
      }
      v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
      v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
      v128_t vout89AB = wasm_f32x4_pmin(vmax, vacc89AB);
      v128_t voutCDEF = wasm_f32x4_pmin(vmax, vaccCDEF);
      v128_t voutGHIJ = wasm_f32x4_pmin(vmax, vaccGHIJ);
      v128_t voutKLMN = wasm_f32x4_pmin(vmax, vaccKLMN);
      v128_t voutOPQR = wasm_f32x4_pmin(vmax, vaccOPQR);
      v128_t voutSTUV = wasm_f32x4_pmin(vmax, vaccSTUV);
      vout0123 = wasm_f32x4_pmax(vmin, vout0123);
      vout4567 = wasm_f32x4_pmax(vmin, vout4567);
      vout89AB = wasm_f32x4_pmax(vmin, vout89AB);
      voutCDEF = wasm_f32x4_pmax(vmin, voutCDEF);
      voutGHIJ = wasm_f32x4_pmax(vmin, voutGHIJ);
      voutKLMN = wasm_f32x4_pmax(vmin, voutKLMN);
      voutOPQR = wasm_f32x4_pmax(vmin, voutOPQR);
      voutSTUV = wasm_f32x4_pmax(vmin, voutSTUV);
      wasm_v128_store(output, vout0123);
      wasm_v128_store(output + 4, vout4567);
      wasm_v128_store(output + 8, vout89AB);
      wasm_v128_store(output + 12, voutCDEF);
      wasm_v128_store(output + 16, voutGHIJ);
      wasm_v128_store(output + 20, voutKLMN);
      wasm_v128_store(output + 24, voutOPQR);
      wasm_v128_store(output + 28, voutSTUV);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        v128_t vacc89AB = vacc0123;
        v128_t vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            const v128_t vi89AB = wasm_v128_load(input + 8);
            const v128_t viCDEF = wasm_v128_load(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
            vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
            vacc89AB = wasm_f32x4_relaxed_madd(vi89AB, vw, vacc89AB);
            vaccCDEF = wasm_f32x4_relaxed_madd(viCDEF, vw, vaccCDEF);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
        v128_t vout89AB = wasm_f32x4_pmin(vmax, vacc89AB);
        v128_t voutCDEF = wasm_f32x4_pmin(vmax, vaccCDEF);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
        vout4567 = wasm_f32x4_pmax(vmin, vout4567);
        vout89AB = wasm_f32x4_pmax(vmin, vout89AB);
        voutCDEF = wasm_f32x4_pmax(vmin, voutCDEF);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        wasm_v128_store(output + 8, vout89AB);
        wasm_v128_store(output + 12, voutCDEF);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        v128_t vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            const v128_t vi4567 = wasm_v128_load(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
            vacc4567 = wasm_f32x4_relaxed_madd(vi4567, vw, vacc4567);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        v128_t vout4567 = wasm_f32x4_pmin(vmax, vacc4567);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
        vout4567 = wasm_f32x4_pmax(vmin, vout4567);
        wasm_v128_store(output, vout0123);

        wasm_v128_store(output + 4, vout4567);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0123 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0123 = wasm_v128_load(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0123 = wasm_f32x4_relaxed_madd(vi0123, vw, vacc0123);
          } while (--nnz != 0);
        }
        v128_t vout0123 = wasm_f32x4_pmin(vmax, vacc0123);
        vout0123 = wasm_f32x4_pmax(vmin, vout0123);
        wasm_v128_store(output, vout0123);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 4;
    }
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc01 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi01 = wasm_v128_load64_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc01 = wasm_f32x4_relaxed_madd(vi01, vw, vacc01);
          } while (--nnz != 0);
        }
        v128_t vout01 = wasm_f32x4_pmin(vmax, vacc01);
        vout01 = wasm_f32x4_pmax(vmin, vout01);
        wasm_v128_store64_lane(output, vout01, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        v128_t vacc0 = wasm_v128_load32_splat(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const v128_t vi0 = wasm_v128_load32_splat(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const v128_t vw = wasm_v128_load32_splat(w); w += 1;
            vacc0 = wasm_f32x4_relaxed_madd(vi0, vw, vacc0);
          } while (--nnz != 0);
        }
        v128_t vout0 = wasm_f32x4_pmin(vmax, vacc0);
        vout0 = wasm_f32x4_pmax(vmin, vout0);
        wasm_v128_store32_lane(output, vout0, 0);

        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}

void xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vsat_cutoff = wasm_f32x4_const_splat(-0x1.154246p+4f);
  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vlog2e = wasm_f32x4_const_splat(0x1.715476p+0f);
  const v128_t vminus_ln2_hi = wasm_f32x4_const_splat(-0x1.62E440p-1f);
  const v128_t vminus_ln2_lo = wasm_f32x4_const_splat(0x1.0105C6p-21f);
  const v128_t vc6 = wasm_f32x4_const_splat(0x1.6b7338p-10f);
  const v128_t vc5 = wasm_f32x4_const_splat(0x1.12278Ep-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.555716p-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(0x1.5554B0p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFFFEp-2f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc6);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const v128_t vprescale = wasm_v128_load32_splat(&params->scalar.prescale);
  const v128_t valpha = wasm_v128_load32_splat(&params->scalar.alpha);
  const v128_t vbeta = wasm_v128_load32_splat(&params->scalar.beta);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    v128_t vx0123 = wasm_v128_load(input);
    v128_t vx4567 = wasm_v128_load(input + 4);
    v128_t vx89AB = wasm_v128_load(input + 8);
    v128_t vxCDEF = wasm_v128_load(input + 12);
    v128_t vxGHIJ = wasm_v128_load(input + 16);
    v128_t vxKLMN = wasm_v128_load(input + 20);
    input += 24;

    const v128_t vz0123 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx0123, vprescale));
    const v128_t vz4567 = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx4567, vprescale));
    const v128_t vz89AB = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx89AB, vprescale));
    const v128_t vzCDEF = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vxCDEF, vprescale));
    const v128_t vzGHIJ = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vxGHIJ, vprescale));
    const v128_t vzKLMN = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vxKLMN, vprescale));

    v128_t vn0123 = wasm_f32x4_relaxed_madd(vz0123, vlog2e, vmagic_bias);
    v128_t vn4567 = wasm_f32x4_relaxed_madd(vz4567, vlog2e, vmagic_bias);
    v128_t vn89AB = wasm_f32x4_relaxed_madd(vz89AB, vlog2e, vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_relaxed_madd(vzCDEF, vlog2e, vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_relaxed_madd(vzGHIJ, vlog2e, vmagic_bias);
    v128_t vnKLMN = wasm_f32x4_relaxed_madd(vzKLMN, vlog2e, vmagic_bias);

    v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);
    v128_t vsKLMN = wasm_i32x4_shl(vnKLMN, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);
    vnKLMN = wasm_f32x4_sub(vnKLMN, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_hi, vz0123);
    v128_t vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_hi, vz4567);
    v128_t vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_hi, vz89AB);
    v128_t vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_hi, vzCDEF);
    v128_t vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_hi, vzGHIJ);
    v128_t vtKLMN = wasm_f32x4_relaxed_madd(vnKLMN, vminus_ln2_hi, vzKLMN);

    vt0123 = wasm_f32x4_relaxed_madd(vn0123, vminus_ln2_lo, vt0123);
    vt4567 = wasm_f32x4_relaxed_madd(vn4567, vminus_ln2_lo, vt4567);
    vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vminus_ln2_lo, vt89AB);
    vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vminus_ln2_lo, vtCDEF);
    vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vminus_ln2_lo, vtGHIJ);
    vtKLMN = wasm_f32x4_relaxed_madd(vnKLMN, vminus_ln2_lo, vtKLMN);

    v128_t vp0123 = wasm_f32x4_relaxed_madd(vc6, vt0123, vc5);
    v128_t vp4567 = wasm_f32x4_relaxed_madd(vc6, vt4567, vc5);
    v128_t vp89AB = wasm_f32x4_relaxed_madd(vc6, vt89AB, vc5);
    v128_t vpCDEF = wasm_f32x4_relaxed_madd(vc6, vtCDEF, vc5);
    v128_t vpGHIJ = wasm_f32x4_relaxed_madd(vc6, vtGHIJ, vc5);
    v128_t vpKLMN = wasm_f32x4_relaxed_madd(vc6, vtKLMN, vc5);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc4);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc4);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc4);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc4);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc4);
    vpKLMN = wasm_f32x4_relaxed_madd(vpKLMN, vtKLMN, vc4);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc3);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc3);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc3);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc3);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc3);
    vpKLMN = wasm_f32x4_relaxed_madd(vpKLMN, vtKLMN, vc3);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vc2);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vc2);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vc2);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vc2);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vc2);
    vpKLMN = wasm_f32x4_relaxed_madd(vpKLMN, vtKLMN, vc2);

    vp0123 = wasm_f32x4_mul(vp0123, vt0123);
    vp4567 = wasm_f32x4_mul(vp4567, vt4567);
    vp89AB = wasm_f32x4_mul(vp89AB, vt89AB);
    vpCDEF = wasm_f32x4_mul(vpCDEF, vtCDEF);
    vpGHIJ = wasm_f32x4_mul(vpGHIJ, vtGHIJ);
    vpKLMN = wasm_f32x4_mul(vpKLMN, vtKLMN);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vs0123 = wasm_f32x4_sub(vs0123, vone);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vs4567 = wasm_f32x4_sub(vs4567, vone);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vs89AB = wasm_f32x4_sub(vs89AB, vone);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vsCDEF = wasm_f32x4_sub(vsCDEF, vone);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vsGHIJ = wasm_f32x4_sub(vsGHIJ, vone);
    vtKLMN = wasm_f32x4_mul(vtKLMN, vsKLMN);
    vsKLMN = wasm_f32x4_sub(vsKLMN, vone);

    vp0123 = wasm_f32x4_relaxed_madd(vp0123, vt0123, vt0123);
    vp4567 = wasm_f32x4_relaxed_madd(vp4567, vt4567, vt4567);
    vp89AB = wasm_f32x4_relaxed_madd(vp89AB, vt89AB, vt89AB);
    vpCDEF = wasm_f32x4_relaxed_madd(vpCDEF, vtCDEF, vtCDEF);
    vpGHIJ = wasm_f32x4_relaxed_madd(vpGHIJ, vtGHIJ, vtGHIJ);
    vpKLMN = wasm_f32x4_relaxed_madd(vpKLMN, vtKLMN, vtKLMN);

    const v128_t ve0123 = wasm_f32x4_mul(wasm_f32x4_add(vp0123, vs0123), valpha);
    const v128_t ve4567 = wasm_f32x4_mul(wasm_f32x4_add(vp4567, vs4567), valpha);
    const v128_t ve89AB = wasm_f32x4_mul(wasm_f32x4_add(vp89AB, vs89AB), valpha);
    const v128_t veCDEF = wasm_f32x4_mul(wasm_f32x4_add(vpCDEF, vsCDEF), valpha);
    const v128_t veGHIJ = wasm_f32x4_mul(wasm_f32x4_add(vpGHIJ, vsGHIJ), valpha);
    const v128_t veKLMN = wasm_f32x4_mul(wasm_f32x4_add(vpKLMN, vsKLMN), valpha);

    const v128_t vsignm0123 = wasm_i32x4_shr(vx0123, 31);
    vx0123 = wasm_f32x4_mul(vx0123, vbeta);
    const v128_t vsignm4567 = wasm_i32x4_shr(vx4567, 31);
    vx4567 = wasm_f32x4_mul(vx4567, vbeta);
    const v128_t vsignm89AB = wasm_i32x4_shr(vx89AB, 31);
    vx89AB = wasm_f32x4_mul(vx89AB, vbeta);
    const v128_t vsignmCDEF = wasm_i32x4_shr(vxCDEF, 31);
    vxCDEF = wasm_f32x4_mul(vxCDEF, vbeta);
    const v128_t vsignmGHIJ = wasm_i32x4_shr(vxGHIJ, 31);
    vxGHIJ = wasm_f32x4_mul(vxGHIJ, vbeta);
    const v128_t vsignmKLMN = wasm_i32x4_shr(vxKLMN, 31);
    vxKLMN = wasm_f32x4_mul(vxKLMN, vbeta);

    const v128_t vy0123 = wasm_i32x4_relaxed_laneselect(ve0123, vx0123, vsignm0123);
    const v128_t vy4567 = wasm_i32x4_relaxed_laneselect(ve4567, vx4567, vsignm4567);
    const v128_t vy89AB = wasm_i32x4_relaxed_laneselect(ve89AB, vx89AB, vsignm89AB);
    const v128_t vyCDEF = wasm_i32x4_relaxed_laneselect(veCDEF, vxCDEF, vsignmCDEF);
    const v128_t vyGHIJ = wasm_i32x4_relaxed_laneselect(veGHIJ, vxGHIJ, vsignmGHIJ);
    const v128_t vyKLMN = wasm_i32x4_relaxed_laneselect(veKLMN, vxKLMN, vsignmKLMN);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    wasm_v128_store(output + 12, vyCDEF);
    wasm_v128_store(output + 16, vyGHIJ);
    wasm_v128_store(output + 20, vyKLMN);
    output += 24;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_relaxed_max(vsat_cutoff, wasm_f32x4_mul(vx, vprescale));

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vlog2e, vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc6, vt, vc5);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    const v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_relaxed_max(wasm_f32x4_mul(vx, vprescale), vsat_cutoff);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vlog2e, vmagic_bias);
    v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vminus_ln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vc6, vt, vc5);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc4);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc3);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vc2);
    vp = wasm_f32x4_mul(vp, vt);

    vt = wasm_f32x4_mul(vt, vs);
    vs = wasm_f32x4_sub(vs, vone);
    vp = wasm_f32x4_relaxed_madd(vp, vt, vt);
    const v128_t ve = wasm_f32x4_mul(wasm_f32x4_add(vp, vs), valpha);

    const v128_t vsignm = wasm_i32x4_shr(vx, 31);
    vx = wasm_f32x4_mul(vx, vbeta);
    v128_t vy = wasm_i32x4_relaxed_laneselect(ve, vx, vsignm);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}

void xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vslope = wasm_v128_load32_splat(&params->scalar.slope);
  const v128_t vzero = wasm_i32x4_const_splat(0);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);
    input += 4;
    v128_t vacc = wasm_i32x4_max(vx, vzero);
    vx = wasm_i32x4_min(vx, vzero);
    vacc = wasm_f32x4_relaxed_madd(vx, vslope, vacc);
    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);
    v128_t vacc = wasm_i32x4_max(vx, vzero);
    vx = wasm_i32x4_min(vx, vzero);
    vacc = wasm_f32x4_relaxed_madd(vx, vslope, vacc);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vacc, 0);
      vacc = wasm_v64x2_shuffle(vacc, vacc, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vacc, 0);
    }
  }
}

void xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vslope = wasm_v128_load32_splat(&params->scalar.slope);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;
    v128_t vacc = wasm_f32x4_mul(vx, vslope);
    const v128_t vmask = wasm_i32x4_shr(vx, 31);
    vacc = wasm_i32x4_relaxed_laneselect(vacc, vx, vmask);
    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);
    v128_t vacc = wasm_f32x4_mul(vx, vslope);
    const v128_t vmask = wasm_i32x4_shr(vx, 31);
    vacc = wasm_i32x4_relaxed_laneselect(vacc, vx, vmask);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vacc, 0);
      vacc = wasm_v64x2_shuffle(vacc, vacc, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vacc, 0);
    }
  }
}

void xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const v128_t vscale0123 = wasm_v128_load(w);

      v128_t vacc0x0123 = wasm_v128_load(i0);
      i0 += 4;
      v128_t vacc1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vbias0123 = wasm_v128_load(w + 4);

      vacc0x0123 = wasm_f32x4_relaxed_madd(vscale0123, vacc0x0123, vbias0123);
      vacc1x0123 = wasm_f32x4_relaxed_madd(vscale0123, vacc1x0123, vbias0123);

      vacc0x0123 = wasm_f32x4_relaxed_max(vmin, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_max(vmin, vacc1x0123);

      vacc0x0123 = wasm_f32x4_relaxed_min(vmax, vacc0x0123);
      vacc1x0123 = wasm_f32x4_relaxed_min(vmax, vacc1x0123);

      wasm_v128_store(o0, vacc0x0123);
      o0 += 4;
      wasm_v128_store(o1, vacc1x0123);
      o1 += 4;

      w += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vscale = wasm_v128_load(w);

      v128_t vacc0 = wasm_v128_load(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      v128_t vacc1 = wasm_v128_load(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const v128_t vbias = wasm_v128_load(w + 4);

      vacc0 = wasm_f32x4_relaxed_madd(vscale, vacc0, vbias);
      vacc1 = wasm_f32x4_relaxed_madd(vscale, vacc1, vbias);

      vacc0 = wasm_f32x4_relaxed_max(vmin, vacc0);
      vacc1 = wasm_f32x4_relaxed_max(vmin, vacc1);

      vacc0 = wasm_f32x4_relaxed_min(vmax, vacc0);
      vacc1 = wasm_f32x4_relaxed_min(vmax, vacc1);

      if (c & (2 * sizeof(float))) {
        wasm_v128_store64_lane(o0, vacc0, 0);
        wasm_v128_store64_lane(o1, vacc1, 0);

        vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
        vacc1 = wasm_v64x2_shuffle(vacc1, vacc1, 1, 1);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        wasm_v128_store32_lane(o0, vacc0, 0);
        o0 += 1;
        wasm_v128_store32_lane(o1, vacc1, 0);
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vmagic_bias = wasm_f32x4_const_splat(0x1.8000FEp23f);
  const v128_t vminus_log2e = wasm_f32x4_const_splat(-0x1.715476p+0f);
  const v128_t vln2_hi = wasm_f32x4_const_splat(0x1.62E400p-1f);
  const v128_t vln2_lo = wasm_f32x4_const_splat(0x1.7F7D1Cp-20f);
  const v128_t vc5 = wasm_f32x4_const_splat(-0x1.0F9F9Cp-7f);
  const v128_t vc4 = wasm_f32x4_const_splat(0x1.573A1Ap-5f);
  const v128_t vc3 = wasm_f32x4_const_splat(-0x1.555A80p-3f);
  const v128_t vc2 = wasm_f32x4_const_splat(0x1.FFFDC6p-2f);
  const v128_t vc1 = wasm_f32x4_const_splat(-0x1.FFFFF6p-1f);
  const v128_t vone = wasm_f32x4_const_splat(1.0f);
  const v128_t vdenorm_cutoff = wasm_f32x4_const_splat(0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  // XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const v128_t vx0123 = wasm_v128_load(input);
    const v128_t vx4567 = wasm_v128_load(input + 4);
    const v128_t vx89AB = wasm_v128_load(input + 8);
    const v128_t vxCDEF = wasm_v128_load(input + 12);
    const v128_t vxGHIJ = wasm_v128_load(input + 16);
    const v128_t vxKLMN = wasm_v128_load(input + 20);
    input += 24;

    const v128_t vz0123 = wasm_f32x4_abs(vx0123);
    const v128_t vz4567 = wasm_f32x4_abs(vx4567);
    const v128_t vz89AB = wasm_f32x4_abs(vx89AB);
    const v128_t vzCDEF = wasm_f32x4_abs(vxCDEF);
    const v128_t vzGHIJ = wasm_f32x4_abs(vxGHIJ);
    const v128_t vzKLMN = wasm_f32x4_abs(vxKLMN);

    v128_t vn0123 = wasm_f32x4_relaxed_madd(vz0123, vminus_log2e, vmagic_bias);
    v128_t vn4567 = wasm_f32x4_relaxed_madd(vz4567, vminus_log2e, vmagic_bias);
    v128_t vn89AB = wasm_f32x4_relaxed_madd(vz89AB, vminus_log2e, vmagic_bias);
    v128_t vnCDEF = wasm_f32x4_relaxed_madd(vzCDEF, vminus_log2e, vmagic_bias);
    v128_t vnGHIJ = wasm_f32x4_relaxed_madd(vzGHIJ, vminus_log2e, vmagic_bias);
    v128_t vnKLMN = wasm_f32x4_relaxed_madd(vzKLMN, vminus_log2e, vmagic_bias);

    const v128_t vs0123 = wasm_i32x4_shl(vn0123, 23);
    const v128_t vs4567 = wasm_i32x4_shl(vn4567, 23);
    const v128_t vs89AB = wasm_i32x4_shl(vn89AB, 23);
    const v128_t vsCDEF = wasm_i32x4_shl(vnCDEF, 23);
    const v128_t vsGHIJ = wasm_i32x4_shl(vnGHIJ, 23);
    const v128_t vsKLMN = wasm_i32x4_shl(vnKLMN, 23);

    vn0123 = wasm_f32x4_sub(vn0123, vmagic_bias);
    vn4567 = wasm_f32x4_sub(vn4567, vmagic_bias);
    vn89AB = wasm_f32x4_sub(vn89AB, vmagic_bias);
    vnCDEF = wasm_f32x4_sub(vnCDEF, vmagic_bias);
    vnGHIJ = wasm_f32x4_sub(vnGHIJ, vmagic_bias);
    vnKLMN = wasm_f32x4_sub(vnKLMN, vmagic_bias);

    v128_t vt0123 = wasm_f32x4_relaxed_madd(vn0123, vln2_hi, vz0123);
    v128_t vt4567 = wasm_f32x4_relaxed_madd(vn4567, vln2_hi, vz4567);
    v128_t vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vln2_hi, vz89AB);
    v128_t vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vln2_hi, vzCDEF);
    v128_t vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vln2_hi, vzGHIJ);
    v128_t vtKLMN = wasm_f32x4_relaxed_madd(vnKLMN, vln2_hi, vzKLMN);

    vt0123 = wasm_f32x4_relaxed_madd(vn0123, vln2_lo, vt0123);
    vt4567 = wasm_f32x4_relaxed_madd(vn4567, vln2_lo, vt4567);
    vt89AB = wasm_f32x4_relaxed_madd(vn89AB, vln2_lo, vt89AB);
    vtCDEF = wasm_f32x4_relaxed_madd(vnCDEF, vln2_lo, vtCDEF);
    vtGHIJ = wasm_f32x4_relaxed_madd(vnGHIJ, vln2_lo, vtGHIJ);
    vtKLMN = wasm_f32x4_relaxed_madd(vnKLMN, vln2_lo, vtKLMN);

    v128_t vp0123 = wasm_f32x4_relaxed_madd(vt0123, vc5, vc4);
    v128_t vp4567 = wasm_f32x4_relaxed_madd(vt4567, vc5, vc4);
    v128_t vp89AB = wasm_f32x4_relaxed_madd(vt89AB, vc5, vc4);
    v128_t vpCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vc5, vc4);
    v128_t vpGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vc5, vc4);
    v128_t vpKLMN = wasm_f32x4_relaxed_madd(vtKLMN, vc5, vc4);

    vp0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vc3);
    vp4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vc3);
    vp89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vc3);
    vpCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vc3);
    vpGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vpGHIJ, vc3);
    vpKLMN = wasm_f32x4_relaxed_madd(vtKLMN, vpKLMN, vc3);

    vp0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vc2);
    vp4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vc2);
    vp89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vc2);
    vpCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vc2);
    vpGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vpGHIJ, vc2);
    vpKLMN = wasm_f32x4_relaxed_madd(vtKLMN, vpKLMN, vc2);

    vp0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vc1);
    vp4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vc1);
    vp89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vc1);
    vpCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vc1);
    vpGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vpGHIJ, vc1);
    vpKLMN = wasm_f32x4_relaxed_madd(vtKLMN, vpKLMN, vc1);

    vt0123 = wasm_f32x4_mul(vt0123, vs0123);
    vt4567 = wasm_f32x4_mul(vt4567, vs4567);
    vt89AB = wasm_f32x4_mul(vt89AB, vs89AB);
    vtCDEF = wasm_f32x4_mul(vtCDEF, vsCDEF);
    vtGHIJ = wasm_f32x4_mul(vtGHIJ, vsGHIJ);
    vtKLMN = wasm_f32x4_mul(vtKLMN, vsKLMN);

    const v128_t ve0123 = wasm_f32x4_relaxed_madd(vt0123, vp0123, vs0123);
    const v128_t ve4567 = wasm_f32x4_relaxed_madd(vt4567, vp4567, vs4567);
    const v128_t ve89AB = wasm_f32x4_relaxed_madd(vt89AB, vp89AB, vs89AB);
    const v128_t veCDEF = wasm_f32x4_relaxed_madd(vtCDEF, vpCDEF, vsCDEF);
    const v128_t veGHIJ = wasm_f32x4_relaxed_madd(vtGHIJ, vpGHIJ, vsGHIJ);
    const v128_t veKLMN = wasm_f32x4_relaxed_madd(vtKLMN, vpKLMN, vsKLMN);

    const v128_t vd0123 = wasm_f32x4_add(ve0123, vone);
    const v128_t vd4567 = wasm_f32x4_add(ve4567, vone);
    const v128_t vd89AB = wasm_f32x4_add(ve89AB, vone);
    const v128_t vdCDEF = wasm_f32x4_add(veCDEF, vone);
    const v128_t vdGHIJ = wasm_f32x4_add(veGHIJ, vone);
    const v128_t vdKLMN = wasm_f32x4_add(veKLMN, vone);

    v128_t vf0123 = wasm_f32x4_div(ve0123, vd0123);
    v128_t vf4567 = wasm_f32x4_div(ve4567, vd4567);
    v128_t vf89AB = wasm_f32x4_div(ve89AB, vd89AB);
    v128_t vfCDEF = wasm_f32x4_div(veCDEF, vdCDEF);
    v128_t vfGHIJ = wasm_f32x4_div(veGHIJ, vdGHIJ);
    v128_t vfKLMN = wasm_f32x4_div(veKLMN, vdKLMN);

    vf0123 = wasm_v128_andnot(vf0123, wasm_f32x4_gt(vz0123, vdenorm_cutoff));
    vf4567 = wasm_v128_andnot(vf4567, wasm_f32x4_gt(vz4567, vdenorm_cutoff));
    vf89AB = wasm_v128_andnot(vf89AB, wasm_f32x4_gt(vz89AB, vdenorm_cutoff));
    vfCDEF = wasm_v128_andnot(vfCDEF, wasm_f32x4_gt(vzCDEF, vdenorm_cutoff));
    vfGHIJ = wasm_v128_andnot(vfGHIJ, wasm_f32x4_gt(vzGHIJ, vdenorm_cutoff));
    vfKLMN = wasm_v128_andnot(vfKLMN, wasm_f32x4_gt(vzKLMN, vdenorm_cutoff));

    const v128_t vcf0123 = wasm_f32x4_sub(vone, vf0123);
    const v128_t vcf4567 = wasm_f32x4_sub(vone, vf4567);
    const v128_t vcf89AB = wasm_f32x4_sub(vone, vf89AB);
    const v128_t vcfCDEF = wasm_f32x4_sub(vone, vfCDEF);
    const v128_t vcfGHIJ = wasm_f32x4_sub(vone, vfGHIJ);
    const v128_t vcfKLMN = wasm_f32x4_sub(vone, vfKLMN);

    vf0123 = wasm_i32x4_relaxed_laneselect(vf0123, vcf0123, wasm_i32x4_shr(vx0123, 31));
    vf4567 = wasm_i32x4_relaxed_laneselect(vf4567, vcf4567, wasm_i32x4_shr(vx4567, 31));
    vf89AB = wasm_i32x4_relaxed_laneselect(vf89AB, vcf89AB, wasm_i32x4_shr(vx89AB, 31));
    vfCDEF = wasm_i32x4_relaxed_laneselect(vfCDEF, vcfCDEF, wasm_i32x4_shr(vxCDEF, 31));
    vfGHIJ = wasm_i32x4_relaxed_laneselect(vfGHIJ, vcfGHIJ, wasm_i32x4_shr(vxGHIJ, 31));
    vfKLMN = wasm_i32x4_relaxed_laneselect(vfKLMN, vcfKLMN, wasm_i32x4_shr(vxKLMN, 31));

    wasm_v128_store(output, vf0123);
    wasm_v128_store(output + 4, vf4567);
    wasm_v128_store(output + 8, vf89AB);
    wasm_v128_store(output + 12, vfCDEF);
    wasm_v128_store(output + 16, vfGHIJ);
    wasm_v128_store(output + 20, vfKLMN);
    output += 24;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vt, vc5, vc4);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc3);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc2);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_relaxed_madd(vt, vp, vs);
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, wasm_i32x4_shr(vx, 31));

    wasm_v128_store(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);

    const v128_t vz = wasm_f32x4_abs(vx);

    v128_t vn = wasm_f32x4_relaxed_madd(vz, vminus_log2e, vmagic_bias);
    const v128_t vs = wasm_i32x4_shl(vn, 23);
    vn = wasm_f32x4_sub(vn, vmagic_bias);

    v128_t vt = wasm_f32x4_relaxed_madd(vn, vln2_hi, vz);
    vt = wasm_f32x4_relaxed_madd(vn, vln2_lo, vt);

    v128_t vp = wasm_f32x4_relaxed_madd(vt, vc5, vc4);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc3);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc2);
    vp = wasm_f32x4_relaxed_madd(vt, vp, vc1);

    vt = wasm_f32x4_mul(vt, vs);
    const v128_t ve = wasm_f32x4_relaxed_madd(vt, vp, vs);
    const v128_t vd = wasm_f32x4_add(ve, vone);

    v128_t vf = wasm_f32x4_div(ve, vd);
    vf = wasm_v128_andnot(vf, wasm_f32x4_gt(vz, vdenorm_cutoff));
    const v128_t vcf = wasm_f32x4_sub(vone, vf);
    vf = wasm_i32x4_relaxed_laneselect(vf, vcf, wasm_i32x4_shr(vx, 31));

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vf, 0);
      vf = wasm_v64x2_shuffle(vf, vf, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vf, 0);
    }
  }
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    const v128_t vinput_zero_point0 = wasm_v128_load32_splat(&quantization_params[0].zero_point);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point0);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point0);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point0);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point0);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

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

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    const v128_t vinput_zero_point0 = wasm_v128_load32_splat(&quantization_params[0].zero_point);
    const v128_t vinput_zero_point1 = wasm_v128_load32_splat(&quantization_params[1].zero_point);
    const v128_t vinput_zero_point2 = wasm_v128_load32_splat(&quantization_params[2].zero_point);
    const v128_t vinput_zero_point3 = wasm_v128_load32_splat(&quantization_params[3].zero_point);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point0);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point0);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point0);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point0);
    v128_t vacc1x0 = wasm_i32x4_mul(vksum0, vinput_zero_point1);
    v128_t vacc1x1 = wasm_i32x4_mul(vksum1, vinput_zero_point1);
    v128_t vacc1x2 = wasm_i32x4_mul(vksum2, vinput_zero_point1);
    v128_t vacc1x3 = wasm_i32x4_mul(vksum3, vinput_zero_point1);
    v128_t vacc2x0 = wasm_i32x4_mul(vksum0, vinput_zero_point2);
    v128_t vacc2x1 = wasm_i32x4_mul(vksum1, vinput_zero_point2);
    v128_t vacc2x2 = wasm_i32x4_mul(vksum2, vinput_zero_point2);
    v128_t vacc2x3 = wasm_i32x4_mul(vksum3, vinput_zero_point2);
    v128_t vacc3x0 = wasm_i32x4_mul(vksum0, vinput_zero_point3);
    v128_t vacc3x1 = wasm_i32x4_mul(vksum1, vinput_zero_point3);
    v128_t vacc3x2 = wasm_i32x4_mul(vksum2, vinput_zero_point3);
    v128_t vacc3x3 = wasm_i32x4_mul(vksum3, vinput_zero_point3);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 16;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 16;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 16;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
      vacc2x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va2, vacc2x0);
      vacc3x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va3, vacc3x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
      vacc2x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va2, vacc2x1);
      vacc3x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va3, vacc3x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
      vacc2x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va2, vacc2x2);
      vacc3x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va3, vacc3x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
      vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
      vacc2x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va2, vacc2x3);
      vacc3x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va3, vacc3x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));
    const v128_t vacc3x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x0, vacc3x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x0, vacc3x2, 2, 6, 3, 7));
    const v128_t vacc3x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x1, vacc3x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x1, vacc3x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x02, vacc3x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x02, vacc3x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vinput_scale0 = wasm_v128_load32_splat(&quantization_params[0].inv_scale);
    const v128_t vinput_scale1 = wasm_v128_load32_splat(&quantization_params[1].inv_scale);
    const v128_t vinput_scale2 = wasm_v128_load32_splat(&quantization_params[2].inv_scale);
    const v128_t vinput_scale3 = wasm_v128_load32_splat(&quantization_params[3].inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale0);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale1);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vinput_scale2);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vinput_scale3);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vfilter_output_scale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vfilter_output_scale0123);

    const v128_t vbias0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vbias0123);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vbias0123);

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_pmax(vacc2x0123, vmin);
    vacc3x0123 = wasm_f32x4_pmax(vacc3x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_pmin(vacc2x0123, vmax);
    vacc3x0123 = wasm_f32x4_pmin(vacc3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c0, vacc0x0123);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c3, vacc3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        c3 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_1x4c16__wasmsdot(
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

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  float* c0 = c;

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    const v128_t vinput_zero_point = wasm_v128_load32_splat(&quantization_params->zero_point);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    w = (const int32_t*) w + 4;

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
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

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

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

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

void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x4c16__wasmsdot(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    v128_t vksum0 = wasm_v128_load32_zero(w);
    v128_t vksum1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vksum2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vksum3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    const v128_t vinput_zero_point = wasm_v128_load32_splat(&quantization_params->zero_point);
    v128_t vacc0x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc0x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc0x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc0x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    v128_t vacc1x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc1x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc1x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc1x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    v128_t vacc2x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc2x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc2x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc2x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    v128_t vacc3x0 = wasm_i32x4_mul(vksum0, vinput_zero_point);
    v128_t vacc3x1 = wasm_i32x4_mul(vksum1, vinput_zero_point);
    v128_t vacc3x2 = wasm_i32x4_mul(vksum2, vinput_zero_point);
    v128_t vacc3x3 = wasm_i32x4_mul(vksum3, vinput_zero_point);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 16;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 16;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
        vacc2x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va2, vacc2x0);
        vacc3x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va3, vacc3x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
        vacc2x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va2, vacc2x1);
        vacc3x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va3, vacc3x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
        vacc2x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va2, vacc2x2);
        vacc3x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va3, vacc3x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
        vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
        vacc2x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va2, vacc2x3);
        vacc3x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va3, vacc3x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));
    const v128_t vacc3x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x0, vacc3x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x0, vacc3x2, 2, 6, 3, 7));
    const v128_t vacc3x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x1, vacc3x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x1, vacc3x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x02, vacc3x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x02, vacc3x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vinput_scale = wasm_v128_load32_splat(&quantization_params->inv_scale);

    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vinput_scale);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vinput_scale);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vinput_scale);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vinput_scale);

    const v128_t vfilter_output_scale0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vfilter_output_scale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vfilter_output_scale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vfilter_output_scale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vfilter_output_scale0123);
    w = (const float*) w + 4;

    const v128_t vbias0123 = wasm_v128_load(w);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vbias0123);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vbias0123);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vbias0123);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vbias0123);
    w = (const float*) w + 4;

    vacc0x0123 = wasm_f32x4_pmax(vacc0x0123, vmin);
    vacc1x0123 = wasm_f32x4_pmax(vacc1x0123, vmin);
    vacc2x0123 = wasm_f32x4_pmax(vacc2x0123, vmin);
    vacc3x0123 = wasm_f32x4_pmax(vacc3x0123, vmin);

    vacc0x0123 = wasm_f32x4_pmin(vacc0x0123, vmax);
    vacc1x0123 = wasm_f32x4_pmin(vacc1x0123, vmax);
    vacc2x0123 = wasm_f32x4_pmin(vacc2x0123, vmax);
    vacc3x0123 = wasm_f32x4_pmin(vacc3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store(c3, vacc3x0123);
      wasm_v128_store(c2, vacc2x0123);
      wasm_v128_store(c1, vacc1x0123);
      wasm_v128_store(c0, vacc0x0123);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store64_lane(c3, vacc3x0123, 0);
        vacc3x0123 = wasm_v64x2_shuffle(vacc3x0123, vacc3x0123, 1, 1);
        c3 += 2;
        wasm_v128_store64_lane(c2, vacc2x0123, 0);
        vacc2x0123 = wasm_v64x2_shuffle(vacc2x0123, vacc2x0123, 1, 1);
        c2 += 2;
        wasm_v128_store64_lane(c1, vacc1x0123, 0);
        vacc1x0123 = wasm_v64x2_shuffle(vacc1x0123, vacc1x0123, 1, 1);
        c1 += 2;
        wasm_v128_store64_lane(c0, vacc0x0123, 0);
        vacc0x0123 = wasm_v64x2_shuffle(vacc0x0123, vacc0x0123, 1, 1);
        c0 += 2;
      }
      if (nc & 1) {
        wasm_v128_store32_lane(c3, vacc3x0123, 0);
        wasm_v128_store32_lane(c2, vacc2x0123, 0);
        wasm_v128_store32_lane(c1, vacc1x0123, 0);
        wasm_v128_store32_lane(c0, vacc0x0123, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;


  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vacc = wasm_i8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc = wasm_i8x16_min(vacc, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmusdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;


  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_xor(wasm_v128_load(a0), vsign_mask);
      a0 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vacc = wasm_i8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc = wasm_i8x16_min(vacc, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmusdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }


  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_xor(wasm_v128_load(a0), vsign_mask);
      a0 += 16;
      const v128_t va1 = wasm_v128_xor(wasm_v128_load(a1), vsign_mask);
      a1 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
      vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);

    v128_t vacc = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc01x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc = wasm_i8x16_min(vacc, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);
      wasm_v128_store32_lane(c1, vacc, 1);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;
        wasm_v128_store16_lane(c1, vacc, 2);
        c1 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
        wasm_v128_store8_lane(c1, vacc, 4);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }


  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    v128_t vacc2x0 = vacc0x0;
    v128_t vacc2x1 = vacc0x1;
    v128_t vacc2x2 = vacc0x2;
    v128_t vacc2x3 = vacc0x3;
    v128_t vacc3x0 = vacc0x0;
    v128_t vacc3x1 = vacc0x1;
    v128_t vacc3x2 = vacc0x2;
    v128_t vacc3x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const v128_t va0 = wasm_v128_load(a0);
      a0 += 16;
      const v128_t va1 = wasm_v128_load(a1);
      a1 += 16;
      const v128_t va2 = wasm_v128_load(a2);
      a2 += 16;
      const v128_t va3 = wasm_v128_load(a3);
      a3 += 16;

      const v128_t vb0 = wasm_v128_load(w);

      vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
      vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
      vacc2x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va2, vacc2x0);
      vacc3x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va3, vacc3x0);
      const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);

      vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
      vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
      vacc2x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va2, vacc2x1);
      vacc3x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va3, vacc3x1);
      const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);

      vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
      vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
      vacc2x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va2, vacc2x2);
      vacc3x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va3, vacc3x2);
      const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);

      vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
      vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
      vacc2x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va2, vacc2x3);
      vacc3x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va3, vacc3x3);

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    } while (k != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));
    const v128_t vacc3x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x0, vacc3x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x0, vacc3x2, 2, 6, 3, 7));
    const v128_t vacc3x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x1, vacc3x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x1, vacc3x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x02, vacc3x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x02, vacc3x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vmagic_bias);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vmagic_min);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc2x0123 = wasm_i32x4_sub(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc3x0123 = wasm_i32x4_sub(vacc3x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);
    v128_t vacc23x0123 = wasm_i16x8_narrow_i32x4(vacc2x0123, vacc3x0123);

    v128_t vacc = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc23x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vacc = wasm_i8x16_min(vacc, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      wasm_v128_store32_lane(c0, vacc, 0);
      wasm_v128_store32_lane(c1, vacc, 1);
      wasm_v128_store32_lane(c2, vacc, 2);
      wasm_v128_store32_lane(c3, vacc, 3);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c0, vacc, 0);
        c0 += 2;
        wasm_v128_store16_lane(c1, vacc, 2);
        c1 += 2;
        wasm_v128_store16_lane(c2, vacc, 4);
        c2 += 2;
        wasm_v128_store16_lane(c3, vacc, 6);
        c3 += 2;

        vacc = wasm_u32x4_shr(vacc, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c0, vacc, 0);
        wasm_v128_store8_lane(c1, vacc, 4);
        wasm_v128_store8_lane(c2, vacc, 8);
        wasm_v128_store8_lane(c3, vacc, 12);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  int8_t* c0 = c;


  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c0, vout, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

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

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmusdot(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  int8_t* c0 = c;


  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_xor(wasm_v128_load(a0), vsign_mask);
        a0 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc00x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc0x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc00x0123, vacc00x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c0, vout, 0);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

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

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c16__wasmusdot(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }


  const v128_t vsign_mask = wasm_u8x16_const_splat(UINT8_C(0x80));
  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_xor(wasm_v128_load(a0), vsign_mask);
        a0 += 16;
        const v128_t va1 = wasm_v128_xor(wasm_v128_load(a1), vsign_mask);
        a1 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
        vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc01x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c1, vout, 1);
      wasm_v128_store32_lane(c0, vout, 0);

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c1, vout, 2);
        c1 += 2;
        wasm_v128_store16_lane(c0, vout, 0);
        c0 += 2;

        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c1, vout, 4);
        wasm_v128_store8_lane(c0, vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot(
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
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 16 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }


  do {
    v128_t vacc0x0 = wasm_v128_load32_zero(w);
    v128_t vacc0x1 = wasm_v128_load32_zero((const int32_t*) w + 1);
    v128_t vacc0x2 = wasm_v128_load32_zero((const int32_t*) w + 2);
    v128_t vacc0x3 = wasm_v128_load32_zero((const int32_t*) w + 3);
    v128_t vacc1x0 = vacc0x0;
    v128_t vacc1x1 = vacc0x1;
    v128_t vacc1x2 = vacc0x2;
    v128_t vacc1x3 = vacc0x3;
    v128_t vacc2x0 = vacc0x0;
    v128_t vacc2x1 = vacc0x1;
    v128_t vacc2x2 = vacc0x2;
    v128_t vacc2x3 = vacc0x3;
    v128_t vacc3x0 = vacc0x0;
    v128_t vacc3x1 = vacc0x1;
    v128_t vacc3x2 = vacc0x2;
    v128_t vacc3x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const v128_t va0 = wasm_v128_load(a0);
        a0 += 16;
        const v128_t va1 = wasm_v128_load(a1);
        a1 += 16;
        const v128_t va2 = wasm_v128_load(a2);
        a2 += 16;
        const v128_t va3 = wasm_v128_load(a3);
        a3 += 16;

        const v128_t vb0 = wasm_v128_load(w);
        vacc0x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va0, vacc0x0);
        vacc1x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va1, vacc1x0);
        vacc2x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va2, vacc2x0);
        vacc3x0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb0, va3, vacc3x0);
        const v128_t vb1 = wasm_v128_load((const int8_t*) w + 16);
        vacc0x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va0, vacc0x1);
        vacc1x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va1, vacc1x1);
        vacc2x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va2, vacc2x1);
        vacc3x1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb1, va3, vacc3x1);
        const v128_t vb2 = wasm_v128_load((const int8_t*) w + 32);
        vacc0x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va0, vacc0x2);
        vacc1x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va1, vacc1x2);
        vacc2x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va2, vacc2x2);
        vacc3x2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb2, va3, vacc3x2);
        const v128_t vb3 = wasm_v128_load((const int8_t*) w + 48);
        vacc0x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va0, vacc0x3);
        vacc1x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va1, vacc1x3);
        vacc2x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va2, vacc2x3);
        vacc3x3 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(vb3, va3, vacc3x3);

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const v128_t vacc0x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x0, vacc0x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x0, vacc0x2, 2, 6, 3, 7));
    const v128_t vacc0x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x1, vacc0x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x1, vacc0x3, 2, 6, 3, 7));
    const v128_t vacc1x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x0, vacc1x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x0, vacc1x2, 2, 6, 3, 7));
    const v128_t vacc1x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x1, vacc1x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x1, vacc1x3, 2, 6, 3, 7));
    const v128_t vacc2x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x0, vacc2x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x0, vacc2x2, 2, 6, 3, 7));
    const v128_t vacc2x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x1, vacc2x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x1, vacc2x3, 2, 6, 3, 7));
    const v128_t vacc3x02 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x0, vacc3x2, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x0, vacc3x2, 2, 6, 3, 7));
    const v128_t vacc3x13 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x1, vacc3x3, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x1, vacc3x3, 2, 6, 3, 7));

    v128_t vacc0x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc0x02, vacc0x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc0x02, vacc0x13, 2, 6, 3, 7));
    v128_t vacc1x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc1x02, vacc1x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc1x02, vacc1x13, 2, 6, 3, 7));
    v128_t vacc2x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc2x02, vacc2x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc2x02, vacc2x13, 2, 6, 3, 7));
    v128_t vacc3x0123 = wasm_i32x4_add(wasm_v32x4_shuffle(vacc3x02, vacc3x13, 0, 4, 1, 5), wasm_v32x4_shuffle(vacc3x02, vacc3x13, 2, 6, 3, 7));

    vacc0x0123 = wasm_f32x4_convert_i32x4(vacc0x0123);
    vacc1x0123 = wasm_f32x4_convert_i32x4(vacc1x0123);
    vacc2x0123 = wasm_f32x4_convert_i32x4(vacc2x0123);
    vacc3x0123 = wasm_f32x4_convert_i32x4(vacc3x0123);

    const v128_t vscale0123 = wasm_v128_load(w);
    w = (const float*) w + 4;
    vacc0x0123 = wasm_f32x4_mul(vacc0x0123, vscale0123);
    vacc1x0123 = wasm_f32x4_mul(vacc1x0123, vscale0123);
    vacc2x0123 = wasm_f32x4_mul(vacc2x0123, vscale0123);
    vacc3x0123 = wasm_f32x4_mul(vacc3x0123, vscale0123);

    const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
    vacc0x0123 = wasm_f32x4_add(vacc0x0123, vmagic_bias);
    vacc1x0123 = wasm_f32x4_add(vacc1x0123, vmagic_bias);
    vacc2x0123 = wasm_f32x4_add(vacc2x0123, vmagic_bias);
    vacc3x0123 = wasm_f32x4_add(vacc3x0123, vmagic_bias);

    const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
    vacc0x0123 = wasm_i32x4_max(vacc0x0123, vmagic_min);
    vacc1x0123 = wasm_i32x4_max(vacc1x0123, vmagic_min);
    vacc2x0123 = wasm_i32x4_max(vacc2x0123, vmagic_min);
    vacc3x0123 = wasm_i32x4_max(vacc3x0123, vmagic_min);

    const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
    vacc0x0123 = wasm_i32x4_sub(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc1x0123 = wasm_i32x4_sub(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc2x0123 = wasm_i32x4_sub(vacc2x0123, vmagic_bias_less_output_zero_point);
    vacc3x0123 = wasm_i32x4_sub(vacc3x0123, vmagic_bias_less_output_zero_point);

    v128_t vacc01x0123 = wasm_i16x8_narrow_i32x4(vacc0x0123, vacc1x0123);
    v128_t vacc23x0123 = wasm_i16x8_narrow_i32x4(vacc2x0123, vacc3x0123);

    v128_t vout = wasm_i8x16_narrow_i16x8(vacc01x0123, vacc23x0123);

    const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
    vout = wasm_i8x16_min(vout, voutput_max);

    if (nc >= 4) {
      wasm_v128_store32_lane(c3, vout, 3);
      wasm_v128_store32_lane(c2, vout, 2);
      wasm_v128_store32_lane(c1, vout, 1);
      wasm_v128_store32_lane(c0, vout, 0);

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        wasm_v128_store16_lane(c3, vout, 6);
        c3 += 2;
        wasm_v128_store16_lane(c2, vout, 4);
        c2 += 2;
        wasm_v128_store16_lane(c1, vout, 2);
        c1 += 2;
        wasm_v128_store16_lane(c0, vout, 0);
        c0 += 2;

        vout = wasm_u32x4_shr(vout, 16);
      }
      if (nc & 1) {
        wasm_v128_store8_lane(c3, vout, 12);
        wasm_v128_store8_lane(c2, vout, 8);
        wasm_v128_store8_lane(c1, vout, 4);
        wasm_v128_store8_lane(c0, vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vcvt_ukernel__wasmrelaxedsimd_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vmultiplier = wasm_i16x8_splat(-params->scalar.multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(input);
    v128_t vacc1 = wasm_i16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_i16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_i16x8_load8x8(input + 24);
    input += 32;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vacc3 = wasm_i16x8_shl(vacc3, 7);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vpositive_multiplier = wasm_i16x8_splat(-params->scalar.positive_multiplier);
  const v128_t vnegative_multiplier = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vpositive_multiplier);
  XNN_FORCE_REALIZATION(vnegative_multiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_low_i8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_high_i8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);
    v128_t vacc2 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_low_i8x16(vx1));
    v128_t vacc3 = wasm_i16x8_sub(vinput_zero_point, wasm_i16x8_extend_high_i8x16(vx1));
    v128_t vmultiplier2 = wasm_i16x8_shr(vacc2, 15);
    v128_t vmultiplier3 = wasm_i16x8_shr(vacc3, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier1);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier2);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier3);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier2);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const v128_t vx = wasm_i16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vmultiplier_diff = wasm_i16x8_splat(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const v128_t vmultiplier_base = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    v128_t vacc0 = wasm_i16x8_load8x8(input);
    v128_t vacc1 = wasm_i16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_i16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_i16x8_load8x8(input + 24);
    input += 32;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    v128_t vmultiplier2 = wasm_i16x8_gt(vacc2, vinput_zero_point);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    v128_t vmultiplier3 = wasm_i16x8_gt(vacc3, vinput_zero_point);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);
    vmultiplier2 = wasm_v128_and(vmultiplier2, vmultiplier_diff);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_v128_xor(vmultiplier2, vmultiplier_base);
    vmultiplier3 = wasm_v128_and(vmultiplier3, vmultiplier_diff);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_v128_xor(vmultiplier3, vmultiplier_base);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier2);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_i8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_i8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    v128_t vacc = wasm_i16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_i8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(int8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(int8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vcvt_ukernel__wasmrelaxedsimd_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vmultiplier = wasm_i16x8_splat(-params->scalar.multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(input);
    v128_t vacc1 = wasm_u16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_u16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_u16x8_load8x8(input + 24);
    input += 32;

    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vacc3 = wasm_i16x8_shl(vacc3, 7);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vacc = wasm_i16x8_shl(vacc, 7);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vpositive_multiplier = wasm_i16x8_splat(-params->scalar.positive_multiplier);
  const v128_t vnegative_multiplier = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vpositive_multiplier);
  XNN_FORCE_REALIZATION(vnegative_multiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load(input);
    v128_t vx1 = wasm_v128_load(input + 16);
    input += 32;

    v128_t vacc0 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx0));
    v128_t vacc1 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx0));
    v128_t vmultiplier0 = wasm_i16x8_shr(vacc0, 15);
    v128_t vmultiplier1 = wasm_i16x8_shr(vacc1, 15);
    v128_t vacc2 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_low_u8x16(vx1));
    v128_t vacc3 = wasm_i16x8_sub(vinput_zero_point, wasm_u16x8_extend_high_u8x16(vx1));
    v128_t vmultiplier2 = wasm_i16x8_shr(vacc2, 15);
    v128_t vmultiplier3 = wasm_i16x8_shr(vacc3, 15);

    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier0);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier1);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier2);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier3);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier2);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    const v128_t vx = wasm_u16x8_load8x8(input);
    v128_t vacc = wasm_i16x8_sub(vinput_zero_point, vx);
    v128_t vmultiplier = wasm_i16x8_shr(vacc, 15);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_i16x8_relaxed_laneselect(vpositive_multiplier, vnegative_multiplier, vmultiplier);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vinput_zero_point = wasm_v128_load16_splat(&params->scalar.input_zero_point);
  const v128_t vmultiplier_diff = wasm_i16x8_splat(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const v128_t vmultiplier_base = wasm_i16x8_splat(-params->scalar.negative_multiplier);
  const v128_t voutput_zero_point = wasm_v128_load16_splat(&params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vacc0 = wasm_u16x8_load8x8(input);
    v128_t vacc1 = wasm_u16x8_load8x8(input + 8);
    v128_t vacc2 = wasm_u16x8_load8x8(input + 16);
    v128_t vacc3 = wasm_u16x8_load8x8(input + 24);
    input += 32;

    v128_t vmultiplier0 = wasm_i16x8_gt(vacc0, vinput_zero_point);
    vacc0 = wasm_i16x8_sub(vinput_zero_point, vacc0);
    v128_t vmultiplier1 = wasm_i16x8_gt(vacc1, vinput_zero_point);
    vacc1 = wasm_i16x8_sub(vinput_zero_point, vacc1);
    v128_t vmultiplier2 = wasm_i16x8_gt(vacc2, vinput_zero_point);
    vacc2 = wasm_i16x8_sub(vinput_zero_point, vacc2);
    v128_t vmultiplier3 = wasm_i16x8_gt(vacc3, vinput_zero_point);
    vacc3 = wasm_i16x8_sub(vinput_zero_point, vacc3);

    vmultiplier0 = wasm_v128_and(vmultiplier0, vmultiplier_diff);
    vacc0 = wasm_i16x8_shl(vacc0, 7);
    vmultiplier0 = wasm_v128_xor(vmultiplier0, vmultiplier_base);
    vmultiplier1 = wasm_v128_and(vmultiplier1, vmultiplier_diff);
    vacc1 = wasm_i16x8_shl(vacc1, 7);
    vmultiplier1 = wasm_v128_xor(vmultiplier1, vmultiplier_base);
    vmultiplier2 = wasm_v128_and(vmultiplier2, vmultiplier_diff);
    vacc2 = wasm_i16x8_shl(vacc2, 7);
    vmultiplier2 = wasm_v128_xor(vmultiplier2, vmultiplier_base);
    vmultiplier3 = wasm_v128_and(vmultiplier3, vmultiplier_diff);
    vacc3 = wasm_i16x8_shl(vacc3, 7);
    vmultiplier3 = wasm_v128_xor(vmultiplier3, vmultiplier_base);

    vacc0 = wasm_i16x8_relaxed_q15mulr(vacc0, vmultiplier0);
    vacc1 = wasm_i16x8_relaxed_q15mulr(vacc1, vmultiplier1);
    vacc2 = wasm_i16x8_relaxed_q15mulr(vacc2, vmultiplier2);
    vacc3 = wasm_i16x8_relaxed_q15mulr(vacc3, vmultiplier3);

    vacc0 = wasm_i16x8_add_sat(vacc0, voutput_zero_point);
    vacc1 = wasm_i16x8_add_sat(vacc1, voutput_zero_point);
    vacc2 = wasm_i16x8_add_sat(vacc2, voutput_zero_point);
    vacc3 = wasm_i16x8_add_sat(vacc3, voutput_zero_point);

    const v128_t vy0 = wasm_u8x16_narrow_i16x8(vacc0, vacc1);
    const v128_t vy1 = wasm_u8x16_narrow_i16x8(vacc2, vacc3);

    wasm_v128_store(output, vy0);
    wasm_v128_store((output + 16), vy1);
    output += 32;
  }
  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);
    input += 8;

    const v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    wasm_v128_store64_lane(output, vy, 0);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 7 * sizeof(uint8_t));

    v128_t vacc = wasm_u16x8_load8x8(input);
    v128_t vmultiplier = wasm_i16x8_gt(vacc, vinput_zero_point);
    vacc = wasm_i16x8_sub(vinput_zero_point, vacc);
    vmultiplier = wasm_v128_and(vmultiplier, vmultiplier_diff);
    vacc = wasm_i16x8_shl(vacc, 7);
    vmultiplier = wasm_v128_xor(vmultiplier, vmultiplier_base);
    vacc = wasm_i16x8_relaxed_q15mulr(vacc, vmultiplier);
    vacc = wasm_i16x8_add_sat(vacc, voutput_zero_point);

    v128_t vy = wasm_u8x16_narrow_i16x8(vacc, vacc);
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}

void xnn_x8_lut_ukernel__wasmpshufb_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vt0 = wasm_v128_load(table);
  const v128_t vt1 = wasm_v128_load(table + 16);
  const v128_t vt2 = wasm_v128_load(table + 32);
  const v128_t vt3 = wasm_v128_load(table + 48);
  const v128_t vt4 = wasm_v128_load(table + 64);
  const v128_t vt5 = wasm_v128_load(table + 80);
  const v128_t vt6 = wasm_v128_load(table + 96);
  const v128_t vt7 = wasm_v128_load(table + 112);
  const v128_t vt8 = wasm_v128_load(table + 128);
  const v128_t vt9 = wasm_v128_load(table + 144);
  const v128_t vtA = wasm_v128_load(table + 160);
  const v128_t vtB = wasm_v128_load(table + 176);
  const v128_t vtC = wasm_v128_load(table + 192);
  const v128_t vtD = wasm_v128_load(table + 208);
  const v128_t vtE = wasm_v128_load(table + 224);
  const v128_t vtF = wasm_v128_load(table + 240);

  const v128_t vtable0 = vt0;
  const v128_t vtable1 = wasm_v128_xor(vt0, vt1);
  const v128_t vtable2 = wasm_v128_xor(vt1, vt2);
  const v128_t vtable3 = wasm_v128_xor(vt2, vt3);
  const v128_t vtable4 = wasm_v128_xor(vt3, vt4);
  const v128_t vtable5 = wasm_v128_xor(vt4, vt5);
  const v128_t vtable6 = wasm_v128_xor(vt5, vt6);
  const v128_t vtable7 = wasm_v128_xor(vt6, vt7);
  const v128_t vtable8 = wasm_v128_xor(wasm_v128_xor(vt7, vt8), vtable0);
  const v128_t vtable9 = wasm_v128_xor(wasm_v128_xor(vt8, vt9), vtable1);
  const v128_t vtableA = wasm_v128_xor(wasm_v128_xor(vt9, vtA), vtable2);
  const v128_t vtableB = wasm_v128_xor(wasm_v128_xor(vtA, vtB), vtable3);
  const v128_t vtableC = wasm_v128_xor(wasm_v128_xor(vtB, vtC), vtable4);
  const v128_t vtableD = wasm_v128_xor(wasm_v128_xor(vtC, vtD), vtable5);
  const v128_t vtableE = wasm_v128_xor(wasm_v128_xor(vtD, vtE), vtable6);
  const v128_t vtableF = wasm_v128_xor(wasm_v128_xor(vtE, vtF), vtable7);

  const v128_t voffset = wasm_i8x16_const_splat(16);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    v128_t vx0 = wasm_v128_load((const v128_t*) input);
    v128_t vx1 = wasm_v128_load((const v128_t*) (input + 16));
    input += 32;

    v128_t vy0 = wasm_i8x16_relaxed_swizzle(vtable0, vx0);
    v128_t vy1 = wasm_i8x16_relaxed_swizzle(vtable0, vx1);

    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable1, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable1, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable2, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable2, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable3, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable3, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable4, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable4, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable5, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable5, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable6, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable6, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable7, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable7, vx1));
    vx0 = wasm_i8x16_sub(vx0, voffset);
    vx1 = wasm_i8x16_sub(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable8, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable8, vx1));

    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtable9, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtable9, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableA, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableA, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableB, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableB, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableC, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableC, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableD, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableD, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableE, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableE, vx1));
    vx0 = wasm_i8x16_sub_sat(vx0, voffset);
    vx1 = wasm_i8x16_sub_sat(vx1, voffset);
    vy0 = wasm_v128_xor(vy0, wasm_i8x16_relaxed_swizzle(vtableF, vx0));
    vy1 = wasm_v128_xor(vy1, wasm_i8x16_relaxed_swizzle(vtableF, vx1));

    wasm_v128_store(output, vy0);
    wasm_v128_store(output + 16, vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    v128_t vx = wasm_v128_load(input);
    input += 16;

    v128_t vy = wasm_i8x16_relaxed_swizzle(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable8, vx));

    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable9, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableA, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableB, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableC, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableD, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableE, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableF, vx));

    wasm_v128_store(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load((const v128_t*) input);

    v128_t vy = wasm_i8x16_relaxed_swizzle(vtable0, vx);

    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable1, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable2, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable3, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable4, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable5, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable6, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable7, vx));
    vx = wasm_i8x16_sub(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable8, vx));

    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtable9, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableA, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableB, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableC, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableD, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableE, vx));
    vx = wasm_i8x16_sub_sat(vx, voffset);
    vy = wasm_v128_xor(vy, wasm_i8x16_relaxed_swizzle(vtableF, vx));

    if (batch & (8 * sizeof(uint8_t))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      wasm_v128_store32_lane(output, vy, 0);
      vy = wasm_u64x2_shr(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      wasm_v128_store16_lane(output, vy, 0);
      vy = wasm_u32x4_shr(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      wasm_v128_store8_lane(output, vy, 0);
    }
  }
}
