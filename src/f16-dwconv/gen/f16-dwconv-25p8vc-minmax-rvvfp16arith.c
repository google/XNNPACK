// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/dwconv.h"

void xnn_f16_dwconv_minmax_ukernel_25p8vc__rvvfp16arith(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const xnn_float16 vmin = params->scalar.min;
  const xnn_float16 vmax = params->scalar.max;
  xnn_float16* o = output;

  do {
    const xnn_float16* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const xnn_float16*) zero) {
      i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
    }
    const xnn_float16* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const xnn_float16*) zero) {
      i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
    }
    const xnn_float16* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const xnn_float16*) zero) {
      i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
    }
    const xnn_float16* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const xnn_float16*) zero) {
      i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
    }
    const xnn_float16* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const xnn_float16*) zero) {
      i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
    }
    const xnn_float16* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const xnn_float16*) zero) {
      i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
    }
    const xnn_float16* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const xnn_float16*) zero) {
      i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
    }
    const xnn_float16* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const xnn_float16*) zero) {
      i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
    }
    const xnn_float16* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const xnn_float16*) zero) {
      i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
    }
    const xnn_float16* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const xnn_float16*) zero) {
      i9 = (const xnn_float16*) ((uintptr_t) i9 + input_offset);
    }
    const xnn_float16* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const xnn_float16*) zero) {
      i10 = (const xnn_float16*) ((uintptr_t) i10 + input_offset);
    }
    const xnn_float16* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const xnn_float16*) zero) {
      i11 = (const xnn_float16*) ((uintptr_t) i11 + input_offset);
    }
    const xnn_float16* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const xnn_float16*) zero) {
      i12 = (const xnn_float16*) ((uintptr_t) i12 + input_offset);
    }
    const xnn_float16* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const xnn_float16*) zero) {
      i13 = (const xnn_float16*) ((uintptr_t) i13 + input_offset);
    }
    const xnn_float16* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const xnn_float16*) zero) {
      i14 = (const xnn_float16*) ((uintptr_t) i14 + input_offset);
    }
    const xnn_float16* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const xnn_float16*) zero) {
      i15 = (const xnn_float16*) ((uintptr_t) i15 + input_offset);
    }
    const xnn_float16* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const xnn_float16*) zero) {
      i16 = (const xnn_float16*) ((uintptr_t) i16 + input_offset);
    }
    const xnn_float16* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const xnn_float16*) zero) {
      i17 = (const xnn_float16*) ((uintptr_t) i17 + input_offset);
    }
    const xnn_float16* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const xnn_float16*) zero) {
      i18 = (const xnn_float16*) ((uintptr_t) i18 + input_offset);
    }
    const xnn_float16* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const xnn_float16*) zero) {
      i19 = (const xnn_float16*) ((uintptr_t) i19 + input_offset);
    }
    const xnn_float16* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const xnn_float16*) zero) {
      i20 = (const xnn_float16*) ((uintptr_t) i20 + input_offset);
    }
    const xnn_float16* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const xnn_float16*) zero) {
      i21 = (const xnn_float16*) ((uintptr_t) i21 + input_offset);
    }
    const xnn_float16* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const xnn_float16*) zero) {
      i22 = (const xnn_float16*) ((uintptr_t) i22 + input_offset);
    }
    const xnn_float16* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const xnn_float16*) zero) {
      i23 = (const xnn_float16*) ((uintptr_t) i23 + input_offset);
    }
    const xnn_float16* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const xnn_float16*) zero) {
      i24 = (const xnn_float16*) ((uintptr_t) i24 + input_offset);
    }

    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const xnn_float16* w = weights;
    const size_t vlmax = __riscv_vsetvlmax_e16m8();

    do {
      size_t vl = __riscv_vsetvl_e16m8(c);
      vfloat16m8_t vacc = __riscv_vle16_v_f16m8(w, vl);
      w += vlmax;

      vfloat16m8_t va = __riscv_vundefined_f16m8();
      vfloat16m8_t vb = __riscv_vundefined_f16m8();
      va = __riscv_vle16_v_f16m8(i0, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i0 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i1, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i1 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i2, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i2 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i3, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i3 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i4, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i4 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i5, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i5 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i6, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i6 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i7, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i7 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i8, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i8 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i9, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i9 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i10, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i10 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i11, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i11 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i12, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i12 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i13, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i13 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i14, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i14 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i15, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i15 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i16, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i16 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i17, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i17 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i18, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i18 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i19, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i19 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i20, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i20 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i21, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i21 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i22, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i22 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i23, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i23 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);
      va = __riscv_vle16_v_f16m8(i24, vl);
      vb = __riscv_vle16_v_f16m8(w, vl);
      w  += vlmax;
      i24 += vlmax;
      vacc = __riscv_vfmacc_vv_f16m8(vacc, va, vb, vl);

      vacc = __riscv_vfmax_vf_f16m8(vacc, vmin, vl);
      vacc = __riscv_vfmin_vf_f16m8(vacc, vmax, vl);
      __riscv_vse16_v_f16m8(o, vacc, vl);
      o += vl;
      c -= vl;
    } while(c != 0);
    input_offset += input_pixel_stride;
    o = (xnn_float16*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}
