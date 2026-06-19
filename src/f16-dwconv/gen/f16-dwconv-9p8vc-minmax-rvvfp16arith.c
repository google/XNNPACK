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

void xnn_f16_dwconv_minmax_ukernel_9p8vc__rvvfp16arith(
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
