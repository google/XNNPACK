// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <riscv_vector.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

void xnn_qs8_rdsum_ukernel_7p7x__rvv_u1v(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;

  do {
    size_t vl = __riscv_vsetvl_e8m1(channels); channels -= vl;

    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);

    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

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

      vint8m1_t vinput;
      vint16m2_t vinput16;
      vinput = __riscv_vle8_v_i8m1(i0, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i1, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i2, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i3, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i4, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i5, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
      vinput = __riscv_vle8_v_i8m1(i6, vl);
      vinput16 = __riscv_vsext_vf2_i16m2(vinput, vl);  
      vacc = __riscv_vwadd_wv_i32m4(vacc, vinput16, vl);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
    }

    vint32m4_t voutput = __riscv_vle32_v_i32m4(output, vl);
    voutput = __riscv_vadd_vv_i32m4(voutput, vacc, vl);
    __riscv_vse32_v_i32m4(output, voutput, vl); output += vl;

    input = (const int8_t*) ((uintptr_t) input + vl);

  } while (channels != 0);
}
