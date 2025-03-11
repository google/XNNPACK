// clang-format off
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
//#include "src/xnnpack/math.h"

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

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {

      vint16m2_t vacc16 = __riscv_vmv_v_x_i16m2(0, vl);

      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }

        vint8m1_t vinput0 = __riscv_vle8_v_i8m1(i0, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput0, vl);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        vint8m1_t vinput1 = __riscv_vle8_v_i8m1(i1, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput1, vl);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        vint8m1_t vinput2 = __riscv_vle8_v_i8m1(i2, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput2, vl);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        vint8m1_t vinput3 = __riscv_vle8_v_i8m1(i3, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput3, vl);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        vint8m1_t vinput4 = __riscv_vle8_v_i8m1(i4, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput4, vl);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        vint8m1_t vinput5 = __riscv_vle8_v_i8m1(i5, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput5, vl);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        vint8m1_t vinput6 = __riscv_vle8_v_i8m1(i6, vl);
        vacc16 = __riscv_vwadd_wv_i16m2(vacc16, vinput6, vl);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }

      vacc = __riscv_vwadd_wv_i32m4(vacc, vacc16, vl);
      r = doz(r, 252);
    }

    vint32m4_t voutput = __riscv_vle32_v_i32m4(output, vl);
    voutput = __riscv_vadd_vv_i32m4(voutput, vacc, vl);
    __riscv_vse32_v_i32m4(output, voutput, vl); output += vl;

    input = (const int8_t*) ((uintptr_t) input + vl);

  } while (channels != 0);
}
