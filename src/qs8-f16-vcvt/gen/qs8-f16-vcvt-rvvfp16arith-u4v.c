// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-f16-vcvt/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_f16_vcvt_ukernel__rvvfp16arith_u4v(
    size_t batch,
    const int8_t* input,
    xnn_float16* output,
    const struct xnn_qs8_f16_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= XNN_LOG2_SIZEOF_INT8_T;

  const int16_t vminus_zero_point = -params->scalar.zero_point;
  const xnn_float16 vscale = params->scalar.scale;

  do {
    const size_t n = __riscv_vsetvl_e8m4(batch);

    vint8m4_t vx = __riscv_vle8_v_i8m4(input, n);
    input += n;

    // int8 minus the zero point is in [-255, 255], which fp16 represents
    // exactly, so the widened value converts without rounding.
    vint16m8_t vacc = __riscv_vsext_vf2_i16m8(vx, n);
    vacc = __riscv_vadd_vx_i16m8(vacc, vminus_zero_point, n);

    vfloat16m8_t vy = __riscv_vfcvt_f_x_v_f16m8(vacc, n);
    vy = __riscv_vfmul_vf_f16m8(vy, vscale, n);

    __riscv_vse16_v_f16m8(output, vy, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
