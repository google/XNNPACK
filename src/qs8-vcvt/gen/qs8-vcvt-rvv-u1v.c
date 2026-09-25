// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Microchip
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>
#include "src/xnnpack/vcvt.h"

void xnn_qs8_vcvt_ukernel__rvv_u1v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vbias =
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000);
  const int32_t multiplier = params->scalar.multiplier;

  do {
    size_t vl = __riscv_vsetvl_e8m1(batch); batch -= vl;
    vint8m1_t in_i8v = __riscv_vle8_v_i8m1(input, vl); input += vl;
    vint32m4_t acc_i32v = __riscv_vsext_vf4(in_i8v, vl);
    vint32m4_t bias_i32v = __riscv_vmv_v_x_i32m4(vbias, vl);
    acc_i32v = __riscv_vmacc(bias_i32v, multiplier, acc_i32v, vl);
    vint16m2_t out_i16v = __riscv_vnclip(acc_i32v, 16, __RISCV_VXRM_RDN, vl);
    vint8m1_t out_i8v = __riscv_vnclip(out_i16v, 0, __RISCV_VXRM_RNU, vl);
    __riscv_vse8(output, out_i8v, vl); output += vl;
  } while (batch != 0);
}
