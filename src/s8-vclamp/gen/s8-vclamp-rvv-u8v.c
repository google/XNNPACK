// Auto-generated file. Do not edit!
//   Template: src/s8-vclamp/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vunary.h"


void xnn_s8_vclamp_ukernel__rvv_u8v(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int8_t vmin = params->scalar.min;
  const int8_t vmax = params->scalar.max;

  do {
    const size_t n = __riscv_vsetvl_e8m8(batch);
    vint8m8_t vacc = __riscv_vle8_v_i8m8(input, n);
    vacc = __riscv_vmax_vx_i8m8(vacc, vmin, n);
    vacc = __riscv_vmin_vx_i8m8(vacc, vmax, n);
    __riscv_vse8_v_i8m8(output, vacc, n);
    input += n;
    output += n;
    batch -= n;
  } while (batch != 0);
}
