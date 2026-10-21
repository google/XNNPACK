// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__rvv_u1v(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  batch >>= XNN_LOG2_SIZEOF_FLOAT16;
  do {
    const size_t n = __riscv_vsetvl_e16m1(batch);
    const vuint16m1_t vi = __riscv_vle16_v_u16m1(i, n);
    i += n;
    const vuint16m1_t vo = __riscv_vand_vx_u16m1(vi, 0x7FFF, n);
    __riscv_vse16_v_u16m1(o, vo, n);
    o += n;

    batch -= n;
  } while (batch != 0);
}
