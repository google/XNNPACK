// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/rvvfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vsqr_ukernel__rvvfp16arith_u4v(
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
    const size_t n = __riscv_vsetvl_e16m4(batch);
    const vfloat16m4_t vi = __riscv_vle16_v_f16m4((const void *) i, n);
    i += n;
    const vfloat16m4_t vo = __riscv_vfmul_vv_f16m4(vi, vi, n);
    __riscv_vse16_v_f16m4((void *) o, vo, n);
    o += n;

    batch -= n;
  } while (batch != 0);
}
