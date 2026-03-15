// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vcopysign/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/vunary.h"

void xnn_f32_vcopysignc_ukernel__rvv_u4v(
      size_t batch,
      const float* mag,
      const float* sign,
      float* output,
      const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);

  const float signc = *sign;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t vl = __riscv_vsetvl_e32m4(batch); batch -= vl;

    vfloat32m4_t vmag = __riscv_vle32_v_f32m4(mag, vl); mag += vl;
    vfloat32m4_t vy = __riscv_vfsgnj(vmag, signc, vl);

    __riscv_vse32(output, vy, vl); output += vl;
 
  } while (batch != 0);
}
