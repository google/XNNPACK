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

void xnn_f32_vrcopysignc_ukernel__rvv_u8v(
      size_t batch,
      const float* sign,
      const float* mag,
      float* output,
      const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(mag != NULL);
  assert(sign != NULL);
  assert(output != NULL);

  const float magc = *mag;

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t vl = __riscv_vsetvl_e32m8(batch); batch -= vl;

    vfloat32m8_t vmag = __riscv_vfmv_v_f_f32m8(magc, vl);
    vfloat32m8_t vsign = __riscv_vle32_v_f32m8(sign, vl); sign += vl;
    vfloat32m8_t vy = __riscv_vfsgnj(vmag, vsign, vl);

    __riscv_vse32(output, vy, vl); output += vl;
 
  } while (batch != 0);
}
