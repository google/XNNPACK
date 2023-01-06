// Auto-generated file. Do not edit!
//   Template: src/f32-vsqrt/rvv-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsqrt_ukernel__rvv_sqrt_x4v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  batch >>= 2;  // log2(sizeof(float))
  do {
    const size_t n = vsetvl_e32m4(batch);
    vfloat32m4_t vx = vle32_v_f32m4(input, n);
    input += n;
    vfloat32m4_t vacc = vfsqrt_v_f32m4(vx, n);
    vse32_v_f32m4(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
