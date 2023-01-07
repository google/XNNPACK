// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vclamp_ukernel__rvv_x2v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  batch >>= 2;  // log2(sizeof(float))
  do {
    const size_t n = vsetvl_e32m2(batch);
    vfloat32m2_t vacc = vle32_v_f32m2(input, n);
    input += n;
    vacc = vfmax_vf_f32m2(vacc, vmin, n);
    vacc = vfmin_vf_f32m2(vacc, vmax, n);
    vse32_v_f32m2(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
