// Auto-generated file. Do not edit!
//   Template: src/f32-gavgpool-cw/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies, inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_f32_gavgpool_cw_ukernel__rvv_u2v(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;

  elements >>= XNN_LOG2_SIZEOF_FLOAT;
  const float multiplier = params->scalar.multiplier;
  const float output_max = params->scalar.output_max;
  const float output_min = params->scalar.output_min;

  while (channels != 0) {
    int32_t fin = __riscv_vsetvl_e32m2(elements);
    vfloat32m2_t sum_f32v = __riscv_vfmv_v_f_f32m2(0.f, __riscv_vsetvl_e32m2(elements));

    size_t ele = elements;
    for( ; ele > 0; ){
      int32_t n =  __riscv_vsetvl_e32m2(ele); ele -= n;
      vfloat32m2_t i0_f32v = __riscv_vle32_v_f32m2(i0, n); i0 += n;
      sum_f32v = __riscv_vfadd_vv_f32m2_tu(sum_f32v, i0_f32v, sum_f32v, n);
    }
    vfloat32m1_t scl_f32v = __riscv_vfmv_s_f_f32m1(0.f, 1);
    scl_f32v = __riscv_vfredusum_vs_f32m2_f32m1(sum_f32v, scl_f32v, fin);

    float result = __riscv_vfmv_f_s_f32m1_f32(scl_f32v);
    result *= multiplier;
    result = math_min_f32(result, output_max);
    result = math_max_f32(result, output_min);

    *output = result; output += 1;
    channels -= 1;
  }
}
