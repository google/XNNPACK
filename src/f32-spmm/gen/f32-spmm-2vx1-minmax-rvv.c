// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Imagination Technologies inc.
// Copyright 2025 Andes Technology
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/prefetch.h"
#include "src/xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_2vx1__rvv(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  mc >>= XNN_LOG2_SIZEOF_FLOAT;
  const float min = params->scalar.min;
  const float max = params->scalar.max;
  size_t vlmax = __riscv_vsetvlmax_e32m2();
  size_t output_decrement = output_stride * nc - vlmax * sizeof(float);
  while (mc > 0) {
    size_t vl = __riscv_vsetvl_e32m2(mc);
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      vfloat32m2_t acc_f32v = __riscv_vfmv_v_f_f32m2(0.0f, vl);
      acc_f32v = __riscv_vfadd_vf_f32m2(acc_f32v, *w++, vl);
      uint32_t nnz = *nnzmap++;

      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          vfloat32m2_t input_f32v = __riscv_vle32_v_f32m2(input, vl);
          acc_f32v = __riscv_vfmacc_vf_f32m2(acc_f32v, *w++, input_f32v, vl);
          input = (const float*) ((uintptr_t) input + (uintptr_t) diff);
        } while (--nnz != 0);
      }

      acc_f32v = __riscv_vfmin_vf_f32m2(acc_f32v, max, vl);
      acc_f32v = __riscv_vfmax_vf_f32m2(acc_f32v, min, vl);
      __riscv_vse32_v_f32m2(output, acc_f32v, vl);
      output = (float*) ((uintptr_t) output + output_stride);
    } while(--n != 0);
    output = (float*) ((uintptr_t) output - output_decrement);
    input += vl;
    mc -= vl;
  }
}
