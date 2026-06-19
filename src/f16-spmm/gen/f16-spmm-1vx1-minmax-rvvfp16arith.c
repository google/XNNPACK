// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-spmm/rvv.c.in
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


void xnn_f16_spmm_minmax_ukernel_1vx1__rvvfp16arith(
    size_t mc,
    size_t nc,
    const xnn_float16* input,
    const xnn_float16* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    xnn_float16* output,
    size_t output_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mc != 0);
  assert(mc % sizeof(xnn_float16) == 0);
  assert(nc != 0);

  mc >>= XNN_LOG2_SIZEOF_FLOAT16;
  const xnn_float16 min = params->scalar.min;
  const xnn_float16 max = params->scalar.max;
  size_t vlmax = __riscv_vsetvlmax_e16m1();
  size_t output_decrement = output_stride * nc - vlmax * sizeof(xnn_float16);
  while (mc > 0) {
    size_t vl = __riscv_vsetvl_e16m1(mc);
    const xnn_float16* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      vfloat16m1_t acc_f16v = __riscv_vfmv_v_f_f16m1(0.0f, vl);
      acc_f16v = __riscv_vfadd_vf_f16m1(acc_f16v, *w++, vl);
      uint32_t nnz = *nnzmap++;

      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          vfloat16m1_t input_f16v = __riscv_vle16_v_f16m1(input, vl);
          acc_f16v = __riscv_vfmacc_vf_f16m1(acc_f16v, *w++, input_f16v, vl);
          input = (const xnn_float16*) ((uintptr_t) input + (uintptr_t) diff);
        } while (--nnz != 0);
      }

      acc_f16v = __riscv_vfmin_vf_f16m1(acc_f16v, max, vl);
      acc_f16v = __riscv_vfmax_vf_f16m1(acc_f16v, min, vl);
      __riscv_vse16_v_f16m1(output, acc_f16v, vl);
      output = (xnn_float16*) ((uintptr_t) output + output_stride);
    } while(--n != 0);
    output = (xnn_float16*) ((uintptr_t) output - output_decrement);
    input += vl;
    mc -= vl;
  }
}
