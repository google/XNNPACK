// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/gemm.h"

void xnn_f16_gemm_minmax_ukernel_1x4v__rvvfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const xnn_float16 vmin = params->scalar.min;
  const xnn_float16 vmax = params->scalar.max;

  const xnn_float16* a0 = a;
  xnn_float16* c0 = c;

  const size_t nr = __riscv_vsetvlmax_e16m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e16m4(nc);
    }
    nc = nc - vl;

    vfloat16m4_t vacc0 =  __riscv_vle16_v_f16m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const xnn_float16 va0 = *a0++;
      vfloat16m4_t vb = __riscv_vle16_v_f16m4(w, vl);
      w = w + nr;
      vacc0 = __riscv_vfmacc_vf_f16m4(vacc0, va0, vb, vl);
      k -= sizeof(xnn_float16);
    } while (k != 0);

    // clamp results with min & max
    vacc0 = __riscv_vfmax_vf_f16m4(vacc0, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f16m4(vacc0, vmax, vl);

    // store 1 x vl results to c
    __riscv_vse16_v_f16m4(c0, vacc0, vl);
    c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);
    a0 = (const xnn_float16*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}
