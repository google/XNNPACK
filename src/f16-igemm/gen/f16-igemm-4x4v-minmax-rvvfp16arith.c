// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "src/xnnpack/igemm.h"

void xnn_f16_igemm_minmax_ukernel_4x4v__rvvfp16arith(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_float16** restrict a,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const xnn_float16 vmin = params->scalar.min;
  const xnn_float16 vmax = params->scalar.max;

  xnn_float16* c0 = c;
  xnn_float16* c1 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  xnn_float16* c2 = (xnn_float16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  xnn_float16* c3 = (xnn_float16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const size_t nr = __riscv_vsetvlmax_e16m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e16m4(nc);
    }
    nc = nc - vl;
    vfloat16m4_t vacc0 =  __riscv_vle16_v_f16m4(w, vl);
    w = w + nr;
    vfloat16m4_t vacc1 =  __riscv_vmv_v_v_f16m4(vacc0, vl);
    vfloat16m4_t vacc2 =  __riscv_vmv_v_v_f16m4(vacc0, vl);
    vfloat16m4_t vacc3 =  __riscv_vmv_v_v_f16m4(vacc0, vl);

    size_t p = ks;
    do {
      const xnn_float16* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const xnn_float16*) zero) {
        a0 = (const xnn_float16*) ((uintptr_t) a0 + a_offset);
      }
      const xnn_float16* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != (const xnn_float16*) zero) {
        a1 = (const xnn_float16*) ((uintptr_t) a1 + a_offset);
      }
      const xnn_float16* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != (const xnn_float16*) zero) {
        a2 = (const xnn_float16*) ((uintptr_t) a2 + a_offset);
      }
      const xnn_float16* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != (const xnn_float16*) zero) {
        a3 = (const xnn_float16*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const xnn_float16 va0 = *a0++;
        const xnn_float16 va1 = *a1++;
        const xnn_float16 va2 = *a2++;
        const xnn_float16 va3 = *a3++;
        vfloat16m4_t vb = __riscv_vle16_v_f16m4(w, vl);
        w = w + nr;
        vacc0 = __riscv_vfmacc_vf_f16m4(vacc0, va0, vb, vl);
        vacc1 = __riscv_vfmacc_vf_f16m4(vacc1, va1, vb, vl);
        vacc2 = __riscv_vfmacc_vf_f16m4(vacc2, va2, vb, vl);
        vacc3 = __riscv_vfmacc_vf_f16m4(vacc3, va3, vb, vl);

        k -= sizeof(xnn_float16);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    // clamp results with min & max
    vacc0 = __riscv_vfmax_vf_f16m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f16m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f16m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f16m4(vacc3, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f16m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f16m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f16m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f16m4(vacc3, vmax, vl);
    // store 4 x vl results to c
    __riscv_vse16_v_f16m4(c3, vacc3, vl);
    c3 = (xnn_float16*) ((uintptr_t) c3 + cn_stride);
    __riscv_vse16_v_f16m4(c2, vacc2, vl);
    c2 = (xnn_float16*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse16_v_f16m4(c1, vacc1, vl);
    c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse16_v_f16m4(c0, vacc0, vl);
    c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);

    a = (const xnn_float16** restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}
