// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/MRxNRv-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include "xnnpack/igemm.h"


void xnn_f32_igemm_relu_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  float* c0 = c;

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);
    // apply ReLU to results
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    // store 1 x vl results to c
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}
