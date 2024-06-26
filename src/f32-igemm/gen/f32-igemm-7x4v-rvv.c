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


void xnn_f32_igemm_ukernel_7x4v__rvv(
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
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  __riscv_vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  __riscv_vmv_v_v_f32m4(vacc0, vl);

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;
        const float va4 = *a4++;
        const float va5 = *a5++;
        const float va6 = *a6++;
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
        vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
        vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
        vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);
        vacc4 = __riscv_vfmacc_vf_f32m4(vacc4, va4, vb, vl);
        vacc5 = __riscv_vfmacc_vf_f32m4(vacc5, va5, vb, vl);
        vacc6 = __riscv_vfmacc_vf_f32m4(vacc6, va6, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);
    // store 7 x vl results to c
    __riscv_vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    __riscv_vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    __riscv_vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}
