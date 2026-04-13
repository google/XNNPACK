// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/ppmm.h"


void xnn_f32_ppmm_minmax_ukernel_4x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // Load bias.
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4(w, vl);
    w += nr;
    vfloat32m4_t vacc1 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 = __riscv_vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 = __riscv_vmv_v_v_f32m4(vacc0, vl);

    // Inner product loop.
    size_t k = kc;
    do {
      const float va0 = a[0];
      const float va1 = a[1];
      const float va2 = a[2];
      const float va3 = a[3];
      a += 4;

      vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
      w += nr;

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m4(vacc3, va3, vb, vl);

      k -= sizeof(float);
    } while (k != 0);

    // Clamp with min & max.
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m4(vacc3, vmin, vl);

    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m4(vacc3, vmax, vl);

    // Store results.
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);

    a = (const float*) ((uintptr_t) a - kc * 4);
  } while (nc != 0);
}
