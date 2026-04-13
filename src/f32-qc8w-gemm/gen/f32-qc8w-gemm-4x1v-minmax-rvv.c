// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qc8w-gemm/MRxNRv-rvv.c.in
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
#include "src/xnnpack/gemm.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_4x1v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
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
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const size_t nr = __riscv_vsetvlmax_e32m1();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m1(nc);
    }
    nc -= vl;

    // Load bias (float).
    vfloat32m1_t vacc0 = __riscv_vle32_v_f32m1((const float*) w, vl);
    w = (const float*) w + nr;
    vfloat32m1_t vacc1 = __riscv_vmv_v_v_f32m1(vacc0, vl);
    vfloat32m1_t vacc2 = __riscv_vmv_v_v_f32m1(vacc0, vl);
    vfloat32m1_t vacc3 = __riscv_vmv_v_v_f32m1(vacc0, vl);

    // Inner loop: accumulate int8 weights * float activations.
    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      // Load int8 weights and widen to int32, then convert to float.
      vint8mf4_t vb_i8 = __riscv_vle8_v_i8mf4((const int8_t*) w, vl);
      w = (const int8_t*) w + nr;
      vint16mf2_t vb_i16 = __riscv_vsext_vf2_i16mf2(vb_i8, vl);
      vint32m1_t vb_i32 = __riscv_vsext_vf2_i32m1(vb_i16, vl);
      vfloat32m1_t vb = __riscv_vfcvt_f_x_v_f32m1(vb_i32, vl);

      vacc0 = __riscv_vfmacc_vf_f32m1(vacc0, va0, vb, vl);
      vacc1 = __riscv_vfmacc_vf_f32m1(vacc1, va1, vb, vl);
      vacc2 = __riscv_vfmacc_vf_f32m1(vacc2, va2, vb, vl);
      vacc3 = __riscv_vfmacc_vf_f32m1(vacc3, va3, vb, vl);

      k -= sizeof(float);
    } while (k != 0);

    // Load per-channel scales and apply.
    vfloat32m1_t vscale = __riscv_vle32_v_f32m1((const float*) w, vl);
    w = (const float*) w + nr;
    vacc0 = __riscv_vfmul_vv_f32m1(vacc0, vscale, vl);
    vacc1 = __riscv_vfmul_vv_f32m1(vacc1, vscale, vl);
    vacc2 = __riscv_vfmul_vv_f32m1(vacc2, vscale, vl);
    vacc3 = __riscv_vfmul_vv_f32m1(vacc3, vscale, vl);

    // Clamp results.
    vacc0 = __riscv_vfmax_vf_f32m1(vacc0, vmin, vl);
    vacc1 = __riscv_vfmax_vf_f32m1(vacc1, vmin, vl);
    vacc2 = __riscv_vfmax_vf_f32m1(vacc2, vmin, vl);
    vacc3 = __riscv_vfmax_vf_f32m1(vacc3, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m1(vacc0, vmax, vl);
    vacc1 = __riscv_vfmin_vf_f32m1(vacc1, vmax, vl);
    vacc2 = __riscv_vfmin_vf_f32m1(vacc2, vmax, vl);
    vacc3 = __riscv_vfmin_vf_f32m1(vacc3, vmax, vl);

    // Store results.
    __riscv_vse32_v_f32m1(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    __riscv_vse32_v_f32m1(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    __riscv_vse32_v_f32m1(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    __riscv_vse32_v_f32m1(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
  } while (nc != 0);
}
