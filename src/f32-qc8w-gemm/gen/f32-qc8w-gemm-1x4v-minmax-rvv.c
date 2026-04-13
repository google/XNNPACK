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


void xnn_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv(
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
  assert(mr <= 1);
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

  const size_t nr = __riscv_vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = __riscv_vsetvl_e32m4(nc);
    }
    nc -= vl;

    // Load bias (float).
    vfloat32m4_t vacc0 = __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;

    // Inner loop: accumulate int8 weights * float activations.
    size_t k = kc;
    do {
      const float va0 = *a0++;

      // Load int8 weights and widen to int32, then convert to float.
      vint8m1_t vb_i8 = __riscv_vle8_v_i8m1((const int8_t*) w, vl);
      w = (const int8_t*) w + nr;
      vint16m2_t vb_i16 = __riscv_vsext_vf2_i16m2(vb_i8, vl);
      vint32m4_t vb_i32 = __riscv_vsext_vf2_i32m4(vb_i16, vl);
      vfloat32m4_t vb = __riscv_vfcvt_f_x_v_f32m4(vb_i32, vl);

      vacc0 = __riscv_vfmacc_vf_f32m4(vacc0, va0, vb, vl);

      k -= sizeof(float);
    } while (k != 0);

    // Load per-channel scales and apply.
    vfloat32m4_t vscale = __riscv_vle32_v_f32m4((const float*) w, vl);
    w = (const float*) w + nr;
    vacc0 = __riscv_vfmul_vv_f32m4(vacc0, vscale, vl);

    // Clamp results.
    vacc0 = __riscv_vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc0 = __riscv_vfmin_vf_f32m4(vacc0, vmax, vl);

    // Store results.
    __riscv_vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}
