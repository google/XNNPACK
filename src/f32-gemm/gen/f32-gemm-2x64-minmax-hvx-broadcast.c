// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/hvx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/gemm.h"

void xnn_f32_gemm_minmax_ukernel_2x64__hvx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    HVX_Vector vacc0x0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 0));
    HVX_Vector vacc0x1 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 32));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    w += 64;

    size_t k = kc;
    do {
      const HVX_Vector va0 = xnn_set1_f32(*a0);
      a0 += 1;
      const HVX_Vector va1 = xnn_set1_f32(*a1);
      a1 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w + 0));
      const HVX_Vector vb1 = *((const HVX_Vector *)(w + 32));
      w += 64;

      const HVX_Vector vtemp0x0 = Q6_Vqf32_vmpy_VsfVsf(va0, vb0);
      const HVX_Vector vtemp1x0 = Q6_Vqf32_vmpy_VsfVsf(va1, vb0);
      const HVX_Vector vtemp0x1 = Q6_Vqf32_vmpy_VsfVsf(va0, vb1);
      const HVX_Vector vtemp1x1 = Q6_Vqf32_vmpy_VsfVsf(va1, vb1);

      vacc0x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x0, vtemp0x0);
      vacc1x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x0, vtemp1x0);
      vacc0x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x1, vtemp0x1);
      vacc1x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x1, vtemp1x1);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0 = Q6_Vsf_equals_Vqf32(vacc0x0);
    vacc1x0 = Q6_Vsf_equals_Vqf32(vacc1x0);
    vacc0x1 = Q6_Vsf_equals_Vqf32(vacc0x1);
    vacc1x1 = Q6_Vsf_equals_Vqf32(vacc1x1);

    HVX_Vector vmin = xnn_set1_f32(params->scalar.min);
    vacc0x0 = xnn_max_f32(vmin, vacc0x0);
    vacc1x0 = xnn_max_f32(vmin, vacc1x0);
    vacc0x1 = xnn_max_f32(vmin, vacc0x1);
    vacc1x1 = xnn_max_f32(vmin, vacc1x1);

    HVX_Vector vmax = xnn_set1_f32(params->scalar.max);
    vacc0x0 = xnn_min_f32(vmax, vacc0x0);
    vacc1x0 = xnn_min_f32(vmax, vacc1x0);
    vacc0x1 = xnn_min_f32(vmax, vacc0x1);
    vacc1x1 = xnn_min_f32(vmax, vacc1x1);

    if XNN_LIKELY(nc >= 64) {
      *((HVX_UVector *)c0) = vacc0x0;
      *((HVX_UVector *)(c0 + 32)) = vacc0x1;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      *((HVX_UVector *)c1) = vacc1x0;
      *((HVX_UVector *)(c1 + 32)) = vacc1x1;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);

      nc -= 64;
    } else {
      if (nc & 32) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)c1) = vacc1x0;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;

        c0 += 32;
        c1 += 32;
        nc ^= 32;
      }
      xnn_store_tail_f32(c0, vacc0x0, nc);
      xnn_store_tail_f32(c1, vacc1x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
