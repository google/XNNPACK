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

void xnn_f32_gemm_minmax_ukernel_6x128__hvx_broadcast(
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
  assert(mr <= 6);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  do {
    HVX_Vector vacc0x0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 0));
    HVX_Vector vacc0x1 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 32));
    HVX_Vector vacc0x2 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 64));
    HVX_Vector vacc0x3 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 96));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc1x2 = vacc0x2;
    HVX_Vector vacc1x3 = vacc0x3;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc2x2 = vacc0x2;
    HVX_Vector vacc2x3 = vacc0x3;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc3x1 = vacc0x1;
    HVX_Vector vacc3x2 = vacc0x2;
    HVX_Vector vacc3x3 = vacc0x3;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc4x1 = vacc0x1;
    HVX_Vector vacc4x2 = vacc0x2;
    HVX_Vector vacc4x3 = vacc0x3;
    HVX_Vector vacc5x0 = vacc0x0;
    HVX_Vector vacc5x1 = vacc0x1;
    HVX_Vector vacc5x2 = vacc0x2;
    HVX_Vector vacc5x3 = vacc0x3;
    w += 128;

    size_t k = kc;
    do {
      const HVX_Vector va0 = xnn_set1_f32(*a0);
      a0 += 1;
      const HVX_Vector va1 = xnn_set1_f32(*a1);
      a1 += 1;
      const HVX_Vector va2 = xnn_set1_f32(*a2);
      a2 += 1;
      const HVX_Vector va3 = xnn_set1_f32(*a3);
      a3 += 1;
      const HVX_Vector va4 = xnn_set1_f32(*a4);
      a4 += 1;
      const HVX_Vector va5 = xnn_set1_f32(*a5);
      a5 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w + 0));
      const HVX_Vector vb1 = *((const HVX_Vector *)(w + 32));
      const HVX_Vector vb2 = *((const HVX_Vector *)(w + 64));
      const HVX_Vector vb3 = *((const HVX_Vector *)(w + 96));
      w += 128;

      const HVX_Vector vtemp0x0 = Q6_Vqf32_vmpy_VsfVsf(va0, vb0);
      const HVX_Vector vtemp1x0 = Q6_Vqf32_vmpy_VsfVsf(va1, vb0);
      const HVX_Vector vtemp2x0 = Q6_Vqf32_vmpy_VsfVsf(va2, vb0);
      const HVX_Vector vtemp3x0 = Q6_Vqf32_vmpy_VsfVsf(va3, vb0);
      const HVX_Vector vtemp4x0 = Q6_Vqf32_vmpy_VsfVsf(va4, vb0);
      const HVX_Vector vtemp5x0 = Q6_Vqf32_vmpy_VsfVsf(va5, vb0);
      const HVX_Vector vtemp0x1 = Q6_Vqf32_vmpy_VsfVsf(va0, vb1);
      const HVX_Vector vtemp1x1 = Q6_Vqf32_vmpy_VsfVsf(va1, vb1);
      const HVX_Vector vtemp2x1 = Q6_Vqf32_vmpy_VsfVsf(va2, vb1);
      const HVX_Vector vtemp3x1 = Q6_Vqf32_vmpy_VsfVsf(va3, vb1);
      const HVX_Vector vtemp4x1 = Q6_Vqf32_vmpy_VsfVsf(va4, vb1);
      const HVX_Vector vtemp5x1 = Q6_Vqf32_vmpy_VsfVsf(va5, vb1);
      const HVX_Vector vtemp0x2 = Q6_Vqf32_vmpy_VsfVsf(va0, vb2);
      const HVX_Vector vtemp1x2 = Q6_Vqf32_vmpy_VsfVsf(va1, vb2);
      const HVX_Vector vtemp2x2 = Q6_Vqf32_vmpy_VsfVsf(va2, vb2);
      const HVX_Vector vtemp3x2 = Q6_Vqf32_vmpy_VsfVsf(va3, vb2);
      const HVX_Vector vtemp4x2 = Q6_Vqf32_vmpy_VsfVsf(va4, vb2);
      const HVX_Vector vtemp5x2 = Q6_Vqf32_vmpy_VsfVsf(va5, vb2);
      const HVX_Vector vtemp0x3 = Q6_Vqf32_vmpy_VsfVsf(va0, vb3);
      const HVX_Vector vtemp1x3 = Q6_Vqf32_vmpy_VsfVsf(va1, vb3);
      const HVX_Vector vtemp2x3 = Q6_Vqf32_vmpy_VsfVsf(va2, vb3);
      const HVX_Vector vtemp3x3 = Q6_Vqf32_vmpy_VsfVsf(va3, vb3);
      const HVX_Vector vtemp4x3 = Q6_Vqf32_vmpy_VsfVsf(va4, vb3);
      const HVX_Vector vtemp5x3 = Q6_Vqf32_vmpy_VsfVsf(va5, vb3);

      vacc0x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x0, vtemp0x0);
      vacc1x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x0, vtemp1x0);
      vacc2x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x0, vtemp2x0);
      vacc3x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x0, vtemp3x0);
      vacc4x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x0, vtemp4x0);
      vacc5x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x0, vtemp5x0);
      vacc0x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x1, vtemp0x1);
      vacc1x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x1, vtemp1x1);
      vacc2x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x1, vtemp2x1);
      vacc3x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x1, vtemp3x1);
      vacc4x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x1, vtemp4x1);
      vacc5x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x1, vtemp5x1);
      vacc0x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x2, vtemp0x2);
      vacc1x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x2, vtemp1x2);
      vacc2x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x2, vtemp2x2);
      vacc3x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x2, vtemp3x2);
      vacc4x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x2, vtemp4x2);
      vacc5x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x2, vtemp5x2);
      vacc0x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x3, vtemp0x3);
      vacc1x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x3, vtemp1x3);
      vacc2x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x3, vtemp2x3);
      vacc3x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x3, vtemp3x3);
      vacc4x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x3, vtemp4x3);
      vacc5x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x3, vtemp5x3);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0 = Q6_Vsf_equals_Vqf32(vacc0x0);
    vacc1x0 = Q6_Vsf_equals_Vqf32(vacc1x0);
    vacc2x0 = Q6_Vsf_equals_Vqf32(vacc2x0);
    vacc3x0 = Q6_Vsf_equals_Vqf32(vacc3x0);
    vacc4x0 = Q6_Vsf_equals_Vqf32(vacc4x0);
    vacc5x0 = Q6_Vsf_equals_Vqf32(vacc5x0);
    vacc0x1 = Q6_Vsf_equals_Vqf32(vacc0x1);
    vacc1x1 = Q6_Vsf_equals_Vqf32(vacc1x1);
    vacc2x1 = Q6_Vsf_equals_Vqf32(vacc2x1);
    vacc3x1 = Q6_Vsf_equals_Vqf32(vacc3x1);
    vacc4x1 = Q6_Vsf_equals_Vqf32(vacc4x1);
    vacc5x1 = Q6_Vsf_equals_Vqf32(vacc5x1);
    vacc0x2 = Q6_Vsf_equals_Vqf32(vacc0x2);
    vacc1x2 = Q6_Vsf_equals_Vqf32(vacc1x2);
    vacc2x2 = Q6_Vsf_equals_Vqf32(vacc2x2);
    vacc3x2 = Q6_Vsf_equals_Vqf32(vacc3x2);
    vacc4x2 = Q6_Vsf_equals_Vqf32(vacc4x2);
    vacc5x2 = Q6_Vsf_equals_Vqf32(vacc5x2);
    vacc0x3 = Q6_Vsf_equals_Vqf32(vacc0x3);
    vacc1x3 = Q6_Vsf_equals_Vqf32(vacc1x3);
    vacc2x3 = Q6_Vsf_equals_Vqf32(vacc2x3);
    vacc3x3 = Q6_Vsf_equals_Vqf32(vacc3x3);
    vacc4x3 = Q6_Vsf_equals_Vqf32(vacc4x3);
    vacc5x3 = Q6_Vsf_equals_Vqf32(vacc5x3);

    HVX_Vector vmin = xnn_set1_f32(params->scalar.min);
    vacc0x0 = xnn_max_f32(vmin, vacc0x0);
    vacc1x0 = xnn_max_f32(vmin, vacc1x0);
    vacc2x0 = xnn_max_f32(vmin, vacc2x0);
    vacc3x0 = xnn_max_f32(vmin, vacc3x0);
    vacc4x0 = xnn_max_f32(vmin, vacc4x0);
    vacc5x0 = xnn_max_f32(vmin, vacc5x0);
    vacc0x1 = xnn_max_f32(vmin, vacc0x1);
    vacc1x1 = xnn_max_f32(vmin, vacc1x1);
    vacc2x1 = xnn_max_f32(vmin, vacc2x1);
    vacc3x1 = xnn_max_f32(vmin, vacc3x1);
    vacc4x1 = xnn_max_f32(vmin, vacc4x1);
    vacc5x1 = xnn_max_f32(vmin, vacc5x1);
    vacc0x2 = xnn_max_f32(vmin, vacc0x2);
    vacc1x2 = xnn_max_f32(vmin, vacc1x2);
    vacc2x2 = xnn_max_f32(vmin, vacc2x2);
    vacc3x2 = xnn_max_f32(vmin, vacc3x2);
    vacc4x2 = xnn_max_f32(vmin, vacc4x2);
    vacc5x2 = xnn_max_f32(vmin, vacc5x2);
    vacc0x3 = xnn_max_f32(vmin, vacc0x3);
    vacc1x3 = xnn_max_f32(vmin, vacc1x3);
    vacc2x3 = xnn_max_f32(vmin, vacc2x3);
    vacc3x3 = xnn_max_f32(vmin, vacc3x3);
    vacc4x3 = xnn_max_f32(vmin, vacc4x3);
    vacc5x3 = xnn_max_f32(vmin, vacc5x3);

    HVX_Vector vmax = xnn_set1_f32(params->scalar.max);
    vacc0x0 = xnn_min_f32(vmax, vacc0x0);
    vacc1x0 = xnn_min_f32(vmax, vacc1x0);
    vacc2x0 = xnn_min_f32(vmax, vacc2x0);
    vacc3x0 = xnn_min_f32(vmax, vacc3x0);
    vacc4x0 = xnn_min_f32(vmax, vacc4x0);
    vacc5x0 = xnn_min_f32(vmax, vacc5x0);
    vacc0x1 = xnn_min_f32(vmax, vacc0x1);
    vacc1x1 = xnn_min_f32(vmax, vacc1x1);
    vacc2x1 = xnn_min_f32(vmax, vacc2x1);
    vacc3x1 = xnn_min_f32(vmax, vacc3x1);
    vacc4x1 = xnn_min_f32(vmax, vacc4x1);
    vacc5x1 = xnn_min_f32(vmax, vacc5x1);
    vacc0x2 = xnn_min_f32(vmax, vacc0x2);
    vacc1x2 = xnn_min_f32(vmax, vacc1x2);
    vacc2x2 = xnn_min_f32(vmax, vacc2x2);
    vacc3x2 = xnn_min_f32(vmax, vacc3x2);
    vacc4x2 = xnn_min_f32(vmax, vacc4x2);
    vacc5x2 = xnn_min_f32(vmax, vacc5x2);
    vacc0x3 = xnn_min_f32(vmax, vacc0x3);
    vacc1x3 = xnn_min_f32(vmax, vacc1x3);
    vacc2x3 = xnn_min_f32(vmax, vacc2x3);
    vacc3x3 = xnn_min_f32(vmax, vacc3x3);
    vacc4x3 = xnn_min_f32(vmax, vacc4x3);
    vacc5x3 = xnn_min_f32(vmax, vacc5x3);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector *)c0) = vacc0x0;
      *((HVX_UVector *)(c0 + 32)) = vacc0x1;
      *((HVX_UVector *)(c0 + 64)) = vacc0x2;
      *((HVX_UVector *)(c0 + 96)) = vacc0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      *((HVX_UVector *)c1) = vacc1x0;
      *((HVX_UVector *)(c1 + 32)) = vacc1x1;
      *((HVX_UVector *)(c1 + 64)) = vacc1x2;
      *((HVX_UVector *)(c1 + 96)) = vacc1x3;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)c2) = vacc2x0;
      *((HVX_UVector *)(c2 + 32)) = vacc2x1;
      *((HVX_UVector *)(c2 + 64)) = vacc2x2;
      *((HVX_UVector *)(c2 + 96)) = vacc2x3;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)c3) = vacc3x0;
      *((HVX_UVector *)(c3 + 32)) = vacc3x1;
      *((HVX_UVector *)(c3 + 64)) = vacc3x2;
      *((HVX_UVector *)(c3 + 96)) = vacc3x3;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      *((HVX_UVector *)c4) = vacc4x0;
      *((HVX_UVector *)(c4 + 32)) = vacc4x1;
      *((HVX_UVector *)(c4 + 64)) = vacc4x2;
      *((HVX_UVector *)(c4 + 96)) = vacc4x3;
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      *((HVX_UVector *)c5) = vacc5x0;
      *((HVX_UVector *)(c5 + 32)) = vacc5x1;
      *((HVX_UVector *)(c5 + 64)) = vacc5x2;
      *((HVX_UVector *)(c5 + 96)) = vacc5x3;
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);

      nc -= 128;
    } else {
      if (nc & 64) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)(c0 + 32)) = vacc0x1;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)(c1 + 32)) = vacc1x1;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)(c2 + 32)) = vacc2x1;
        *((HVX_UVector *)c3) = vacc3x0;
        *((HVX_UVector *)(c3 + 32)) = vacc3x1;
        *((HVX_UVector *)c4) = vacc4x0;
        *((HVX_UVector *)(c4 + 32)) = vacc4x1;
        *((HVX_UVector *)c5) = vacc5x0;
        *((HVX_UVector *)(c5 + 32)) = vacc5x1;

        vacc0x0 = vacc0x2;
        vacc0x1 = vacc0x3;
        vacc1x0 = vacc1x2;
        vacc1x1 = vacc1x3;
        vacc2x0 = vacc2x2;
        vacc2x1 = vacc2x3;
        vacc3x0 = vacc3x2;
        vacc3x1 = vacc3x3;
        vacc4x0 = vacc4x2;
        vacc4x1 = vacc4x3;
        vacc5x0 = vacc5x2;
        vacc5x1 = vacc5x3;

        c0 += 64;
        c1 += 64;
        c2 += 64;
        c3 += 64;
        c4 += 64;
        c5 += 64;
        nc ^= 64;
      }
      if (nc & 32) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)c3) = vacc3x0;
        *((HVX_UVector *)c4) = vacc4x0;
        *((HVX_UVector *)c5) = vacc5x0;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;
        vacc3x0 = vacc3x1;
        vacc4x0 = vacc4x1;
        vacc5x0 = vacc5x1;

        c0 += 32;
        c1 += 32;
        c2 += 32;
        c3 += 32;
        c4 += 32;
        c5 += 32;
        nc ^= 32;
      }
      xnn_store_tail_f32(c0, vacc0x0, nc);
      xnn_store_tail_f32(c1, vacc1x0, nc);
      xnn_store_tail_f32(c2, vacc2x0, nc);
      xnn_store_tail_f32(c3, vacc3x0, nc);
      xnn_store_tail_f32(c4, vacc4x0, nc);
      xnn_store_tail_f32(c5, vacc5x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
