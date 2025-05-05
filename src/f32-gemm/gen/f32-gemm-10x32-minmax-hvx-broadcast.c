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

void xnn_f32_gemm_minmax_ukernel_10x32__hvx_broadcast(
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
  assert(mr <= 10);
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
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const float* a7 = (const float*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const float* a8 = (const float*) ((uintptr_t) a7 + a_stride);
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const float* a9 = (const float*) ((uintptr_t) a8 + a_stride);
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 10) {
    a9 = a8;
    c9 = c8;
  }

  do {
    HVX_Vector vacc0x0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), xnn_load_f32(w + 0));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc5x0 = vacc0x0;
    HVX_Vector vacc6x0 = vacc0x0;
    HVX_Vector vacc7x0 = vacc0x0;
    HVX_Vector vacc8x0 = vacc0x0;
    HVX_Vector vacc9x0 = vacc0x0;
    w += 32;

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
      const HVX_Vector va6 = xnn_set1_f32(*a6);
      a6 += 1;
      const HVX_Vector va7 = xnn_set1_f32(*a7);
      a7 += 1;
      const HVX_Vector va8 = xnn_set1_f32(*a8);
      a8 += 1;
      const HVX_Vector va9 = xnn_set1_f32(*a9);
      a9 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w + 0));
      w += 32;

      const HVX_Vector vtemp0x0 = Q6_Vqf32_vmpy_VsfVsf(va0, vb0);
      const HVX_Vector vtemp1x0 = Q6_Vqf32_vmpy_VsfVsf(va1, vb0);
      const HVX_Vector vtemp2x0 = Q6_Vqf32_vmpy_VsfVsf(va2, vb0);
      const HVX_Vector vtemp3x0 = Q6_Vqf32_vmpy_VsfVsf(va3, vb0);
      const HVX_Vector vtemp4x0 = Q6_Vqf32_vmpy_VsfVsf(va4, vb0);
      const HVX_Vector vtemp5x0 = Q6_Vqf32_vmpy_VsfVsf(va5, vb0);
      const HVX_Vector vtemp6x0 = Q6_Vqf32_vmpy_VsfVsf(va6, vb0);
      const HVX_Vector vtemp7x0 = Q6_Vqf32_vmpy_VsfVsf(va7, vb0);
      const HVX_Vector vtemp8x0 = Q6_Vqf32_vmpy_VsfVsf(va8, vb0);
      const HVX_Vector vtemp9x0 = Q6_Vqf32_vmpy_VsfVsf(va9, vb0);

      vacc0x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x0, vtemp0x0);
      vacc1x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x0, vtemp1x0);
      vacc2x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x0, vtemp2x0);
      vacc3x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x0, vtemp3x0);
      vacc4x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x0, vtemp4x0);
      vacc5x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x0, vtemp5x0);
      vacc6x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc6x0, vtemp6x0);
      vacc7x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc7x0, vtemp7x0);
      vacc8x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc8x0, vtemp8x0);
      vacc9x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc9x0, vtemp9x0);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x0 = Q6_Vsf_equals_Vqf32(vacc0x0);
    vacc1x0 = Q6_Vsf_equals_Vqf32(vacc1x0);
    vacc2x0 = Q6_Vsf_equals_Vqf32(vacc2x0);
    vacc3x0 = Q6_Vsf_equals_Vqf32(vacc3x0);
    vacc4x0 = Q6_Vsf_equals_Vqf32(vacc4x0);
    vacc5x0 = Q6_Vsf_equals_Vqf32(vacc5x0);
    vacc6x0 = Q6_Vsf_equals_Vqf32(vacc6x0);
    vacc7x0 = Q6_Vsf_equals_Vqf32(vacc7x0);
    vacc8x0 = Q6_Vsf_equals_Vqf32(vacc8x0);
    vacc9x0 = Q6_Vsf_equals_Vqf32(vacc9x0);

    HVX_Vector vmin = xnn_set1_f32(params->scalar.min);
    vacc0x0 = xnn_max_f32(vmin, vacc0x0);
    vacc1x0 = xnn_max_f32(vmin, vacc1x0);
    vacc2x0 = xnn_max_f32(vmin, vacc2x0);
    vacc3x0 = xnn_max_f32(vmin, vacc3x0);
    vacc4x0 = xnn_max_f32(vmin, vacc4x0);
    vacc5x0 = xnn_max_f32(vmin, vacc5x0);
    vacc6x0 = xnn_max_f32(vmin, vacc6x0);
    vacc7x0 = xnn_max_f32(vmin, vacc7x0);
    vacc8x0 = xnn_max_f32(vmin, vacc8x0);
    vacc9x0 = xnn_max_f32(vmin, vacc9x0);

    HVX_Vector vmax = xnn_set1_f32(params->scalar.max);
    vacc0x0 = xnn_min_f32(vmax, vacc0x0);
    vacc1x0 = xnn_min_f32(vmax, vacc1x0);
    vacc2x0 = xnn_min_f32(vmax, vacc2x0);
    vacc3x0 = xnn_min_f32(vmax, vacc3x0);
    vacc4x0 = xnn_min_f32(vmax, vacc4x0);
    vacc5x0 = xnn_min_f32(vmax, vacc5x0);
    vacc6x0 = xnn_min_f32(vmax, vacc6x0);
    vacc7x0 = xnn_min_f32(vmax, vacc7x0);
    vacc8x0 = xnn_min_f32(vmax, vacc8x0);
    vacc9x0 = xnn_min_f32(vmax, vacc9x0);

    if XNN_LIKELY(nc >= 32) {
      *((HVX_UVector *)c0) = vacc0x0;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      *((HVX_UVector *)c1) = vacc1x0;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)c2) = vacc2x0;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)c3) = vacc3x0;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      *((HVX_UVector *)c4) = vacc4x0;
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      *((HVX_UVector *)c5) = vacc5x0;
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      *((HVX_UVector *)c6) = vacc6x0;
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      *((HVX_UVector *)c7) = vacc7x0;
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      *((HVX_UVector *)c8) = vacc8x0;
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      *((HVX_UVector *)c9) = vacc9x0;
      c9 = (float*) ((uintptr_t) c9 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);
      a7 = (const float*) ((uintptr_t) a7 - kc);
      a8 = (const float*) ((uintptr_t) a8 - kc);
      a9 = (const float*) ((uintptr_t) a9 - kc);

      nc -= 32;
    } else {
      xnn_store_tail_f32(c0, vacc0x0, nc);
      xnn_store_tail_f32(c1, vacc1x0, nc);
      xnn_store_tail_f32(c2, vacc2x0, nc);
      xnn_store_tail_f32(c3, vacc3x0, nc);
      xnn_store_tail_f32(c4, vacc4x0, nc);
      xnn_store_tail_f32(c5, vacc5x0, nc);
      xnn_store_tail_f32(c6, vacc6x0, nc);
      xnn_store_tail_f32(c7, vacc7x0, nc);
      xnn_store_tail_f32(c8, vacc8x0, nc);
      xnn_store_tail_f32(c9, vacc9x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
