// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/hvx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"

void xnn_f32_igemm_minmax_ukernel_5x128__hvx_broadcast(
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
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
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

  do {
    HVX_Vector vacc0x0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 0)));
    HVX_Vector vacc0x1 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 32)));
    HVX_Vector vacc0x2 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 64)));
    HVX_Vector vacc0x3 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 96)));
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
    w += 128;

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
      a += 5;

      size_t k = kc;
      do {
        const HVX_Vector vb0 = *((HVX_Vector *)(w));
        const HVX_Vector vb1 = *((HVX_Vector *)(w + 32));
        const HVX_Vector vb2 = *((HVX_Vector *)(w + 64));
        const HVX_Vector vb3 = *((HVX_Vector *)(w + 96));
        w += 128;

        const HVX_Vector va0 =  Q6_V_vsplat_R(*(uint32_t *)a0);
        a0 += 1;
        const HVX_Vector va1 =  Q6_V_vsplat_R(*(uint32_t *)a1);
        a1 += 1;
        const HVX_Vector va2 =  Q6_V_vsplat_R(*(uint32_t *)a2);
        a2 += 1;
        const HVX_Vector va3 =  Q6_V_vsplat_R(*(uint32_t *)a3);
        a3 += 1;
        const HVX_Vector va4 =  Q6_V_vsplat_R(*(uint32_t *)a4);
        a4 += 1;

        vacc0x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x0, Q6_Vqf32_vmpy_VsfVsf(va0, vb0));
        vacc0x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x1, Q6_Vqf32_vmpy_VsfVsf(va0, vb1));
        vacc0x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x2, Q6_Vqf32_vmpy_VsfVsf(va0, vb2));
        vacc0x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x3, Q6_Vqf32_vmpy_VsfVsf(va0, vb3));
        vacc1x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x0, Q6_Vqf32_vmpy_VsfVsf(va1, vb0));
        vacc1x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x1, Q6_Vqf32_vmpy_VsfVsf(va1, vb1));
        vacc1x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x2, Q6_Vqf32_vmpy_VsfVsf(va1, vb2));
        vacc1x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x3, Q6_Vqf32_vmpy_VsfVsf(va1, vb3));
        vacc2x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x0, Q6_Vqf32_vmpy_VsfVsf(va2, vb0));
        vacc2x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x1, Q6_Vqf32_vmpy_VsfVsf(va2, vb1));
        vacc2x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x2, Q6_Vqf32_vmpy_VsfVsf(va2, vb2));
        vacc2x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x3, Q6_Vqf32_vmpy_VsfVsf(va2, vb3));
        vacc3x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x0, Q6_Vqf32_vmpy_VsfVsf(va3, vb0));
        vacc3x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x1, Q6_Vqf32_vmpy_VsfVsf(va3, vb1));
        vacc3x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x2, Q6_Vqf32_vmpy_VsfVsf(va3, vb2));
        vacc3x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x3, Q6_Vqf32_vmpy_VsfVsf(va3, vb3));
        vacc4x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x0, Q6_Vqf32_vmpy_VsfVsf(va4, vb0));
        vacc4x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x1, Q6_Vqf32_vmpy_VsfVsf(va4, vb1));
        vacc4x2 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x2, Q6_Vqf32_vmpy_VsfVsf(va4, vb2));
        vacc4x3 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x3, Q6_Vqf32_vmpy_VsfVsf(va4, vb3));
        k -= sizeof(float);
      } while (k != 0);
      p -= 5 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = Q6_Vsf_equals_Vqf32(vacc0x0);
    vacc1x0 = Q6_Vsf_equals_Vqf32(vacc1x0);
    vacc2x0 = Q6_Vsf_equals_Vqf32(vacc2x0);
    vacc3x0 = Q6_Vsf_equals_Vqf32(vacc3x0);
    vacc4x0 = Q6_Vsf_equals_Vqf32(vacc4x0);
    vacc0x1 = Q6_Vsf_equals_Vqf32(vacc0x1);
    vacc1x1 = Q6_Vsf_equals_Vqf32(vacc1x1);
    vacc2x1 = Q6_Vsf_equals_Vqf32(vacc2x1);
    vacc3x1 = Q6_Vsf_equals_Vqf32(vacc3x1);
    vacc4x1 = Q6_Vsf_equals_Vqf32(vacc4x1);
    vacc0x2 = Q6_Vsf_equals_Vqf32(vacc0x2);
    vacc1x2 = Q6_Vsf_equals_Vqf32(vacc1x2);
    vacc2x2 = Q6_Vsf_equals_Vqf32(vacc2x2);
    vacc3x2 = Q6_Vsf_equals_Vqf32(vacc3x2);
    vacc4x2 = Q6_Vsf_equals_Vqf32(vacc4x2);
    vacc0x3 = Q6_Vsf_equals_Vqf32(vacc0x3);
    vacc1x3 = Q6_Vsf_equals_Vqf32(vacc1x3);
    vacc2x3 = Q6_Vsf_equals_Vqf32(vacc2x3);
    vacc3x3 = Q6_Vsf_equals_Vqf32(vacc3x3);
    vacc4x3 = Q6_Vsf_equals_Vqf32(vacc4x3);

    const HVX_Vector vmin = Q6_V_vsplat_R(params->scalar.min);
    vacc0x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x0);
    vacc1x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x0);
    vacc2x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x0);
    vacc3x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x0);
    vacc4x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x0);
    vacc0x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x1);
    vacc1x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x1);
    vacc2x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x1);
    vacc3x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x1);
    vacc4x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x1);
    vacc0x2 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x2);
    vacc1x2 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x2);
    vacc2x2 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x2);
    vacc3x2 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x2);
    vacc4x2 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x2);
    vacc0x3 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x3);
    vacc1x3 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x3);
    vacc2x3 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x3);
    vacc3x3 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x3);
    vacc4x3 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x3);

    const HVX_Vector vmax = Q6_V_vsplat_R(params->scalar.max);
    vacc0x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x0);
    vacc1x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x0);
    vacc2x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x0);
    vacc3x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x0);
    vacc4x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x0);
    vacc0x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x1);
    vacc1x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x1);
    vacc2x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x1);
    vacc3x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x1);
    vacc4x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x1);
    vacc0x2 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x2);
    vacc1x2 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x2);
    vacc2x2 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x2);
    vacc3x2 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x2);
    vacc4x2 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x2);
    vacc0x3 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x3);
    vacc1x3 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x3);
    vacc2x3 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x3);
    vacc3x3 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x3);
    vacc4x3 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x3);

    if XNN_LIKELY(nc >= 128) {
      *((HVX_UVector *)(c4)) = vacc4x0;
      *((HVX_UVector *)(c4 + 32)) = vacc4x1;
      *((HVX_UVector *)(c4 + 64)) = vacc4x2;
      *((HVX_UVector *)(c4 + 96)) = vacc4x3;
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      *((HVX_UVector *)(c3)) = vacc3x0;
      *((HVX_UVector *)(c3 + 32)) = vacc3x1;
      *((HVX_UVector *)(c3 + 64)) = vacc3x2;
      *((HVX_UVector *)(c3 + 96)) = vacc3x3;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      *((HVX_UVector *)(c2)) = vacc2x0;
      *((HVX_UVector *)(c2 + 32)) = vacc2x1;
      *((HVX_UVector *)(c2 + 64)) = vacc2x2;
      *((HVX_UVector *)(c2 + 96)) = vacc2x3;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)(c1)) = vacc1x0;
      *((HVX_UVector *)(c1 + 32)) = vacc1x1;
      *((HVX_UVector *)(c1 + 64)) = vacc1x2;
      *((HVX_UVector *)(c1 + 96)) = vacc1x3;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)(c0)) = vacc0x0;
      *((HVX_UVector *)(c0 + 32)) = vacc0x1;
      *((HVX_UVector *)(c0 + 64)) = vacc0x2;
      *((HVX_UVector *)(c0 + 96)) = vacc0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 128;
    } else {
      if (nc & 64) {
        *((HVX_UVector *)c4) = vacc4x0;
        *((HVX_UVector *)(c4 + 32)) = vacc4x1;
        *((HVX_UVector *)c3) = vacc3x0;
        *((HVX_UVector *)(c3 + 32)) = vacc3x1;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)(c2 + 32)) = vacc2x1;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)(c1 + 32)) = vacc1x1;
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)(c0 + 32)) = vacc0x1;

        vacc4x0 = vacc4x2;
        vacc4x1 = vacc4x3;
        vacc3x0 = vacc3x2;
        vacc3x1 = vacc3x3;
        vacc2x0 = vacc2x2;
        vacc2x1 = vacc2x3;
        vacc1x0 = vacc1x2;
        vacc1x1 = vacc1x3;
        vacc0x0 = vacc0x2;
        vacc0x1 = vacc0x3;

        c4 += 64;
        c3 += 64;
        c2 += 64;
        c1 += 64;
        c0 += 64;
        nc ^= 64;
      }
      if (nc & 32) {
        *((HVX_UVector *)c4) = vacc4x0;
        *((HVX_UVector *)c3) = vacc3x0;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)c0) = vacc0x0;

        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;

        c4 += 32;
        c3 += 32;
        c2 += 32;
        c1 += 32;
        c0 += 32;
        nc ^= 32;
      }
      Q6_V_vstu_variable(c4, nc * sizeof(float), vacc4x0);
      Q6_V_vstu_variable(c3, nc * sizeof(float), vacc3x0);
      Q6_V_vstu_variable(c2, nc * sizeof(float), vacc2x0);
      Q6_V_vstu_variable(c1, nc * sizeof(float), vacc1x0);
      Q6_V_vstu_variable(c0, nc * sizeof(float), vacc0x0);

      nc = 0;
    }
  } while (nc != 0);
}
