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

#include "xnnpack/igemm.h"

static XNN_INLINE
void vstu_variable_scalar(char *bytes, size_t num_bytes, HVX_Vector vin) {
  char temp[128]  __attribute__((aligned(128)));
  *((HVX_Vector *)temp) = vin;
  for (size_t idx = 0; idx < num_bytes; idx++){
     *bytes = temp[idx];
     bytes++;
  }
}

void xnn_f32_igemm_minmax_ukernel_16x32__hvx_broadcast(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 16);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (16 * sizeof(void*)) == 0);
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
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    c13 = c12;
  }
  float* c14 = (float*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    c14 = c13;
  }
  float* c15 = (float*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    c15 = c14;
  }

  do {
    HVX_Vector vacc0x0 = *((HVX_Vector *)(w));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc5x0 = vacc0x0;
    HVX_Vector vacc6x0 = vacc0x0;
    HVX_Vector vacc7x0 = vacc0x0;
    HVX_Vector vacc8x0 = vacc0x0;
    HVX_Vector vacc9x0 = vacc0x0;
    HVX_Vector vacc10x0 = vacc0x0;
    HVX_Vector vacc11x0 = vacc0x0;
    HVX_Vector vacc12x0 = vacc0x0;
    HVX_Vector vacc13x0 = vacc0x0;
    HVX_Vector vacc14x0 = vacc0x0;
    HVX_Vector vacc15x0 = vacc0x0;
    w += 32;

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
      const float* restrict a7 = a[7];
      assert(a7 != NULL);
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const float*) ((uintptr_t) a7 + a_offset);
      }
      const float* restrict a8 = a[8];
      assert(a8 != NULL);
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const float*) ((uintptr_t) a8 + a_offset);
      }
      const float* restrict a9 = a[9];
      assert(a9 != NULL);
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const float*) ((uintptr_t) a9 + a_offset);
      }
      const float* restrict a10 = a[10];
      assert(a10 != NULL);
      if XNN_UNPREDICTABLE(a10 != zero) {
        a10 = (const float*) ((uintptr_t) a10 + a_offset);
      }
      const float* restrict a11 = a[11];
      assert(a11 != NULL);
      if XNN_UNPREDICTABLE(a11 != zero) {
        a11 = (const float*) ((uintptr_t) a11 + a_offset);
      }
      const float* restrict a12 = a[12];
      assert(a12 != NULL);
      if XNN_UNPREDICTABLE(a12 != zero) {
        a12 = (const float*) ((uintptr_t) a12 + a_offset);
      }
      const float* restrict a13 = a[13];
      assert(a13 != NULL);
      if XNN_UNPREDICTABLE(a13 != zero) {
        a13 = (const float*) ((uintptr_t) a13 + a_offset);
      }
      const float* restrict a14 = a[14];
      assert(a14 != NULL);
      if XNN_UNPREDICTABLE(a14 != zero) {
        a14 = (const float*) ((uintptr_t) a14 + a_offset);
      }
      const float* restrict a15 = a[15];
      assert(a15 != NULL);
      if XNN_UNPREDICTABLE(a15 != zero) {
        a15 = (const float*) ((uintptr_t) a15 + a_offset);
      }
      a += 16;

      size_t k = kc;
      do {
        const HVX_Vector vb0 = *((HVX_Vector *)(w));
        w += 32;

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
        const HVX_Vector va5 =  Q6_V_vsplat_R(*(uint32_t *)a5);
        a5 += 1;
        const HVX_Vector va6 =  Q6_V_vsplat_R(*(uint32_t *)a6);
        a6 += 1;
        const HVX_Vector va7 =  Q6_V_vsplat_R(*(uint32_t *)a7);
        a7 += 1;
        const HVX_Vector va8 =  Q6_V_vsplat_R(*(uint32_t *)a8);
        a8 += 1;
        const HVX_Vector va9 =  Q6_V_vsplat_R(*(uint32_t *)a9);
        a9 += 1;
        const HVX_Vector va10 =  Q6_V_vsplat_R(*(uint32_t *)a10);
        a10 += 1;
        const HVX_Vector va11 =  Q6_V_vsplat_R(*(uint32_t *)a11);
        a11 += 1;
        const HVX_Vector va12 =  Q6_V_vsplat_R(*(uint32_t *)a12);
        a12 += 1;
        const HVX_Vector va13 =  Q6_V_vsplat_R(*(uint32_t *)a13);
        a13 += 1;
        const HVX_Vector va14 =  Q6_V_vsplat_R(*(uint32_t *)a14);
        a14 += 1;
        const HVX_Vector va15 =  Q6_V_vsplat_R(*(uint32_t *)a15);
        a15 += 1;

        vacc0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va0, vb0), vacc0x0));
        vacc1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va1, vb0), vacc1x0));
        vacc2x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va2, vb0), vacc2x0));
        vacc3x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va3, vb0), vacc3x0));
        vacc4x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va4, vb0), vacc4x0));
        vacc5x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va5, vb0), vacc5x0));
        vacc6x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va6, vb0), vacc6x0));
        vacc7x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va7, vb0), vacc7x0));
        vacc8x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va8, vb0), vacc8x0));
        vacc9x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va9, vb0), vacc9x0));
        vacc10x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va10, vb0), vacc10x0));
        vacc11x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va11, vb0), vacc11x0));
        vacc12x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va12, vb0), vacc12x0));
        vacc13x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va13, vb0), vacc13x0));
        vacc14x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va14, vb0), vacc14x0));
        vacc15x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va15, vb0), vacc15x0));
        k -= sizeof(float);
      } while (k != 0);
      p -= 16 * sizeof(void*);
    } while (p != 0);

    const HVX_Vector vmin = *((HVX_Vector *)(params->hvx.min));
    vacc0x0 = Q6_Vw_vmax_VwVw(vmin, vacc0x0);
    vacc1x0 = Q6_Vw_vmax_VwVw(vmin, vacc1x0);
    vacc2x0 = Q6_Vw_vmax_VwVw(vmin, vacc2x0);
    vacc3x0 = Q6_Vw_vmax_VwVw(vmin, vacc3x0);
    vacc4x0 = Q6_Vw_vmax_VwVw(vmin, vacc4x0);
    vacc5x0 = Q6_Vw_vmax_VwVw(vmin, vacc5x0);
    vacc6x0 = Q6_Vw_vmax_VwVw(vmin, vacc6x0);
    vacc7x0 = Q6_Vw_vmax_VwVw(vmin, vacc7x0);
    vacc8x0 = Q6_Vw_vmax_VwVw(vmin, vacc8x0);
    vacc9x0 = Q6_Vw_vmax_VwVw(vmin, vacc9x0);
    vacc10x0 = Q6_Vw_vmax_VwVw(vmin, vacc10x0);
    vacc11x0 = Q6_Vw_vmax_VwVw(vmin, vacc11x0);
    vacc12x0 = Q6_Vw_vmax_VwVw(vmin, vacc12x0);
    vacc13x0 = Q6_Vw_vmax_VwVw(vmin, vacc13x0);
    vacc14x0 = Q6_Vw_vmax_VwVw(vmin, vacc14x0);
    vacc15x0 = Q6_Vw_vmax_VwVw(vmin, vacc15x0);

    const HVX_Vector vmax = *((HVX_Vector *)(params->hvx.max));
    vacc0x0 = Q6_Vw_vmin_VwVw(vmax, vacc0x0);
    vacc1x0 = Q6_Vw_vmin_VwVw(vmax, vacc1x0);
    vacc2x0 = Q6_Vw_vmin_VwVw(vmax, vacc2x0);
    vacc3x0 = Q6_Vw_vmin_VwVw(vmax, vacc3x0);
    vacc4x0 = Q6_Vw_vmin_VwVw(vmax, vacc4x0);
    vacc5x0 = Q6_Vw_vmin_VwVw(vmax, vacc5x0);
    vacc6x0 = Q6_Vw_vmin_VwVw(vmax, vacc6x0);
    vacc7x0 = Q6_Vw_vmin_VwVw(vmax, vacc7x0);
    vacc8x0 = Q6_Vw_vmin_VwVw(vmax, vacc8x0);
    vacc9x0 = Q6_Vw_vmin_VwVw(vmax, vacc9x0);
    vacc10x0 = Q6_Vw_vmin_VwVw(vmax, vacc10x0);
    vacc11x0 = Q6_Vw_vmin_VwVw(vmax, vacc11x0);
    vacc12x0 = Q6_Vw_vmin_VwVw(vmax, vacc12x0);
    vacc13x0 = Q6_Vw_vmin_VwVw(vmax, vacc13x0);
    vacc14x0 = Q6_Vw_vmin_VwVw(vmax, vacc14x0);
    vacc15x0 = Q6_Vw_vmin_VwVw(vmax, vacc15x0);

    if XNN_LIKELY(nc >= 32) {
      *((HVX_UVector *)(c15)) = vacc15x0;
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      *((HVX_UVector *)(c14)) = vacc14x0;
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      *((HVX_UVector *)(c13)) = vacc13x0;
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      *((HVX_UVector *)(c12)) = vacc12x0;
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      *((HVX_UVector *)(c11)) = vacc11x0;
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      *((HVX_UVector *)(c10)) = vacc10x0;
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      *((HVX_UVector *)(c9)) = vacc9x0;
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      *((HVX_UVector *)(c8)) = vacc8x0;
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      *((HVX_UVector *)(c7)) = vacc7x0;
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      *((HVX_UVector *)(c6)) = vacc6x0;
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      *((HVX_UVector *)(c5)) = vacc5x0;
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      *((HVX_UVector *)(c4)) = vacc4x0;
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      *((HVX_UVector *)(c3)) = vacc3x0;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      *((HVX_UVector *)(c2)) = vacc2x0;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)(c1)) = vacc1x0;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)(c0)) = vacc0x0;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      vstu_variable_scalar((char*)c15, nc*sizeof(float), vacc15x0);
      vstu_variable_scalar((char*)c14, nc*sizeof(float), vacc14x0);
      vstu_variable_scalar((char*)c13, nc*sizeof(float), vacc13x0);
      vstu_variable_scalar((char*)c12, nc*sizeof(float), vacc12x0);
      vstu_variable_scalar((char*)c11, nc*sizeof(float), vacc11x0);
      vstu_variable_scalar((char*)c10, nc*sizeof(float), vacc10x0);
      vstu_variable_scalar((char*)c9, nc*sizeof(float), vacc9x0);
      vstu_variable_scalar((char*)c8, nc*sizeof(float), vacc8x0);
      vstu_variable_scalar((char*)c7, nc*sizeof(float), vacc7x0);
      vstu_variable_scalar((char*)c6, nc*sizeof(float), vacc6x0);
      vstu_variable_scalar((char*)c5, nc*sizeof(float), vacc5x0);
      vstu_variable_scalar((char*)c4, nc*sizeof(float), vacc4x0);
      vstu_variable_scalar((char*)c3, nc*sizeof(float), vacc3x0);
      vstu_variable_scalar((char*)c2, nc*sizeof(float), vacc2x0);
      vstu_variable_scalar((char*)c1, nc*sizeof(float), vacc1x0);
      vstu_variable_scalar((char*)c0, nc*sizeof(float), vacc0x0);

      nc = 0;
    }
  } while (nc != 0);
}
