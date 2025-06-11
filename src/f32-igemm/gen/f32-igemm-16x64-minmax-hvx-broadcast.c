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

void xnn_f32_igemm_minmax_ukernel_16x64__hvx_broadcast(
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
    const struct xnn_f32_minmax_params* restrict params)
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
    HVX_Vector vacc0x0 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 0)));
    HVX_Vector vacc0x1 = Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_vzero(), *((HVX_Vector *)(w + 32)));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc2x0 = vacc0x0;
    HVX_Vector vacc2x1 = vacc0x1;
    HVX_Vector vacc3x0 = vacc0x0;
    HVX_Vector vacc3x1 = vacc0x1;
    HVX_Vector vacc4x0 = vacc0x0;
    HVX_Vector vacc4x1 = vacc0x1;
    HVX_Vector vacc5x0 = vacc0x0;
    HVX_Vector vacc5x1 = vacc0x1;
    HVX_Vector vacc6x0 = vacc0x0;
    HVX_Vector vacc6x1 = vacc0x1;
    HVX_Vector vacc7x0 = vacc0x0;
    HVX_Vector vacc7x1 = vacc0x1;
    HVX_Vector vacc8x0 = vacc0x0;
    HVX_Vector vacc8x1 = vacc0x1;
    HVX_Vector vacc9x0 = vacc0x0;
    HVX_Vector vacc9x1 = vacc0x1;
    HVX_Vector vacc10x0 = vacc0x0;
    HVX_Vector vacc10x1 = vacc0x1;
    HVX_Vector vacc11x0 = vacc0x0;
    HVX_Vector vacc11x1 = vacc0x1;
    HVX_Vector vacc12x0 = vacc0x0;
    HVX_Vector vacc12x1 = vacc0x1;
    HVX_Vector vacc13x0 = vacc0x0;
    HVX_Vector vacc13x1 = vacc0x1;
    HVX_Vector vacc14x0 = vacc0x0;
    HVX_Vector vacc14x1 = vacc0x1;
    HVX_Vector vacc15x0 = vacc0x0;
    HVX_Vector vacc15x1 = vacc0x1;
    w += 64;

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
        const HVX_Vector vb1 = *((HVX_Vector *)(w + 32));
        w += 64;

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

        vacc0x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x0, Q6_Vqf32_vmpy_VsfVsf(va0, vb0));
        vacc0x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0x1, Q6_Vqf32_vmpy_VsfVsf(va0, vb1));
        vacc1x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x0, Q6_Vqf32_vmpy_VsfVsf(va1, vb0));
        vacc1x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1x1, Q6_Vqf32_vmpy_VsfVsf(va1, vb1));
        vacc2x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x0, Q6_Vqf32_vmpy_VsfVsf(va2, vb0));
        vacc2x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2x1, Q6_Vqf32_vmpy_VsfVsf(va2, vb1));
        vacc3x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x0, Q6_Vqf32_vmpy_VsfVsf(va3, vb0));
        vacc3x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3x1, Q6_Vqf32_vmpy_VsfVsf(va3, vb1));
        vacc4x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x0, Q6_Vqf32_vmpy_VsfVsf(va4, vb0));
        vacc4x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc4x1, Q6_Vqf32_vmpy_VsfVsf(va4, vb1));
        vacc5x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x0, Q6_Vqf32_vmpy_VsfVsf(va5, vb0));
        vacc5x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc5x1, Q6_Vqf32_vmpy_VsfVsf(va5, vb1));
        vacc6x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc6x0, Q6_Vqf32_vmpy_VsfVsf(va6, vb0));
        vacc6x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc6x1, Q6_Vqf32_vmpy_VsfVsf(va6, vb1));
        vacc7x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc7x0, Q6_Vqf32_vmpy_VsfVsf(va7, vb0));
        vacc7x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc7x1, Q6_Vqf32_vmpy_VsfVsf(va7, vb1));
        vacc8x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc8x0, Q6_Vqf32_vmpy_VsfVsf(va8, vb0));
        vacc8x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc8x1, Q6_Vqf32_vmpy_VsfVsf(va8, vb1));
        vacc9x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc9x0, Q6_Vqf32_vmpy_VsfVsf(va9, vb0));
        vacc9x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc9x1, Q6_Vqf32_vmpy_VsfVsf(va9, vb1));
        vacc10x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc10x0, Q6_Vqf32_vmpy_VsfVsf(va10, vb0));
        vacc10x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc10x1, Q6_Vqf32_vmpy_VsfVsf(va10, vb1));
        vacc11x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc11x0, Q6_Vqf32_vmpy_VsfVsf(va11, vb0));
        vacc11x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc11x1, Q6_Vqf32_vmpy_VsfVsf(va11, vb1));
        vacc12x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc12x0, Q6_Vqf32_vmpy_VsfVsf(va12, vb0));
        vacc12x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc12x1, Q6_Vqf32_vmpy_VsfVsf(va12, vb1));
        vacc13x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc13x0, Q6_Vqf32_vmpy_VsfVsf(va13, vb0));
        vacc13x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc13x1, Q6_Vqf32_vmpy_VsfVsf(va13, vb1));
        vacc14x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc14x0, Q6_Vqf32_vmpy_VsfVsf(va14, vb0));
        vacc14x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc14x1, Q6_Vqf32_vmpy_VsfVsf(va14, vb1));
        vacc15x0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc15x0, Q6_Vqf32_vmpy_VsfVsf(va15, vb0));
        vacc15x1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc15x1, Q6_Vqf32_vmpy_VsfVsf(va15, vb1));
        k -= sizeof(float);
      } while (k != 0);
      p -= 16 * sizeof(void*);
    } while (p != 0);

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
    vacc10x0 = Q6_Vsf_equals_Vqf32(vacc10x0);
    vacc11x0 = Q6_Vsf_equals_Vqf32(vacc11x0);
    vacc12x0 = Q6_Vsf_equals_Vqf32(vacc12x0);
    vacc13x0 = Q6_Vsf_equals_Vqf32(vacc13x0);
    vacc14x0 = Q6_Vsf_equals_Vqf32(vacc14x0);
    vacc15x0 = Q6_Vsf_equals_Vqf32(vacc15x0);
    vacc0x1 = Q6_Vsf_equals_Vqf32(vacc0x1);
    vacc1x1 = Q6_Vsf_equals_Vqf32(vacc1x1);
    vacc2x1 = Q6_Vsf_equals_Vqf32(vacc2x1);
    vacc3x1 = Q6_Vsf_equals_Vqf32(vacc3x1);
    vacc4x1 = Q6_Vsf_equals_Vqf32(vacc4x1);
    vacc5x1 = Q6_Vsf_equals_Vqf32(vacc5x1);
    vacc6x1 = Q6_Vsf_equals_Vqf32(vacc6x1);
    vacc7x1 = Q6_Vsf_equals_Vqf32(vacc7x1);
    vacc8x1 = Q6_Vsf_equals_Vqf32(vacc8x1);
    vacc9x1 = Q6_Vsf_equals_Vqf32(vacc9x1);
    vacc10x1 = Q6_Vsf_equals_Vqf32(vacc10x1);
    vacc11x1 = Q6_Vsf_equals_Vqf32(vacc11x1);
    vacc12x1 = Q6_Vsf_equals_Vqf32(vacc12x1);
    vacc13x1 = Q6_Vsf_equals_Vqf32(vacc13x1);
    vacc14x1 = Q6_Vsf_equals_Vqf32(vacc14x1);
    vacc15x1 = Q6_Vsf_equals_Vqf32(vacc15x1);

    const HVX_Vector vmin = Q6_V_vsplat_R(params->scalar.min);
    vacc0x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x0);
    vacc1x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x0);
    vacc2x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x0);
    vacc3x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x0);
    vacc4x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x0);
    vacc5x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc5x0);
    vacc6x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc6x0);
    vacc7x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc7x0);
    vacc8x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc8x0);
    vacc9x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc9x0);
    vacc10x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc10x0);
    vacc11x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc11x0);
    vacc12x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc12x0);
    vacc13x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc13x0);
    vacc14x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc14x0);
    vacc15x0 = Q6_Vsf_vmax_VsfVsf(vmin, vacc15x0);
    vacc0x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc0x1);
    vacc1x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc1x1);
    vacc2x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc2x1);
    vacc3x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc3x1);
    vacc4x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc4x1);
    vacc5x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc5x1);
    vacc6x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc6x1);
    vacc7x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc7x1);
    vacc8x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc8x1);
    vacc9x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc9x1);
    vacc10x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc10x1);
    vacc11x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc11x1);
    vacc12x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc12x1);
    vacc13x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc13x1);
    vacc14x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc14x1);
    vacc15x1 = Q6_Vsf_vmax_VsfVsf(vmin, vacc15x1);

    const HVX_Vector vmax = Q6_V_vsplat_R(params->scalar.max);
    vacc0x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x0);
    vacc1x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x0);
    vacc2x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x0);
    vacc3x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x0);
    vacc4x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x0);
    vacc5x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc5x0);
    vacc6x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc6x0);
    vacc7x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc7x0);
    vacc8x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc8x0);
    vacc9x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc9x0);
    vacc10x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc10x0);
    vacc11x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc11x0);
    vacc12x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc12x0);
    vacc13x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc13x0);
    vacc14x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc14x0);
    vacc15x0 = Q6_Vsf_vmin_VsfVsf(vmax, vacc15x0);
    vacc0x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc0x1);
    vacc1x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc1x1);
    vacc2x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc2x1);
    vacc3x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc3x1);
    vacc4x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc4x1);
    vacc5x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc5x1);
    vacc6x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc6x1);
    vacc7x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc7x1);
    vacc8x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc8x1);
    vacc9x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc9x1);
    vacc10x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc10x1);
    vacc11x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc11x1);
    vacc12x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc12x1);
    vacc13x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc13x1);
    vacc14x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc14x1);
    vacc15x1 = Q6_Vsf_vmin_VsfVsf(vmax, vacc15x1);

    if XNN_LIKELY(nc >= 64) {
      *((HVX_UVector *)(c15)) = vacc15x0;
      *((HVX_UVector *)(c15 + 32)) = vacc15x1;
      c15 = (float*) ((uintptr_t) c15 + cn_stride);
      *((HVX_UVector *)(c14)) = vacc14x0;
      *((HVX_UVector *)(c14 + 32)) = vacc14x1;
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      *((HVX_UVector *)(c13)) = vacc13x0;
      *((HVX_UVector *)(c13 + 32)) = vacc13x1;
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      *((HVX_UVector *)(c12)) = vacc12x0;
      *((HVX_UVector *)(c12 + 32)) = vacc12x1;
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      *((HVX_UVector *)(c11)) = vacc11x0;
      *((HVX_UVector *)(c11 + 32)) = vacc11x1;
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      *((HVX_UVector *)(c10)) = vacc10x0;
      *((HVX_UVector *)(c10 + 32)) = vacc10x1;
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      *((HVX_UVector *)(c9)) = vacc9x0;
      *((HVX_UVector *)(c9 + 32)) = vacc9x1;
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      *((HVX_UVector *)(c8)) = vacc8x0;
      *((HVX_UVector *)(c8 + 32)) = vacc8x1;
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      *((HVX_UVector *)(c7)) = vacc7x0;
      *((HVX_UVector *)(c7 + 32)) = vacc7x1;
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      *((HVX_UVector *)(c6)) = vacc6x0;
      *((HVX_UVector *)(c6 + 32)) = vacc6x1;
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      *((HVX_UVector *)(c5)) = vacc5x0;
      *((HVX_UVector *)(c5 + 32)) = vacc5x1;
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      *((HVX_UVector *)(c4)) = vacc4x0;
      *((HVX_UVector *)(c4 + 32)) = vacc4x1;
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      *((HVX_UVector *)(c3)) = vacc3x0;
      *((HVX_UVector *)(c3 + 32)) = vacc3x1;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      *((HVX_UVector *)(c2)) = vacc2x0;
      *((HVX_UVector *)(c2 + 32)) = vacc2x1;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      *((HVX_UVector *)(c1)) = vacc1x0;
      *((HVX_UVector *)(c1 + 32)) = vacc1x1;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      *((HVX_UVector *)(c0)) = vacc0x0;
      *((HVX_UVector *)(c0 + 32)) = vacc0x1;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      if (nc & 32) {
        *((HVX_UVector *)c15) = vacc15x0;
        *((HVX_UVector *)c14) = vacc14x0;
        *((HVX_UVector *)c13) = vacc13x0;
        *((HVX_UVector *)c12) = vacc12x0;
        *((HVX_UVector *)c11) = vacc11x0;
        *((HVX_UVector *)c10) = vacc10x0;
        *((HVX_UVector *)c9) = vacc9x0;
        *((HVX_UVector *)c8) = vacc8x0;
        *((HVX_UVector *)c7) = vacc7x0;
        *((HVX_UVector *)c6) = vacc6x0;
        *((HVX_UVector *)c5) = vacc5x0;
        *((HVX_UVector *)c4) = vacc4x0;
        *((HVX_UVector *)c3) = vacc3x0;
        *((HVX_UVector *)c2) = vacc2x0;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)c0) = vacc0x0;

        vacc15x0 = vacc15x1;
        vacc14x0 = vacc14x1;
        vacc13x0 = vacc13x1;
        vacc12x0 = vacc12x1;
        vacc11x0 = vacc11x1;
        vacc10x0 = vacc10x1;
        vacc9x0 = vacc9x1;
        vacc8x0 = vacc8x1;
        vacc7x0 = vacc7x1;
        vacc6x0 = vacc6x1;
        vacc5x0 = vacc5x1;
        vacc4x0 = vacc4x1;
        vacc3x0 = vacc3x1;
        vacc2x0 = vacc2x1;
        vacc1x0 = vacc1x1;
        vacc0x0 = vacc0x1;

        c15 += 32;
        c14 += 32;
        c13 += 32;
        c12 += 32;
        c11 += 32;
        c10 += 32;
        c9 += 32;
        c8 += 32;
        c7 += 32;
        c6 += 32;
        c5 += 32;
        c4 += 32;
        c3 += 32;
        c2 += 32;
        c1 += 32;
        c0 += 32;
        nc ^= 32;
      }
      Q6_V_vstu_variable(c15, nc * sizeof(float), vacc15x0);
      Q6_V_vstu_variable(c14, nc * sizeof(float), vacc14x0);
      Q6_V_vstu_variable(c13, nc * sizeof(float), vacc13x0);
      Q6_V_vstu_variable(c12, nc * sizeof(float), vacc12x0);
      Q6_V_vstu_variable(c11, nc * sizeof(float), vacc11x0);
      Q6_V_vstu_variable(c10, nc * sizeof(float), vacc10x0);
      Q6_V_vstu_variable(c9, nc * sizeof(float), vacc9x0);
      Q6_V_vstu_variable(c8, nc * sizeof(float), vacc8x0);
      Q6_V_vstu_variable(c7, nc * sizeof(float), vacc7x0);
      Q6_V_vstu_variable(c6, nc * sizeof(float), vacc6x0);
      Q6_V_vstu_variable(c5, nc * sizeof(float), vacc5x0);
      Q6_V_vstu_variable(c4, nc * sizeof(float), vacc4x0);
      Q6_V_vstu_variable(c3, nc * sizeof(float), vacc3x0);
      Q6_V_vstu_variable(c2, nc * sizeof(float), vacc2x0);
      Q6_V_vstu_variable(c1, nc * sizeof(float), vacc1x0);
      Q6_V_vstu_variable(c0, nc * sizeof(float), vacc0x0);

      nc = 0;
    }
  } while (nc != 0);
}
