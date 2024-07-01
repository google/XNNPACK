// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/hvx-broadcast.c.in
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
#include "xnnpack/gemm.h"

static XNN_INLINE
void vstu_variable_scalar(char *bytes, size_t num_bytes, HVX_Vector vin) {
  char temp[128]  __attribute__((aligned(128)));
  *((HVX_Vector *)temp) = vin;
  for (size_t idx = 0; idx < num_bytes; idx++){
     *bytes = temp[idx];
     bytes++;
  }
}

void xnn_f32_gemm_minmax_ukernel_2x128__hvx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
    HVX_Vector vacc0x0 = *((HVX_Vector *)(w + 0));
    HVX_Vector vacc0x1 = *((HVX_Vector *)(w + 32));
    HVX_Vector vacc0x2 = *((HVX_Vector *)(w + 64));
    HVX_Vector vacc0x3 = *((HVX_Vector *)(w + 96));
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc1x2 = vacc0x2;
    HVX_Vector vacc1x3 = vacc0x3;
    w += 128;

    size_t k = kc;
    do {
      const HVX_Vector va0 = Q6_V_vsplat_R(*(uint32_t *)a0);
      a0 += 1;
      const HVX_Vector va1 = Q6_V_vsplat_R(*(uint32_t *)a1);
      a1 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w));
      const HVX_Vector vb1 = *((const HVX_Vector *)(w + 32));
      const HVX_Vector vb2 = *((const HVX_Vector *)(w + 64));
      const HVX_Vector vb3 = *((const HVX_Vector *)(w + 96));
      w += 128;

      vacc0x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va0, vb0),vacc0x0));
      vacc1x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va1, vb0),vacc1x0));
      vacc0x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va0, vb1),vacc0x1));
      vacc1x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va1, vb1),vacc1x1));
      vacc0x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va0, vb2),vacc0x2));
      vacc1x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va1, vb2),vacc1x2));
      vacc0x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va0, vb3),vacc0x3));
      vacc1x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(va1, vb3),vacc1x3));

      k -= sizeof(float);
    } while (k != 0);

    const HVX_Vector vmin = *(const HVX_Vector *)((params->hvx.min));
    vacc0x0 = Q6_Vw_vmax_VwVw(vmin, vacc0x0);
    vacc1x0 = Q6_Vw_vmax_VwVw(vmin, vacc1x0);
    vacc0x1 = Q6_Vw_vmax_VwVw(vmin, vacc0x1);
    vacc1x1 = Q6_Vw_vmax_VwVw(vmin, vacc1x1);
    vacc0x2 = Q6_Vw_vmax_VwVw(vmin, vacc0x2);
    vacc1x2 = Q6_Vw_vmax_VwVw(vmin, vacc1x2);
    vacc0x3 = Q6_Vw_vmax_VwVw(vmin, vacc0x3);
    vacc1x3 = Q6_Vw_vmax_VwVw(vmin, vacc1x3);

    const HVX_Vector vmax = *(const HVX_Vector *)((params->hvx.max));
    vacc0x0 = Q6_Vw_vmin_VwVw(vmax, vacc0x0);
    vacc1x0 = Q6_Vw_vmin_VwVw(vmax, vacc1x0);
    vacc0x1 = Q6_Vw_vmin_VwVw(vmax, vacc0x1);
    vacc1x1 = Q6_Vw_vmin_VwVw(vmax, vacc1x1);
    vacc0x2 = Q6_Vw_vmin_VwVw(vmax, vacc0x2);
    vacc1x2 = Q6_Vw_vmin_VwVw(vmax, vacc1x2);
    vacc0x3 = Q6_Vw_vmin_VwVw(vmax, vacc0x3);
    vacc1x3 = Q6_Vw_vmin_VwVw(vmax, vacc1x3);

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

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);

      nc -= 128;
    } else {
      if (nc & 64) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)(c0 + 32)) = vacc0x1;
        *((HVX_UVector *)c1) = vacc1x0;
        *((HVX_UVector *)(c1 + 32)) = vacc1x1;

        vacc0x0 = vacc0x2;
        vacc0x1 = vacc0x3;
        vacc1x0 = vacc1x2;
        vacc1x1 = vacc1x3;

        c0 += 64;
        c1 += 64;
        nc ^= 64;
      }
      if (nc & 32) {
        *((HVX_UVector *)c0) = vacc0x0;
        *((HVX_UVector *)c1) = vacc1x0;

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;

        c0 += 32;
        c1 += 32;
        nc ^= 32;
      }
      vstu_variable_scalar((char*)c0, nc*sizeof(float), vacc0x0);
      vstu_variable_scalar((char*)c1, nc*sizeof(float), vacc1x0);
      nc = 0;
    }
  } while (nc != 0);
}
