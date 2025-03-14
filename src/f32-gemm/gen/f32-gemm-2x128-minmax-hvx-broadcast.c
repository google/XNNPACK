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
    HVX_Vector vacc0x0 = xnn_load_f32(w + 0);
    HVX_Vector vacc0x1 = xnn_load_f32(w + 32);
    HVX_Vector vacc0x2 = xnn_load_f32(w + 64);
    HVX_Vector vacc0x3 = xnn_load_f32(w + 96);
    HVX_Vector vacc1x0 = vacc0x0;
    HVX_Vector vacc1x1 = vacc0x1;
    HVX_Vector vacc1x2 = vacc0x2;
    HVX_Vector vacc1x3 = vacc0x3;
    w += 128;

    size_t k = kc;
    do {
      const HVX_Vector va0 = xnn_set1_f32(*a0);
      a0 += 1;
      const HVX_Vector va1 = xnn_set1_f32(*a1);
      a1 += 1;

      const HVX_Vector vb0 = *((const HVX_Vector *)(w));
      const HVX_Vector vb1 = *((const HVX_Vector *)(w + 32));
      const HVX_Vector vb2 = *((const HVX_Vector *)(w + 64));
      const HVX_Vector vb3 = *((const HVX_Vector *)(w + 96));
      w += 128;

      vacc0x0 = xnn_fmadd_f32(va0, vb0, vacc0x0);
      vacc1x0 = xnn_fmadd_f32(va1, vb0, vacc1x0);
      vacc0x1 = xnn_fmadd_f32(va0, vb1, vacc0x1);
      vacc1x1 = xnn_fmadd_f32(va1, vb1, vacc1x1);
      vacc0x2 = xnn_fmadd_f32(va0, vb2, vacc0x2);
      vacc1x2 = xnn_fmadd_f32(va1, vb2, vacc1x2);
      vacc0x3 = xnn_fmadd_f32(va0, vb3, vacc0x3);
      vacc1x3 = xnn_fmadd_f32(va1, vb3, vacc1x3);

      k -= sizeof(float);
    } while (k != 0);

    HVX_Vector vmin = xnn_set1_f32(params->scalar.min);
    vacc0x0 = xnn_max_f32(vmin, vacc0x0);
    vacc1x0 = xnn_max_f32(vmin, vacc1x0);
    vacc0x1 = xnn_max_f32(vmin, vacc0x1);
    vacc1x1 = xnn_max_f32(vmin, vacc1x1);
    vacc0x2 = xnn_max_f32(vmin, vacc0x2);
    vacc1x2 = xnn_max_f32(vmin, vacc1x2);
    vacc0x3 = xnn_max_f32(vmin, vacc0x3);
    vacc1x3 = xnn_max_f32(vmin, vacc1x3);

    HVX_Vector vmax = xnn_set1_f32(params->scalar.max);
    vacc0x0 = xnn_min_f32(vmax, vacc0x0);
    vacc1x0 = xnn_min_f32(vmax, vacc1x0);
    vacc0x1 = xnn_min_f32(vmax, vacc0x1);
    vacc1x1 = xnn_min_f32(vmax, vacc1x1);
    vacc0x2 = xnn_min_f32(vmax, vacc0x2);
    vacc1x2 = xnn_min_f32(vmax, vacc1x2);
    vacc0x3 = xnn_min_f32(vmax, vacc0x3);
    vacc1x3 = xnn_min_f32(vmax, vacc1x3);

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
      xnn_store_tail_f32(c0, vacc0x0, nc);
      xnn_store_tail_f32(c1, vacc1x0, nc);
      nc = 0;
    }
  } while (nc != 0);
}
