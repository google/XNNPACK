// Auto-generated file. Do not edit!
//   Template: src/x32-packw/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <emmintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/prefetch.h"

void xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w0 = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const __m128 vb0123 = _mm_loadu_ps(b);
        const __m128 vb4567 = _mm_loadu_ps(b + 4);
        const __m128 vb89AB = _mm_loadu_ps(b + 8);
        const __m128 vbCDEF = _mm_loadu_ps(b + 12);
        b += 16;

        _mm_store_ps(packed_w, vb0123);
        _mm_store_ps(packed_w + 4, vb4567);
        _mm_store_ps(packed_w + 8, vb89AB);
        _mm_store_ps(packed_w + 12, vbCDEF);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        _mm_store_ps(packed_w + 8, vzero);
        _mm_store_ps(packed_w + 12, vzero);
      }
      packed_w += 16;

      const float* w1 = w0 + kc;
      const float* w2 = w1 + kc;
      const float* w3 = w2 + kc;
      const float* w4 = w3 + kc;
      const float* w5 = w4 + kc;
      const float* w6 = w5 + kc;
      const float* w7 = w6 + kc;
      const float* w8 = w7 + kc;
      const float* w9 = w8 + kc;
      const float* w10 = w9 + kc;
      const float* w11 = w10 + kc;
      const float* w12 = w11 + kc;
      const float* w13 = w12 + kc;
      const float* w14 = w13 + kc;
      const float* w15 = w14 + kc;
      xnn_prefetch_to_l1((const int8_t*) w0);
      xnn_prefetch_to_l1((const int8_t*) w0 + 64);
      xnn_prefetch_to_l1((const int8_t*) w1);
      xnn_prefetch_to_l1((const int8_t*) w1 + 64);
      xnn_prefetch_to_l1((const int8_t*) w2);
      xnn_prefetch_to_l1((const int8_t*) w2 + 64);
      xnn_prefetch_to_l1((const int8_t*) w3);
      xnn_prefetch_to_l1((const int8_t*) w3 + 64);
      xnn_prefetch_to_l1((const int8_t*) w4);
      xnn_prefetch_to_l1((const int8_t*) w4 + 64);
      xnn_prefetch_to_l1((const int8_t*) w5);
      xnn_prefetch_to_l1((const int8_t*) w5 + 64);
      xnn_prefetch_to_l1((const int8_t*) w6);
      xnn_prefetch_to_l1((const int8_t*) w6 + 64);
      xnn_prefetch_to_l1((const int8_t*) w7);
      xnn_prefetch_to_l1((const int8_t*) w7 + 64);
      xnn_prefetch_to_l1((const int8_t*) w8);
      xnn_prefetch_to_l1((const int8_t*) w8 + 64);
      xnn_prefetch_to_l1((const int8_t*) w9);
      xnn_prefetch_to_l1((const int8_t*) w9 + 64);
      xnn_prefetch_to_l1((const int8_t*) w10);
      xnn_prefetch_to_l1((const int8_t*) w10 + 64);
      xnn_prefetch_to_l1((const int8_t*) w11);
      xnn_prefetch_to_l1((const int8_t*) w11 + 64);
      xnn_prefetch_to_l1((const int8_t*) w12);
      xnn_prefetch_to_l1((const int8_t*) w12 + 64);
      xnn_prefetch_to_l1((const int8_t*) w13);
      xnn_prefetch_to_l1((const int8_t*) w13 + 64);
      xnn_prefetch_to_l1((const int8_t*) w14);
      xnn_prefetch_to_l1((const int8_t*) w14 + 64);
      xnn_prefetch_to_l1((const int8_t*) w15);
      xnn_prefetch_to_l1((const int8_t*) w15 + 64);

      size_t k = kc;
      // KC main loop multiple of 8
      for (; k >= 8; k -= 8) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        const __m128 v0x4567 = _mm_loadu_ps(w0 + 4);
        w0 += 8;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        const __m128 v1x4567 = _mm_loadu_ps(w1 + 4);
        w1 += 8;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        const __m128 v2x4567 = _mm_loadu_ps(w2 + 4);
        w2 += 8;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        const __m128 v3x4567 = _mm_loadu_ps(w3 + 4);
        w3 += 8;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        const __m128 v4x4567 = _mm_loadu_ps(w4 + 4);
        w4 += 8;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        const __m128 v5x4567 = _mm_loadu_ps(w5 + 4);
        w5 += 8;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        const __m128 v6x4567 = _mm_loadu_ps(w6 + 4);
        w6 += 8;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        const __m128 v7x4567 = _mm_loadu_ps(w7 + 4);
        w7 += 8;
        const __m128 v8x0123 = _mm_loadu_ps(w8);
        const __m128 v8x4567 = _mm_loadu_ps(w8 + 4);
        w8 += 8;
        const __m128 v9x0123 = _mm_loadu_ps(w9);
        const __m128 v9x4567 = _mm_loadu_ps(w9 + 4);
        w9 += 8;
        const __m128 vAx0123 = _mm_loadu_ps(w10);
        const __m128 vAx4567 = _mm_loadu_ps(w10 + 4);
        w10 += 8;
        const __m128 vBx0123 = _mm_loadu_ps(w11);
        const __m128 vBx4567 = _mm_loadu_ps(w11 + 4);
        w11 += 8;
        const __m128 vCx0123 = _mm_loadu_ps(w12);
        const __m128 vCx4567 = _mm_loadu_ps(w12 + 4);
        w12 += 8;
        const __m128 vDx0123 = _mm_loadu_ps(w13);
        const __m128 vDx4567 = _mm_loadu_ps(w13 + 4);
        w13 += 8;
        const __m128 vEx0123 = _mm_loadu_ps(w14);
        const __m128 vEx4567 = _mm_loadu_ps(w14 + 4);
        w14 += 8;
        const __m128 vFx0123 = _mm_loadu_ps(w15);
        const __m128 vFx4567 = _mm_loadu_ps(w15 + 4);
        w15 += 8;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v01x4_01x5 = _mm_unpacklo_ps(v0x4567, v1x4567);
        const __m128 v23x4_23x5 = _mm_unpacklo_ps(v2x4567, v3x4567);
        const __m128 v01x6_01x7 = _mm_unpackhi_ps(v0x4567, v1x4567);
        const __m128 v23x6_23x7 = _mm_unpackhi_ps(v2x4567, v3x4567);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v45x4_45x5 = _mm_unpacklo_ps(v4x4567, v5x4567);
        const __m128 v67x4_67x5 = _mm_unpacklo_ps(v6x4567, v7x4567);
        const __m128 v45x6_45x7 = _mm_unpackhi_ps(v4x4567, v5x4567);
        const __m128 v67x6_67x7 = _mm_unpackhi_ps(v6x4567, v7x4567);
        const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x0123, v9x0123);
        const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx0123, vBx0123);
        const __m128 v89x2_89x3 = _mm_unpackhi_ps(v8x0123, v9x0123);
        const __m128 vABx2_ABx3 = _mm_unpackhi_ps(vAx0123, vBx0123);
        const __m128 v89x4_89x5 = _mm_unpacklo_ps(v8x4567, v9x4567);
        const __m128 vABx4_ABx5 = _mm_unpacklo_ps(vAx4567, vBx4567);
        const __m128 v89x6_89x7 = _mm_unpackhi_ps(v8x4567, v9x4567);
        const __m128 vABx6_ABx7 = _mm_unpackhi_ps(vAx4567, vBx4567);
        const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx0123, vDx0123);
        const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx0123, vFx0123);
        const __m128 vCDx2_CDx3 = _mm_unpackhi_ps(vCx0123, vDx0123);
        const __m128 vEFx2_EFx3 = _mm_unpackhi_ps(vEx0123, vFx0123);
        const __m128 vCDx4_CDx5 = _mm_unpacklo_ps(vCx4567, vDx4567);
        const __m128 vEFx4_EFx5 = _mm_unpacklo_ps(vEx4567, vFx4567);
        const __m128 vCDx6_CDx7 = _mm_unpackhi_ps(vCx4567, vDx4567);
        const __m128 vEFx6_EFx7 = _mm_unpackhi_ps(vEx4567, vFx4567);
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v0123x4 = _mm_movelh_ps(v01x4_01x5, v23x4_23x5);
        const __m128 v0123x5 = _mm_movehl_ps(v23x4_23x5, v01x4_01x5);
        const __m128 v0123x6 = _mm_movelh_ps(v01x6_01x7, v23x6_23x7);
        const __m128 v0123x7 = _mm_movehl_ps(v23x6_23x7, v01x6_01x7);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);
        const __m128 v4567x4 = _mm_movelh_ps(v45x4_45x5, v67x4_67x5);
        const __m128 v4567x5 = _mm_movehl_ps(v67x4_67x5, v45x4_45x5);
        const __m128 v4567x6 = _mm_movelh_ps(v45x6_45x7, v67x6_67x7);
        const __m128 v4567x7 = _mm_movehl_ps(v67x6_67x7, v45x6_45x7);
        const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
        const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
        const __m128 v89ABx2 = _mm_movelh_ps(v89x2_89x3, vABx2_ABx3);
        const __m128 v89ABx3 = _mm_movehl_ps(vABx2_ABx3, v89x2_89x3);
        const __m128 v89ABx4 = _mm_movelh_ps(v89x4_89x5, vABx4_ABx5);
        const __m128 v89ABx5 = _mm_movehl_ps(vABx4_ABx5, v89x4_89x5);
        const __m128 v89ABx6 = _mm_movelh_ps(v89x6_89x7, vABx6_ABx7);
        const __m128 v89ABx7 = _mm_movehl_ps(vABx6_ABx7, v89x6_89x7);
        const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
        const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
        const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2_CDx3, vEFx2_EFx3);
        const __m128 vCDEFx3 = _mm_movehl_ps(vEFx2_EFx3, vCDx2_CDx3);
        const __m128 vCDEFx4 = _mm_movelh_ps(vCDx4_CDx5, vEFx4_EFx5);
        const __m128 vCDEFx5 = _mm_movehl_ps(vEFx4_EFx5, vCDx4_CDx5);
        const __m128 vCDEFx6 = _mm_movelh_ps(vCDx6_CDx7, vEFx6_EFx7);
        const __m128 vCDEFx7 = _mm_movehl_ps(vEFx6_EFx7, vCDx6_CDx7);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v89ABx0);
        _mm_store_ps(packed_w + 12, vCDEFx0);
        _mm_store_ps(packed_w + 16, v0123x1);
        _mm_store_ps(packed_w + 20, v4567x1);
        _mm_store_ps(packed_w + 24, v89ABx1);
        _mm_store_ps(packed_w + 28, vCDEFx1);
        _mm_store_ps(packed_w + 32, v0123x2);
        _mm_store_ps(packed_w + 36, v4567x2);
        _mm_store_ps(packed_w + 40, v89ABx2);
        _mm_store_ps(packed_w + 44, vCDEFx2);
        _mm_store_ps(packed_w + 48, v0123x3);
        _mm_store_ps(packed_w + 52, v4567x3);
        _mm_store_ps(packed_w + 56, v89ABx3);
        _mm_store_ps(packed_w + 60, vCDEFx3);
        _mm_store_ps(packed_w + 64, v0123x4);
        _mm_store_ps(packed_w + 68, v4567x4);
        _mm_store_ps(packed_w + 72, v89ABx4);
        _mm_store_ps(packed_w + 76, vCDEFx4);
        _mm_store_ps(packed_w + 80, v0123x5);
        _mm_store_ps(packed_w + 84, v4567x5);
        _mm_store_ps(packed_w + 88, v89ABx5);
        _mm_store_ps(packed_w + 92, vCDEFx5);
        _mm_store_ps(packed_w + 96, v0123x6);
        _mm_store_ps(packed_w + 100, v4567x6);
        _mm_store_ps(packed_w + 104, v89ABx6);
        _mm_store_ps(packed_w + 108, vCDEFx6);
        _mm_store_ps(packed_w + 112, v0123x7);
        _mm_store_ps(packed_w + 116, v4567x7);
        _mm_store_ps(packed_w + 120, v89ABx7);
        _mm_store_ps(packed_w + 124, vCDEFx7);
        packed_w += 128;
      }

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;
        const __m128 v8x0123 = _mm_loadu_ps(w8);
        w8 += 4;
        const __m128 v9x0123 = _mm_loadu_ps(w9);
        w9 += 4;
        const __m128 vAx0123 = _mm_loadu_ps(w10);
        w10 += 4;
        const __m128 vBx0123 = _mm_loadu_ps(w11);
        w11 += 4;
        const __m128 vCx0123 = _mm_loadu_ps(w12);
        w12 += 4;
        const __m128 vDx0123 = _mm_loadu_ps(w13);
        w13 += 4;
        const __m128 vEx0123 = _mm_loadu_ps(w14);
        w14 += 4;
        const __m128 vFx0123 = _mm_loadu_ps(w15);
        w15 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x0123, v9x0123);
        const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx0123, vBx0123);
        const __m128 v89x2_89x3 = _mm_unpackhi_ps(v8x0123, v9x0123);
        const __m128 vABx2_ABx3 = _mm_unpackhi_ps(vAx0123, vBx0123);
        const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx0123, vDx0123);
        const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx0123, vFx0123);
        const __m128 vCDx2_CDx3 = _mm_unpackhi_ps(vCx0123, vDx0123);
        const __m128 vEFx2_EFx3 = _mm_unpackhi_ps(vEx0123, vFx0123);
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        xnn_prefetch_to_l1((const int8_t*) w15 + 128);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);
        const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
        const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
        const __m128 v89ABx2 = _mm_movelh_ps(v89x2_89x3, vABx2_ABx3);
        const __m128 v89ABx3 = _mm_movehl_ps(vABx2_ABx3, v89x2_89x3);
        const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
        const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
        const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2_CDx3, vEFx2_EFx3);
        const __m128 vCDEFx3 = _mm_movehl_ps(vEFx2_EFx3, vCDx2_CDx3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v89ABx0);
        _mm_store_ps(packed_w + 12, vCDEFx0);
        _mm_store_ps(packed_w + 16, v0123x1);
        _mm_store_ps(packed_w + 20, v4567x1);
        _mm_store_ps(packed_w + 24, v89ABx1);
        _mm_store_ps(packed_w + 28, vCDEFx1);
        _mm_store_ps(packed_w + 32, v0123x2);
        _mm_store_ps(packed_w + 36, v4567x2);
        _mm_store_ps(packed_w + 40, v89ABx2);
        _mm_store_ps(packed_w + 44, vCDEFx2);
        _mm_store_ps(packed_w + 48, v0123x3);
        _mm_store_ps(packed_w + 52, v4567x3);
        _mm_store_ps(packed_w + 56, v89ABx3);
        _mm_store_ps(packed_w + 60, vCDEFx3);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            w0 += 1;
            const __m128 v1x0 = _mm_load_ss(w1);
            w1 += 1;
            const __m128 v2x0 = _mm_load_ss(w2);
            w2 += 1;
            const __m128 v3x0 = _mm_load_ss(w3);
            w3 += 1;
            const __m128 v4x0 = _mm_load_ss(w4);
            w4 += 1;
            const __m128 v5x0 = _mm_load_ss(w5);
            w5 += 1;
            const __m128 v6x0 = _mm_load_ss(w6);
            w6 += 1;
            const __m128 v7x0 = _mm_load_ss(w7);
            w7 += 1;
            const __m128 v8x0 = _mm_load_ss(w8);
            w8 += 1;
            const __m128 v9x0 = _mm_load_ss(w9);
            w9 += 1;
            const __m128 vAx0 = _mm_load_ss(w10);
            w10 += 1;
            const __m128 vBx0 = _mm_load_ss(w11);
            w11 += 1;
            const __m128 vCx0 = _mm_load_ss(w12);
            w12 += 1;
            const __m128 vDx0 = _mm_load_ss(w13);
            w13 += 1;
            const __m128 vEx0 = _mm_load_ss(w14);
            w14 += 1;
            const __m128 vFx0 = _mm_load_ss(w15);
            w15 += 1;

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v7x0);
            const __m128 v89x0 = _mm_unpacklo_ps(v8x0, v9x0);
            const __m128 vABx0 = _mm_unpacklo_ps(vAx0, vBx0);
            const __m128 vCDx0 = _mm_unpacklo_ps(vCx0, vDx0);
            const __m128 vEFx0 = _mm_unpacklo_ps(vEx0, vFx0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0, vABx0);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0, vEFx0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            packed_w += 16;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            w0 += 2;
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            w1 += 2;
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            w2 += 2;
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            w3 += 2;
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            w4 += 2;
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            w5 += 2;
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
            w6 += 2;
            const __m128 v7x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
            w7 += 2;
            const __m128 v8x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
            w8 += 2;
            const __m128 v9x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
            w9 += 2;
            const __m128 vAx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
            w10 += 2;
            const __m128 vBx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
            w11 += 2;
            const __m128 vCx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
            w12 += 2;
            const __m128 vDx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
            w13 += 2;
            const __m128 vEx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));
            w14 += 2;
            const __m128 vFx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w15));
            w15 += 2;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v7x01);
            const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x01, v9x01);
            const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx01, vBx01);
            const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx01, vDx01);
            const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx01, vFx01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
            const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
            const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            _mm_store_ps(packed_w + 16, v0123x1);
            _mm_store_ps(packed_w + 20, v4567x1);
            _mm_store_ps(packed_w + 24, v89ABx1);
            _mm_store_ps(packed_w + 28, vCDEFx1);
            packed_w += 32;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);
            __m128 v7x012 = _mm_load_ss(w7 + 2);
            __m128 v8x012 = _mm_load_ss(w8 + 2);
            __m128 v9x012 = _mm_load_ss(w9 + 2);
            __m128 vAx012 = _mm_load_ss(w10 + 2);
            __m128 vBx012 = _mm_load_ss(w11 + 2);
            __m128 vCx012 = _mm_load_ss(w12 + 2);
            __m128 vDx012 = _mm_load_ss(w13 + 2);
            __m128 vEx012 = _mm_load_ss(w14 + 2);
            __m128 vFx012 = _mm_load_ss(w15 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);
            v7x012 = _mm_movelh_ps(v7x012, v7x012);
            v8x012 = _mm_movelh_ps(v8x012, v8x012);
            v9x012 = _mm_movelh_ps(v9x012, v9x012);
            vAx012 = _mm_movelh_ps(vAx012, vAx012);
            vBx012 = _mm_movelh_ps(vBx012, vBx012);
            vCx012 = _mm_movelh_ps(vCx012, vCx012);
            vDx012 = _mm_movelh_ps(vDx012, vDx012);
            vEx012 = _mm_movelh_ps(vEx012, vEx012);
            vFx012 = _mm_movelh_ps(vFx012, vFx012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            w0 += 3;
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            w1 += 3;
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            w2 += 3;
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            w3 += 3;
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            w4 += 3;
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            w5 += 3;
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);
            w6 += 3;
            v7x012 = _mm_loadl_pi(v7x012, (const __m64*) w7);
            w7 += 3;
            v8x012 = _mm_loadl_pi(v8x012, (const __m64*) w8);
            w8 += 3;
            v9x012 = _mm_loadl_pi(v9x012, (const __m64*) w9);
            w9 += 3;
            vAx012 = _mm_loadl_pi(vAx012, (const __m64*) w10);
            w10 += 3;
            vBx012 = _mm_loadl_pi(vBx012, (const __m64*) w11);
            w11 += 3;
            vCx012 = _mm_loadl_pi(vCx012, (const __m64*) w12);
            w12 += 3;
            vDx012 = _mm_loadl_pi(vDx012, (const __m64*) w13);
            w13 += 3;
            vEx012 = _mm_loadl_pi(vEx012, (const __m64*) w14);
            w14 += 3;
            vFx012 = _mm_loadl_pi(vFx012, (const __m64*) w15);
            w15 += 3;

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v7x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v7x012);
            const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x012, v9x012);
            const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx012, vBx012);
            const __m128 v89x2 = _mm_unpackhi_ps(v8x012, v9x012);
            const __m128 vABx2 = _mm_unpackhi_ps(vAx012, vBx012);
            const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx012, vDx012);
            const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx012, vFx012);
            const __m128 vCDx2 = _mm_unpackhi_ps(vCx012, vDx012);
            const __m128 vEFx2 = _mm_unpackhi_ps(vEx012, vFx012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
            const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
            const __m128 v89ABx2 = _mm_movelh_ps(v89x2, vABx2);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
            const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
            const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2, vEFx2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            _mm_store_ps(packed_w + 16, v0123x1);
            _mm_store_ps(packed_w + 20, v4567x1);
            _mm_store_ps(packed_w + 24, v89ABx1);
            _mm_store_ps(packed_w + 28, vCDEFx1);
            _mm_store_ps(packed_w + 32, v0123x2);
            _mm_store_ps(packed_w + 36, v4567x2);
            _mm_store_ps(packed_w + 40, v89ABx2);
            _mm_store_ps(packed_w + 44, vCDEFx2);
            packed_w += 48;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
      w0 = w15;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *packed_w++  = *b++;
        } while (--nb != 0);
        packed_w += (16 - n);
      } else {
        const __m128 vzero = _mm_setzero_ps();
        _mm_store_ps(packed_w, vzero);
        _mm_store_ps(packed_w + 4, vzero);
        _mm_store_ps(packed_w + 8, vzero);
        _mm_store_ps(packed_w + 12, vzero);
        packed_w += 16;
      }

      const float* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const float* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const float* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const float* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const float* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const float* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const float* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const float* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const float* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const float* w10 = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const float* w11 = w10 + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const float* w12 = w11 + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const float* w13 = w12 + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const float* w14 = w13 + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }

      size_t k = kc;
      // KC main loop multiple of 8
      for (; k >= 8; k -= 8) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        const __m128 v0x4567 = _mm_loadu_ps(w0 + 4);
        w0 += 8;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        const __m128 v1x4567 = _mm_loadu_ps(w1 + 4);
        w1 += 8;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        const __m128 v2x4567 = _mm_loadu_ps(w2 + 4);
        w2 += 8;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        const __m128 v3x4567 = _mm_loadu_ps(w3 + 4);
        w3 += 8;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        const __m128 v4x4567 = _mm_loadu_ps(w4 + 4);
        w4 += 8;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        const __m128 v5x4567 = _mm_loadu_ps(w5 + 4);
        w5 += 8;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        const __m128 v6x4567 = _mm_loadu_ps(w6 + 4);
        w6 += 8;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        const __m128 v7x4567 = _mm_loadu_ps(w7 + 4);
        w7 += 8;
        const __m128 v8x0123 = _mm_loadu_ps(w8);
        const __m128 v8x4567 = _mm_loadu_ps(w8 + 4);
        w8 += 8;
        const __m128 v9x0123 = _mm_loadu_ps(w9);
        const __m128 v9x4567 = _mm_loadu_ps(w9 + 4);
        w9 += 8;
        const __m128 vAx0123 = _mm_loadu_ps(w10);
        const __m128 vAx4567 = _mm_loadu_ps(w10 + 4);
        w10 += 8;
        const __m128 vBx0123 = _mm_loadu_ps(w11);
        const __m128 vBx4567 = _mm_loadu_ps(w11 + 4);
        w11 += 8;
        const __m128 vCx0123 = _mm_loadu_ps(w12);
        const __m128 vCx4567 = _mm_loadu_ps(w12 + 4);
        w12 += 8;
        const __m128 vDx0123 = _mm_loadu_ps(w13);
        const __m128 vDx4567 = _mm_loadu_ps(w13 + 4);
        w13 += 8;
        const __m128 vEx0123 = _mm_loadu_ps(w14);
        const __m128 vEx4567 = _mm_loadu_ps(w14 + 4);
        w14 += 8;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v01x4_01x5 = _mm_unpacklo_ps(v0x4567, v1x4567);
        const __m128 v23x4_23x5 = _mm_unpacklo_ps(v2x4567, v3x4567);
        const __m128 v01x6_01x7 = _mm_unpackhi_ps(v0x4567, v1x4567);
        const __m128 v23x6_23x7 = _mm_unpackhi_ps(v2x4567, v3x4567);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v45x4_45x5 = _mm_unpacklo_ps(v4x4567, v5x4567);
        const __m128 v67x4_67x5 = _mm_unpacklo_ps(v6x4567, v7x4567);
        const __m128 v45x6_45x7 = _mm_unpackhi_ps(v4x4567, v5x4567);
        const __m128 v67x6_67x7 = _mm_unpackhi_ps(v6x4567, v7x4567);
        const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x0123, v9x0123);
        const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx0123, vBx0123);
        const __m128 v89x2_89x3 = _mm_unpackhi_ps(v8x0123, v9x0123);
        const __m128 vABx2_ABx3 = _mm_unpackhi_ps(vAx0123, vBx0123);
        const __m128 v89x4_89x5 = _mm_unpacklo_ps(v8x4567, v9x4567);
        const __m128 vABx4_ABx5 = _mm_unpacklo_ps(vAx4567, vBx4567);
        const __m128 v89x6_89x7 = _mm_unpackhi_ps(v8x4567, v9x4567);
        const __m128 vABx6_ABx7 = _mm_unpackhi_ps(vAx4567, vBx4567);
        const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx0123, vDx0123);
        const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx0123, vEx0123);
        const __m128 vCDx2_CDx3 = _mm_unpackhi_ps(vCx0123, vDx0123);
        const __m128 vEFx2_EFx3 = _mm_unpackhi_ps(vEx0123, vEx0123);
        const __m128 vCDx4_CDx5 = _mm_unpacklo_ps(vCx4567, vDx4567);
        const __m128 vEFx4_EFx5 = _mm_unpacklo_ps(vEx4567, vEx4567);
        const __m128 vCDx6_CDx7 = _mm_unpackhi_ps(vCx4567, vDx4567);
        const __m128 vEFx6_EFx7 = _mm_unpackhi_ps(vEx4567, vEx4567);
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v0123x4 = _mm_movelh_ps(v01x4_01x5, v23x4_23x5);
        const __m128 v0123x5 = _mm_movehl_ps(v23x4_23x5, v01x4_01x5);
        const __m128 v0123x6 = _mm_movelh_ps(v01x6_01x7, v23x6_23x7);
        const __m128 v0123x7 = _mm_movehl_ps(v23x6_23x7, v01x6_01x7);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);
        const __m128 v4567x4 = _mm_movelh_ps(v45x4_45x5, v67x4_67x5);
        const __m128 v4567x5 = _mm_movehl_ps(v67x4_67x5, v45x4_45x5);
        const __m128 v4567x6 = _mm_movelh_ps(v45x6_45x7, v67x6_67x7);
        const __m128 v4567x7 = _mm_movehl_ps(v67x6_67x7, v45x6_45x7);
        const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
        const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
        const __m128 v89ABx2 = _mm_movelh_ps(v89x2_89x3, vABx2_ABx3);
        const __m128 v89ABx3 = _mm_movehl_ps(vABx2_ABx3, v89x2_89x3);
        const __m128 v89ABx4 = _mm_movelh_ps(v89x4_89x5, vABx4_ABx5);
        const __m128 v89ABx5 = _mm_movehl_ps(vABx4_ABx5, v89x4_89x5);
        const __m128 v89ABx6 = _mm_movelh_ps(v89x6_89x7, vABx6_ABx7);
        const __m128 v89ABx7 = _mm_movehl_ps(vABx6_ABx7, v89x6_89x7);
        const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
        const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
        const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2_CDx3, vEFx2_EFx3);
        const __m128 vCDEFx3 = _mm_movehl_ps(vEFx2_EFx3, vCDx2_CDx3);
        const __m128 vCDEFx4 = _mm_movelh_ps(vCDx4_CDx5, vEFx4_EFx5);
        const __m128 vCDEFx5 = _mm_movehl_ps(vEFx4_EFx5, vCDx4_CDx5);
        const __m128 vCDEFx6 = _mm_movelh_ps(vCDx6_CDx7, vEFx6_EFx7);
        const __m128 vCDEFx7 = _mm_movehl_ps(vEFx6_EFx7, vCDx6_CDx7);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v89ABx0);
        _mm_store_ps(packed_w + 12, vCDEFx0);
        _mm_store_ps(packed_w + 16, v0123x1);
        _mm_store_ps(packed_w + 20, v4567x1);
        _mm_store_ps(packed_w + 24, v89ABx1);
        _mm_store_ps(packed_w + 28, vCDEFx1);
        _mm_store_ps(packed_w + 32, v0123x2);
        _mm_store_ps(packed_w + 36, v4567x2);
        _mm_store_ps(packed_w + 40, v89ABx2);
        _mm_store_ps(packed_w + 44, vCDEFx2);
        _mm_store_ps(packed_w + 48, v0123x3);
        _mm_store_ps(packed_w + 52, v4567x3);
        _mm_store_ps(packed_w + 56, v89ABx3);
        _mm_store_ps(packed_w + 60, vCDEFx3);
        _mm_store_ps(packed_w + 64, v0123x4);
        _mm_store_ps(packed_w + 68, v4567x4);
        _mm_store_ps(packed_w + 72, v89ABx4);
        _mm_store_ps(packed_w + 76, vCDEFx4);
        _mm_store_ps(packed_w + 80, v0123x5);
        _mm_store_ps(packed_w + 84, v4567x5);
        _mm_store_ps(packed_w + 88, v89ABx5);
        _mm_store_ps(packed_w + 92, vCDEFx5);
        _mm_store_ps(packed_w + 96, v0123x6);
        _mm_store_ps(packed_w + 100, v4567x6);
        _mm_store_ps(packed_w + 104, v89ABx6);
        _mm_store_ps(packed_w + 108, vCDEFx6);
        _mm_store_ps(packed_w + 112, v0123x7);
        _mm_store_ps(packed_w + 116, v4567x7);
        _mm_store_ps(packed_w + 120, v89ABx7);
        _mm_store_ps(packed_w + 124, vCDEFx7);
        packed_w += 128;
      }

      // KC multiple of 4
      for (; k >= 4; k -= 4) {
        const __m128 v0x0123 = _mm_loadu_ps(w0);
        w0 += 4;
        const __m128 v1x0123 = _mm_loadu_ps(w1);
        w1 += 4;
        const __m128 v2x0123 = _mm_loadu_ps(w2);
        w2 += 4;
        const __m128 v3x0123 = _mm_loadu_ps(w3);
        w3 += 4;
        const __m128 v4x0123 = _mm_loadu_ps(w4);
        w4 += 4;
        const __m128 v5x0123 = _mm_loadu_ps(w5);
        w5 += 4;
        const __m128 v6x0123 = _mm_loadu_ps(w6);
        w6 += 4;
        const __m128 v7x0123 = _mm_loadu_ps(w7);
        w7 += 4;
        const __m128 v8x0123 = _mm_loadu_ps(w8);
        w8 += 4;
        const __m128 v9x0123 = _mm_loadu_ps(w9);
        w9 += 4;
        const __m128 vAx0123 = _mm_loadu_ps(w10);
        w10 += 4;
        const __m128 vBx0123 = _mm_loadu_ps(w11);
        w11 += 4;
        const __m128 vCx0123 = _mm_loadu_ps(w12);
        w12 += 4;
        const __m128 vDx0123 = _mm_loadu_ps(w13);
        w13 += 4;
        const __m128 vEx0123 = _mm_loadu_ps(w14);
        w14 += 4;

        const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x0123, v1x0123);
        const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x0123, v3x0123);
        const __m128 v01x2_01x3 = _mm_unpackhi_ps(v0x0123, v1x0123);
        const __m128 v23x2_23x3 = _mm_unpackhi_ps(v2x0123, v3x0123);
        const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x0123, v5x0123);
        const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x0123, v7x0123);
        const __m128 v45x2_45x3 = _mm_unpackhi_ps(v4x0123, v5x0123);
        const __m128 v67x2_67x3 = _mm_unpackhi_ps(v6x0123, v7x0123);
        const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x0123, v9x0123);
        const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx0123, vBx0123);
        const __m128 v89x2_89x3 = _mm_unpackhi_ps(v8x0123, v9x0123);
        const __m128 vABx2_ABx3 = _mm_unpackhi_ps(vAx0123, vBx0123);
        const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx0123, vDx0123);
        const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx0123, vEx0123);
        const __m128 vCDx2_CDx3 = _mm_unpackhi_ps(vCx0123, vDx0123);
        const __m128 vEFx2_EFx3 = _mm_unpackhi_ps(vEx0123, vEx0123);
        xnn_prefetch_to_l1((const int8_t*) w0 + 128);
        xnn_prefetch_to_l1((const int8_t*) w1 + 128);
        xnn_prefetch_to_l1((const int8_t*) w2 + 128);
        xnn_prefetch_to_l1((const int8_t*) w3 + 128);
        xnn_prefetch_to_l1((const int8_t*) w4 + 128);
        xnn_prefetch_to_l1((const int8_t*) w5 + 128);
        xnn_prefetch_to_l1((const int8_t*) w6 + 128);
        xnn_prefetch_to_l1((const int8_t*) w7 + 128);
        xnn_prefetch_to_l1((const int8_t*) w8 + 128);
        xnn_prefetch_to_l1((const int8_t*) w9 + 128);
        xnn_prefetch_to_l1((const int8_t*) w10 + 128);
        xnn_prefetch_to_l1((const int8_t*) w11 + 128);
        xnn_prefetch_to_l1((const int8_t*) w12 + 128);
        xnn_prefetch_to_l1((const int8_t*) w13 + 128);
        xnn_prefetch_to_l1((const int8_t*) w14 + 128);
        const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
        const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
        const __m128 v0123x2 = _mm_movelh_ps(v01x2_01x3, v23x2_23x3);
        const __m128 v0123x3 = _mm_movehl_ps(v23x2_23x3, v01x2_01x3);
        const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
        const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
        const __m128 v4567x2 = _mm_movelh_ps(v45x2_45x3, v67x2_67x3);
        const __m128 v4567x3 = _mm_movehl_ps(v67x2_67x3, v45x2_45x3);
        const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
        const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
        const __m128 v89ABx2 = _mm_movelh_ps(v89x2_89x3, vABx2_ABx3);
        const __m128 v89ABx3 = _mm_movehl_ps(vABx2_ABx3, v89x2_89x3);
        const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
        const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
        const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2_CDx3, vEFx2_EFx3);
        const __m128 vCDEFx3 = _mm_movehl_ps(vEFx2_EFx3, vCDx2_CDx3);

        _mm_store_ps(packed_w, v0123x0);
        _mm_store_ps(packed_w + 4, v4567x0);
        _mm_store_ps(packed_w + 8, v89ABx0);
        _mm_store_ps(packed_w + 12, vCDEFx0);
        _mm_store_ps(packed_w + 16, v0123x1);
        _mm_store_ps(packed_w + 20, v4567x1);
        _mm_store_ps(packed_w + 24, v89ABx1);
        _mm_store_ps(packed_w + 28, vCDEFx1);
        _mm_store_ps(packed_w + 32, v0123x2);
        _mm_store_ps(packed_w + 36, v4567x2);
        _mm_store_ps(packed_w + 40, v89ABx2);
        _mm_store_ps(packed_w + 44, vCDEFx2);
        _mm_store_ps(packed_w + 48, v0123x3);
        _mm_store_ps(packed_w + 52, v4567x3);
        _mm_store_ps(packed_w + 56, v89ABx3);
        _mm_store_ps(packed_w + 60, vCDEFx3);
        packed_w += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            const __m128 v0x0 = _mm_load_ss(w0);
            const __m128 v1x0 = _mm_load_ss(w1);
            const __m128 v2x0 = _mm_load_ss(w2);
            const __m128 v3x0 = _mm_load_ss(w3);
            const __m128 v4x0 = _mm_load_ss(w4);
            const __m128 v5x0 = _mm_load_ss(w5);
            const __m128 v6x0 = _mm_load_ss(w6);
            const __m128 v7x0 = _mm_load_ss(w7);
            const __m128 v8x0 = _mm_load_ss(w8);
            const __m128 v9x0 = _mm_load_ss(w9);
            const __m128 vAx0 = _mm_load_ss(w10);
            const __m128 vBx0 = _mm_load_ss(w11);
            const __m128 vCx0 = _mm_load_ss(w12);
            const __m128 vDx0 = _mm_load_ss(w13);
            const __m128 vEx0 = _mm_load_ss(w14);

            const __m128 v01x0 = _mm_unpacklo_ps(v0x0, v1x0);
            const __m128 v23x0 = _mm_unpacklo_ps(v2x0, v3x0);
            const __m128 v45x0 = _mm_unpacklo_ps(v4x0, v5x0);
            const __m128 v67x0 = _mm_unpacklo_ps(v6x0, v7x0);
            const __m128 v89x0 = _mm_unpacklo_ps(v8x0, v9x0);
            const __m128 vABx0 = _mm_unpacklo_ps(vAx0, vBx0);
            const __m128 vCDx0 = _mm_unpacklo_ps(vCx0, vDx0);
            const __m128 vEFx0 = _mm_unpacklo_ps(vEx0, vEx0);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0, v23x0);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0, v67x0);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0, vABx0);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0, vEFx0);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            packed_w += 16;
            break;
          }
          case 2:
          {
            const __m128 v0x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w0));
            const __m128 v1x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w1));
            const __m128 v2x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w2));
            const __m128 v3x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w3));
            const __m128 v4x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w4));
            const __m128 v5x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w5));
            const __m128 v6x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w6));
            const __m128 v7x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w7));
            const __m128 v8x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w8));
            const __m128 v9x01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w9));
            const __m128 vAx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w10));
            const __m128 vBx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w11));
            const __m128 vCx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w12));
            const __m128 vDx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w13));
            const __m128 vEx01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) w14));

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x01, v1x01);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x01, v3x01);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x01, v5x01);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x01, v7x01);
            const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x01, v9x01);
            const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx01, vBx01);
            const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx01, vDx01);
            const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx01, vEx01);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
            const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
            const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            _mm_store_ps(packed_w + 16, v0123x1);
            _mm_store_ps(packed_w + 20, v4567x1);
            _mm_store_ps(packed_w + 24, v89ABx1);
            _mm_store_ps(packed_w + 28, vCDEFx1);
            packed_w += 32;
            break;
          }
          case 3:
          {
            __m128 v0x012 = _mm_load_ss(w0 + 2);
            __m128 v1x012 = _mm_load_ss(w1 + 2);
            __m128 v2x012 = _mm_load_ss(w2 + 2);
            __m128 v3x012 = _mm_load_ss(w3 + 2);
            __m128 v4x012 = _mm_load_ss(w4 + 2);
            __m128 v5x012 = _mm_load_ss(w5 + 2);
            __m128 v6x012 = _mm_load_ss(w6 + 2);
            __m128 v7x012 = _mm_load_ss(w7 + 2);
            __m128 v8x012 = _mm_load_ss(w8 + 2);
            __m128 v9x012 = _mm_load_ss(w9 + 2);
            __m128 vAx012 = _mm_load_ss(w10 + 2);
            __m128 vBx012 = _mm_load_ss(w11 + 2);
            __m128 vCx012 = _mm_load_ss(w12 + 2);
            __m128 vDx012 = _mm_load_ss(w13 + 2);
            __m128 vEx012 = _mm_load_ss(w14 + 2);

            v0x012 = _mm_movelh_ps(v0x012, v0x012);
            v1x012 = _mm_movelh_ps(v1x012, v1x012);
            v2x012 = _mm_movelh_ps(v2x012, v2x012);
            v3x012 = _mm_movelh_ps(v3x012, v3x012);
            v4x012 = _mm_movelh_ps(v4x012, v4x012);
            v5x012 = _mm_movelh_ps(v5x012, v5x012);
            v6x012 = _mm_movelh_ps(v6x012, v6x012);
            v7x012 = _mm_movelh_ps(v7x012, v7x012);
            v8x012 = _mm_movelh_ps(v8x012, v8x012);
            v9x012 = _mm_movelh_ps(v9x012, v9x012);
            vAx012 = _mm_movelh_ps(vAx012, vAx012);
            vBx012 = _mm_movelh_ps(vBx012, vBx012);
            vCx012 = _mm_movelh_ps(vCx012, vCx012);
            vDx012 = _mm_movelh_ps(vDx012, vDx012);
            vEx012 = _mm_movelh_ps(vEx012, vEx012);

            v0x012 = _mm_loadl_pi(v0x012, (const __m64*) w0);
            v1x012 = _mm_loadl_pi(v1x012, (const __m64*) w1);
            v2x012 = _mm_loadl_pi(v2x012, (const __m64*) w2);
            v3x012 = _mm_loadl_pi(v3x012, (const __m64*) w3);
            v4x012 = _mm_loadl_pi(v4x012, (const __m64*) w4);
            v5x012 = _mm_loadl_pi(v5x012, (const __m64*) w5);
            v6x012 = _mm_loadl_pi(v6x012, (const __m64*) w6);
            v7x012 = _mm_loadl_pi(v7x012, (const __m64*) w7);
            v8x012 = _mm_loadl_pi(v8x012, (const __m64*) w8);
            v9x012 = _mm_loadl_pi(v9x012, (const __m64*) w9);
            vAx012 = _mm_loadl_pi(vAx012, (const __m64*) w10);
            vBx012 = _mm_loadl_pi(vBx012, (const __m64*) w11);
            vCx012 = _mm_loadl_pi(vCx012, (const __m64*) w12);
            vDx012 = _mm_loadl_pi(vDx012, (const __m64*) w13);
            vEx012 = _mm_loadl_pi(vEx012, (const __m64*) w14);

            const __m128 v01x0_01x1 = _mm_unpacklo_ps(v0x012, v1x012);
            const __m128 v23x0_23x1 = _mm_unpacklo_ps(v2x012, v3x012);
            const __m128 v01x2 = _mm_unpackhi_ps(v0x012, v1x012);
            const __m128 v23x2 = _mm_unpackhi_ps(v2x012, v3x012);
            const __m128 v45x0_45x1 = _mm_unpacklo_ps(v4x012, v5x012);
            const __m128 v67x0_67x1 = _mm_unpacklo_ps(v6x012, v7x012);
            const __m128 v45x2 = _mm_unpackhi_ps(v4x012, v5x012);
            const __m128 v67x2 = _mm_unpackhi_ps(v6x012, v7x012);
            const __m128 v89x0_89x1 = _mm_unpacklo_ps(v8x012, v9x012);
            const __m128 vABx0_ABx1 = _mm_unpacklo_ps(vAx012, vBx012);
            const __m128 v89x2 = _mm_unpackhi_ps(v8x012, v9x012);
            const __m128 vABx2 = _mm_unpackhi_ps(vAx012, vBx012);
            const __m128 vCDx0_CDx1 = _mm_unpacklo_ps(vCx012, vDx012);
            const __m128 vEFx0_EFx1 = _mm_unpacklo_ps(vEx012, vEx012);
            const __m128 vCDx2 = _mm_unpackhi_ps(vCx012, vDx012);
            const __m128 vEFx2 = _mm_unpackhi_ps(vEx012, vEx012);

            const __m128 v0123x0 = _mm_movelh_ps(v01x0_01x1, v23x0_23x1);
            const __m128 v0123x1 = _mm_movehl_ps(v23x0_23x1, v01x0_01x1);
            const __m128 v0123x2 = _mm_movelh_ps(v01x2, v23x2);
            const __m128 v4567x0 = _mm_movelh_ps(v45x0_45x1, v67x0_67x1);
            const __m128 v4567x1 = _mm_movehl_ps(v67x0_67x1, v45x0_45x1);
            const __m128 v4567x2 = _mm_movelh_ps(v45x2, v67x2);
            const __m128 v89ABx0 = _mm_movelh_ps(v89x0_89x1, vABx0_ABx1);
            const __m128 v89ABx1 = _mm_movehl_ps(vABx0_ABx1, v89x0_89x1);
            const __m128 v89ABx2 = _mm_movelh_ps(v89x2, vABx2);
            const __m128 vCDEFx0 = _mm_movelh_ps(vCDx0_CDx1, vEFx0_EFx1);
            const __m128 vCDEFx1 = _mm_movehl_ps(vEFx0_EFx1, vCDx0_CDx1);
            const __m128 vCDEFx2 = _mm_movelh_ps(vCDx2, vEFx2);

            _mm_store_ps(packed_w, v0123x0);
            _mm_store_ps(packed_w + 4, v4567x0);
            _mm_store_ps(packed_w + 8, v89ABx0);
            _mm_store_ps(packed_w + 12, vCDEFx0);
            _mm_store_ps(packed_w + 16, v0123x1);
            _mm_store_ps(packed_w + 20, v4567x1);
            _mm_store_ps(packed_w + 24, v89ABx1);
            _mm_store_ps(packed_w + 28, vCDEFx1);
            _mm_store_ps(packed_w + 32, v0123x2);
            _mm_store_ps(packed_w + 36, v4567x2);
            _mm_store_ps(packed_w + 40, v89ABx2);
            _mm_store_ps(packed_w + 44, vCDEFx2);
            packed_w += 48;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_w = (float*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
