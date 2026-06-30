// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x16__wasmsimd_u4(
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

  do {
    // NC main loop multiple of 16
    const uint32_t* w0 = (const uint32_t*) weights;

    size_t n = nc;
    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(bias != NULL) {
        const v128_t vb0123 = wasm_v128_load(bias);
        const v128_t vb4567 = wasm_v128_load(bias + 4);
        const v128_t vb89AB = wasm_v128_load(bias + 8);
        const v128_t vbCDEF = wasm_v128_load(bias + 12);
        bias += 16;

        wasm_v128_store(packed_weights, vb0123);
        wasm_v128_store(packed_weights + 4, vb4567);
        wasm_v128_store(packed_weights + 8, vb89AB);
        wasm_v128_store(packed_weights + 12, vbCDEF);
      } else {
        const v128_t vzero = wasm_i32x4_const_splat(0);
        wasm_v128_store(packed_weights, vzero);
        wasm_v128_store(packed_weights + 4, vzero);
        wasm_v128_store(packed_weights + 8, vzero);
        wasm_v128_store(packed_weights + 12, vzero);
      }
      packed_weights += 16;

      const uint32_t* w1 = w0 + kc;
      const uint32_t* w2 = w1 + kc;
      const uint32_t* w3 = w2 + kc;
      const uint32_t* w4 = w3 + kc;
      const uint32_t* w5 = w4 + kc;
      const uint32_t* w6 = w5 + kc;
      const uint32_t* w7 = w6 + kc;
      const uint32_t* w8 = w7 + kc;
      const uint32_t* w9 = w8 + kc;
      const uint32_t* wA = w9 + kc;
      const uint32_t* wB = wA + kc;
      const uint32_t* wC = wB + kc;
      const uint32_t* wD = wC + kc;
      const uint32_t* wE = wD + kc;
      const uint32_t* wF = wE + kc;

      // KC main loop multiple of 16x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const v128_t v0x0123 = wasm_v128_load(w0);
        w0 += 4;
        const v128_t v1x0123 = wasm_v128_load(w1);
        w1 += 4;
        const v128_t v2x0123 = wasm_v128_load(w2);
        w2 += 4;
        const v128_t v3x0123 = wasm_v128_load(w3);
        w3 += 4;
        const v128_t v4x0123 = wasm_v128_load(w4);
        w4 += 4;
        const v128_t v5x0123 = wasm_v128_load(w5);
        w5 += 4;
        const v128_t v6x0123 = wasm_v128_load(w6);
        w6 += 4;
        const v128_t v7x0123 = wasm_v128_load(w7);
        w7 += 4;
        const v128_t v8x0123 = wasm_v128_load(w8);
        w8 += 4;
        const v128_t v9x0123 = wasm_v128_load(w9);
        w9 += 4;
        const v128_t vAx0123 = wasm_v128_load(wA);
        wA += 4;
        const v128_t vBx0123 = wasm_v128_load(wB);
        wB += 4;
        const v128_t vCx0123 = wasm_v128_load(wC);
        wC += 4;
        const v128_t vDx0123 = wasm_v128_load(wD);
        wD += 4;
        const v128_t vEx0123 = wasm_v128_load(wE);
        wE += 4;
        const v128_t vFx0123 = wasm_v128_load(wF);
        wF += 4;

        const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x0123, v1x0123, 0, 4, 1, 5);
        const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x0123, v3x0123, 0, 4, 1, 5);
        const v128_t v01x2_01x3 = wasm_v32x4_shuffle(v0x0123, v1x0123, 2, 6, 3, 7);
        const v128_t v23x2_23x3 = wasm_v32x4_shuffle(v2x0123, v3x0123, 2, 6, 3, 7);
        const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x0123, v5x0123, 0, 4, 1, 5);
        const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x0123, v7x0123, 0, 4, 1, 5);
        const v128_t v45x2_45x3 = wasm_v32x4_shuffle(v4x0123, v5x0123, 2, 6, 3, 7);
        const v128_t v67x2_67x3 = wasm_v32x4_shuffle(v6x0123, v7x0123, 2, 6, 3, 7);
        const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x0123, v9x0123, 0, 4, 1, 5);
        const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx0123, vBx0123, 0, 4, 1, 5);
        const v128_t v89x2_89x3 = wasm_v32x4_shuffle(v8x0123, v9x0123, 2, 6, 3, 7);
        const v128_t vABx2_ABx3 = wasm_v32x4_shuffle(vAx0123, vBx0123, 2, 6, 3, 7);
        const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx0123, vDx0123, 0, 4, 1, 5);
        const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx0123, vFx0123, 0, 4, 1, 5);
        const v128_t vCDx2_CDx3 = wasm_v32x4_shuffle(vCx0123, vDx0123, 2, 6, 3, 7);
        const v128_t vEFx2_EFx3 = wasm_v32x4_shuffle(vEx0123, vFx0123, 2, 6, 3, 7);

        const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
        const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
        const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 0, 2);
        const v128_t v0123x3 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 1, 3);
        const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
        const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
        const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 0, 2);
        const v128_t v4567x3 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 1, 3);
        const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
        const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
        const v128_t v89ABx2 = wasm_v64x2_shuffle(v89x2_89x3, vABx2_ABx3, 0, 2);
        const v128_t v89ABx3 = wasm_v64x2_shuffle(v89x2_89x3, vABx2_ABx3, 1, 3);
        const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
        const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);
        const v128_t vCDEFx2 = wasm_v64x2_shuffle(vCDx2_CDx3, vEFx2_EFx3, 0, 2);
        const v128_t vCDEFx3 = wasm_v64x2_shuffle(vCDx2_CDx3, vEFx2_EFx3, 1, 3);

        wasm_v128_store(packed_weights, v0123x0);
        wasm_v128_store(packed_weights + 4, v4567x0);
        wasm_v128_store(packed_weights + 8, v89ABx0);
        wasm_v128_store(packed_weights + 12, vCDEFx0);
        wasm_v128_store(packed_weights + 16, v0123x1);
        wasm_v128_store(packed_weights + 20, v4567x1);
        wasm_v128_store(packed_weights + 24, v89ABx1);
        wasm_v128_store(packed_weights + 28, vCDEFx1);
        wasm_v128_store(packed_weights + 32, v0123x2);
        wasm_v128_store(packed_weights + 36, v4567x2);
        wasm_v128_store(packed_weights + 40, v89ABx2);
        wasm_v128_store(packed_weights + 44, vCDEFx2);
        wasm_v128_store(packed_weights + 48, v0123x3);
        wasm_v128_store(packed_weights + 52, v4567x3);
        wasm_v128_store(packed_weights + 56, v89ABx3);
        wasm_v128_store(packed_weights + 60, vCDEFx3);
        packed_weights += 64;
      }

      if XNN_UNLIKELY(k != 0) {
        // KC remainder (1..3)
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            v128_t v0123x0 = wasm_v128_load32_zero(w0);
            w0 += 1;
            v128_t v4567x0 = wasm_v128_load32_zero(w4);
            w4 += 1;
            v128_t v89ABx0 = wasm_v128_load32_zero(w8);
            w8 += 1;
            v128_t vCDEFx0 = wasm_v128_load32_zero(wC);
            wC += 1;

            v0123x0 = wasm_v128_load32_lane(w1, v0123x0, 1);
            w1 += 1;
            v4567x0 = wasm_v128_load32_lane(w5, v4567x0, 1);
            w5 += 1;
            v89ABx0 = wasm_v128_load32_lane(w9, v89ABx0, 1);
            w9 += 1;
            vCDEFx0 = wasm_v128_load32_lane(wD, vCDEFx0, 1);
            wD += 1;
            v0123x0 = wasm_v128_load32_lane(w2, v0123x0, 2);
            w2 += 1;
            v4567x0 = wasm_v128_load32_lane(w6, v4567x0, 2);
            w6 += 1;
            v89ABx0 = wasm_v128_load32_lane(wA, v89ABx0, 2);
            wA += 1;
            vCDEFx0 = wasm_v128_load32_lane(wE, vCDEFx0, 2);
            wE += 1;
            v0123x0 = wasm_v128_load32_lane(w3, v0123x0, 3);
            w3 += 1;
            v4567x0 = wasm_v128_load32_lane(w7, v4567x0, 3);
            w7 += 1;
            v89ABx0 = wasm_v128_load32_lane(wB, v89ABx0, 3);
            wB += 1;
            vCDEFx0 = wasm_v128_load32_lane(wF, vCDEFx0, 3);
            wF += 1;

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            packed_weights += 16;
            break;
          }
          case 2:
          {
            const v128_t v0x01 = wasm_v128_load64_zero(w0);
            w0 += 2;
            const v128_t v1x01 = wasm_v128_load64_zero(w1);
            w1 += 2;
            const v128_t v2x01 = wasm_v128_load64_zero(w2);
            w2 += 2;
            const v128_t v3x01 = wasm_v128_load64_zero(w3);
            w3 += 2;
            const v128_t v4x01 = wasm_v128_load64_zero(w4);
            w4 += 2;
            const v128_t v5x01 = wasm_v128_load64_zero(w5);
            w5 += 2;
            const v128_t v6x01 = wasm_v128_load64_zero(w6);
            w6 += 2;
            const v128_t v7x01 = wasm_v128_load64_zero(w7);
            w7 += 2;
            const v128_t v8x01 = wasm_v128_load64_zero(w8);
            w8 += 2;
            const v128_t v9x01 = wasm_v128_load64_zero(w9);
            w9 += 2;
            const v128_t vAx01 = wasm_v128_load64_zero(wA);
            wA += 2;
            const v128_t vBx01 = wasm_v128_load64_zero(wB);
            wB += 2;
            const v128_t vCx01 = wasm_v128_load64_zero(wC);
            wC += 2;
            const v128_t vDx01 = wasm_v128_load64_zero(wD);
            wD += 2;
            const v128_t vEx01 = wasm_v128_load64_zero(wE);
            wE += 2;
            const v128_t vFx01 = wasm_v128_load64_zero(wF);
            wF += 2;

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x01, v1x01, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x01, v3x01, 0, 4, 1, 5);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x01, v5x01, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x01, v7x01, 0, 4, 1, 5);
            const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x01, v9x01, 0, 4, 1, 5);
            const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx01, vBx01, 0, 4, 1, 5);
            const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx01, vDx01, 0, 4, 1, 5);
            const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx01, vFx01, 0, 4, 1, 5);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
            const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
            const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
            const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            wasm_v128_store(packed_weights + 16, v0123x1);
            wasm_v128_store(packed_weights + 20, v4567x1);
            wasm_v128_store(packed_weights + 24, v89ABx1);
            wasm_v128_store(packed_weights + 28, vCDEFx1);
            packed_weights += 32;
            break;
          }
          case 3:
          {
            v128_t v0x012 = wasm_v128_load64_zero(w0);
            w0 += 2;
            v128_t v1x012 = wasm_v128_load64_zero(w1);
            w1 += 2;
            v128_t v2x012 = wasm_v128_load64_zero(w2);
            w2 += 2;
            v128_t v3x012 = wasm_v128_load64_zero(w3);
            w3 += 2;
            v128_t v4x012 = wasm_v128_load64_zero(w4);
            w4 += 2;
            v128_t v5x012 = wasm_v128_load64_zero(w5);
            w5 += 2;
            v128_t v6x012 = wasm_v128_load64_zero(w6);
            w6 += 2;
            v128_t v7x012 = wasm_v128_load64_zero(w7);
            w7 += 2;
            v128_t v8x012 = wasm_v128_load64_zero(w8);
            w8 += 2;
            v128_t v9x012 = wasm_v128_load64_zero(w9);
            w9 += 2;
            v128_t vAx012 = wasm_v128_load64_zero(wA);
            wA += 2;
            v128_t vBx012 = wasm_v128_load64_zero(wB);
            wB += 2;
            v128_t vCx012 = wasm_v128_load64_zero(wC);
            wC += 2;
            v128_t vDx012 = wasm_v128_load64_zero(wD);
            wD += 2;
            v128_t vEx012 = wasm_v128_load64_zero(wE);
            wE += 2;
            v128_t vFx012 = wasm_v128_load64_zero(wF);
            wF += 2;

            v0x012 = wasm_v128_load32_lane(w0, v0x012, 2);
            w0 += 1;
            v1x012 = wasm_v128_load32_lane(w1, v1x012, 2);
            w1 += 1;
            v2x012 = wasm_v128_load32_lane(w2, v2x012, 2);
            w2 += 1;
            v3x012 = wasm_v128_load32_lane(w3, v3x012, 2);
            w3 += 1;
            v4x012 = wasm_v128_load32_lane(w4, v4x012, 2);
            w4 += 1;
            v5x012 = wasm_v128_load32_lane(w5, v5x012, 2);
            w5 += 1;
            v6x012 = wasm_v128_load32_lane(w6, v6x012, 2);
            w6 += 1;
            v7x012 = wasm_v128_load32_lane(w7, v7x012, 2);
            w7 += 1;
            v8x012 = wasm_v128_load32_lane(w8, v8x012, 2);
            w8 += 1;
            v9x012 = wasm_v128_load32_lane(w9, v9x012, 2);
            w9 += 1;
            vAx012 = wasm_v128_load32_lane(wA, vAx012, 2);
            wA += 1;
            vBx012 = wasm_v128_load32_lane(wB, vBx012, 2);
            wB += 1;
            vCx012 = wasm_v128_load32_lane(wC, vCx012, 2);
            wC += 1;
            vDx012 = wasm_v128_load32_lane(wD, vDx012, 2);
            wD += 1;
            vEx012 = wasm_v128_load32_lane(wE, vEx012, 2);
            wE += 1;
            vFx012 = wasm_v128_load32_lane(wF, vFx012, 2);
            wF += 1;

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x012, v1x012, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x012, v3x012, 0, 4, 1, 5);
            const v128_t v01x2 = wasm_v32x4_shuffle(v0x012, v1x012, 2, 6, 3, 7);
            const v128_t v23x2 = wasm_v32x4_shuffle(v2x012, v3x012, 2, 6, 3, 7);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x012, v5x012, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x012, v7x012, 0, 4, 1, 5);
            const v128_t v45x2 = wasm_v32x4_shuffle(v4x012, v5x012, 2, 6, 3, 7);
            const v128_t v67x2 = wasm_v32x4_shuffle(v6x012, v7x012, 2, 6, 3, 7);
            const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x012, v9x012, 0, 4, 1, 5);
            const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx012, vBx012, 0, 4, 1, 5);
            const v128_t v89x2 = wasm_v32x4_shuffle(v8x012, v9x012, 2, 6, 3, 7);
            const v128_t vABx2 = wasm_v32x4_shuffle(vAx012, vBx012, 2, 6, 3, 7);
            const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx012, vDx012, 0, 4, 1, 5);
            const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx012, vFx012, 0, 4, 1, 5);
            const v128_t vCDx2 = wasm_v32x4_shuffle(vCx012, vDx012, 2, 6, 3, 7);
            const v128_t vEFx2 = wasm_v32x4_shuffle(vEx012, vFx012, 2, 6, 3, 7);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2, v23x2, 0, 2);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2, v67x2, 0, 2);
            const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
            const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
            const v128_t v89ABx2 = wasm_v64x2_shuffle(v89x2, vABx2, 0, 2);
            const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
            const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);
            const v128_t vCDEFx2 = wasm_v64x2_shuffle(vCDx2, vEFx2, 0, 2);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            wasm_v128_store(packed_weights + 16, v0123x1);
            wasm_v128_store(packed_weights + 20, v4567x1);
            wasm_v128_store(packed_weights + 24, v89ABx1);
            wasm_v128_store(packed_weights + 28, vCDEFx1);
            wasm_v128_store(packed_weights + 32, v0123x2);
            wasm_v128_store(packed_weights + 36, v4567x2);
            wasm_v128_store(packed_weights + 40, v89ABx2);
            wasm_v128_store(packed_weights + 44, vCDEFx2);
            packed_weights += 48;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = wF;
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (16 - n);
      } else {
        const v128_t vzero = wasm_i32x4_const_splat(0);
        wasm_v128_store(packed_weights, vzero);
        wasm_v128_store(packed_weights + 4, vzero);
        wasm_v128_store(packed_weights + 8, vzero);
        wasm_v128_store(packed_weights + 12, vzero);
        packed_weights += 16;
      }

      const uint32_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const uint32_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const uint32_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const uint32_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const uint32_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const uint32_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const uint32_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const uint32_t* w8 = w7 + kc;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const uint32_t* w9 = w8 + kc;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const uint32_t* wA = w9 + kc;
      if XNN_UNPREDICTABLE(n <= 10) {
        wA = w9;
      }
      const uint32_t* wB = wA + kc;
      if XNN_UNPREDICTABLE(n < 12) {
        wB = wA;
      }
      const uint32_t* wC = wB + kc;
      if XNN_UNPREDICTABLE(n <= 12) {
        wC = wB;
      }
      const uint32_t* wD = wC + kc;
      if XNN_UNPREDICTABLE(n < 14) {
        wD = wC;
      }
      const uint32_t* wE = wD + kc;
      if XNN_UNPREDICTABLE(n <= 14) {
        wE = wD;
      }

      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const v128_t v0x0123 = wasm_v128_load(w0);
        w0 += 4;
        const v128_t v1x0123 = wasm_v128_load(w1);
        w1 += 4;
        const v128_t v2x0123 = wasm_v128_load(w2);
        w2 += 4;
        const v128_t v3x0123 = wasm_v128_load(w3);
        w3 += 4;
        const v128_t v4x0123 = wasm_v128_load(w4);
        w4 += 4;
        const v128_t v5x0123 = wasm_v128_load(w5);
        w5 += 4;
        const v128_t v6x0123 = wasm_v128_load(w6);
        w6 += 4;
        const v128_t v7x0123 = wasm_v128_load(w7);
        w7 += 4;
        const v128_t v8x0123 = wasm_v128_load(w8);
        w8 += 4;
        const v128_t v9x0123 = wasm_v128_load(w9);
        w9 += 4;
        const v128_t vAx0123 = wasm_v128_load(wA);
        wA += 4;
        const v128_t vBx0123 = wasm_v128_load(wB);
        wB += 4;
        const v128_t vCx0123 = wasm_v128_load(wC);
        wC += 4;
        const v128_t vDx0123 = wasm_v128_load(wD);
        wD += 4;
        const v128_t vEx0123 = wasm_v128_load(wE);
        wE += 4;

        const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x0123, v1x0123, 0, 4, 1, 5);
        const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x0123, v3x0123, 0, 4, 1, 5);
        const v128_t v01x2_01x3 = wasm_v32x4_shuffle(v0x0123, v1x0123, 2, 6, 3, 7);
        const v128_t v23x2_23x3 = wasm_v32x4_shuffle(v2x0123, v3x0123, 2, 6, 3, 7);
        const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x0123, v5x0123, 0, 4, 1, 5);
        const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x0123, v7x0123, 0, 4, 1, 5);
        const v128_t v45x2_45x3 = wasm_v32x4_shuffle(v4x0123, v5x0123, 2, 6, 3, 7);
        const v128_t v67x2_67x3 = wasm_v32x4_shuffle(v6x0123, v7x0123, 2, 6, 3, 7);
        const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x0123, v9x0123, 0, 4, 1, 5);
        const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx0123, vBx0123, 0, 4, 1, 5);
        const v128_t v89x2_89x3 = wasm_v32x4_shuffle(v8x0123, v9x0123, 2, 6, 3, 7);
        const v128_t vABx2_ABx3 = wasm_v32x4_shuffle(vAx0123, vBx0123, 2, 6, 3, 7);
        const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx0123, vDx0123, 0, 4, 1, 5);
        const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx0123, vEx0123, 0, 4, 1, 5);
        const v128_t vCDx2_CDx3 = wasm_v32x4_shuffle(vCx0123, vDx0123, 2, 6, 3, 7);
        const v128_t vEFx2_EFx3 = wasm_v32x4_shuffle(vEx0123, vEx0123, 2, 6, 3, 7);

        const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
        const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
        const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 0, 2);
        const v128_t v0123x3 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 1, 3);
        const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
        const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
        const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 0, 2);
        const v128_t v4567x3 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 1, 3);
        const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
        const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
        const v128_t v89ABx2 = wasm_v64x2_shuffle(v89x2_89x3, vABx2_ABx3, 0, 2);
        const v128_t v89ABx3 = wasm_v64x2_shuffle(v89x2_89x3, vABx2_ABx3, 1, 3);
        const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
        const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);
        const v128_t vCDEFx2 = wasm_v64x2_shuffle(vCDx2_CDx3, vEFx2_EFx3, 0, 2);
        const v128_t vCDEFx3 = wasm_v64x2_shuffle(vCDx2_CDx3, vEFx2_EFx3, 1, 3);

        wasm_v128_store(packed_weights, v0123x0);
        wasm_v128_store(packed_weights + 4, v4567x0);
        wasm_v128_store(packed_weights + 8, v89ABx0);
        wasm_v128_store(packed_weights + 12, vCDEFx0);
        wasm_v128_store(packed_weights + 16, v0123x1);
        wasm_v128_store(packed_weights + 20, v4567x1);
        wasm_v128_store(packed_weights + 24, v89ABx1);
        wasm_v128_store(packed_weights + 28, vCDEFx1);
        wasm_v128_store(packed_weights + 32, v0123x2);
        wasm_v128_store(packed_weights + 36, v4567x2);
        wasm_v128_store(packed_weights + 40, v89ABx2);
        wasm_v128_store(packed_weights + 44, vCDEFx2);
        wasm_v128_store(packed_weights + 48, v0123x3);
        wasm_v128_store(packed_weights + 52, v4567x3);
        wasm_v128_store(packed_weights + 56, v89ABx3);
        wasm_v128_store(packed_weights + 60, vCDEFx3);
        packed_weights += 64;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            v128_t v0123x0 = wasm_v128_load32_zero(w0);
            w0 += 1;
            v128_t v4567x0 = wasm_v128_load32_zero(w4);
            w4 += 1;
            v128_t v89ABx0 = wasm_v128_load32_zero(w8);
            w8 += 1;
            v128_t vCDEFx0 = wasm_v128_load32_zero(wC);
            wC += 1;

            v0123x0 = wasm_v128_load32_lane(w1, v0123x0, 1);
            w1 += 1;
            v4567x0 = wasm_v128_load32_lane(w5, v4567x0, 1);
            w5 += 1;
            v89ABx0 = wasm_v128_load32_lane(w9, v89ABx0, 1);
            w9 += 1;
            vCDEFx0 = wasm_v128_load32_lane(wD, vCDEFx0, 1);
            wD += 1;
            v0123x0 = wasm_v128_load32_lane(w2, v0123x0, 2);
            w2 += 1;
            v4567x0 = wasm_v128_load32_lane(w6, v4567x0, 2);
            w6 += 1;
            v89ABx0 = wasm_v128_load32_lane(wA, v89ABx0, 2);
            wA += 1;
            vCDEFx0 = wasm_v128_load32_lane(wE, vCDEFx0, 2);
            wE += 1;
            v0123x0 = wasm_v128_load32_lane(w3, v0123x0, 3);
            w3 += 1;
            v4567x0 = wasm_v128_load32_lane(w7, v4567x0, 3);
            w7 += 1;
            v89ABx0 = wasm_v128_load32_lane(wB, v89ABx0, 3);
            wB += 1;

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            packed_weights += 16;
            break;
          }
          case 2:
          {
            const v128_t v0x01 = wasm_v128_load64_zero(w0);
            w0 += 2;
            const v128_t v1x01 = wasm_v128_load64_zero(w1);
            w1 += 2;
            const v128_t v2x01 = wasm_v128_load64_zero(w2);
            w2 += 2;
            const v128_t v3x01 = wasm_v128_load64_zero(w3);
            w3 += 2;
            const v128_t v4x01 = wasm_v128_load64_zero(w4);
            w4 += 2;
            const v128_t v5x01 = wasm_v128_load64_zero(w5);
            w5 += 2;
            const v128_t v6x01 = wasm_v128_load64_zero(w6);
            w6 += 2;
            const v128_t v7x01 = wasm_v128_load64_zero(w7);
            w7 += 2;
            const v128_t v8x01 = wasm_v128_load64_zero(w8);
            w8 += 2;
            const v128_t v9x01 = wasm_v128_load64_zero(w9);
            w9 += 2;
            const v128_t vAx01 = wasm_v128_load64_zero(wA);
            wA += 2;
            const v128_t vBx01 = wasm_v128_load64_zero(wB);
            wB += 2;
            const v128_t vCx01 = wasm_v128_load64_zero(wC);
            wC += 2;
            const v128_t vDx01 = wasm_v128_load64_zero(wD);
            wD += 2;
            const v128_t vEx01 = wasm_v128_load64_zero(wE);
            wE += 2;

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x01, v1x01, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x01, v3x01, 0, 4, 1, 5);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x01, v5x01, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x01, v7x01, 0, 4, 1, 5);
            const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x01, v9x01, 0, 4, 1, 5);
            const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx01, vBx01, 0, 4, 1, 5);
            const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx01, vDx01, 0, 4, 1, 5);
            const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx01, vEx01, 0, 4, 1, 5);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
            const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
            const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
            const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            wasm_v128_store(packed_weights + 16, v0123x1);
            wasm_v128_store(packed_weights + 20, v4567x1);
            wasm_v128_store(packed_weights + 24, v89ABx1);
            wasm_v128_store(packed_weights + 28, vCDEFx1);
            packed_weights += 32;
            break;
          }
          case 3:
          {
            v128_t v0x012 = wasm_v128_load64_zero(w0);
            w0 += 2;
            v128_t v1x012 = wasm_v128_load64_zero(w1);
            w1 += 2;
            v128_t v2x012 = wasm_v128_load64_zero(w2);
            w2 += 2;
            v128_t v3x012 = wasm_v128_load64_zero(w3);
            w3 += 2;
            v128_t v4x012 = wasm_v128_load64_zero(w4);
            w4 += 2;
            v128_t v5x012 = wasm_v128_load64_zero(w5);
            w5 += 2;
            v128_t v6x012 = wasm_v128_load64_zero(w6);
            w6 += 2;
            v128_t v7x012 = wasm_v128_load64_zero(w7);
            w7 += 2;
            v128_t v8x012 = wasm_v128_load64_zero(w8);
            w8 += 2;
            v128_t v9x012 = wasm_v128_load64_zero(w9);
            w9 += 2;
            v128_t vAx012 = wasm_v128_load64_zero(wA);
            wA += 2;
            v128_t vBx012 = wasm_v128_load64_zero(wB);
            wB += 2;
            v128_t vCx012 = wasm_v128_load64_zero(wC);
            wC += 2;
            v128_t vDx012 = wasm_v128_load64_zero(wD);
            wD += 2;
            v128_t vEx012 = wasm_v128_load64_zero(wE);
            wE += 2;

            v0x012 = wasm_v128_load32_lane(w0, v0x012, 2);
            w0 += 1;
            v1x012 = wasm_v128_load32_lane(w1, v1x012, 2);
            w1 += 1;
            v2x012 = wasm_v128_load32_lane(w2, v2x012, 2);
            w2 += 1;
            v3x012 = wasm_v128_load32_lane(w3, v3x012, 2);
            w3 += 1;
            v4x012 = wasm_v128_load32_lane(w4, v4x012, 2);
            w4 += 1;
            v5x012 = wasm_v128_load32_lane(w5, v5x012, 2);
            w5 += 1;
            v6x012 = wasm_v128_load32_lane(w6, v6x012, 2);
            w6 += 1;
            v7x012 = wasm_v128_load32_lane(w7, v7x012, 2);
            w7 += 1;
            v8x012 = wasm_v128_load32_lane(w8, v8x012, 2);
            w8 += 1;
            v9x012 = wasm_v128_load32_lane(w9, v9x012, 2);
            w9 += 1;
            vAx012 = wasm_v128_load32_lane(wA, vAx012, 2);
            wA += 1;
            vBx012 = wasm_v128_load32_lane(wB, vBx012, 2);
            wB += 1;
            vCx012 = wasm_v128_load32_lane(wC, vCx012, 2);
            wC += 1;
            vDx012 = wasm_v128_load32_lane(wD, vDx012, 2);
            wD += 1;
            vEx012 = wasm_v128_load32_lane(wE, vEx012, 2);
            wE += 1;

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x012, v1x012, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x012, v3x012, 0, 4, 1, 5);
            const v128_t v01x2 = wasm_v32x4_shuffle(v0x012, v1x012, 2, 6, 3, 7);
            const v128_t v23x2 = wasm_v32x4_shuffle(v2x012, v3x012, 2, 6, 3, 7);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x012, v5x012, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x012, v7x012, 0, 4, 1, 5);
            const v128_t v45x2 = wasm_v32x4_shuffle(v4x012, v5x012, 2, 6, 3, 7);
            const v128_t v67x2 = wasm_v32x4_shuffle(v6x012, v7x012, 2, 6, 3, 7);
            const v128_t v89x0_89x1 = wasm_v32x4_shuffle(v8x012, v9x012, 0, 4, 1, 5);
            const v128_t vABx0_ABx1 = wasm_v32x4_shuffle(vAx012, vBx012, 0, 4, 1, 5);
            const v128_t v89x2 = wasm_v32x4_shuffle(v8x012, v9x012, 2, 6, 3, 7);
            const v128_t vABx2 = wasm_v32x4_shuffle(vAx012, vBx012, 2, 6, 3, 7);
            const v128_t vCDx0_CDx1 = wasm_v32x4_shuffle(vCx012, vDx012, 0, 4, 1, 5);
            const v128_t vEFx0_EFx1 = wasm_v32x4_shuffle(vEx012, vEx012, 0, 4, 1, 5);
            const v128_t vCDx2 = wasm_v32x4_shuffle(vCx012, vDx012, 2, 6, 3, 7);
            const v128_t vEFx2 = wasm_v32x4_shuffle(vEx012, vEx012, 2, 6, 3, 7);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2, v23x2, 0, 2);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2, v67x2, 0, 2);
            const v128_t v89ABx0 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 0, 2);
            const v128_t v89ABx1 = wasm_v64x2_shuffle(v89x0_89x1, vABx0_ABx1, 1, 3);
            const v128_t v89ABx2 = wasm_v64x2_shuffle(v89x2, vABx2, 0, 2);
            const v128_t vCDEFx0 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 0, 2);
            const v128_t vCDEFx1 = wasm_v64x2_shuffle(vCDx0_CDx1, vEFx0_EFx1, 1, 3);
            const v128_t vCDEFx2 = wasm_v64x2_shuffle(vCDx2, vEFx2, 0, 2);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v89ABx0);
            wasm_v128_store(packed_weights + 12, vCDEFx0);
            wasm_v128_store(packed_weights + 16, v0123x1);
            wasm_v128_store(packed_weights + 20, v4567x1);
            wasm_v128_store(packed_weights + 24, v89ABx1);
            wasm_v128_store(packed_weights + 28, vCDEFx1);
            wasm_v128_store(packed_weights + 32, v0123x2);
            wasm_v128_store(packed_weights + 36, v4567x2);
            wasm_v128_store(packed_weights + 40, v89ABx2);
            wasm_v128_store(packed_weights + 44, vCDEFx2);
            packed_weights += 48;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
