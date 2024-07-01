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

#include "xnnpack/common.h"
#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4(
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
  assert(nr == 8);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  do {
    // NC main loop multiple of 8
    const uint32_t* w0 = (const uint32_t*) weights;

    size_t n = nc;
    for (; n >= 8; n -= 8) {
      if XNN_LIKELY(bias != NULL) {
        const v128_t vb0123 = wasm_v128_load(bias);
        const v128_t vb4567 = wasm_v128_load(bias + 4);
        bias += 8;

        wasm_v128_store(packed_weights, vb0123);
        wasm_v128_store(packed_weights + 4, vb4567);
      } else {
        const v128_t vzero = wasm_i32x4_const_splat(0);
        wasm_v128_store(packed_weights, vzero);
        wasm_v128_store(packed_weights + 4, vzero);
      }
      packed_weights += 8;

      const uint32_t* w1 = w0 + kc;
      const uint32_t* w2 = w1 + kc;
      const uint32_t* w3 = w2 + kc;
      const uint32_t* w4 = w3 + kc;
      const uint32_t* w5 = w4 + kc;
      const uint32_t* w6 = w5 + kc;
      const uint32_t* w7 = w6 + kc;

      // KC main loop multiple of 8x4
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

        const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x0123, v1x0123, 0, 4, 1, 5);
        const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x0123, v3x0123, 0, 4, 1, 5);
        const v128_t v01x2_01x3 = wasm_v32x4_shuffle(v0x0123, v1x0123, 2, 6, 3, 7);
        const v128_t v23x2_23x3 = wasm_v32x4_shuffle(v2x0123, v3x0123, 2, 6, 3, 7);
        const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x0123, v5x0123, 0, 4, 1, 5);
        const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x0123, v7x0123, 0, 4, 1, 5);
        const v128_t v45x2_45x3 = wasm_v32x4_shuffle(v4x0123, v5x0123, 2, 6, 3, 7);
        const v128_t v67x2_67x3 = wasm_v32x4_shuffle(v6x0123, v7x0123, 2, 6, 3, 7);

        const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
        const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
        const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 0, 2);
        const v128_t v0123x3 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 1, 3);
        const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
        const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
        const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 0, 2);
        const v128_t v4567x3 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 1, 3);

        wasm_v128_store(packed_weights, v0123x0);
        wasm_v128_store(packed_weights + 4, v4567x0);
        wasm_v128_store(packed_weights + 8, v0123x1);
        wasm_v128_store(packed_weights + 12, v4567x1);
        wasm_v128_store(packed_weights + 16, v0123x2);
        wasm_v128_store(packed_weights + 20, v4567x2);
        wasm_v128_store(packed_weights + 24, v0123x3);
        wasm_v128_store(packed_weights + 28, v4567x3);
        packed_weights += 32;
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

            v0123x0 = wasm_v128_load32_lane(w1, v0123x0, 1);
            w1 += 1;
            v4567x0 = wasm_v128_load32_lane(w5, v4567x0, 1);
            w5 += 1;
            v0123x0 = wasm_v128_load32_lane(w2, v0123x0, 2);
            w2 += 1;
            v4567x0 = wasm_v128_load32_lane(w6, v4567x0, 2);
            w6 += 1;
            v0123x0 = wasm_v128_load32_lane(w3, v0123x0, 3);
            w3 += 1;
            v4567x0 = wasm_v128_load32_lane(w7, v4567x0, 3);
            w7 += 1;

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            packed_weights += 8;
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

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x01, v1x01, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x01, v3x01, 0, 4, 1, 5);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x01, v5x01, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x01, v7x01, 0, 4, 1, 5);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v0123x1);
            wasm_v128_store(packed_weights + 12, v4567x1);
            packed_weights += 16;
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

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x012, v1x012, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x012, v3x012, 0, 4, 1, 5);
            const v128_t v01x2 = wasm_v32x4_shuffle(v0x012, v1x012, 2, 6, 3, 7);
            const v128_t v23x2 = wasm_v32x4_shuffle(v2x012, v3x012, 2, 6, 3, 7);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x012, v5x012, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x012, v7x012, 0, 4, 1, 5);
            const v128_t v45x2 = wasm_v32x4_shuffle(v4x012, v5x012, 2, 6, 3, 7);
            const v128_t v67x2 = wasm_v32x4_shuffle(v6x012, v7x012, 2, 6, 3, 7);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2, v23x2, 0, 2);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2, v67x2, 0, 2);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v0123x1);
            wasm_v128_store(packed_weights + 12, v4567x1);
            wasm_v128_store(packed_weights + 16, v0123x2);
            wasm_v128_store(packed_weights + 20, v4567x2);
            packed_weights += 24;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 7);
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (8 - n);
      } else {
        const v128_t vzero = wasm_i32x4_const_splat(0);
        wasm_v128_store(packed_weights, vzero);
        wasm_v128_store(packed_weights + 4, vzero);
        packed_weights += 8;
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

        const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x0123, v1x0123, 0, 4, 1, 5);
        const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x0123, v3x0123, 0, 4, 1, 5);
        const v128_t v01x2_01x3 = wasm_v32x4_shuffle(v0x0123, v1x0123, 2, 6, 3, 7);
        const v128_t v23x2_23x3 = wasm_v32x4_shuffle(v2x0123, v3x0123, 2, 6, 3, 7);
        const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x0123, v5x0123, 0, 4, 1, 5);
        const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x0123, v6x0123, 0, 4, 1, 5);
        const v128_t v45x2_45x3 = wasm_v32x4_shuffle(v4x0123, v5x0123, 2, 6, 3, 7);
        const v128_t v67x2_67x3 = wasm_v32x4_shuffle(v6x0123, v6x0123, 2, 6, 3, 7);

        const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
        const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
        const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 0, 2);
        const v128_t v0123x3 = wasm_v64x2_shuffle(v01x2_01x3, v23x2_23x3, 1, 3);
        const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
        const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
        const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 0, 2);
        const v128_t v4567x3 = wasm_v64x2_shuffle(v45x2_45x3, v67x2_67x3, 1, 3);

        wasm_v128_store(packed_weights, v0123x0);
        wasm_v128_store(packed_weights + 4, v4567x0);
        wasm_v128_store(packed_weights + 8, v0123x1);
        wasm_v128_store(packed_weights + 12, v4567x1);
        wasm_v128_store(packed_weights + 16, v0123x2);
        wasm_v128_store(packed_weights + 20, v4567x2);
        wasm_v128_store(packed_weights + 24, v0123x3);
        wasm_v128_store(packed_weights + 28, v4567x3);
        packed_weights += 32;
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

            v0123x0 = wasm_v128_load32_lane(w1, v0123x0, 1);
            w1 += 1;
            v4567x0 = wasm_v128_load32_lane(w5, v4567x0, 1);
            w5 += 1;
            v0123x0 = wasm_v128_load32_lane(w2, v0123x0, 2);
            w2 += 1;
            v4567x0 = wasm_v128_load32_lane(w6, v4567x0, 2);
            w6 += 1;
            v0123x0 = wasm_v128_load32_lane(w3, v0123x0, 3);
            w3 += 1;

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            packed_weights += 8;
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

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x01, v1x01, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x01, v3x01, 0, 4, 1, 5);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x01, v5x01, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x01, v6x01, 0, 4, 1, 5);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v0123x1);
            wasm_v128_store(packed_weights + 12, v4567x1);
            packed_weights += 16;
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

            const v128_t v01x0_01x1 = wasm_v32x4_shuffle(v0x012, v1x012, 0, 4, 1, 5);
            const v128_t v23x0_23x1 = wasm_v32x4_shuffle(v2x012, v3x012, 0, 4, 1, 5);
            const v128_t v01x2 = wasm_v32x4_shuffle(v0x012, v1x012, 2, 6, 3, 7);
            const v128_t v23x2 = wasm_v32x4_shuffle(v2x012, v3x012, 2, 6, 3, 7);
            const v128_t v45x0_45x1 = wasm_v32x4_shuffle(v4x012, v5x012, 0, 4, 1, 5);
            const v128_t v67x0_67x1 = wasm_v32x4_shuffle(v6x012, v6x012, 0, 4, 1, 5);
            const v128_t v45x2 = wasm_v32x4_shuffle(v4x012, v5x012, 2, 6, 3, 7);
            const v128_t v67x2 = wasm_v32x4_shuffle(v6x012, v6x012, 2, 6, 3, 7);

            const v128_t v0123x0 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 0, 2);
            const v128_t v0123x1 = wasm_v64x2_shuffle(v01x0_01x1, v23x0_23x1, 1, 3);
            const v128_t v0123x2 = wasm_v64x2_shuffle(v01x2, v23x2, 0, 2);
            const v128_t v4567x0 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 0, 2);
            const v128_t v4567x1 = wasm_v64x2_shuffle(v45x0_45x1, v67x0_67x1, 1, 3);
            const v128_t v4567x2 = wasm_v64x2_shuffle(v45x2, v67x2, 0, 2);

            wasm_v128_store(packed_weights, v0123x0);
            wasm_v128_store(packed_weights + 4, v4567x0);
            wasm_v128_store(packed_weights + 8, v0123x1);
            wasm_v128_store(packed_weights + 12, v4567x1);
            wasm_v128_store(packed_weights + 16, v0123x2);
            wasm_v128_store(packed_weights + 20, v4567x2);
            packed_weights += 24;
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
