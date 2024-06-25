// Auto-generated file. Do not edit!
//   Template: src/x32-packw/c4-wasmsimd.c.in
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

#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4(
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
  assert(nr == 2);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  do {
    // NC main loop multiple of 2
    const uint32_t* w0 = (const uint32_t*) weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(bias != NULL) {
        packed_weights[0] = bias[0];
        packed_weights[1] = bias[1];
        bias += 2;
      } else {
        packed_weights[0] = 0;
        packed_weights[1] = 0;
      }
      packed_weights += 2;

      const uint32_t* w1 = w0 + kc;

      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        const v128_t v0 = wasm_v128_load(w0);
        w0 += 4;
        const v128_t v1 = wasm_v128_load(w1);
        w1 += 4;

        wasm_v128_store(packed_weights, v0);
        wasm_v128_store(packed_weights + 4, v1);
        packed_weights += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            // Read blocks of 2x1
            // a
            // e
            const v128_t v0 = wasm_v128_load32_zero(w0);
            ++w0;
            const v128_t v1 = wasm_v128_load32_zero(w1);
            ++w1;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v1);
            packed_weights += 8;
            break;
          }
          case 2:
          {
            // Read blocks of 2x2
            // a b
            // e f
            const v128_t v0 = wasm_v128_load64_zero(w0);
            w0 += 2;
            const v128_t v1 = wasm_v128_load64_zero(w1);
            w1 += 2;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v1);
            packed_weights += 8;
            break;
          }
          case 3:
          {
            // Read blocks of 2x3
            // a b c
            // e f g
            v128_t v0 = wasm_v128_load64_zero(w0);
            v0 = wasm_v128_load32_lane(w0 + 2, v0, 2);
            w0 += 3;
            v128_t v1 = wasm_v128_load64_zero(w1);
            v1 = wasm_v128_load32_lane(w1 + 2, v1, 2);
            w1 += 3;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v1);
            packed_weights += 8;
            break;
          }
          default:
            XNN_UNREACHABLE;
        }
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w1;
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 1);
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (2 - n);
      } else {
        packed_weights[0] = 0;
        packed_weights[1] = 0;
        packed_weights += 2;
      }

      // NR remainder has less than 2 rows so last row is not loaded


      // KC main loop multiple of 2x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        // Read blocks of 2x4
        // a b c d
        // e f g h
        const v128_t v0 = wasm_v128_load(w0);
        w0 += 4;

        wasm_v128_store(packed_weights, v0);
        wasm_v128_store(packed_weights + 4, v0);
        packed_weights += 8;
      }

      // KC remainder (1..3)
      if XNN_UNLIKELY(k != 0) {
        assert(k >= 1);
        assert(k <= 3);
        switch (k) {
          case 1:
          {
            // Read blocks of 1x1
            // a
            const v128_t v0 = wasm_v128_load32_zero(w0);
            ++w0;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v0);
            packed_weights += 8;
            break;
          }
          case 2:
          {
            // Read blocks of 1x2
            // a b
            const v128_t v0 = wasm_v128_load64_zero(w0);
            w0 += 2;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v0);
            packed_weights += 8;
            break;
          }
          case 3:
          {
            // Read blocks of 1x3
            // a b c
            v128_t v0 = wasm_v128_load64_zero(w0);
            v0 = wasm_v128_load32_lane(w0 + 2, v0, 2);
            w0 += 3;
            wasm_v128_store(packed_weights, v0);
            wasm_v128_store(packed_weights + 4, v0);
            packed_weights += 8;
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
