// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"


void xnn_x8_packw_gemm_gio_ukernel_x32__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* weights,
  const uint32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint32_t* b = bias;
  int8_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 32
    const int8_t* w = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
        b += 32;
      } else {
        memset(packed_w, 0, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        memcpy(packed_w, w, 32 * sizeof(int8_t));
        w += k_stride;
        packed_w += 32;
      }
      packed_w += extra_bytes;
      w = w - kc * k_stride + 32;  // Advance to next column of 32 int8_t
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);

      if XNN_LIKELY(b != NULL) {
        memcpy(packed_w, b, n * sizeof(uint32_t));
        memset(packed_w + n * sizeof(uint32_t), 0, (32 - n) * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
        b += n;
      } else {
        memset(packed_w, 0, 32 * sizeof(uint32_t));
        packed_w += 32 * sizeof(uint32_t);
      }

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        memcpy(packed_w, w, n * sizeof(int8_t));
        memset(packed_w + n, 0, (32 - n) * sizeof(int8_t));
        w += k_stride;
        packed_w += 32;
      }
      packed_w += extra_bytes;
    }
    weights += nc * kc;
  } while (--g != 0);
}
