// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"


void xnn_x32_packw_gemm_gio_ukernel_x2__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
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
  assert(nr == 2);   // This kernel is for NR=2
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const uint32_t* b = bias;
  uint32_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 2
    const uint32_t* w = weights;
    size_t n = nc;

    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(b != NULL) {
        const uint32_t v0 = b[0];
        const uint32_t v1 = b[1];
        packed_w[0] = v0;
        packed_w[1] = v1;
        b += 2;
      } else {
        packed_w[0] = 0;
        packed_w[1] = 0;
      }
      packed_w += 2;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint32_t v0 = w[0];
        const uint32_t v1 = w[1];
        packed_w[0] = v0;
        packed_w[1] = v1;
        w += k_stride;
        packed_w += 2;
      }
      w = w - kc * k_stride + 2;  // Advance to next column of 2 uint32_t
    }

    // NC remainder (1..1)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 1);

      if XNN_LIKELY(b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = b[i];
        }
        b += n;
      } else {
        packed_w[0] = 0;
        packed_w[1] = 0;
      }
      packed_w += 2;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = w[i];
        }
        w += k_stride;
        packed_w += 2;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
