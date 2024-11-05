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

#include "xnnpack/packw.h"


void xnn_x32_packw_gemm_gio_ukernel_x4__scalar(
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
  assert(nr == 4);   // This kernel is for NR=4
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 4
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 4; n -= 4) {
      if XNN_LIKELY(b != NULL) {
        const uint64_t v0 = ((const uint64_t*)b)[0];
        const uint64_t v1 = ((const uint64_t*)b)[1];
        ((uint64_t*)packed_w)[0] = v0;
        ((uint64_t*)packed_w)[1] = v1;
        b += 4;
      } else {
        ((uint64_t*)packed_w)[0] = 0;
        ((uint64_t*)packed_w)[1] = 0;
      }
      packed_w += 4;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint64_t v0 = ((const uint64_t*)w)[0];
        const uint64_t v1 = ((const uint64_t*)w)[1];
        ((uint64_t*)packed_w)[0] = v0;
        ((uint64_t*)packed_w)[1] = v1;
        w += k_stride;
        packed_w += 4;
      }
      w = w - kc * k_stride + 4;  // Advance to next column of 4 floats
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 3);

      if XNN_LIKELY(b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = b[i];
        }
        b += n;
      } else {
        ((uint64_t*)packed_w)[0] = 0;
        ((uint64_t*)packed_w)[1] = 0;
      }
      packed_w += 4;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = w[i];
        }
        w += k_stride;
        packed_w += 4;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
