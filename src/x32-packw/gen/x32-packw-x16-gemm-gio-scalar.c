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


void xnn_x32_packw_gemm_gio_ukernel_x16__scalar(
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
  assert(nr == 16);   // This kernel is for NR=16
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 16
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 16; n -= 16) {
      if XNN_LIKELY(b != NULL) {
        const uint64_t v0 = ((const uint64_t*)b)[0];
        const uint64_t v1 = ((const uint64_t*)b)[1];
        const uint64_t v2 = ((const uint64_t*)b)[2];
        const uint64_t v3 = ((const uint64_t*)b)[3];
        const uint64_t v4 = ((const uint64_t*)b)[4];
        const uint64_t v5 = ((const uint64_t*)b)[5];
        const uint64_t v6 = ((const uint64_t*)b)[6];
        const uint64_t v7 = ((const uint64_t*)b)[7];
        ((uint64_t*)packed_w)[0] = v0;
        ((uint64_t*)packed_w)[1] = v1;
        ((uint64_t*)packed_w)[2] = v2;
        ((uint64_t*)packed_w)[3] = v3;
        ((uint64_t*)packed_w)[4] = v4;
        ((uint64_t*)packed_w)[5] = v5;
        ((uint64_t*)packed_w)[6] = v6;
        ((uint64_t*)packed_w)[7] = v7;
        b += 16;
      } else {
        ((uint64_t*)packed_w)[0] = 0;
        ((uint64_t*)packed_w)[1] = 0;
        ((uint64_t*)packed_w)[2] = 0;
        ((uint64_t*)packed_w)[3] = 0;
        ((uint64_t*)packed_w)[4] = 0;
        ((uint64_t*)packed_w)[5] = 0;
        ((uint64_t*)packed_w)[6] = 0;
        ((uint64_t*)packed_w)[7] = 0;
      }
      packed_w += 16;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint64_t v0 = ((const uint64_t*)w)[0];
        const uint64_t v1 = ((const uint64_t*)w)[1];
        const uint64_t v2 = ((const uint64_t*)w)[2];
        const uint64_t v3 = ((const uint64_t*)w)[3];
        const uint64_t v4 = ((const uint64_t*)w)[4];
        const uint64_t v5 = ((const uint64_t*)w)[5];
        const uint64_t v6 = ((const uint64_t*)w)[6];
        const uint64_t v7 = ((const uint64_t*)w)[7];
        ((uint64_t*)packed_w)[0] = v0;
        ((uint64_t*)packed_w)[1] = v1;
        ((uint64_t*)packed_w)[2] = v2;
        ((uint64_t*)packed_w)[3] = v3;
        ((uint64_t*)packed_w)[4] = v4;
        ((uint64_t*)packed_w)[5] = v5;
        ((uint64_t*)packed_w)[6] = v6;
        ((uint64_t*)packed_w)[7] = v7;
        w += k_stride;
        packed_w += 16;
      }
      w = w - kc * k_stride + 16;  // Advance to next column of 16 floats
    }

    // NC remainder (1..15)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 15);

      if XNN_LIKELY(b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = b[i];
        }
        b += n;
      } else {
        ((uint64_t*)packed_w)[0] = 0;
        ((uint64_t*)packed_w)[1] = 0;
        ((uint64_t*)packed_w)[2] = 0;
        ((uint64_t*)packed_w)[3] = 0;
        ((uint64_t*)packed_w)[4] = 0;
        ((uint64_t*)packed_w)[5] = 0;
        ((uint64_t*)packed_w)[6] = 0;
        ((uint64_t*)packed_w)[7] = 0;
      }
      packed_w += 16;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = w[i];
        }
        w += k_stride;
        packed_w += 16;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
