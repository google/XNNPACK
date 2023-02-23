// Auto-generated file. Do not edit!
//   Template: src/x32-packw/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/packw.h>

void xnn_x32_packw_gemm_goi_ukernel_x2__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
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
  assert(nr >= sr);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  do {

    // NC main loop multiple of 2
    const uint32_t* w = weights;

    size_t n = nc;
    if XNN_LIKELY(n >= 2) {
      do {
        if XNN_LIKELY(bias != NULL) {
          packed_weights[0] = bias[0];
          packed_weights[1] = bias[1];
          bias += 2;
        }
        packed_weights += 2;

        const uint32_t* w0 = w;
        const uint32_t* w1 = w0 + kc;

        // KC main loop multiple of 2x4
        size_t k = kc;
        for (; k >= 4; k -= 4) {
          const uint32_t v00 = w0[0];
          const uint32_t v01 = w0[1];
          const uint32_t v02 = w0[2];
          const uint32_t v03 = w0[3];
          w0 += 4;
          const uint32_t v10 = w1[0];
          const uint32_t v11 = w1[1];
          const uint32_t v12 = w1[2];
          const uint32_t v13 = w1[3];
          w1 += 4;
          packed_weights[0] = v00;
          packed_weights[1] = v10;
          packed_weights[2] = v01;
          packed_weights[3] = v11;
          packed_weights[4] = v02;
          packed_weights[5] = v12;
          packed_weights[6] = v03;
          packed_weights[7] = v13;
          packed_weights += 8;
        }

        // KC remainder
        for (; k >= 1; --k) {
          packed_weights[0] = *w0++;
          packed_weights[1] = *w1++;
          packed_weights += 2;
        }
        packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
        w = w1;
        n -= 2;
      } while (n >= 2);
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder of 1
      if XNN_LIKELY(bias != NULL) {
        *packed_weights = *bias++;
      }
      packed_weights += 2;

      size_t k = kc;
      do {
        *packed_weights = *w++;
        packed_weights += 2;
      } while (--k);

      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      }

    weights += nc * kc;
  } while (--g != 0);
}
