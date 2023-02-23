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

void xnn_x32_packw_gemm_goi_ukernel_x4__scalar(
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
  assert(nr == 4);   // This kernel is for NR=4
  assert(kr == 1);
  assert(sr == 1);
  assert(nr >= sr);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  do {

    // NC main loop multiple of 4
    const uint32_t* w = weights;

    size_t n = nc;
    if XNN_LIKELY(n >= 4) {
      do {
        if XNN_LIKELY(bias != NULL) {
          packed_weights[0] = bias[0];
          packed_weights[1] = bias[1];
          packed_weights[2] = bias[2];
          packed_weights[3] = bias[3];
          bias += 4;
        }
        packed_weights += 4;

        const uint32_t* w0 = w;
        const uint32_t* w1 = w0 + kc;
        const uint32_t* w2 = w1 + kc;
        const uint32_t* w3 = w2 + kc;

        // KC main loop multiple of 4x4
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
          const uint32_t v20 = w2[0];
          const uint32_t v21 = w2[1];
          const uint32_t v22 = w2[2];
          const uint32_t v23 = w2[3];
          w2 += 4;
          const uint32_t v30 = w3[0];
          const uint32_t v31 = w3[1];
          const uint32_t v32 = w3[2];
          const uint32_t v33 = w3[3];
          w3 += 4;
          packed_weights[0] = v00;
          packed_weights[1] = v10;
          packed_weights[2] = v20;
          packed_weights[3] = v30;
          packed_weights[4] = v01;
          packed_weights[5] = v11;
          packed_weights[6] = v21;
          packed_weights[7] = v31;
          packed_weights[8] = v02;
          packed_weights[9] = v12;
          packed_weights[10] = v22;
          packed_weights[11] = v32;
          packed_weights[12] = v03;
          packed_weights[13] = v13;
          packed_weights[14] = v23;
          packed_weights[15] = v33;
          packed_weights += 16;
        }

        // KC remainder
        for (; k >= 1; --k) {
          packed_weights[0] = *w0++;
          packed_weights[1] = *w1++;
          packed_weights[2] = *w2++;
          packed_weights[3] = *w3++;
          packed_weights += 4;
        }
        packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
        w = w3;
        n -= 4;
      } while (n >= 4);
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder (1..3)
      if XNN_LIKELY(bias != NULL) {
        size_t nb = n;
        do {
          *packed_weights++  = *bias++;
        } while (--nb != 0);
        packed_weights += (4 - n);
      } else {
        packed_weights += 4;
      }

      size_t k = kc;
      do {
        const uint32_t* wn = w;
        size_t nw = n;
        do {
          *packed_weights++ = wn[0];
          wn += kc;
        } while (--nw != 0);
        ++w;
        packed_weights += (4 - n);
      } while (--k);
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      }

    weights += nc * kc;
  } while (--g != 0);
}
