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


void xnn_x32_packw_gemm_gio_ukernel_x32__scalar(
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
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 1);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const float* b = (const float*) bias;
  float* packed_w = (float*) packed_weights;
  do {
    // NC main loop multiple of 32
    const float* w = (const float*) weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const uint32_t v0 = ((const uint32_t*)b)[0];
        const uint32_t v1 = ((const uint32_t*)b)[1];
        const uint32_t v2 = ((const uint32_t*)b)[2];
        const uint32_t v3 = ((const uint32_t*)b)[3];
        const uint32_t v4 = ((const uint32_t*)b)[4];
        const uint32_t v5 = ((const uint32_t*)b)[5];
        const uint32_t v6 = ((const uint32_t*)b)[6];
        const uint32_t v7 = ((const uint32_t*)b)[7];
        const uint32_t v8 = ((const uint32_t*)b)[8];
        const uint32_t v9 = ((const uint32_t*)b)[9];
        const uint32_t v10 = ((const uint32_t*)b)[10];
        const uint32_t v11 = ((const uint32_t*)b)[11];
        const uint32_t v12 = ((const uint32_t*)b)[12];
        const uint32_t v13 = ((const uint32_t*)b)[13];
        const uint32_t v14 = ((const uint32_t*)b)[14];
        const uint32_t v15 = ((const uint32_t*)b)[15];
        const uint32_t v16 = ((const uint32_t*)b)[16];
        const uint32_t v17 = ((const uint32_t*)b)[17];
        const uint32_t v18 = ((const uint32_t*)b)[18];
        const uint32_t v19 = ((const uint32_t*)b)[19];
        const uint32_t v20 = ((const uint32_t*)b)[20];
        const uint32_t v21 = ((const uint32_t*)b)[21];
        const uint32_t v22 = ((const uint32_t*)b)[22];
        const uint32_t v23 = ((const uint32_t*)b)[23];
        const uint32_t v24 = ((const uint32_t*)b)[24];
        const uint32_t v25 = ((const uint32_t*)b)[25];
        const uint32_t v26 = ((const uint32_t*)b)[26];
        const uint32_t v27 = ((const uint32_t*)b)[27];
        const uint32_t v28 = ((const uint32_t*)b)[28];
        const uint32_t v29 = ((const uint32_t*)b)[29];
        const uint32_t v30 = ((const uint32_t*)b)[30];
        const uint32_t v31 = ((const uint32_t*)b)[31];
        ((uint32_t*)packed_w)[0] = v0;
        ((uint32_t*)packed_w)[1] = v1;
        ((uint32_t*)packed_w)[2] = v2;
        ((uint32_t*)packed_w)[3] = v3;
        ((uint32_t*)packed_w)[4] = v4;
        ((uint32_t*)packed_w)[5] = v5;
        ((uint32_t*)packed_w)[6] = v6;
        ((uint32_t*)packed_w)[7] = v7;
        ((uint32_t*)packed_w)[8] = v8;
        ((uint32_t*)packed_w)[9] = v9;
        ((uint32_t*)packed_w)[10] = v10;
        ((uint32_t*)packed_w)[11] = v11;
        ((uint32_t*)packed_w)[12] = v12;
        ((uint32_t*)packed_w)[13] = v13;
        ((uint32_t*)packed_w)[14] = v14;
        ((uint32_t*)packed_w)[15] = v15;
        ((uint32_t*)packed_w)[16] = v16;
        ((uint32_t*)packed_w)[17] = v17;
        ((uint32_t*)packed_w)[18] = v18;
        ((uint32_t*)packed_w)[19] = v19;
        ((uint32_t*)packed_w)[20] = v20;
        ((uint32_t*)packed_w)[21] = v21;
        ((uint32_t*)packed_w)[22] = v22;
        ((uint32_t*)packed_w)[23] = v23;
        ((uint32_t*)packed_w)[24] = v24;
        ((uint32_t*)packed_w)[25] = v25;
        ((uint32_t*)packed_w)[26] = v26;
        ((uint32_t*)packed_w)[27] = v27;
        ((uint32_t*)packed_w)[28] = v28;
        ((uint32_t*)packed_w)[29] = v29;
        ((uint32_t*)packed_w)[30] = v30;
        ((uint32_t*)packed_w)[31] = v31;
        b += 32;
      } else {
        ((uint32_t*)packed_w)[0] = 0;
        ((uint32_t*)packed_w)[1] = 0;
        ((uint32_t*)packed_w)[2] = 0;
        ((uint32_t*)packed_w)[3] = 0;
        ((uint32_t*)packed_w)[4] = 0;
        ((uint32_t*)packed_w)[5] = 0;
        ((uint32_t*)packed_w)[6] = 0;
        ((uint32_t*)packed_w)[7] = 0;
        ((uint32_t*)packed_w)[8] = 0;
        ((uint32_t*)packed_w)[9] = 0;
        ((uint32_t*)packed_w)[10] = 0;
        ((uint32_t*)packed_w)[11] = 0;
        ((uint32_t*)packed_w)[12] = 0;
        ((uint32_t*)packed_w)[13] = 0;
        ((uint32_t*)packed_w)[14] = 0;
        ((uint32_t*)packed_w)[15] = 0;
        ((uint32_t*)packed_w)[16] = 0;
        ((uint32_t*)packed_w)[17] = 0;
        ((uint32_t*)packed_w)[18] = 0;
        ((uint32_t*)packed_w)[19] = 0;
        ((uint32_t*)packed_w)[20] = 0;
        ((uint32_t*)packed_w)[21] = 0;
        ((uint32_t*)packed_w)[22] = 0;
        ((uint32_t*)packed_w)[23] = 0;
        ((uint32_t*)packed_w)[24] = 0;
        ((uint32_t*)packed_w)[25] = 0;
        ((uint32_t*)packed_w)[26] = 0;
        ((uint32_t*)packed_w)[27] = 0;
        ((uint32_t*)packed_w)[28] = 0;
        ((uint32_t*)packed_w)[29] = 0;
        ((uint32_t*)packed_w)[30] = 0;
        ((uint32_t*)packed_w)[31] = 0;
      }
      packed_w += 32;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint32_t v0 = ((const uint32_t*)w)[0];
        const uint32_t v1 = ((const uint32_t*)w)[1];
        const uint32_t v2 = ((const uint32_t*)w)[2];
        const uint32_t v3 = ((const uint32_t*)w)[3];
        const uint32_t v4 = ((const uint32_t*)w)[4];
        const uint32_t v5 = ((const uint32_t*)w)[5];
        const uint32_t v6 = ((const uint32_t*)w)[6];
        const uint32_t v7 = ((const uint32_t*)w)[7];
        const uint32_t v8 = ((const uint32_t*)w)[8];
        const uint32_t v9 = ((const uint32_t*)w)[9];
        const uint32_t v10 = ((const uint32_t*)w)[10];
        const uint32_t v11 = ((const uint32_t*)w)[11];
        const uint32_t v12 = ((const uint32_t*)w)[12];
        const uint32_t v13 = ((const uint32_t*)w)[13];
        const uint32_t v14 = ((const uint32_t*)w)[14];
        const uint32_t v15 = ((const uint32_t*)w)[15];
        const uint32_t v16 = ((const uint32_t*)w)[16];
        const uint32_t v17 = ((const uint32_t*)w)[17];
        const uint32_t v18 = ((const uint32_t*)w)[18];
        const uint32_t v19 = ((const uint32_t*)w)[19];
        const uint32_t v20 = ((const uint32_t*)w)[20];
        const uint32_t v21 = ((const uint32_t*)w)[21];
        const uint32_t v22 = ((const uint32_t*)w)[22];
        const uint32_t v23 = ((const uint32_t*)w)[23];
        const uint32_t v24 = ((const uint32_t*)w)[24];
        const uint32_t v25 = ((const uint32_t*)w)[25];
        const uint32_t v26 = ((const uint32_t*)w)[26];
        const uint32_t v27 = ((const uint32_t*)w)[27];
        const uint32_t v28 = ((const uint32_t*)w)[28];
        const uint32_t v29 = ((const uint32_t*)w)[29];
        const uint32_t v30 = ((const uint32_t*)w)[30];
        const uint32_t v31 = ((const uint32_t*)w)[31];
        ((uint32_t*)packed_w)[0] = v0;
        ((uint32_t*)packed_w)[1] = v1;
        ((uint32_t*)packed_w)[2] = v2;
        ((uint32_t*)packed_w)[3] = v3;
        ((uint32_t*)packed_w)[4] = v4;
        ((uint32_t*)packed_w)[5] = v5;
        ((uint32_t*)packed_w)[6] = v6;
        ((uint32_t*)packed_w)[7] = v7;
        ((uint32_t*)packed_w)[8] = v8;
        ((uint32_t*)packed_w)[9] = v9;
        ((uint32_t*)packed_w)[10] = v10;
        ((uint32_t*)packed_w)[11] = v11;
        ((uint32_t*)packed_w)[12] = v12;
        ((uint32_t*)packed_w)[13] = v13;
        ((uint32_t*)packed_w)[14] = v14;
        ((uint32_t*)packed_w)[15] = v15;
        ((uint32_t*)packed_w)[16] = v16;
        ((uint32_t*)packed_w)[17] = v17;
        ((uint32_t*)packed_w)[18] = v18;
        ((uint32_t*)packed_w)[19] = v19;
        ((uint32_t*)packed_w)[20] = v20;
        ((uint32_t*)packed_w)[21] = v21;
        ((uint32_t*)packed_w)[22] = v22;
        ((uint32_t*)packed_w)[23] = v23;
        ((uint32_t*)packed_w)[24] = v24;
        ((uint32_t*)packed_w)[25] = v25;
        ((uint32_t*)packed_w)[26] = v26;
        ((uint32_t*)packed_w)[27] = v27;
        ((uint32_t*)packed_w)[28] = v28;
        ((uint32_t*)packed_w)[29] = v29;
        ((uint32_t*)packed_w)[30] = v30;
        ((uint32_t*)packed_w)[31] = v31;
        w += k_stride;
        packed_w += 32;
      }
      w = w - kc * k_stride + 32;  // Advance to next column of 32 floats
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);

      if XNN_LIKELY(b != NULL) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = b[i];
        }
        b += n;
      } else {
        ((uint32_t*)packed_w)[0] = 0;
        ((uint32_t*)packed_w)[1] = 0;
        ((uint32_t*)packed_w)[2] = 0;
        ((uint32_t*)packed_w)[3] = 0;
        ((uint32_t*)packed_w)[4] = 0;
        ((uint32_t*)packed_w)[5] = 0;
        ((uint32_t*)packed_w)[6] = 0;
        ((uint32_t*)packed_w)[7] = 0;
        ((uint32_t*)packed_w)[8] = 0;
        ((uint32_t*)packed_w)[9] = 0;
        ((uint32_t*)packed_w)[10] = 0;
        ((uint32_t*)packed_w)[11] = 0;
        ((uint32_t*)packed_w)[12] = 0;
        ((uint32_t*)packed_w)[13] = 0;
        ((uint32_t*)packed_w)[14] = 0;
        ((uint32_t*)packed_w)[15] = 0;
        ((uint32_t*)packed_w)[16] = 0;
        ((uint32_t*)packed_w)[17] = 0;
        ((uint32_t*)packed_w)[18] = 0;
        ((uint32_t*)packed_w)[19] = 0;
        ((uint32_t*)packed_w)[20] = 0;
        ((uint32_t*)packed_w)[21] = 0;
        ((uint32_t*)packed_w)[22] = 0;
        ((uint32_t*)packed_w)[23] = 0;
        ((uint32_t*)packed_w)[24] = 0;
        ((uint32_t*)packed_w)[25] = 0;
        ((uint32_t*)packed_w)[26] = 0;
        ((uint32_t*)packed_w)[27] = 0;
        ((uint32_t*)packed_w)[28] = 0;
        ((uint32_t*)packed_w)[29] = 0;
        ((uint32_t*)packed_w)[30] = 0;
        ((uint32_t*)packed_w)[31] = 0;
      }
      packed_w += 32;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        for (size_t i = 0; i < n; ++i) {
          packed_w[i] = w[i];
        }
        w += k_stride;
        packed_w += 32;
      }
    }
    weights += nc * kc;
  } while (--g != 0);
}
