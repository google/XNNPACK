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

  const uint32_t* b = bias;
  uint32_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 32
    const uint32_t* w = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const uint32_t v0 = b[0];
        const uint32_t v1 = b[1];
        const uint32_t v2 = b[2];
        const uint32_t v3 = b[3];
        const uint32_t v4 = b[4];
        const uint32_t v5 = b[5];
        const uint32_t v6 = b[6];
        const uint32_t v7 = b[7];
        const uint32_t v8 = b[8];
        const uint32_t v9 = b[9];
        const uint32_t v10 = b[10];
        const uint32_t v11 = b[11];
        const uint32_t v12 = b[12];
        const uint32_t v13 = b[13];
        const uint32_t v14 = b[14];
        const uint32_t v15 = b[15];
        const uint32_t v16 = b[16];
        const uint32_t v17 = b[17];
        const uint32_t v18 = b[18];
        const uint32_t v19 = b[19];
        const uint32_t v20 = b[20];
        const uint32_t v21 = b[21];
        const uint32_t v22 = b[22];
        const uint32_t v23 = b[23];
        const uint32_t v24 = b[24];
        const uint32_t v25 = b[25];
        const uint32_t v26 = b[26];
        const uint32_t v27 = b[27];
        const uint32_t v28 = b[28];
        const uint32_t v29 = b[29];
        const uint32_t v30 = b[30];
        const uint32_t v31 = b[31];
        packed_w[0] = v0;
        packed_w[1] = v1;
        packed_w[2] = v2;
        packed_w[3] = v3;
        packed_w[4] = v4;
        packed_w[5] = v5;
        packed_w[6] = v6;
        packed_w[7] = v7;
        packed_w[8] = v8;
        packed_w[9] = v9;
        packed_w[10] = v10;
        packed_w[11] = v11;
        packed_w[12] = v12;
        packed_w[13] = v13;
        packed_w[14] = v14;
        packed_w[15] = v15;
        packed_w[16] = v16;
        packed_w[17] = v17;
        packed_w[18] = v18;
        packed_w[19] = v19;
        packed_w[20] = v20;
        packed_w[21] = v21;
        packed_w[22] = v22;
        packed_w[23] = v23;
        packed_w[24] = v24;
        packed_w[25] = v25;
        packed_w[26] = v26;
        packed_w[27] = v27;
        packed_w[28] = v28;
        packed_w[29] = v29;
        packed_w[30] = v30;
        packed_w[31] = v31;
        b += 32;
      } else {
        packed_w[0] = 0;
        packed_w[1] = 0;
        packed_w[2] = 0;
        packed_w[3] = 0;
        packed_w[4] = 0;
        packed_w[5] = 0;
        packed_w[6] = 0;
        packed_w[7] = 0;
        packed_w[8] = 0;
        packed_w[9] = 0;
        packed_w[10] = 0;
        packed_w[11] = 0;
        packed_w[12] = 0;
        packed_w[13] = 0;
        packed_w[14] = 0;
        packed_w[15] = 0;
        packed_w[16] = 0;
        packed_w[17] = 0;
        packed_w[18] = 0;
        packed_w[19] = 0;
        packed_w[20] = 0;
        packed_w[21] = 0;
        packed_w[22] = 0;
        packed_w[23] = 0;
        packed_w[24] = 0;
        packed_w[25] = 0;
        packed_w[26] = 0;
        packed_w[27] = 0;
        packed_w[28] = 0;
        packed_w[29] = 0;
        packed_w[30] = 0;
        packed_w[31] = 0;
      }
      packed_w += 32;

      // KC main loop
      for (size_t k = kc; k > 0; --k) {
        const uint32_t v0 = w[0];
        const uint32_t v1 = w[1];
        const uint32_t v2 = w[2];
        const uint32_t v3 = w[3];
        const uint32_t v4 = w[4];
        const uint32_t v5 = w[5];
        const uint32_t v6 = w[6];
        const uint32_t v7 = w[7];
        const uint32_t v8 = w[8];
        const uint32_t v9 = w[9];
        const uint32_t v10 = w[10];
        const uint32_t v11 = w[11];
        const uint32_t v12 = w[12];
        const uint32_t v13 = w[13];
        const uint32_t v14 = w[14];
        const uint32_t v15 = w[15];
        const uint32_t v16 = w[16];
        const uint32_t v17 = w[17];
        const uint32_t v18 = w[18];
        const uint32_t v19 = w[19];
        const uint32_t v20 = w[20];
        const uint32_t v21 = w[21];
        const uint32_t v22 = w[22];
        const uint32_t v23 = w[23];
        const uint32_t v24 = w[24];
        const uint32_t v25 = w[25];
        const uint32_t v26 = w[26];
        const uint32_t v27 = w[27];
        const uint32_t v28 = w[28];
        const uint32_t v29 = w[29];
        const uint32_t v30 = w[30];
        const uint32_t v31 = w[31];
        packed_w[0] = v0;
        packed_w[1] = v1;
        packed_w[2] = v2;
        packed_w[3] = v3;
        packed_w[4] = v4;
        packed_w[5] = v5;
        packed_w[6] = v6;
        packed_w[7] = v7;
        packed_w[8] = v8;
        packed_w[9] = v9;
        packed_w[10] = v10;
        packed_w[11] = v11;
        packed_w[12] = v12;
        packed_w[13] = v13;
        packed_w[14] = v14;
        packed_w[15] = v15;
        packed_w[16] = v16;
        packed_w[17] = v17;
        packed_w[18] = v18;
        packed_w[19] = v19;
        packed_w[20] = v20;
        packed_w[21] = v21;
        packed_w[22] = v22;
        packed_w[23] = v23;
        packed_w[24] = v24;
        packed_w[25] = v25;
        packed_w[26] = v26;
        packed_w[27] = v27;
        packed_w[28] = v28;
        packed_w[29] = v29;
        packed_w[30] = v30;
        packed_w[31] = v31;
        w += k_stride;
        packed_w += 32;
      }
      w = w - kc * k_stride + 32;  // Advance to next column of 32 uint32_t
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
        packed_w[0] = 0;
        packed_w[1] = 0;
        packed_w[2] = 0;
        packed_w[3] = 0;
        packed_w[4] = 0;
        packed_w[5] = 0;
        packed_w[6] = 0;
        packed_w[7] = 0;
        packed_w[8] = 0;
        packed_w[9] = 0;
        packed_w[10] = 0;
        packed_w[11] = 0;
        packed_w[12] = 0;
        packed_w[13] = 0;
        packed_w[14] = 0;
        packed_w[15] = 0;
        packed_w[16] = 0;
        packed_w[17] = 0;
        packed_w[18] = 0;
        packed_w[19] = 0;
        packed_w[20] = 0;
        packed_w[21] = 0;
        packed_w[22] = 0;
        packed_w[23] = 0;
        packed_w[24] = 0;
        packed_w[25] = 0;
        packed_w[26] = 0;
        packed_w[27] = 0;
        packed_w[28] = 0;
        packed_w[29] = 0;
        packed_w[30] = 0;
        packed_w[31] = 0;
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
