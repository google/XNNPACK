// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"

void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x4c8__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 4);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);

  do {
    // NC main loop multiple of 4
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 4; n -= 4) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        ((int32_t*) out)[0] = b[0];
        ((int32_t*) out)[1] = b[1];
        ((int32_t*) out)[2] = b[2];
        ((int32_t*) out)[3] = b[3];
        b += 4;
      } else {
        ((int32_t*) out)[0] = 0;
        ((int32_t*) out)[1] = 0;
        ((int32_t*) out)[2] = 0;
        ((int32_t*) out)[3] = 0;
      }
      out += 4 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;
      uint32_t ksum3 = 0;

      // KC main loop multiple of 4x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        const int8_t v0x4 = w0[4];
        const int8_t v0x5 = w0[5];
        const int8_t v0x6 = w0[6];
        const int8_t v0x7 = w0[7];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += 8;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        const int8_t v1x4 = w1[4];
        const int8_t v1x5 = w1[5];
        const int8_t v1x6 = w1[6];
        const int8_t v1x7 = w1[7];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += 8;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        const int8_t v2x4 = w2[4];
        const int8_t v2x5 = w2[5];
        const int8_t v2x6 = w2[6];
        const int8_t v2x7 = w2[7];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += 8;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = w3[1];
        const int8_t v3x2 = w3[2];
        const int8_t v3x3 = w3[3];
        const int8_t v3x4 = w3[4];
        const int8_t v3x5 = w3[5];
        const int8_t v3x6 = w3[6];
        const int8_t v3x7 = w3[7];
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        ksum3 += (uint32_t) v3x4;
        ksum3 += (uint32_t) v3x5;
        ksum3 += (uint32_t) v3x6;
        ksum3 += (uint32_t) v3x7;
        out[24] = v3x0;
        out[25] = v3x1;
        out[26] = v3x2;
        out[27] = v3x3;
        out[28] = v3x4;
        out[29] = v3x5;
        out[30] = v3x6;
        out[31] = v3x7;
        w3 += 8;
        out += 32;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = 1 < k ? w0[1] : 0;
        const int8_t v0x2 = 2 < k ? w0[2] : 0;
        const int8_t v0x3 = 3 < k ? w0[3] : 0;
        const int8_t v0x4 = 4 < k ? w0[4] : 0;
        const int8_t v0x5 = 5 < k ? w0[5] : 0;
        const int8_t v0x6 = 6 < k ? w0[6] : 0;
        const int8_t v0x7 = 7 < k ? w0[7] : 0;
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += k;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = 1 < k ? w1[1] : 0;
        const int8_t v1x2 = 2 < k ? w1[2] : 0;
        const int8_t v1x3 = 3 < k ? w1[3] : 0;
        const int8_t v1x4 = 4 < k ? w1[4] : 0;
        const int8_t v1x5 = 5 < k ? w1[5] : 0;
        const int8_t v1x6 = 6 < k ? w1[6] : 0;
        const int8_t v1x7 = 7 < k ? w1[7] : 0;
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += k;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = 1 < k ? w2[1] : 0;
        const int8_t v2x2 = 2 < k ? w2[2] : 0;
        const int8_t v2x3 = 3 < k ? w2[3] : 0;
        const int8_t v2x4 = 4 < k ? w2[4] : 0;
        const int8_t v2x5 = 5 < k ? w2[5] : 0;
        const int8_t v2x6 = 6 < k ? w2[6] : 0;
        const int8_t v2x7 = 7 < k ? w2[7] : 0;
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += k;
        const int8_t v3x0 = w3[0];
        const int8_t v3x1 = 1 < k ? w3[1] : 0;
        const int8_t v3x2 = 2 < k ? w3[2] : 0;
        const int8_t v3x3 = 3 < k ? w3[3] : 0;
        const int8_t v3x4 = 4 < k ? w3[4] : 0;
        const int8_t v3x5 = 5 < k ? w3[5] : 0;
        const int8_t v3x6 = 6 < k ? w3[6] : 0;
        const int8_t v3x7 = 7 < k ? w3[7] : 0;
        ksum3 += (uint32_t) v3x0;
        ksum3 += (uint32_t) v3x1;
        ksum3 += (uint32_t) v3x2;
        ksum3 += (uint32_t) v3x3;
        ksum3 += (uint32_t) v3x4;
        ksum3 += (uint32_t) v3x5;
        ksum3 += (uint32_t) v3x6;
        ksum3 += (uint32_t) v3x7;
        out[24] = v3x0;
        out[25] = v3x1;
        out[26] = v3x2;
        out[27] = v3x3;
        out[28] = v3x4;
        out[29] = v3x5;
        out[30] = v3x6;
        out[31] = v3x7;
        w3 += k;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      packed_b[3] -= ksum3 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w3;
    }

    // NC remainder (1..3)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((int32_t*) out) = *b++;
          out += sizeof(int32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((int32_t*) out) = 0;
          out += sizeof(int32_t);
        } while (--nb != 0);
      }
      out += (4 - n) * sizeof(int32_t);

     // NR remainder has less than 4 rows so last row is not loaded
      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }

      uint32_t ksum0 = 0;
      uint32_t ksum1 = 0;
      uint32_t ksum2 = 0;

      // KC main loop multiple of 4x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = w0[1];
        const int8_t v0x2 = w0[2];
        const int8_t v0x3 = w0[3];
        const int8_t v0x4 = w0[4];
        const int8_t v0x5 = w0[5];
        const int8_t v0x6 = w0[6];
        const int8_t v0x7 = w0[7];
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += 8;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = w1[1];
        const int8_t v1x2 = w1[2];
        const int8_t v1x3 = w1[3];
        const int8_t v1x4 = w1[4];
        const int8_t v1x5 = w1[5];
        const int8_t v1x6 = w1[6];
        const int8_t v1x7 = w1[7];
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += 8;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = w2[1];
        const int8_t v2x2 = w2[2];
        const int8_t v2x3 = w2[3];
        const int8_t v2x4 = w2[4];
        const int8_t v2x5 = w2[5];
        const int8_t v2x6 = w2[6];
        const int8_t v2x7 = w2[7];
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += 8;
        out += 32;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        const int8_t v0x0 = w0[0];
        const int8_t v0x1 = 1 < k ? w0[1] : 0;
        const int8_t v0x2 = 2 < k ? w0[2] : 0;
        const int8_t v0x3 = 3 < k ? w0[3] : 0;
        const int8_t v0x4 = 4 < k ? w0[4] : 0;
        const int8_t v0x5 = 5 < k ? w0[5] : 0;
        const int8_t v0x6 = 6 < k ? w0[6] : 0;
        const int8_t v0x7 = 7 < k ? w0[7] : 0;
        ksum0 += (uint32_t) v0x0;
        ksum0 += (uint32_t) v0x1;
        ksum0 += (uint32_t) v0x2;
        ksum0 += (uint32_t) v0x3;
        ksum0 += (uint32_t) v0x4;
        ksum0 += (uint32_t) v0x5;
        ksum0 += (uint32_t) v0x6;
        ksum0 += (uint32_t) v0x7;
        out[0] = v0x0;
        out[1] = v0x1;
        out[2] = v0x2;
        out[3] = v0x3;
        out[4] = v0x4;
        out[5] = v0x5;
        out[6] = v0x6;
        out[7] = v0x7;
        w0 += k;
        const int8_t v1x0 = w1[0];
        const int8_t v1x1 = 1 < k ? w1[1] : 0;
        const int8_t v1x2 = 2 < k ? w1[2] : 0;
        const int8_t v1x3 = 3 < k ? w1[3] : 0;
        const int8_t v1x4 = 4 < k ? w1[4] : 0;
        const int8_t v1x5 = 5 < k ? w1[5] : 0;
        const int8_t v1x6 = 6 < k ? w1[6] : 0;
        const int8_t v1x7 = 7 < k ? w1[7] : 0;
        ksum1 += (uint32_t) v1x0;
        ksum1 += (uint32_t) v1x1;
        ksum1 += (uint32_t) v1x2;
        ksum1 += (uint32_t) v1x3;
        ksum1 += (uint32_t) v1x4;
        ksum1 += (uint32_t) v1x5;
        ksum1 += (uint32_t) v1x6;
        ksum1 += (uint32_t) v1x7;
        out[8] = v1x0;
        out[9] = v1x1;
        out[10] = v1x2;
        out[11] = v1x3;
        out[12] = v1x4;
        out[13] = v1x5;
        out[14] = v1x6;
        out[15] = v1x7;
        w1 += k;
        const int8_t v2x0 = w2[0];
        const int8_t v2x1 = 1 < k ? w2[1] : 0;
        const int8_t v2x2 = 2 < k ? w2[2] : 0;
        const int8_t v2x3 = 3 < k ? w2[3] : 0;
        const int8_t v2x4 = 4 < k ? w2[4] : 0;
        const int8_t v2x5 = 5 < k ? w2[5] : 0;
        const int8_t v2x6 = 6 < k ? w2[6] : 0;
        const int8_t v2x7 = 7 < k ? w2[7] : 0;
        ksum2 += (uint32_t) v2x0;
        ksum2 += (uint32_t) v2x1;
        ksum2 += (uint32_t) v2x2;
        ksum2 += (uint32_t) v2x3;
        ksum2 += (uint32_t) v2x4;
        ksum2 += (uint32_t) v2x5;
        ksum2 += (uint32_t) v2x6;
        ksum2 += (uint32_t) v2x7;
        out[16] = v2x0;
        out[17] = v2x1;
        out[18] = v2x2;
        out[19] = v2x3;
        out[20] = v2x4;
        out[21] = v2x5;
        out[22] = v2x6;
        out[23] = v2x7;
        w2 += k;
        out[24] = 0;
        out[25] = 0;
        out[26] = 0;
        out[27] = 0;
        out[28] = 0;
        out[29] = 0;
        out[30] = 0;
        out[31] = 0;
        out += 32;
      }

      packed_b[0] -= ksum0 * izp;
      packed_b[1] -= ksum1 * izp;
      packed_b[2] -= ksum2 * izp;
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
