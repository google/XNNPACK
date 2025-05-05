// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.



#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"

void xnn_qs8_to_qu8_packw_gemm_gio_ukernel_x8c8__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
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
  assert(nr == 8);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128): 128);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, 8 * sizeof(int32_t));
        b += 8;
      } else {
        memset(out, 0, 8 * sizeof(int32_t));
      }
      out += 8 * sizeof(int32_t);

      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;
      const int8_t* w4 = w3 + k_stride;
      const int8_t* w5 = w4 + k_stride;
      const int8_t* w6 = w5 + k_stride;
      const int8_t* w7 = w6 + k_stride;
      uint32_t ksum[8] = {0,};

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        for (size_t no = 0; no < 8; no += 8) {
          const int8_t v0x0 = w0[no + 0];
          const int8_t v1x0 = w1[no + 0];
          const int8_t v2x0 = w2[no + 0];
          const int8_t v3x0 = w3[no + 0];
          const int8_t v4x0 = w4[no + 0];
          const int8_t v5x0 = w5[no + 0];
          const int8_t v6x0 = w6[no + 0];
          const int8_t v7x0 = w7[no + 0];
          ksum[no + 0] += (uint32_t) v0x0;
          ksum[no + 0] += (uint32_t) v1x0;
          ksum[no + 0] += (uint32_t) v2x0;
          ksum[no + 0] += (uint32_t) v3x0;
          ksum[no + 0] += (uint32_t) v4x0;
          ksum[no + 0] += (uint32_t) v5x0;
          ksum[no + 0] += (uint32_t) v6x0;
          ksum[no + 0] += (uint32_t) v7x0;
          out[0] = v0x0;
          out[1] = v1x0;
          out[2] = v2x0;
          out[3] = v3x0;
          out[4] = v4x0;
          out[5] = v5x0;
          out[6] = v6x0;
          out[7] = v7x0;
          const int8_t v0x1 = w0[no + 1];
          const int8_t v1x1 = w1[no + 1];
          const int8_t v2x1 = w2[no + 1];
          const int8_t v3x1 = w3[no + 1];
          const int8_t v4x1 = w4[no + 1];
          const int8_t v5x1 = w5[no + 1];
          const int8_t v6x1 = w6[no + 1];
          const int8_t v7x1 = w7[no + 1];
          ksum[no + 1] += (uint32_t) v0x1;
          ksum[no + 1] += (uint32_t) v1x1;
          ksum[no + 1] += (uint32_t) v2x1;
          ksum[no + 1] += (uint32_t) v3x1;
          ksum[no + 1] += (uint32_t) v4x1;
          ksum[no + 1] += (uint32_t) v5x1;
          ksum[no + 1] += (uint32_t) v6x1;
          ksum[no + 1] += (uint32_t) v7x1;
          out[8] = v0x1;
          out[9] = v1x1;
          out[10] = v2x1;
          out[11] = v3x1;
          out[12] = v4x1;
          out[13] = v5x1;
          out[14] = v6x1;
          out[15] = v7x1;
          const int8_t v0x2 = w0[no + 2];
          const int8_t v1x2 = w1[no + 2];
          const int8_t v2x2 = w2[no + 2];
          const int8_t v3x2 = w3[no + 2];
          const int8_t v4x2 = w4[no + 2];
          const int8_t v5x2 = w5[no + 2];
          const int8_t v6x2 = w6[no + 2];
          const int8_t v7x2 = w7[no + 2];
          ksum[no + 2] += (uint32_t) v0x2;
          ksum[no + 2] += (uint32_t) v1x2;
          ksum[no + 2] += (uint32_t) v2x2;
          ksum[no + 2] += (uint32_t) v3x2;
          ksum[no + 2] += (uint32_t) v4x2;
          ksum[no + 2] += (uint32_t) v5x2;
          ksum[no + 2] += (uint32_t) v6x2;
          ksum[no + 2] += (uint32_t) v7x2;
          out[16] = v0x2;
          out[17] = v1x2;
          out[18] = v2x2;
          out[19] = v3x2;
          out[20] = v4x2;
          out[21] = v5x2;
          out[22] = v6x2;
          out[23] = v7x2;
          const int8_t v0x3 = w0[no + 3];
          const int8_t v1x3 = w1[no + 3];
          const int8_t v2x3 = w2[no + 3];
          const int8_t v3x3 = w3[no + 3];
          const int8_t v4x3 = w4[no + 3];
          const int8_t v5x3 = w5[no + 3];
          const int8_t v6x3 = w6[no + 3];
          const int8_t v7x3 = w7[no + 3];
          ksum[no + 3] += (uint32_t) v0x3;
          ksum[no + 3] += (uint32_t) v1x3;
          ksum[no + 3] += (uint32_t) v2x3;
          ksum[no + 3] += (uint32_t) v3x3;
          ksum[no + 3] += (uint32_t) v4x3;
          ksum[no + 3] += (uint32_t) v5x3;
          ksum[no + 3] += (uint32_t) v6x3;
          ksum[no + 3] += (uint32_t) v7x3;
          out[24] = v0x3;
          out[25] = v1x3;
          out[26] = v2x3;
          out[27] = v3x3;
          out[28] = v4x3;
          out[29] = v5x3;
          out[30] = v6x3;
          out[31] = v7x3;
          const int8_t v0x4 = w0[no + 4];
          const int8_t v1x4 = w1[no + 4];
          const int8_t v2x4 = w2[no + 4];
          const int8_t v3x4 = w3[no + 4];
          const int8_t v4x4 = w4[no + 4];
          const int8_t v5x4 = w5[no + 4];
          const int8_t v6x4 = w6[no + 4];
          const int8_t v7x4 = w7[no + 4];
          ksum[no + 4] += (uint32_t) v0x4;
          ksum[no + 4] += (uint32_t) v1x4;
          ksum[no + 4] += (uint32_t) v2x4;
          ksum[no + 4] += (uint32_t) v3x4;
          ksum[no + 4] += (uint32_t) v4x4;
          ksum[no + 4] += (uint32_t) v5x4;
          ksum[no + 4] += (uint32_t) v6x4;
          ksum[no + 4] += (uint32_t) v7x4;
          out[32] = v0x4;
          out[33] = v1x4;
          out[34] = v2x4;
          out[35] = v3x4;
          out[36] = v4x4;
          out[37] = v5x4;
          out[38] = v6x4;
          out[39] = v7x4;
          const int8_t v0x5 = w0[no + 5];
          const int8_t v1x5 = w1[no + 5];
          const int8_t v2x5 = w2[no + 5];
          const int8_t v3x5 = w3[no + 5];
          const int8_t v4x5 = w4[no + 5];
          const int8_t v5x5 = w5[no + 5];
          const int8_t v6x5 = w6[no + 5];
          const int8_t v7x5 = w7[no + 5];
          ksum[no + 5] += (uint32_t) v0x5;
          ksum[no + 5] += (uint32_t) v1x5;
          ksum[no + 5] += (uint32_t) v2x5;
          ksum[no + 5] += (uint32_t) v3x5;
          ksum[no + 5] += (uint32_t) v4x5;
          ksum[no + 5] += (uint32_t) v5x5;
          ksum[no + 5] += (uint32_t) v6x5;
          ksum[no + 5] += (uint32_t) v7x5;
          out[40] = v0x5;
          out[41] = v1x5;
          out[42] = v2x5;
          out[43] = v3x5;
          out[44] = v4x5;
          out[45] = v5x5;
          out[46] = v6x5;
          out[47] = v7x5;
          const int8_t v0x6 = w0[no + 6];
          const int8_t v1x6 = w1[no + 6];
          const int8_t v2x6 = w2[no + 6];
          const int8_t v3x6 = w3[no + 6];
          const int8_t v4x6 = w4[no + 6];
          const int8_t v5x6 = w5[no + 6];
          const int8_t v6x6 = w6[no + 6];
          const int8_t v7x6 = w7[no + 6];
          ksum[no + 6] += (uint32_t) v0x6;
          ksum[no + 6] += (uint32_t) v1x6;
          ksum[no + 6] += (uint32_t) v2x6;
          ksum[no + 6] += (uint32_t) v3x6;
          ksum[no + 6] += (uint32_t) v4x6;
          ksum[no + 6] += (uint32_t) v5x6;
          ksum[no + 6] += (uint32_t) v6x6;
          ksum[no + 6] += (uint32_t) v7x6;
          out[48] = v0x6;
          out[49] = v1x6;
          out[50] = v2x6;
          out[51] = v3x6;
          out[52] = v4x6;
          out[53] = v5x6;
          out[54] = v6x6;
          out[55] = v7x6;
          const int8_t v0x7 = w0[no + 7];
          const int8_t v1x7 = w1[no + 7];
          const int8_t v2x7 = w2[no + 7];
          const int8_t v3x7 = w3[no + 7];
          const int8_t v4x7 = w4[no + 7];
          const int8_t v5x7 = w5[no + 7];
          const int8_t v6x7 = w6[no + 7];
          const int8_t v7x7 = w7[no + 7];
          ksum[no + 7] += (uint32_t) v0x7;
          ksum[no + 7] += (uint32_t) v1x7;
          ksum[no + 7] += (uint32_t) v2x7;
          ksum[no + 7] += (uint32_t) v3x7;
          ksum[no + 7] += (uint32_t) v4x7;
          ksum[no + 7] += (uint32_t) v5x7;
          ksum[no + 7] += (uint32_t) v6x7;
          ksum[no + 7] += (uint32_t) v7x7;
          out[56] = v0x7;
          out[57] = v1x7;
          out[58] = v2x7;
          out[59] = v3x7;
          out[60] = v4x7;
          out[61] = v5x7;
          out[62] = v6x7;
          out[63] = v7x7;
          out += 64;
        }
        w0 += 8 * k_stride;
        w1 += 8 * k_stride;
        w2 += 8 * k_stride;
        w3 += 8 * k_stride;
        w4 += 8 * k_stride;
        w5 += 8 * k_stride;
        w6 += 8 * k_stride;
        w7 += 8 * k_stride;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        for (size_t no = 0; no < 8; no += 8) {
          const int8_t v0x0 = w0[no + 0];
          const int8_t v1x0 = 1 < k ? w1[no + 0] : 0;
          const int8_t v2x0 = 2 < k ? w2[no + 0] : 0;
          const int8_t v3x0 = 3 < k ? w3[no + 0] : 0;
          const int8_t v4x0 = 4 < k ? w4[no + 0] : 0;
          const int8_t v5x0 = 5 < k ? w5[no + 0] : 0;
          const int8_t v6x0 = 6 < k ? w6[no + 0] : 0;
          const int8_t v7x0 = 7 < k ? w7[no + 0] : 0;
          ksum[no + 0] += (uint32_t) v0x0;
          ksum[no + 0] += (uint32_t) v1x0;
          ksum[no + 0] += (uint32_t) v2x0;
          ksum[no + 0] += (uint32_t) v3x0;
          ksum[no + 0] += (uint32_t) v4x0;
          ksum[no + 0] += (uint32_t) v5x0;
          ksum[no + 0] += (uint32_t) v6x0;
          ksum[no + 0] += (uint32_t) v7x0;
          out[0] = v0x0;
          out[1] = v1x0;
          out[2] = v2x0;
          out[3] = v3x0;
          out[4] = v4x0;
          out[5] = v5x0;
          out[6] = v6x0;
          out[7] = v7x0;
          const int8_t v0x1 = w0[no + 1];
          const int8_t v1x1 = 1 < k ? w1[no + 1] : 0;
          const int8_t v2x1 = 2 < k ? w2[no + 1] : 0;
          const int8_t v3x1 = 3 < k ? w3[no + 1] : 0;
          const int8_t v4x1 = 4 < k ? w4[no + 1] : 0;
          const int8_t v5x1 = 5 < k ? w5[no + 1] : 0;
          const int8_t v6x1 = 6 < k ? w6[no + 1] : 0;
          const int8_t v7x1 = 7 < k ? w7[no + 1] : 0;
          ksum[no + 1] += (uint32_t) v0x1;
          ksum[no + 1] += (uint32_t) v1x1;
          ksum[no + 1] += (uint32_t) v2x1;
          ksum[no + 1] += (uint32_t) v3x1;
          ksum[no + 1] += (uint32_t) v4x1;
          ksum[no + 1] += (uint32_t) v5x1;
          ksum[no + 1] += (uint32_t) v6x1;
          ksum[no + 1] += (uint32_t) v7x1;
          out[8] = v0x1;
          out[9] = v1x1;
          out[10] = v2x1;
          out[11] = v3x1;
          out[12] = v4x1;
          out[13] = v5x1;
          out[14] = v6x1;
          out[15] = v7x1;
          const int8_t v0x2 = w0[no + 2];
          const int8_t v1x2 = 1 < k ? w1[no + 2] : 0;
          const int8_t v2x2 = 2 < k ? w2[no + 2] : 0;
          const int8_t v3x2 = 3 < k ? w3[no + 2] : 0;
          const int8_t v4x2 = 4 < k ? w4[no + 2] : 0;
          const int8_t v5x2 = 5 < k ? w5[no + 2] : 0;
          const int8_t v6x2 = 6 < k ? w6[no + 2] : 0;
          const int8_t v7x2 = 7 < k ? w7[no + 2] : 0;
          ksum[no + 2] += (uint32_t) v0x2;
          ksum[no + 2] += (uint32_t) v1x2;
          ksum[no + 2] += (uint32_t) v2x2;
          ksum[no + 2] += (uint32_t) v3x2;
          ksum[no + 2] += (uint32_t) v4x2;
          ksum[no + 2] += (uint32_t) v5x2;
          ksum[no + 2] += (uint32_t) v6x2;
          ksum[no + 2] += (uint32_t) v7x2;
          out[16] = v0x2;
          out[17] = v1x2;
          out[18] = v2x2;
          out[19] = v3x2;
          out[20] = v4x2;
          out[21] = v5x2;
          out[22] = v6x2;
          out[23] = v7x2;
          const int8_t v0x3 = w0[no + 3];
          const int8_t v1x3 = 1 < k ? w1[no + 3] : 0;
          const int8_t v2x3 = 2 < k ? w2[no + 3] : 0;
          const int8_t v3x3 = 3 < k ? w3[no + 3] : 0;
          const int8_t v4x3 = 4 < k ? w4[no + 3] : 0;
          const int8_t v5x3 = 5 < k ? w5[no + 3] : 0;
          const int8_t v6x3 = 6 < k ? w6[no + 3] : 0;
          const int8_t v7x3 = 7 < k ? w7[no + 3] : 0;
          ksum[no + 3] += (uint32_t) v0x3;
          ksum[no + 3] += (uint32_t) v1x3;
          ksum[no + 3] += (uint32_t) v2x3;
          ksum[no + 3] += (uint32_t) v3x3;
          ksum[no + 3] += (uint32_t) v4x3;
          ksum[no + 3] += (uint32_t) v5x3;
          ksum[no + 3] += (uint32_t) v6x3;
          ksum[no + 3] += (uint32_t) v7x3;
          out[24] = v0x3;
          out[25] = v1x3;
          out[26] = v2x3;
          out[27] = v3x3;
          out[28] = v4x3;
          out[29] = v5x3;
          out[30] = v6x3;
          out[31] = v7x3;
          const int8_t v0x4 = w0[no + 4];
          const int8_t v1x4 = 1 < k ? w1[no + 4] : 0;
          const int8_t v2x4 = 2 < k ? w2[no + 4] : 0;
          const int8_t v3x4 = 3 < k ? w3[no + 4] : 0;
          const int8_t v4x4 = 4 < k ? w4[no + 4] : 0;
          const int8_t v5x4 = 5 < k ? w5[no + 4] : 0;
          const int8_t v6x4 = 6 < k ? w6[no + 4] : 0;
          const int8_t v7x4 = 7 < k ? w7[no + 4] : 0;
          ksum[no + 4] += (uint32_t) v0x4;
          ksum[no + 4] += (uint32_t) v1x4;
          ksum[no + 4] += (uint32_t) v2x4;
          ksum[no + 4] += (uint32_t) v3x4;
          ksum[no + 4] += (uint32_t) v4x4;
          ksum[no + 4] += (uint32_t) v5x4;
          ksum[no + 4] += (uint32_t) v6x4;
          ksum[no + 4] += (uint32_t) v7x4;
          out[32] = v0x4;
          out[33] = v1x4;
          out[34] = v2x4;
          out[35] = v3x4;
          out[36] = v4x4;
          out[37] = v5x4;
          out[38] = v6x4;
          out[39] = v7x4;
          const int8_t v0x5 = w0[no + 5];
          const int8_t v1x5 = 1 < k ? w1[no + 5] : 0;
          const int8_t v2x5 = 2 < k ? w2[no + 5] : 0;
          const int8_t v3x5 = 3 < k ? w3[no + 5] : 0;
          const int8_t v4x5 = 4 < k ? w4[no + 5] : 0;
          const int8_t v5x5 = 5 < k ? w5[no + 5] : 0;
          const int8_t v6x5 = 6 < k ? w6[no + 5] : 0;
          const int8_t v7x5 = 7 < k ? w7[no + 5] : 0;
          ksum[no + 5] += (uint32_t) v0x5;
          ksum[no + 5] += (uint32_t) v1x5;
          ksum[no + 5] += (uint32_t) v2x5;
          ksum[no + 5] += (uint32_t) v3x5;
          ksum[no + 5] += (uint32_t) v4x5;
          ksum[no + 5] += (uint32_t) v5x5;
          ksum[no + 5] += (uint32_t) v6x5;
          ksum[no + 5] += (uint32_t) v7x5;
          out[40] = v0x5;
          out[41] = v1x5;
          out[42] = v2x5;
          out[43] = v3x5;
          out[44] = v4x5;
          out[45] = v5x5;
          out[46] = v6x5;
          out[47] = v7x5;
          const int8_t v0x6 = w0[no + 6];
          const int8_t v1x6 = 1 < k ? w1[no + 6] : 0;
          const int8_t v2x6 = 2 < k ? w2[no + 6] : 0;
          const int8_t v3x6 = 3 < k ? w3[no + 6] : 0;
          const int8_t v4x6 = 4 < k ? w4[no + 6] : 0;
          const int8_t v5x6 = 5 < k ? w5[no + 6] : 0;
          const int8_t v6x6 = 6 < k ? w6[no + 6] : 0;
          const int8_t v7x6 = 7 < k ? w7[no + 6] : 0;
          ksum[no + 6] += (uint32_t) v0x6;
          ksum[no + 6] += (uint32_t) v1x6;
          ksum[no + 6] += (uint32_t) v2x6;
          ksum[no + 6] += (uint32_t) v3x6;
          ksum[no + 6] += (uint32_t) v4x6;
          ksum[no + 6] += (uint32_t) v5x6;
          ksum[no + 6] += (uint32_t) v6x6;
          ksum[no + 6] += (uint32_t) v7x6;
          out[48] = v0x6;
          out[49] = v1x6;
          out[50] = v2x6;
          out[51] = v3x6;
          out[52] = v4x6;
          out[53] = v5x6;
          out[54] = v6x6;
          out[55] = v7x6;
          const int8_t v0x7 = w0[no + 7];
          const int8_t v1x7 = 1 < k ? w1[no + 7] : 0;
          const int8_t v2x7 = 2 < k ? w2[no + 7] : 0;
          const int8_t v3x7 = 3 < k ? w3[no + 7] : 0;
          const int8_t v4x7 = 4 < k ? w4[no + 7] : 0;
          const int8_t v5x7 = 5 < k ? w5[no + 7] : 0;
          const int8_t v6x7 = 6 < k ? w6[no + 7] : 0;
          const int8_t v7x7 = 7 < k ? w7[no + 7] : 0;
          ksum[no + 7] += (uint32_t) v0x7;
          ksum[no + 7] += (uint32_t) v1x7;
          ksum[no + 7] += (uint32_t) v2x7;
          ksum[no + 7] += (uint32_t) v3x7;
          ksum[no + 7] += (uint32_t) v4x7;
          ksum[no + 7] += (uint32_t) v5x7;
          ksum[no + 7] += (uint32_t) v6x7;
          ksum[no + 7] += (uint32_t) v7x7;
          out[56] = v0x7;
          out[57] = v1x7;
          out[58] = v2x7;
          out[59] = v3x7;
          out[60] = v4x7;
          out[61] = v5x7;
          out[62] = v6x7;
          out[63] = v7x7;
          out += 64;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
        w4 += k * k_stride;
        w5 += k * k_stride;
        w6 += k * k_stride;
        w7 += k * k_stride;
      }

      for (size_t no = 0; no < 8; no += 8) {
        packed_b[no + 0] -= ksum[no + 0] * izp;
        packed_b[no + 1] -= ksum[no + 1] * izp;
        packed_b[no + 2] -= ksum[no + 2] * izp;
        packed_b[no + 3] -= ksum[no + 3] * izp;
        packed_b[no + 4] -= ksum[no + 4] * izp;
        packed_b[no + 5] -= ksum[no + 5] * izp;
        packed_b[no + 6] -= ksum[no + 6] * izp;
        packed_b[no + 7] -= ksum[no + 7] * izp;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w0 - kc * k_stride + 8;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, n * sizeof(int32_t));
        b += n;
      } else {
        memset(out, 0, n * sizeof(int32_t));
      }
      out += 8 * sizeof(int32_t);

     // NR remainder has less than 8 rows so last row is not loaded
      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;
      const int8_t* w4 = w3 + k_stride;
      const int8_t* w5 = w4 + k_stride;
      const int8_t* w6 = w5 + k_stride;
      const int8_t* w7 = w6 + k_stride;

      uint32_t ksum[8] = {0,};

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const int8_t v0x0 = w0[0];
        const int8_t v1x0 = w1[0];
        const int8_t v2x0 = w2[0];
        const int8_t v3x0 = w3[0];
        const int8_t v4x0 = w4[0];
        const int8_t v5x0 = w5[0];
        const int8_t v6x0 = w6[0];
        const int8_t v7x0 = w7[0];
        ksum[0] += (uint32_t) v0x0;
        ksum[0] += (uint32_t) v1x0;
        ksum[0] += (uint32_t) v2x0;
        ksum[0] += (uint32_t) v3x0;
        ksum[0] += (uint32_t) v4x0;
        ksum[0] += (uint32_t) v5x0;
        ksum[0] += (uint32_t) v6x0;
        ksum[0] += (uint32_t) v7x0;
        out[0] = v0x0;
        out[1] = v1x0;
        out[2] = v2x0;
        out[3] = v3x0;
        out[4] = v4x0;
        out[5] = v5x0;
        out[6] = v6x0;
        out[7] = v7x0;
        const int8_t v0x1 = 1 < n ? w0[1] : 0;
        const int8_t v1x1 = 1 < n ? w1[1] : 0;
        const int8_t v2x1 = 1 < n ? w2[1] : 0;
        const int8_t v3x1 = 1 < n ? w3[1] : 0;
        const int8_t v4x1 = 1 < n ? w4[1] : 0;
        const int8_t v5x1 = 1 < n ? w5[1] : 0;
        const int8_t v6x1 = 1 < n ? w6[1] : 0;
        const int8_t v7x1 = 1 < n ? w7[1] : 0;
        ksum[1] += (uint32_t) v0x1;
        ksum[1] += (uint32_t) v1x1;
        ksum[1] += (uint32_t) v2x1;
        ksum[1] += (uint32_t) v3x1;
        ksum[1] += (uint32_t) v4x1;
        ksum[1] += (uint32_t) v5x1;
        ksum[1] += (uint32_t) v6x1;
        ksum[1] += (uint32_t) v7x1;
        out[8] = v0x1;
        out[9] = v1x1;
        out[10] = v2x1;
        out[11] = v3x1;
        out[12] = v4x1;
        out[13] = v5x1;
        out[14] = v6x1;
        out[15] = v7x1;
        const int8_t v0x2 = 2 < n ? w0[2] : 0;
        const int8_t v1x2 = 2 < n ? w1[2] : 0;
        const int8_t v2x2 = 2 < n ? w2[2] : 0;
        const int8_t v3x2 = 2 < n ? w3[2] : 0;
        const int8_t v4x2 = 2 < n ? w4[2] : 0;
        const int8_t v5x2 = 2 < n ? w5[2] : 0;
        const int8_t v6x2 = 2 < n ? w6[2] : 0;
        const int8_t v7x2 = 2 < n ? w7[2] : 0;
        ksum[2] += (uint32_t) v0x2;
        ksum[2] += (uint32_t) v1x2;
        ksum[2] += (uint32_t) v2x2;
        ksum[2] += (uint32_t) v3x2;
        ksum[2] += (uint32_t) v4x2;
        ksum[2] += (uint32_t) v5x2;
        ksum[2] += (uint32_t) v6x2;
        ksum[2] += (uint32_t) v7x2;
        out[16] = v0x2;
        out[17] = v1x2;
        out[18] = v2x2;
        out[19] = v3x2;
        out[20] = v4x2;
        out[21] = v5x2;
        out[22] = v6x2;
        out[23] = v7x2;
        const int8_t v0x3 = 3 < n ? w0[3] : 0;
        const int8_t v1x3 = 3 < n ? w1[3] : 0;
        const int8_t v2x3 = 3 < n ? w2[3] : 0;
        const int8_t v3x3 = 3 < n ? w3[3] : 0;
        const int8_t v4x3 = 3 < n ? w4[3] : 0;
        const int8_t v5x3 = 3 < n ? w5[3] : 0;
        const int8_t v6x3 = 3 < n ? w6[3] : 0;
        const int8_t v7x3 = 3 < n ? w7[3] : 0;
        ksum[3] += (uint32_t) v0x3;
        ksum[3] += (uint32_t) v1x3;
        ksum[3] += (uint32_t) v2x3;
        ksum[3] += (uint32_t) v3x3;
        ksum[3] += (uint32_t) v4x3;
        ksum[3] += (uint32_t) v5x3;
        ksum[3] += (uint32_t) v6x3;
        ksum[3] += (uint32_t) v7x3;
        out[24] = v0x3;
        out[25] = v1x3;
        out[26] = v2x3;
        out[27] = v3x3;
        out[28] = v4x3;
        out[29] = v5x3;
        out[30] = v6x3;
        out[31] = v7x3;
        const int8_t v0x4 = 4 < n ? w0[4] : 0;
        const int8_t v1x4 = 4 < n ? w1[4] : 0;
        const int8_t v2x4 = 4 < n ? w2[4] : 0;
        const int8_t v3x4 = 4 < n ? w3[4] : 0;
        const int8_t v4x4 = 4 < n ? w4[4] : 0;
        const int8_t v5x4 = 4 < n ? w5[4] : 0;
        const int8_t v6x4 = 4 < n ? w6[4] : 0;
        const int8_t v7x4 = 4 < n ? w7[4] : 0;
        ksum[4] += (uint32_t) v0x4;
        ksum[4] += (uint32_t) v1x4;
        ksum[4] += (uint32_t) v2x4;
        ksum[4] += (uint32_t) v3x4;
        ksum[4] += (uint32_t) v4x4;
        ksum[4] += (uint32_t) v5x4;
        ksum[4] += (uint32_t) v6x4;
        ksum[4] += (uint32_t) v7x4;
        out[32] = v0x4;
        out[33] = v1x4;
        out[34] = v2x4;
        out[35] = v3x4;
        out[36] = v4x4;
        out[37] = v5x4;
        out[38] = v6x4;
        out[39] = v7x4;
        const int8_t v0x5 = 5 < n ? w0[5] : 0;
        const int8_t v1x5 = 5 < n ? w1[5] : 0;
        const int8_t v2x5 = 5 < n ? w2[5] : 0;
        const int8_t v3x5 = 5 < n ? w3[5] : 0;
        const int8_t v4x5 = 5 < n ? w4[5] : 0;
        const int8_t v5x5 = 5 < n ? w5[5] : 0;
        const int8_t v6x5 = 5 < n ? w6[5] : 0;
        const int8_t v7x5 = 5 < n ? w7[5] : 0;
        ksum[5] += (uint32_t) v0x5;
        ksum[5] += (uint32_t) v1x5;
        ksum[5] += (uint32_t) v2x5;
        ksum[5] += (uint32_t) v3x5;
        ksum[5] += (uint32_t) v4x5;
        ksum[5] += (uint32_t) v5x5;
        ksum[5] += (uint32_t) v6x5;
        ksum[5] += (uint32_t) v7x5;
        out[40] = v0x5;
        out[41] = v1x5;
        out[42] = v2x5;
        out[43] = v3x5;
        out[44] = v4x5;
        out[45] = v5x5;
        out[46] = v6x5;
        out[47] = v7x5;
        const int8_t v0x6 = 6 < n ? w0[6] : 0;
        const int8_t v1x6 = 6 < n ? w1[6] : 0;
        const int8_t v2x6 = 6 < n ? w2[6] : 0;
        const int8_t v3x6 = 6 < n ? w3[6] : 0;
        const int8_t v4x6 = 6 < n ? w4[6] : 0;
        const int8_t v5x6 = 6 < n ? w5[6] : 0;
        const int8_t v6x6 = 6 < n ? w6[6] : 0;
        const int8_t v7x6 = 6 < n ? w7[6] : 0;
        ksum[6] += (uint32_t) v0x6;
        ksum[6] += (uint32_t) v1x6;
        ksum[6] += (uint32_t) v2x6;
        ksum[6] += (uint32_t) v3x6;
        ksum[6] += (uint32_t) v4x6;
        ksum[6] += (uint32_t) v5x6;
        ksum[6] += (uint32_t) v6x6;
        ksum[6] += (uint32_t) v7x6;
        out[48] = v0x6;
        out[49] = v1x6;
        out[50] = v2x6;
        out[51] = v3x6;
        out[52] = v4x6;
        out[53] = v5x6;
        out[54] = v6x6;
        out[55] = v7x6;
        const int8_t v0x7 = 7 < n ? w0[7] : 0;
        const int8_t v1x7 = 7 < n ? w1[7] : 0;
        const int8_t v2x7 = 7 < n ? w2[7] : 0;
        const int8_t v3x7 = 7 < n ? w3[7] : 0;
        const int8_t v4x7 = 7 < n ? w4[7] : 0;
        const int8_t v5x7 = 7 < n ? w5[7] : 0;
        const int8_t v6x7 = 7 < n ? w6[7] : 0;
        const int8_t v7x7 = 7 < n ? w7[7] : 0;
        ksum[7] += (uint32_t) v0x7;
        ksum[7] += (uint32_t) v1x7;
        ksum[7] += (uint32_t) v2x7;
        ksum[7] += (uint32_t) v3x7;
        ksum[7] += (uint32_t) v4x7;
        ksum[7] += (uint32_t) v5x7;
        ksum[7] += (uint32_t) v6x7;
        ksum[7] += (uint32_t) v7x7;
        out[56] = v0x7;
        out[57] = v1x7;
        out[58] = v2x7;
        out[59] = v3x7;
        out[60] = v4x7;
        out[61] = v5x7;
        out[62] = v6x7;
        out[63] = v7x7;
        for (size_t N = 8; N < n; ++N) {
          const int8_t v0 = w0[N];
          const int8_t v1 = w1[N];
          const int8_t v2 = w2[N];
          const int8_t v3 = w3[N];
          const int8_t v4 = w4[N];
          const int8_t v5 = w5[N];
          const int8_t v6 = w6[N];
          const int8_t v7 = w7[N];
          ksum[N] += (uint32_t) v0;
          ksum[N] += (uint32_t) v1;
          ksum[N] += (uint32_t) v2;
          ksum[N] += (uint32_t) v3;
          ksum[N] += (uint32_t) v4;
          ksum[N] += (uint32_t) v5;
          ksum[N] += (uint32_t) v6;
          ksum[N] += (uint32_t) v7;
          out[N*8 + 0] = v0;
          out[N*8 + 1] = v1;
          out[N*8 + 2] = v2;
          out[N*8 + 3] = v3;
          out[N*8 + 4] = v4;
          out[N*8 + 5] = v5;
          out[N*8 + 6] = v6;
          out[N*8 + 7] = v7;
        }
        for (size_t N = n; N < 8; ++N) {
          out[N*8 + 0] = 0;
          out[N*8 + 1] = 0;
          out[N*8 + 2] = 0;
          out[N*8 + 3] = 0;
          out[N*8 + 4] = 0;
          out[N*8 + 5] = 0;
          out[N*8 + 6] = 0;
          out[N*8 + 7] = 0;
        }
        w0 += 8 * k_stride;
        w1 += 8 * k_stride;
        w2 += 8 * k_stride;
        w3 += 8 * k_stride;
        w4 += 8 * k_stride;
        w5 += 8 * k_stride;
        w6 += 8 * k_stride;
        w7 += 8 * k_stride;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);
        for (size_t N = 0; N < n; ++N) {
          const int8_t v0 = w0[N];
          const int8_t v1 = 1 < k ? w1[N] : 0;
          const int8_t v2 = 2 < k ? w2[N] : 0;
          const int8_t v3 = 3 < k ? w3[N] : 0;
          const int8_t v4 = 4 < k ? w4[N] : 0;
          const int8_t v5 = 5 < k ? w5[N] : 0;
          const int8_t v6 = 6 < k ? w6[N] : 0;
          const int8_t v7 = 7 < k ? w7[N] : 0;
          ksum[N] += (uint32_t) v0;
          ksum[N] += (uint32_t) v1;
          ksum[N] += (uint32_t) v2;
          ksum[N] += (uint32_t) v3;
          ksum[N] += (uint32_t) v4;
          ksum[N] += (uint32_t) v5;
          ksum[N] += (uint32_t) v6;
          ksum[N] += (uint32_t) v7;
          out[N*8] = v0;
          out[N*8 + 1] = v1;
          out[N*8 + 2] = v2;
          out[N*8 + 3] = v3;
          out[N*8 + 4] = v4;
          out[N*8 + 5] = v5;
          out[N*8 + 6] = v6;
          out[N*8 + 7] = v7;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
        w4 += k * k_stride;
        w5 += k * k_stride;
        w6 += k * k_stride;
        w7 += k * k_stride;
        out += 64;
      }

      for (size_t N = 0; N < 7; ++N) {
        packed_b[N] -= ksum[N] * izp;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
