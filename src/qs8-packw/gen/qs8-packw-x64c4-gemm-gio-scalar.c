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

void xnn_qs8_packw_gemm_gio_ukernel_x64c4__scalar(
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
  assert(nr == 64);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (params ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 0): 0);

  do {
    // NC main loop multiple of 64
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 64; n -= 64) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, 64 * sizeof(int32_t));
        b += 64;
      } else {
        memset(out, 0, 64 * sizeof(int32_t));
      }
      out += 64 * sizeof(int32_t);

      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;
      uint32_t ksum[64] = {0,};

      // KC main loop multiple of 64x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        for (size_t no = 0; no < 64; no += 16) {
          const int8_t v0x0 = w0[no + 0];
          const int8_t v1x0 = w1[no + 0];
          const int8_t v2x0 = w2[no + 0];
          const int8_t v3x0 = w3[no + 0];
          ksum[no + 0] += (uint32_t) v0x0;
          ksum[no + 0] += (uint32_t) v1x0;
          ksum[no + 0] += (uint32_t) v2x0;
          ksum[no + 0] += (uint32_t) v3x0;
          out[0] = v0x0;
          out[1] = v1x0;
          out[2] = v2x0;
          out[3] = v3x0;
          const int8_t v0x1 = w0[no + 1];
          const int8_t v1x1 = w1[no + 1];
          const int8_t v2x1 = w2[no + 1];
          const int8_t v3x1 = w3[no + 1];
          ksum[no + 1] += (uint32_t) v0x1;
          ksum[no + 1] += (uint32_t) v1x1;
          ksum[no + 1] += (uint32_t) v2x1;
          ksum[no + 1] += (uint32_t) v3x1;
          out[4] = v0x1;
          out[5] = v1x1;
          out[6] = v2x1;
          out[7] = v3x1;
          const int8_t v0x2 = w0[no + 2];
          const int8_t v1x2 = w1[no + 2];
          const int8_t v2x2 = w2[no + 2];
          const int8_t v3x2 = w3[no + 2];
          ksum[no + 2] += (uint32_t) v0x2;
          ksum[no + 2] += (uint32_t) v1x2;
          ksum[no + 2] += (uint32_t) v2x2;
          ksum[no + 2] += (uint32_t) v3x2;
          out[8] = v0x2;
          out[9] = v1x2;
          out[10] = v2x2;
          out[11] = v3x2;
          const int8_t v0x3 = w0[no + 3];
          const int8_t v1x3 = w1[no + 3];
          const int8_t v2x3 = w2[no + 3];
          const int8_t v3x3 = w3[no + 3];
          ksum[no + 3] += (uint32_t) v0x3;
          ksum[no + 3] += (uint32_t) v1x3;
          ksum[no + 3] += (uint32_t) v2x3;
          ksum[no + 3] += (uint32_t) v3x3;
          out[12] = v0x3;
          out[13] = v1x3;
          out[14] = v2x3;
          out[15] = v3x3;
          const int8_t v0x4 = w0[no + 4];
          const int8_t v1x4 = w1[no + 4];
          const int8_t v2x4 = w2[no + 4];
          const int8_t v3x4 = w3[no + 4];
          ksum[no + 4] += (uint32_t) v0x4;
          ksum[no + 4] += (uint32_t) v1x4;
          ksum[no + 4] += (uint32_t) v2x4;
          ksum[no + 4] += (uint32_t) v3x4;
          out[16] = v0x4;
          out[17] = v1x4;
          out[18] = v2x4;
          out[19] = v3x4;
          const int8_t v0x5 = w0[no + 5];
          const int8_t v1x5 = w1[no + 5];
          const int8_t v2x5 = w2[no + 5];
          const int8_t v3x5 = w3[no + 5];
          ksum[no + 5] += (uint32_t) v0x5;
          ksum[no + 5] += (uint32_t) v1x5;
          ksum[no + 5] += (uint32_t) v2x5;
          ksum[no + 5] += (uint32_t) v3x5;
          out[20] = v0x5;
          out[21] = v1x5;
          out[22] = v2x5;
          out[23] = v3x5;
          const int8_t v0x6 = w0[no + 6];
          const int8_t v1x6 = w1[no + 6];
          const int8_t v2x6 = w2[no + 6];
          const int8_t v3x6 = w3[no + 6];
          ksum[no + 6] += (uint32_t) v0x6;
          ksum[no + 6] += (uint32_t) v1x6;
          ksum[no + 6] += (uint32_t) v2x6;
          ksum[no + 6] += (uint32_t) v3x6;
          out[24] = v0x6;
          out[25] = v1x6;
          out[26] = v2x6;
          out[27] = v3x6;
          const int8_t v0x7 = w0[no + 7];
          const int8_t v1x7 = w1[no + 7];
          const int8_t v2x7 = w2[no + 7];
          const int8_t v3x7 = w3[no + 7];
          ksum[no + 7] += (uint32_t) v0x7;
          ksum[no + 7] += (uint32_t) v1x7;
          ksum[no + 7] += (uint32_t) v2x7;
          ksum[no + 7] += (uint32_t) v3x7;
          out[28] = v0x7;
          out[29] = v1x7;
          out[30] = v2x7;
          out[31] = v3x7;
          const int8_t v0x8 = w0[no + 8];
          const int8_t v1x8 = w1[no + 8];
          const int8_t v2x8 = w2[no + 8];
          const int8_t v3x8 = w3[no + 8];
          ksum[no + 8] += (uint32_t) v0x8;
          ksum[no + 8] += (uint32_t) v1x8;
          ksum[no + 8] += (uint32_t) v2x8;
          ksum[no + 8] += (uint32_t) v3x8;
          out[32] = v0x8;
          out[33] = v1x8;
          out[34] = v2x8;
          out[35] = v3x8;
          const int8_t v0x9 = w0[no + 9];
          const int8_t v1x9 = w1[no + 9];
          const int8_t v2x9 = w2[no + 9];
          const int8_t v3x9 = w3[no + 9];
          ksum[no + 9] += (uint32_t) v0x9;
          ksum[no + 9] += (uint32_t) v1x9;
          ksum[no + 9] += (uint32_t) v2x9;
          ksum[no + 9] += (uint32_t) v3x9;
          out[36] = v0x9;
          out[37] = v1x9;
          out[38] = v2x9;
          out[39] = v3x9;
          const int8_t v0x10 = w0[no + 10];
          const int8_t v1x10 = w1[no + 10];
          const int8_t v2x10 = w2[no + 10];
          const int8_t v3x10 = w3[no + 10];
          ksum[no + 10] += (uint32_t) v0x10;
          ksum[no + 10] += (uint32_t) v1x10;
          ksum[no + 10] += (uint32_t) v2x10;
          ksum[no + 10] += (uint32_t) v3x10;
          out[40] = v0x10;
          out[41] = v1x10;
          out[42] = v2x10;
          out[43] = v3x10;
          const int8_t v0x11 = w0[no + 11];
          const int8_t v1x11 = w1[no + 11];
          const int8_t v2x11 = w2[no + 11];
          const int8_t v3x11 = w3[no + 11];
          ksum[no + 11] += (uint32_t) v0x11;
          ksum[no + 11] += (uint32_t) v1x11;
          ksum[no + 11] += (uint32_t) v2x11;
          ksum[no + 11] += (uint32_t) v3x11;
          out[44] = v0x11;
          out[45] = v1x11;
          out[46] = v2x11;
          out[47] = v3x11;
          const int8_t v0x12 = w0[no + 12];
          const int8_t v1x12 = w1[no + 12];
          const int8_t v2x12 = w2[no + 12];
          const int8_t v3x12 = w3[no + 12];
          ksum[no + 12] += (uint32_t) v0x12;
          ksum[no + 12] += (uint32_t) v1x12;
          ksum[no + 12] += (uint32_t) v2x12;
          ksum[no + 12] += (uint32_t) v3x12;
          out[48] = v0x12;
          out[49] = v1x12;
          out[50] = v2x12;
          out[51] = v3x12;
          const int8_t v0x13 = w0[no + 13];
          const int8_t v1x13 = w1[no + 13];
          const int8_t v2x13 = w2[no + 13];
          const int8_t v3x13 = w3[no + 13];
          ksum[no + 13] += (uint32_t) v0x13;
          ksum[no + 13] += (uint32_t) v1x13;
          ksum[no + 13] += (uint32_t) v2x13;
          ksum[no + 13] += (uint32_t) v3x13;
          out[52] = v0x13;
          out[53] = v1x13;
          out[54] = v2x13;
          out[55] = v3x13;
          const int8_t v0x14 = w0[no + 14];
          const int8_t v1x14 = w1[no + 14];
          const int8_t v2x14 = w2[no + 14];
          const int8_t v3x14 = w3[no + 14];
          ksum[no + 14] += (uint32_t) v0x14;
          ksum[no + 14] += (uint32_t) v1x14;
          ksum[no + 14] += (uint32_t) v2x14;
          ksum[no + 14] += (uint32_t) v3x14;
          out[56] = v0x14;
          out[57] = v1x14;
          out[58] = v2x14;
          out[59] = v3x14;
          const int8_t v0x15 = w0[no + 15];
          const int8_t v1x15 = w1[no + 15];
          const int8_t v2x15 = w2[no + 15];
          const int8_t v3x15 = w3[no + 15];
          ksum[no + 15] += (uint32_t) v0x15;
          ksum[no + 15] += (uint32_t) v1x15;
          ksum[no + 15] += (uint32_t) v2x15;
          ksum[no + 15] += (uint32_t) v3x15;
          out[60] = v0x15;
          out[61] = v1x15;
          out[62] = v2x15;
          out[63] = v3x15;
          out += 64;
        }
        w0 += 4 * k_stride;
        w1 += 4 * k_stride;
        w2 += 4 * k_stride;
        w3 += 4 * k_stride;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
        for (size_t no = 0; no < 64; no += 16) {
          const int8_t v0x0 = w0[no + 0];
          const int8_t v1x0 = 1 < k ? w1[no + 0] : 0;
          const int8_t v2x0 = 2 < k ? w2[no + 0] : 0;
          const int8_t v3x0 = 3 < k ? w3[no + 0] : 0;
          ksum[no + 0] += (uint32_t) v0x0;
          ksum[no + 0] += (uint32_t) v1x0;
          ksum[no + 0] += (uint32_t) v2x0;
          ksum[no + 0] += (uint32_t) v3x0;
          out[0] = v0x0;
          out[1] = v1x0;
          out[2] = v2x0;
          out[3] = v3x0;
          const int8_t v0x1 = w0[no + 1];
          const int8_t v1x1 = 1 < k ? w1[no + 1] : 0;
          const int8_t v2x1 = 2 < k ? w2[no + 1] : 0;
          const int8_t v3x1 = 3 < k ? w3[no + 1] : 0;
          ksum[no + 1] += (uint32_t) v0x1;
          ksum[no + 1] += (uint32_t) v1x1;
          ksum[no + 1] += (uint32_t) v2x1;
          ksum[no + 1] += (uint32_t) v3x1;
          out[4] = v0x1;
          out[5] = v1x1;
          out[6] = v2x1;
          out[7] = v3x1;
          const int8_t v0x2 = w0[no + 2];
          const int8_t v1x2 = 1 < k ? w1[no + 2] : 0;
          const int8_t v2x2 = 2 < k ? w2[no + 2] : 0;
          const int8_t v3x2 = 3 < k ? w3[no + 2] : 0;
          ksum[no + 2] += (uint32_t) v0x2;
          ksum[no + 2] += (uint32_t) v1x2;
          ksum[no + 2] += (uint32_t) v2x2;
          ksum[no + 2] += (uint32_t) v3x2;
          out[8] = v0x2;
          out[9] = v1x2;
          out[10] = v2x2;
          out[11] = v3x2;
          const int8_t v0x3 = w0[no + 3];
          const int8_t v1x3 = 1 < k ? w1[no + 3] : 0;
          const int8_t v2x3 = 2 < k ? w2[no + 3] : 0;
          const int8_t v3x3 = 3 < k ? w3[no + 3] : 0;
          ksum[no + 3] += (uint32_t) v0x3;
          ksum[no + 3] += (uint32_t) v1x3;
          ksum[no + 3] += (uint32_t) v2x3;
          ksum[no + 3] += (uint32_t) v3x3;
          out[12] = v0x3;
          out[13] = v1x3;
          out[14] = v2x3;
          out[15] = v3x3;
          const int8_t v0x4 = w0[no + 4];
          const int8_t v1x4 = 1 < k ? w1[no + 4] : 0;
          const int8_t v2x4 = 2 < k ? w2[no + 4] : 0;
          const int8_t v3x4 = 3 < k ? w3[no + 4] : 0;
          ksum[no + 4] += (uint32_t) v0x4;
          ksum[no + 4] += (uint32_t) v1x4;
          ksum[no + 4] += (uint32_t) v2x4;
          ksum[no + 4] += (uint32_t) v3x4;
          out[16] = v0x4;
          out[17] = v1x4;
          out[18] = v2x4;
          out[19] = v3x4;
          const int8_t v0x5 = w0[no + 5];
          const int8_t v1x5 = 1 < k ? w1[no + 5] : 0;
          const int8_t v2x5 = 2 < k ? w2[no + 5] : 0;
          const int8_t v3x5 = 3 < k ? w3[no + 5] : 0;
          ksum[no + 5] += (uint32_t) v0x5;
          ksum[no + 5] += (uint32_t) v1x5;
          ksum[no + 5] += (uint32_t) v2x5;
          ksum[no + 5] += (uint32_t) v3x5;
          out[20] = v0x5;
          out[21] = v1x5;
          out[22] = v2x5;
          out[23] = v3x5;
          const int8_t v0x6 = w0[no + 6];
          const int8_t v1x6 = 1 < k ? w1[no + 6] : 0;
          const int8_t v2x6 = 2 < k ? w2[no + 6] : 0;
          const int8_t v3x6 = 3 < k ? w3[no + 6] : 0;
          ksum[no + 6] += (uint32_t) v0x6;
          ksum[no + 6] += (uint32_t) v1x6;
          ksum[no + 6] += (uint32_t) v2x6;
          ksum[no + 6] += (uint32_t) v3x6;
          out[24] = v0x6;
          out[25] = v1x6;
          out[26] = v2x6;
          out[27] = v3x6;
          const int8_t v0x7 = w0[no + 7];
          const int8_t v1x7 = 1 < k ? w1[no + 7] : 0;
          const int8_t v2x7 = 2 < k ? w2[no + 7] : 0;
          const int8_t v3x7 = 3 < k ? w3[no + 7] : 0;
          ksum[no + 7] += (uint32_t) v0x7;
          ksum[no + 7] += (uint32_t) v1x7;
          ksum[no + 7] += (uint32_t) v2x7;
          ksum[no + 7] += (uint32_t) v3x7;
          out[28] = v0x7;
          out[29] = v1x7;
          out[30] = v2x7;
          out[31] = v3x7;
          const int8_t v0x8 = w0[no + 8];
          const int8_t v1x8 = 1 < k ? w1[no + 8] : 0;
          const int8_t v2x8 = 2 < k ? w2[no + 8] : 0;
          const int8_t v3x8 = 3 < k ? w3[no + 8] : 0;
          ksum[no + 8] += (uint32_t) v0x8;
          ksum[no + 8] += (uint32_t) v1x8;
          ksum[no + 8] += (uint32_t) v2x8;
          ksum[no + 8] += (uint32_t) v3x8;
          out[32] = v0x8;
          out[33] = v1x8;
          out[34] = v2x8;
          out[35] = v3x8;
          const int8_t v0x9 = w0[no + 9];
          const int8_t v1x9 = 1 < k ? w1[no + 9] : 0;
          const int8_t v2x9 = 2 < k ? w2[no + 9] : 0;
          const int8_t v3x9 = 3 < k ? w3[no + 9] : 0;
          ksum[no + 9] += (uint32_t) v0x9;
          ksum[no + 9] += (uint32_t) v1x9;
          ksum[no + 9] += (uint32_t) v2x9;
          ksum[no + 9] += (uint32_t) v3x9;
          out[36] = v0x9;
          out[37] = v1x9;
          out[38] = v2x9;
          out[39] = v3x9;
          const int8_t v0x10 = w0[no + 10];
          const int8_t v1x10 = 1 < k ? w1[no + 10] : 0;
          const int8_t v2x10 = 2 < k ? w2[no + 10] : 0;
          const int8_t v3x10 = 3 < k ? w3[no + 10] : 0;
          ksum[no + 10] += (uint32_t) v0x10;
          ksum[no + 10] += (uint32_t) v1x10;
          ksum[no + 10] += (uint32_t) v2x10;
          ksum[no + 10] += (uint32_t) v3x10;
          out[40] = v0x10;
          out[41] = v1x10;
          out[42] = v2x10;
          out[43] = v3x10;
          const int8_t v0x11 = w0[no + 11];
          const int8_t v1x11 = 1 < k ? w1[no + 11] : 0;
          const int8_t v2x11 = 2 < k ? w2[no + 11] : 0;
          const int8_t v3x11 = 3 < k ? w3[no + 11] : 0;
          ksum[no + 11] += (uint32_t) v0x11;
          ksum[no + 11] += (uint32_t) v1x11;
          ksum[no + 11] += (uint32_t) v2x11;
          ksum[no + 11] += (uint32_t) v3x11;
          out[44] = v0x11;
          out[45] = v1x11;
          out[46] = v2x11;
          out[47] = v3x11;
          const int8_t v0x12 = w0[no + 12];
          const int8_t v1x12 = 1 < k ? w1[no + 12] : 0;
          const int8_t v2x12 = 2 < k ? w2[no + 12] : 0;
          const int8_t v3x12 = 3 < k ? w3[no + 12] : 0;
          ksum[no + 12] += (uint32_t) v0x12;
          ksum[no + 12] += (uint32_t) v1x12;
          ksum[no + 12] += (uint32_t) v2x12;
          ksum[no + 12] += (uint32_t) v3x12;
          out[48] = v0x12;
          out[49] = v1x12;
          out[50] = v2x12;
          out[51] = v3x12;
          const int8_t v0x13 = w0[no + 13];
          const int8_t v1x13 = 1 < k ? w1[no + 13] : 0;
          const int8_t v2x13 = 2 < k ? w2[no + 13] : 0;
          const int8_t v3x13 = 3 < k ? w3[no + 13] : 0;
          ksum[no + 13] += (uint32_t) v0x13;
          ksum[no + 13] += (uint32_t) v1x13;
          ksum[no + 13] += (uint32_t) v2x13;
          ksum[no + 13] += (uint32_t) v3x13;
          out[52] = v0x13;
          out[53] = v1x13;
          out[54] = v2x13;
          out[55] = v3x13;
          const int8_t v0x14 = w0[no + 14];
          const int8_t v1x14 = 1 < k ? w1[no + 14] : 0;
          const int8_t v2x14 = 2 < k ? w2[no + 14] : 0;
          const int8_t v3x14 = 3 < k ? w3[no + 14] : 0;
          ksum[no + 14] += (uint32_t) v0x14;
          ksum[no + 14] += (uint32_t) v1x14;
          ksum[no + 14] += (uint32_t) v2x14;
          ksum[no + 14] += (uint32_t) v3x14;
          out[56] = v0x14;
          out[57] = v1x14;
          out[58] = v2x14;
          out[59] = v3x14;
          const int8_t v0x15 = w0[no + 15];
          const int8_t v1x15 = 1 < k ? w1[no + 15] : 0;
          const int8_t v2x15 = 2 < k ? w2[no + 15] : 0;
          const int8_t v3x15 = 3 < k ? w3[no + 15] : 0;
          ksum[no + 15] += (uint32_t) v0x15;
          ksum[no + 15] += (uint32_t) v1x15;
          ksum[no + 15] += (uint32_t) v2x15;
          ksum[no + 15] += (uint32_t) v3x15;
          out[60] = v0x15;
          out[61] = v1x15;
          out[62] = v2x15;
          out[63] = v3x15;
          out += 64;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
      }

      for (size_t no = 0; no < 64; no += 16) {
        packed_b[no + 0] -= ksum[no + 0] * izp;
        packed_b[no + 1] -= ksum[no + 1] * izp;
        packed_b[no + 2] -= ksum[no + 2] * izp;
        packed_b[no + 3] -= ksum[no + 3] * izp;
        packed_b[no + 4] -= ksum[no + 4] * izp;
        packed_b[no + 5] -= ksum[no + 5] * izp;
        packed_b[no + 6] -= ksum[no + 6] * izp;
        packed_b[no + 7] -= ksum[no + 7] * izp;
        packed_b[no + 8] -= ksum[no + 8] * izp;
        packed_b[no + 9] -= ksum[no + 9] * izp;
        packed_b[no + 10] -= ksum[no + 10] * izp;
        packed_b[no + 11] -= ksum[no + 11] * izp;
        packed_b[no + 12] -= ksum[no + 12] * izp;
        packed_b[no + 13] -= ksum[no + 13] * izp;
        packed_b[no + 14] -= ksum[no + 14] * izp;
        packed_b[no + 15] -= ksum[no + 15] * izp;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w0 - kc * k_stride + 64;
    }

    // NC remainder (1..63)
    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, n * sizeof(int32_t));
        b += n;
      } else {
        memset(out, 0, n * sizeof(int32_t));
      }
      out += 64 * sizeof(int32_t);

     // NR remainder has less than 64 rows so last row is not loaded
      const int8_t* w1 = w0 + k_stride;
      const int8_t* w2 = w1 + k_stride;
      const int8_t* w3 = w2 + k_stride;

      uint32_t ksum[64] = {0,};

      // KC main loop multiple of 64x4
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const int8_t v0x0 = w0[0];
        const int8_t v1x0 = w1[0];
        const int8_t v2x0 = w2[0];
        const int8_t v3x0 = w3[0];
        ksum[0] += (uint32_t) v0x0;
        ksum[0] += (uint32_t) v1x0;
        ksum[0] += (uint32_t) v2x0;
        ksum[0] += (uint32_t) v3x0;
        out[0] = v0x0;
        out[1] = v1x0;
        out[2] = v2x0;
        out[3] = v3x0;
        const int8_t v0x1 = 1 < n ? w0[1] : 0;
        const int8_t v1x1 = 1 < n ? w1[1] : 0;
        const int8_t v2x1 = 1 < n ? w2[1] : 0;
        const int8_t v3x1 = 1 < n ? w3[1] : 0;
        ksum[1] += (uint32_t) v0x1;
        ksum[1] += (uint32_t) v1x1;
        ksum[1] += (uint32_t) v2x1;
        ksum[1] += (uint32_t) v3x1;
        out[4] = v0x1;
        out[5] = v1x1;
        out[6] = v2x1;
        out[7] = v3x1;
        const int8_t v0x2 = 2 < n ? w0[2] : 0;
        const int8_t v1x2 = 2 < n ? w1[2] : 0;
        const int8_t v2x2 = 2 < n ? w2[2] : 0;
        const int8_t v3x2 = 2 < n ? w3[2] : 0;
        ksum[2] += (uint32_t) v0x2;
        ksum[2] += (uint32_t) v1x2;
        ksum[2] += (uint32_t) v2x2;
        ksum[2] += (uint32_t) v3x2;
        out[8] = v0x2;
        out[9] = v1x2;
        out[10] = v2x2;
        out[11] = v3x2;
        const int8_t v0x3 = 3 < n ? w0[3] : 0;
        const int8_t v1x3 = 3 < n ? w1[3] : 0;
        const int8_t v2x3 = 3 < n ? w2[3] : 0;
        const int8_t v3x3 = 3 < n ? w3[3] : 0;
        ksum[3] += (uint32_t) v0x3;
        ksum[3] += (uint32_t) v1x3;
        ksum[3] += (uint32_t) v2x3;
        ksum[3] += (uint32_t) v3x3;
        out[12] = v0x3;
        out[13] = v1x3;
        out[14] = v2x3;
        out[15] = v3x3;
        const int8_t v0x4 = 4 < n ? w0[4] : 0;
        const int8_t v1x4 = 4 < n ? w1[4] : 0;
        const int8_t v2x4 = 4 < n ? w2[4] : 0;
        const int8_t v3x4 = 4 < n ? w3[4] : 0;
        ksum[4] += (uint32_t) v0x4;
        ksum[4] += (uint32_t) v1x4;
        ksum[4] += (uint32_t) v2x4;
        ksum[4] += (uint32_t) v3x4;
        out[16] = v0x4;
        out[17] = v1x4;
        out[18] = v2x4;
        out[19] = v3x4;
        const int8_t v0x5 = 5 < n ? w0[5] : 0;
        const int8_t v1x5 = 5 < n ? w1[5] : 0;
        const int8_t v2x5 = 5 < n ? w2[5] : 0;
        const int8_t v3x5 = 5 < n ? w3[5] : 0;
        ksum[5] += (uint32_t) v0x5;
        ksum[5] += (uint32_t) v1x5;
        ksum[5] += (uint32_t) v2x5;
        ksum[5] += (uint32_t) v3x5;
        out[20] = v0x5;
        out[21] = v1x5;
        out[22] = v2x5;
        out[23] = v3x5;
        const int8_t v0x6 = 6 < n ? w0[6] : 0;
        const int8_t v1x6 = 6 < n ? w1[6] : 0;
        const int8_t v2x6 = 6 < n ? w2[6] : 0;
        const int8_t v3x6 = 6 < n ? w3[6] : 0;
        ksum[6] += (uint32_t) v0x6;
        ksum[6] += (uint32_t) v1x6;
        ksum[6] += (uint32_t) v2x6;
        ksum[6] += (uint32_t) v3x6;
        out[24] = v0x6;
        out[25] = v1x6;
        out[26] = v2x6;
        out[27] = v3x6;
        const int8_t v0x7 = 7 < n ? w0[7] : 0;
        const int8_t v1x7 = 7 < n ? w1[7] : 0;
        const int8_t v2x7 = 7 < n ? w2[7] : 0;
        const int8_t v3x7 = 7 < n ? w3[7] : 0;
        ksum[7] += (uint32_t) v0x7;
        ksum[7] += (uint32_t) v1x7;
        ksum[7] += (uint32_t) v2x7;
        ksum[7] += (uint32_t) v3x7;
        out[28] = v0x7;
        out[29] = v1x7;
        out[30] = v2x7;
        out[31] = v3x7;
        const int8_t v0x8 = 8 < n ? w0[8] : 0;
        const int8_t v1x8 = 8 < n ? w1[8] : 0;
        const int8_t v2x8 = 8 < n ? w2[8] : 0;
        const int8_t v3x8 = 8 < n ? w3[8] : 0;
        ksum[8] += (uint32_t) v0x8;
        ksum[8] += (uint32_t) v1x8;
        ksum[8] += (uint32_t) v2x8;
        ksum[8] += (uint32_t) v3x8;
        out[32] = v0x8;
        out[33] = v1x8;
        out[34] = v2x8;
        out[35] = v3x8;
        const int8_t v0x9 = 9 < n ? w0[9] : 0;
        const int8_t v1x9 = 9 < n ? w1[9] : 0;
        const int8_t v2x9 = 9 < n ? w2[9] : 0;
        const int8_t v3x9 = 9 < n ? w3[9] : 0;
        ksum[9] += (uint32_t) v0x9;
        ksum[9] += (uint32_t) v1x9;
        ksum[9] += (uint32_t) v2x9;
        ksum[9] += (uint32_t) v3x9;
        out[36] = v0x9;
        out[37] = v1x9;
        out[38] = v2x9;
        out[39] = v3x9;
        const int8_t v0x10 = 10 < n ? w0[10] : 0;
        const int8_t v1x10 = 10 < n ? w1[10] : 0;
        const int8_t v2x10 = 10 < n ? w2[10] : 0;
        const int8_t v3x10 = 10 < n ? w3[10] : 0;
        ksum[10] += (uint32_t) v0x10;
        ksum[10] += (uint32_t) v1x10;
        ksum[10] += (uint32_t) v2x10;
        ksum[10] += (uint32_t) v3x10;
        out[40] = v0x10;
        out[41] = v1x10;
        out[42] = v2x10;
        out[43] = v3x10;
        const int8_t v0x11 = 11 < n ? w0[11] : 0;
        const int8_t v1x11 = 11 < n ? w1[11] : 0;
        const int8_t v2x11 = 11 < n ? w2[11] : 0;
        const int8_t v3x11 = 11 < n ? w3[11] : 0;
        ksum[11] += (uint32_t) v0x11;
        ksum[11] += (uint32_t) v1x11;
        ksum[11] += (uint32_t) v2x11;
        ksum[11] += (uint32_t) v3x11;
        out[44] = v0x11;
        out[45] = v1x11;
        out[46] = v2x11;
        out[47] = v3x11;
        const int8_t v0x12 = 12 < n ? w0[12] : 0;
        const int8_t v1x12 = 12 < n ? w1[12] : 0;
        const int8_t v2x12 = 12 < n ? w2[12] : 0;
        const int8_t v3x12 = 12 < n ? w3[12] : 0;
        ksum[12] += (uint32_t) v0x12;
        ksum[12] += (uint32_t) v1x12;
        ksum[12] += (uint32_t) v2x12;
        ksum[12] += (uint32_t) v3x12;
        out[48] = v0x12;
        out[49] = v1x12;
        out[50] = v2x12;
        out[51] = v3x12;
        const int8_t v0x13 = 13 < n ? w0[13] : 0;
        const int8_t v1x13 = 13 < n ? w1[13] : 0;
        const int8_t v2x13 = 13 < n ? w2[13] : 0;
        const int8_t v3x13 = 13 < n ? w3[13] : 0;
        ksum[13] += (uint32_t) v0x13;
        ksum[13] += (uint32_t) v1x13;
        ksum[13] += (uint32_t) v2x13;
        ksum[13] += (uint32_t) v3x13;
        out[52] = v0x13;
        out[53] = v1x13;
        out[54] = v2x13;
        out[55] = v3x13;
        const int8_t v0x14 = 14 < n ? w0[14] : 0;
        const int8_t v1x14 = 14 < n ? w1[14] : 0;
        const int8_t v2x14 = 14 < n ? w2[14] : 0;
        const int8_t v3x14 = 14 < n ? w3[14] : 0;
        ksum[14] += (uint32_t) v0x14;
        ksum[14] += (uint32_t) v1x14;
        ksum[14] += (uint32_t) v2x14;
        ksum[14] += (uint32_t) v3x14;
        out[56] = v0x14;
        out[57] = v1x14;
        out[58] = v2x14;
        out[59] = v3x14;
        const int8_t v0x15 = 15 < n ? w0[15] : 0;
        const int8_t v1x15 = 15 < n ? w1[15] : 0;
        const int8_t v2x15 = 15 < n ? w2[15] : 0;
        const int8_t v3x15 = 15 < n ? w3[15] : 0;
        ksum[15] += (uint32_t) v0x15;
        ksum[15] += (uint32_t) v1x15;
        ksum[15] += (uint32_t) v2x15;
        ksum[15] += (uint32_t) v3x15;
        out[60] = v0x15;
        out[61] = v1x15;
        out[62] = v2x15;
        out[63] = v3x15;
        for (size_t N = 16; N < n; ++N) {
          const int8_t v0 = w0[N];
          const int8_t v1 = w1[N];
          const int8_t v2 = w2[N];
          const int8_t v3 = w3[N];
          ksum[N] += (uint32_t) v0;
          ksum[N] += (uint32_t) v1;
          ksum[N] += (uint32_t) v2;
          ksum[N] += (uint32_t) v3;
          out[N*4 + 0] = v0;
          out[N*4 + 1] = v1;
          out[N*4 + 2] = v2;
          out[N*4 + 3] = v3;
        }
        for (size_t N = n; N < 64; ++N) {
          out[N*4 + 0] = 0;
          out[N*4 + 1] = 0;
          out[N*4 + 2] = 0;
          out[N*4 + 3] = 0;
        }
        w0 += 4 * k_stride;
        w1 += 4 * k_stride;
        w2 += 4 * k_stride;
        w3 += 4 * k_stride;
        out += 256;
      }

      // KC remainder of 1..3
      if (k != 0) {
        assert(k >= 1 && k <= 3);
        for (size_t N = 0; N < n; ++N) {
          const int8_t v0 = w0[N];
          const int8_t v1 = 1 < k ? w1[N] : 0;
          const int8_t v2 = 2 < k ? w2[N] : 0;
          const int8_t v3 = 3 < k ? w3[N] : 0;
          ksum[N] += (uint32_t) v0;
          ksum[N] += (uint32_t) v1;
          ksum[N] += (uint32_t) v2;
          ksum[N] += (uint32_t) v3;
          out[N*4] = v0;
          out[N*4 + 1] = v1;
          out[N*4 + 2] = v2;
          out[N*4 + 3] = v3;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        w2 += k * k_stride;
        w3 += k * k_stride;
        out += 256;
      }

      for (size_t N = 0; N < 63; ++N) {
        packed_b[N] -= ksum[N] * izp;
      }
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
