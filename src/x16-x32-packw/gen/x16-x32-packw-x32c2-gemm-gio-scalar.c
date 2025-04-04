// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-x32-packw/kr-gio-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"


void xnn_x16_x32_packw_gemm_gio_ukernel_x32c2__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint16_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);
  assert(kr == 2);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint16_t* out = (uint16_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;

  do {
    // NC main loop multiple of 32
    const uint16_t* w0 = (const uint16_t*) weights;
    size_t n = nc;
    for (;n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        memcpy(out, b, sizeof(uint32_t) * 32);
        b += 32;
      } else {
        memset(out, 0, sizeof(uint32_t) * 32);
      }
      out += 32 * sizeof(uint32_t) / sizeof(uint16_t);

      const uint16_t* w1 = w0 + k_stride;

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint16_t v0x0 = w0[0];
        const uint16_t v1x0 = w1[0];
        out[0] = v0x0;
        out[1] = v1x0;
        const uint16_t v0x1 = w0[1];
        const uint16_t v1x1 = w1[1];
        out[2] = v0x1;
        out[3] = v1x1;
        const uint16_t v0x2 = w0[2];
        const uint16_t v1x2 = w1[2];
        out[4] = v0x2;
        out[5] = v1x2;
        const uint16_t v0x3 = w0[3];
        const uint16_t v1x3 = w1[3];
        out[6] = v0x3;
        out[7] = v1x3;
        const uint16_t v0x4 = w0[4];
        const uint16_t v1x4 = w1[4];
        out[8] = v0x4;
        out[9] = v1x4;
        const uint16_t v0x5 = w0[5];
        const uint16_t v1x5 = w1[5];
        out[10] = v0x5;
        out[11] = v1x5;
        const uint16_t v0x6 = w0[6];
        const uint16_t v1x6 = w1[6];
        out[12] = v0x6;
        out[13] = v1x6;
        const uint16_t v0x7 = w0[7];
        const uint16_t v1x7 = w1[7];
        out[14] = v0x7;
        out[15] = v1x7;
        const uint16_t v0x8 = w0[8];
        const uint16_t v1x8 = w1[8];
        out[16] = v0x8;
        out[17] = v1x8;
        const uint16_t v0x9 = w0[9];
        const uint16_t v1x9 = w1[9];
        out[18] = v0x9;
        out[19] = v1x9;
        const uint16_t v0x10 = w0[10];
        const uint16_t v1x10 = w1[10];
        out[20] = v0x10;
        out[21] = v1x10;
        const uint16_t v0x11 = w0[11];
        const uint16_t v1x11 = w1[11];
        out[22] = v0x11;
        out[23] = v1x11;
        const uint16_t v0x12 = w0[12];
        const uint16_t v1x12 = w1[12];
        out[24] = v0x12;
        out[25] = v1x12;
        const uint16_t v0x13 = w0[13];
        const uint16_t v1x13 = w1[13];
        out[26] = v0x13;
        out[27] = v1x13;
        const uint16_t v0x14 = w0[14];
        const uint16_t v1x14 = w1[14];
        out[28] = v0x14;
        out[29] = v1x14;
        const uint16_t v0x15 = w0[15];
        const uint16_t v1x15 = w1[15];
        out[30] = v0x15;
        out[31] = v1x15;
        const uint16_t v0x16 = w0[16];
        const uint16_t v1x16 = w1[16];
        out[32] = v0x16;
        out[33] = v1x16;
        const uint16_t v0x17 = w0[17];
        const uint16_t v1x17 = w1[17];
        out[34] = v0x17;
        out[35] = v1x17;
        const uint16_t v0x18 = w0[18];
        const uint16_t v1x18 = w1[18];
        out[36] = v0x18;
        out[37] = v1x18;
        const uint16_t v0x19 = w0[19];
        const uint16_t v1x19 = w1[19];
        out[38] = v0x19;
        out[39] = v1x19;
        const uint16_t v0x20 = w0[20];
        const uint16_t v1x20 = w1[20];
        out[40] = v0x20;
        out[41] = v1x20;
        const uint16_t v0x21 = w0[21];
        const uint16_t v1x21 = w1[21];
        out[42] = v0x21;
        out[43] = v1x21;
        const uint16_t v0x22 = w0[22];
        const uint16_t v1x22 = w1[22];
        out[44] = v0x22;
        out[45] = v1x22;
        const uint16_t v0x23 = w0[23];
        const uint16_t v1x23 = w1[23];
        out[46] = v0x23;
        out[47] = v1x23;
        const uint16_t v0x24 = w0[24];
        const uint16_t v1x24 = w1[24];
        out[48] = v0x24;
        out[49] = v1x24;
        const uint16_t v0x25 = w0[25];
        const uint16_t v1x25 = w1[25];
        out[50] = v0x25;
        out[51] = v1x25;
        const uint16_t v0x26 = w0[26];
        const uint16_t v1x26 = w1[26];
        out[52] = v0x26;
        out[53] = v1x26;
        const uint16_t v0x27 = w0[27];
        const uint16_t v1x27 = w1[27];
        out[54] = v0x27;
        out[55] = v1x27;
        const uint16_t v0x28 = w0[28];
        const uint16_t v1x28 = w1[28];
        out[56] = v0x28;
        out[57] = v1x28;
        const uint16_t v0x29 = w0[29];
        const uint16_t v1x29 = w1[29];
        out[58] = v0x29;
        out[59] = v1x29;
        const uint16_t v0x30 = w0[30];
        const uint16_t v1x30 = w1[30];
        out[60] = v0x30;
        out[61] = v1x30;
        const uint16_t v0x31 = w0[31];
        const uint16_t v1x31 = w1[31];
        out[62] = v0x31;
        out[63] = v1x31;
        w0 += 2 * k_stride;
        w1 += 2 * k_stride;
        out += 64;
      }

      // KC remainder of 1..1
      if (k != 0) {
        assert(k >= 1 && k <= 1);
        const uint16_t v0x0 = w0[0];
        const uint16_t v1x0 = 1 < k ? w1[0] : 0;
        out[0] = v0x0;
        out[1] = v1x0;
        const uint16_t v0x1 = w0[1];
        const uint16_t v1x1 = 1 < k ? w1[1] : 0;
        out[2] = v0x1;
        out[3] = v1x1;
        const uint16_t v0x2 = w0[2];
        const uint16_t v1x2 = 1 < k ? w1[2] : 0;
        out[4] = v0x2;
        out[5] = v1x2;
        const uint16_t v0x3 = w0[3];
        const uint16_t v1x3 = 1 < k ? w1[3] : 0;
        out[6] = v0x3;
        out[7] = v1x3;
        const uint16_t v0x4 = w0[4];
        const uint16_t v1x4 = 1 < k ? w1[4] : 0;
        out[8] = v0x4;
        out[9] = v1x4;
        const uint16_t v0x5 = w0[5];
        const uint16_t v1x5 = 1 < k ? w1[5] : 0;
        out[10] = v0x5;
        out[11] = v1x5;
        const uint16_t v0x6 = w0[6];
        const uint16_t v1x6 = 1 < k ? w1[6] : 0;
        out[12] = v0x6;
        out[13] = v1x6;
        const uint16_t v0x7 = w0[7];
        const uint16_t v1x7 = 1 < k ? w1[7] : 0;
        out[14] = v0x7;
        out[15] = v1x7;
        const uint16_t v0x8 = w0[8];
        const uint16_t v1x8 = 1 < k ? w1[8] : 0;
        out[16] = v0x8;
        out[17] = v1x8;
        const uint16_t v0x9 = w0[9];
        const uint16_t v1x9 = 1 < k ? w1[9] : 0;
        out[18] = v0x9;
        out[19] = v1x9;
        const uint16_t v0x10 = w0[10];
        const uint16_t v1x10 = 1 < k ? w1[10] : 0;
        out[20] = v0x10;
        out[21] = v1x10;
        const uint16_t v0x11 = w0[11];
        const uint16_t v1x11 = 1 < k ? w1[11] : 0;
        out[22] = v0x11;
        out[23] = v1x11;
        const uint16_t v0x12 = w0[12];
        const uint16_t v1x12 = 1 < k ? w1[12] : 0;
        out[24] = v0x12;
        out[25] = v1x12;
        const uint16_t v0x13 = w0[13];
        const uint16_t v1x13 = 1 < k ? w1[13] : 0;
        out[26] = v0x13;
        out[27] = v1x13;
        const uint16_t v0x14 = w0[14];
        const uint16_t v1x14 = 1 < k ? w1[14] : 0;
        out[28] = v0x14;
        out[29] = v1x14;
        const uint16_t v0x15 = w0[15];
        const uint16_t v1x15 = 1 < k ? w1[15] : 0;
        out[30] = v0x15;
        out[31] = v1x15;
        const uint16_t v0x16 = w0[16];
        const uint16_t v1x16 = 1 < k ? w1[16] : 0;
        out[32] = v0x16;
        out[33] = v1x16;
        const uint16_t v0x17 = w0[17];
        const uint16_t v1x17 = 1 < k ? w1[17] : 0;
        out[34] = v0x17;
        out[35] = v1x17;
        const uint16_t v0x18 = w0[18];
        const uint16_t v1x18 = 1 < k ? w1[18] : 0;
        out[36] = v0x18;
        out[37] = v1x18;
        const uint16_t v0x19 = w0[19];
        const uint16_t v1x19 = 1 < k ? w1[19] : 0;
        out[38] = v0x19;
        out[39] = v1x19;
        const uint16_t v0x20 = w0[20];
        const uint16_t v1x20 = 1 < k ? w1[20] : 0;
        out[40] = v0x20;
        out[41] = v1x20;
        const uint16_t v0x21 = w0[21];
        const uint16_t v1x21 = 1 < k ? w1[21] : 0;
        out[42] = v0x21;
        out[43] = v1x21;
        const uint16_t v0x22 = w0[22];
        const uint16_t v1x22 = 1 < k ? w1[22] : 0;
        out[44] = v0x22;
        out[45] = v1x22;
        const uint16_t v0x23 = w0[23];
        const uint16_t v1x23 = 1 < k ? w1[23] : 0;
        out[46] = v0x23;
        out[47] = v1x23;
        const uint16_t v0x24 = w0[24];
        const uint16_t v1x24 = 1 < k ? w1[24] : 0;
        out[48] = v0x24;
        out[49] = v1x24;
        const uint16_t v0x25 = w0[25];
        const uint16_t v1x25 = 1 < k ? w1[25] : 0;
        out[50] = v0x25;
        out[51] = v1x25;
        const uint16_t v0x26 = w0[26];
        const uint16_t v1x26 = 1 < k ? w1[26] : 0;
        out[52] = v0x26;
        out[53] = v1x26;
        const uint16_t v0x27 = w0[27];
        const uint16_t v1x27 = 1 < k ? w1[27] : 0;
        out[54] = v0x27;
        out[55] = v1x27;
        const uint16_t v0x28 = w0[28];
        const uint16_t v1x28 = 1 < k ? w1[28] : 0;
        out[56] = v0x28;
        out[57] = v1x28;
        const uint16_t v0x29 = w0[29];
        const uint16_t v1x29 = 1 < k ? w1[29] : 0;
        out[58] = v0x29;
        out[59] = v1x29;
        const uint16_t v0x30 = w0[30];
        const uint16_t v1x30 = 1 < k ? w1[30] : 0;
        out[60] = v0x30;
        out[61] = v1x30;
        const uint16_t v0x31 = w0[31];
        const uint16_t v1x31 = 1 < k ? w1[31] : 0;
        out[62] = v0x31;
        out[63] = v1x31;
        w0 += k * k_stride;
        w1 += k * k_stride;
        out += 64;
      }

      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
      w0 = w0 - kc * k_stride + 32;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((uint32_t*) out) = *b++;
          out += sizeof(uint32_t)/sizeof(uint16_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((uint32_t*) out) = 0;
          out += sizeof(uint32_t)/sizeof(uint16_t);
        } while (--nb != 0);
      }
      out += (32 - n) * sizeof(uint32_t) / sizeof(uint16_t);

     // NR remainder has less than 32 rows so last row is not loaded
      const uint16_t* w1 = w0 + k_stride;

      // KC main loop multiple of 32x2
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const uint16_t v0x0 = w0[0];
        const uint16_t v1x0 = w1[0];
        out[0] = v0x0;
        out[1] = v1x0;
        if (1 < n) {
          const uint16_t v0x1 = w0[1];
          const uint16_t v1x1 = w1[1];
          out[2] = v0x1;
          out[3] = v1x1;
        }
        if (2 < n) {
          const uint16_t v0x2 = w0[2];
          const uint16_t v1x2 = w1[2];
          out[4] = v0x2;
          out[5] = v1x2;
        }
        if (3 < n) {
          const uint16_t v0x3 = w0[3];
          const uint16_t v1x3 = w1[3];
          out[6] = v0x3;
          out[7] = v1x3;
        }
        if (4 < n) {
          const uint16_t v0x4 = w0[4];
          const uint16_t v1x4 = w1[4];
          out[8] = v0x4;
          out[9] = v1x4;
        }
        if (5 < n) {
          const uint16_t v0x5 = w0[5];
          const uint16_t v1x5 = w1[5];
          out[10] = v0x5;
          out[11] = v1x5;
        }
        if (6 < n) {
          const uint16_t v0x6 = w0[6];
          const uint16_t v1x6 = w1[6];
          out[12] = v0x6;
          out[13] = v1x6;
        }
        if (7 < n) {
          const uint16_t v0x7 = w0[7];
          const uint16_t v1x7 = w1[7];
          out[14] = v0x7;
          out[15] = v1x7;
        }
        if (8 < n) {
          const uint16_t v0x8 = w0[8];
          const uint16_t v1x8 = w1[8];
          out[16] = v0x8;
          out[17] = v1x8;
        }
        if (9 < n) {
          const uint16_t v0x9 = w0[9];
          const uint16_t v1x9 = w1[9];
          out[18] = v0x9;
          out[19] = v1x9;
        }
        if (10 < n) {
          const uint16_t v0x10 = w0[10];
          const uint16_t v1x10 = w1[10];
          out[20] = v0x10;
          out[21] = v1x10;
        }
        if (11 < n) {
          const uint16_t v0x11 = w0[11];
          const uint16_t v1x11 = w1[11];
          out[22] = v0x11;
          out[23] = v1x11;
        }
        if (12 < n) {
          const uint16_t v0x12 = w0[12];
          const uint16_t v1x12 = w1[12];
          out[24] = v0x12;
          out[25] = v1x12;
        }
        if (13 < n) {
          const uint16_t v0x13 = w0[13];
          const uint16_t v1x13 = w1[13];
          out[26] = v0x13;
          out[27] = v1x13;
        }
        if (14 < n) {
          const uint16_t v0x14 = w0[14];
          const uint16_t v1x14 = w1[14];
          out[28] = v0x14;
          out[29] = v1x14;
        }
        if (15 < n) {
          const uint16_t v0x15 = w0[15];
          const uint16_t v1x15 = w1[15];
          out[30] = v0x15;
          out[31] = v1x15;
        }
        if (16 < n) {
          const uint16_t v0x16 = w0[16];
          const uint16_t v1x16 = w1[16];
          out[32] = v0x16;
          out[33] = v1x16;
        }
        if (17 < n) {
          const uint16_t v0x17 = w0[17];
          const uint16_t v1x17 = w1[17];
          out[34] = v0x17;
          out[35] = v1x17;
        }
        if (18 < n) {
          const uint16_t v0x18 = w0[18];
          const uint16_t v1x18 = w1[18];
          out[36] = v0x18;
          out[37] = v1x18;
        }
        if (19 < n) {
          const uint16_t v0x19 = w0[19];
          const uint16_t v1x19 = w1[19];
          out[38] = v0x19;
          out[39] = v1x19;
        }
        if (20 < n) {
          const uint16_t v0x20 = w0[20];
          const uint16_t v1x20 = w1[20];
          out[40] = v0x20;
          out[41] = v1x20;
        }
        if (21 < n) {
          const uint16_t v0x21 = w0[21];
          const uint16_t v1x21 = w1[21];
          out[42] = v0x21;
          out[43] = v1x21;
        }
        if (22 < n) {
          const uint16_t v0x22 = w0[22];
          const uint16_t v1x22 = w1[22];
          out[44] = v0x22;
          out[45] = v1x22;
        }
        if (23 < n) {
          const uint16_t v0x23 = w0[23];
          const uint16_t v1x23 = w1[23];
          out[46] = v0x23;
          out[47] = v1x23;
        }
        if (24 < n) {
          const uint16_t v0x24 = w0[24];
          const uint16_t v1x24 = w1[24];
          out[48] = v0x24;
          out[49] = v1x24;
        }
        if (25 < n) {
          const uint16_t v0x25 = w0[25];
          const uint16_t v1x25 = w1[25];
          out[50] = v0x25;
          out[51] = v1x25;
        }
        if (26 < n) {
          const uint16_t v0x26 = w0[26];
          const uint16_t v1x26 = w1[26];
          out[52] = v0x26;
          out[53] = v1x26;
        }
        if (27 < n) {
          const uint16_t v0x27 = w0[27];
          const uint16_t v1x27 = w1[27];
          out[54] = v0x27;
          out[55] = v1x27;
        }
        if (28 < n) {
          const uint16_t v0x28 = w0[28];
          const uint16_t v1x28 = w1[28];
          out[56] = v0x28;
          out[57] = v1x28;
        }
        if (29 < n) {
          const uint16_t v0x29 = w0[29];
          const uint16_t v1x29 = w1[29];
          out[58] = v0x29;
          out[59] = v1x29;
        }
        if (30 < n) {
          const uint16_t v0x30 = w0[30];
          const uint16_t v1x30 = w1[30];
          out[60] = v0x30;
          out[61] = v1x30;
        }
        w0 += 2 * k_stride;
        w1 += 2 * k_stride;
        out += 64;
      }

      // KC remainder of 1..1
      if (k != 0) {
        assert(k >= 1 && k <= 1);
        if (0 < n) {
          const uint16_t v0x0 = w0[0];
          const uint16_t v1x0 = 1 < k ? w1[0] : 0;
          out[0] = v0x0;
          out[1] = v1x0;
        }
        if (1 < n) {
          const uint16_t v0x1 = w0[1];
          const uint16_t v1x1 = 1 < k ? w1[1] : 0;
          out[2] = v0x1;
          out[3] = v1x1;
        }
        if (2 < n) {
          const uint16_t v0x2 = w0[2];
          const uint16_t v1x2 = 1 < k ? w1[2] : 0;
          out[4] = v0x2;
          out[5] = v1x2;
        }
        if (3 < n) {
          const uint16_t v0x3 = w0[3];
          const uint16_t v1x3 = 1 < k ? w1[3] : 0;
          out[6] = v0x3;
          out[7] = v1x3;
        }
        if (4 < n) {
          const uint16_t v0x4 = w0[4];
          const uint16_t v1x4 = 1 < k ? w1[4] : 0;
          out[8] = v0x4;
          out[9] = v1x4;
        }
        if (5 < n) {
          const uint16_t v0x5 = w0[5];
          const uint16_t v1x5 = 1 < k ? w1[5] : 0;
          out[10] = v0x5;
          out[11] = v1x5;
        }
        if (6 < n) {
          const uint16_t v0x6 = w0[6];
          const uint16_t v1x6 = 1 < k ? w1[6] : 0;
          out[12] = v0x6;
          out[13] = v1x6;
        }
        if (7 < n) {
          const uint16_t v0x7 = w0[7];
          const uint16_t v1x7 = 1 < k ? w1[7] : 0;
          out[14] = v0x7;
          out[15] = v1x7;
        }
        if (8 < n) {
          const uint16_t v0x8 = w0[8];
          const uint16_t v1x8 = 1 < k ? w1[8] : 0;
          out[16] = v0x8;
          out[17] = v1x8;
        }
        if (9 < n) {
          const uint16_t v0x9 = w0[9];
          const uint16_t v1x9 = 1 < k ? w1[9] : 0;
          out[18] = v0x9;
          out[19] = v1x9;
        }
        if (10 < n) {
          const uint16_t v0x10 = w0[10];
          const uint16_t v1x10 = 1 < k ? w1[10] : 0;
          out[20] = v0x10;
          out[21] = v1x10;
        }
        if (11 < n) {
          const uint16_t v0x11 = w0[11];
          const uint16_t v1x11 = 1 < k ? w1[11] : 0;
          out[22] = v0x11;
          out[23] = v1x11;
        }
        if (12 < n) {
          const uint16_t v0x12 = w0[12];
          const uint16_t v1x12 = 1 < k ? w1[12] : 0;
          out[24] = v0x12;
          out[25] = v1x12;
        }
        if (13 < n) {
          const uint16_t v0x13 = w0[13];
          const uint16_t v1x13 = 1 < k ? w1[13] : 0;
          out[26] = v0x13;
          out[27] = v1x13;
        }
        if (14 < n) {
          const uint16_t v0x14 = w0[14];
          const uint16_t v1x14 = 1 < k ? w1[14] : 0;
          out[28] = v0x14;
          out[29] = v1x14;
        }
        if (15 < n) {
          const uint16_t v0x15 = w0[15];
          const uint16_t v1x15 = 1 < k ? w1[15] : 0;
          out[30] = v0x15;
          out[31] = v1x15;
        }
        if (16 < n) {
          const uint16_t v0x16 = w0[16];
          const uint16_t v1x16 = 1 < k ? w1[16] : 0;
          out[32] = v0x16;
          out[33] = v1x16;
        }
        if (17 < n) {
          const uint16_t v0x17 = w0[17];
          const uint16_t v1x17 = 1 < k ? w1[17] : 0;
          out[34] = v0x17;
          out[35] = v1x17;
        }
        if (18 < n) {
          const uint16_t v0x18 = w0[18];
          const uint16_t v1x18 = 1 < k ? w1[18] : 0;
          out[36] = v0x18;
          out[37] = v1x18;
        }
        if (19 < n) {
          const uint16_t v0x19 = w0[19];
          const uint16_t v1x19 = 1 < k ? w1[19] : 0;
          out[38] = v0x19;
          out[39] = v1x19;
        }
        if (20 < n) {
          const uint16_t v0x20 = w0[20];
          const uint16_t v1x20 = 1 < k ? w1[20] : 0;
          out[40] = v0x20;
          out[41] = v1x20;
        }
        if (21 < n) {
          const uint16_t v0x21 = w0[21];
          const uint16_t v1x21 = 1 < k ? w1[21] : 0;
          out[42] = v0x21;
          out[43] = v1x21;
        }
        if (22 < n) {
          const uint16_t v0x22 = w0[22];
          const uint16_t v1x22 = 1 < k ? w1[22] : 0;
          out[44] = v0x22;
          out[45] = v1x22;
        }
        if (23 < n) {
          const uint16_t v0x23 = w0[23];
          const uint16_t v1x23 = 1 < k ? w1[23] : 0;
          out[46] = v0x23;
          out[47] = v1x23;
        }
        if (24 < n) {
          const uint16_t v0x24 = w0[24];
          const uint16_t v1x24 = 1 < k ? w1[24] : 0;
          out[48] = v0x24;
          out[49] = v1x24;
        }
        if (25 < n) {
          const uint16_t v0x25 = w0[25];
          const uint16_t v1x25 = 1 < k ? w1[25] : 0;
          out[50] = v0x25;
          out[51] = v1x25;
        }
        if (26 < n) {
          const uint16_t v0x26 = w0[26];
          const uint16_t v1x26 = 1 < k ? w1[26] : 0;
          out[52] = v0x26;
          out[53] = v1x26;
        }
        if (27 < n) {
          const uint16_t v0x27 = w0[27];
          const uint16_t v1x27 = 1 < k ? w1[27] : 0;
          out[54] = v0x27;
          out[55] = v1x27;
        }
        if (28 < n) {
          const uint16_t v0x28 = w0[28];
          const uint16_t v1x28 = 1 < k ? w1[28] : 0;
          out[56] = v0x28;
          out[57] = v1x28;
        }
        if (29 < n) {
          const uint16_t v0x29 = w0[29];
          const uint16_t v1x29 = 1 < k ? w1[29] : 0;
          out[58] = v0x29;
          out[59] = v1x29;
        }
        if (30 < n) {
          const uint16_t v0x30 = w0[30];
          const uint16_t v1x30 = 1 < k ? w1[30] : 0;
          out[60] = v0x30;
          out[61] = v1x30;
        }
        w0 += k * k_stride;
        w1 += k * k_stride;
        out += 64;
      }

      out = (uint16_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
