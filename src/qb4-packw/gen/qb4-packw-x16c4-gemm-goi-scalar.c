// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qb4-packw/kr-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "src/xnnpack/packw.h"

static XNN_INTRINSIC void xnn_packed2planar(
  int32_t* ksum,
  const uint8_t* weights,
  int8_t* out)
{
  const uint8_t v0 = weights[0] & 0xF;
  const uint8_t v1 = weights[0] >> 4;
  const uint8_t v2 = weights[1] & 0xF;
  const uint8_t v3 = weights[1] >> 4;
  const uint8_t v4 = weights[2] & 0xF;
  const uint8_t v5 = weights[2] >> 4;
  const uint8_t v6 = weights[3] & 0xF;
  const uint8_t v7 = weights[3] >> 4;

  (*ksum) += (uint32_t) (v0);
  (*ksum) += (uint32_t) (v1);
  (*ksum) += (uint32_t) (v2);
  (*ksum) += (uint32_t) (v3);
  (*ksum) += (uint32_t) (v4);
  (*ksum) += (uint32_t) (v5);
  (*ksum) += (uint32_t) (v6);
  (*ksum) += (uint32_t) (v7);

  // Subtract 8 zero points (8)
  (*ksum) -= 64;

  out[0] = (v0 | (v4 << 4)) ^ 0x88;
  out[1] = (v1 | (v5 << 4)) ^ 0x88;
  out[2] = (v2 | (v6 << 4)) ^ 0x88;
  out[3] = (v3 | (v7 << 4)) ^ 0x88;
}

void xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(extra_bytes_bl == nr * sizeof(uint16_t));
  assert(extra_bytes_n == nr * sizeof(float));
  assert(params != NULL);
  assert(kc % bl == 0);
  size_t num_blocks = kc / bl;
  size_t weight_stride = (kc >> 1);

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const uint32_t izp = (uint32_t) (((const struct xnn_qs8_qc4w_packing_params*) params)->input_zero_point + 0);

  do {
    // NC main loop multiple of 16
    const uint8_t* w0 = (const uint8_t*) weights;
    const uint16_t* s0 = (const uint16_t*) scale;
    int n = nc;
    for (;n > 0; n -= 16) {
      float* packed_k_scaled_sum = (float*) out;
      ((float*) out)[0] = 0;
      ((float*) out)[1] = 0;
      ((float*) out)[2] = 0;
      ((float*) out)[3] = 0;
      ((float*) out)[4] = 0;
      ((float*) out)[5] = 0;
      ((float*) out)[6] = 0;
      ((float*) out)[7] = 0;
      ((float*) out)[8] = 0;
      ((float*) out)[9] = 0;
      ((float*) out)[10] = 0;
      ((float*) out)[11] = 0;
      ((float*) out)[12] = 0;
      ((float*) out)[13] = 0;
      ((float*) out)[14] = 0;
      ((float*) out)[15] = 0;
      out += 16 * sizeof(float);

      // KC/2 bytes is KC Nibbles
      const uint8_t* w1 = w0 + weight_stride;
      const uint8_t* w2 = w1 + weight_stride;
      const uint8_t* w3 = w2 + weight_stride;
      const uint8_t* w4 = w3 + weight_stride;
      const uint8_t* w5 = w4 + weight_stride;
      const uint8_t* w6 = w5 + weight_stride;
      const uint8_t* w7 = w6 + weight_stride;
      const uint8_t* w8 = w7 + weight_stride;
      const uint8_t* w9 = w8 + weight_stride;
      const uint8_t* w10 = w9 + weight_stride;
      const uint8_t* w11 = w10 + weight_stride;
      const uint8_t* w12 = w11 + weight_stride;
      const uint8_t* w13 = w12 + weight_stride;
      const uint8_t* w14 = w13 + weight_stride;
      const uint8_t* w15 = w14 + weight_stride;

      // scales
      const uint16_t* s1 = s0 + num_blocks;
      const uint16_t* s2 = s1 + num_blocks;
      const uint16_t* s3 = s2 + num_blocks;
      const uint16_t* s4 = s3 + num_blocks;
      const uint16_t* s5 = s4 + num_blocks;
      const uint16_t* s6 = s5 + num_blocks;
      const uint16_t* s7 = s6 + num_blocks;
      const uint16_t* s8 = s7 + num_blocks;
      const uint16_t* s9 = s8 + num_blocks;
      const uint16_t* s10 = s9 + num_blocks;
      const uint16_t* s11 = s10 + num_blocks;
      const uint16_t* s12 = s11 + num_blocks;
      const uint16_t* s13 = s12 + num_blocks;
      const uint16_t* s14 = s13 + num_blocks;
      const uint16_t* s15 = s14 + num_blocks;

      if XNN_UNLIKELY(n < 16){
        if XNN_UNPREDICTABLE(n < 2) {
          w1 = w0;
          s1 = s0;
        }
        if XNN_UNPREDICTABLE(n <= 2) {
          w2 = w1;
          s2 = s1;
        }
        if XNN_UNPREDICTABLE(n < 4) {
          w3 = w2;
          s3 = s2;
        }
        if XNN_UNPREDICTABLE(n <= 4) {
          w4 = w3;
          s4 = s3;
        }
        if XNN_UNPREDICTABLE(n < 6) {
          w5 = w4;
          s5 = s4;
        }
        if XNN_UNPREDICTABLE(n <= 6) {
          w6 = w5;
          s6 = s5;
        }
        if XNN_UNPREDICTABLE(n < 8) {
          w7 = w6;
          s7 = s6;
        }
        if XNN_UNPREDICTABLE(n <= 8) {
          w8 = w7;
          s8 = s7;
        }
        if XNN_UNPREDICTABLE(n < 10) {
          w9 = w8;
          s9 = s8;
        }
        if XNN_UNPREDICTABLE(n <= 10) {
          w10 = w9;
          s10 = s9;
        }
        if XNN_UNPREDICTABLE(n < 12) {
          w11 = w10;
          s11 = s10;
        }
        if XNN_UNPREDICTABLE(n <= 12) {
          w12 = w11;
          s12 = s11;
        }
        if XNN_UNPREDICTABLE(n < 14) {
          w13 = w12;
          s13 = s12;
        }
        if XNN_UNPREDICTABLE(n <= 14) {
          w14 = w13;
          s14 = s13;
        }
        if XNN_UNPREDICTABLE(n < 16) {
          w15 = w14;
          s15 = s14;
        }
      }

      size_t kb = kc;
      // Process k by blocks (bl)
      for (; kb >= bl; kb-=bl) {
        // Initialize KSum as subtracting bl zero points (8)
        int32_t ksum0 = 0;
        int32_t ksum1 = 0;
        int32_t ksum2 = 0;
        int32_t ksum3 = 0;
        int32_t ksum4 = 0;
        int32_t ksum5 = 0;
        int32_t ksum6 = 0;
        int32_t ksum7 = 0;
        int32_t ksum8 = 0;
        int32_t ksum9 = 0;
        int32_t ksum10 = 0;
        int32_t ksum11 = 0;
        int32_t ksum12 = 0;
        int32_t ksum13 = 0;
        int32_t ksum14 = 0;
        int32_t ksum15 = 0;
        size_t k = bl;
        for(; k >= 8; k-=8) {
          xnn_packed2planar(&ksum0, w0, out + 0); w0 += 4;
          xnn_packed2planar(&ksum1, w1, out + 4); w1 += 4;
          xnn_packed2planar(&ksum2, w2, out + 8); w2 += 4;
          xnn_packed2planar(&ksum3, w3, out + 12); w3 += 4;
          xnn_packed2planar(&ksum4, w4, out + 16); w4 += 4;
          xnn_packed2planar(&ksum5, w5, out + 20); w5 += 4;
          xnn_packed2planar(&ksum6, w6, out + 24); w6 += 4;
          xnn_packed2planar(&ksum7, w7, out + 28); w7 += 4;
          xnn_packed2planar(&ksum8, w8, out + 32); w8 += 4;
          xnn_packed2planar(&ksum9, w9, out + 36); w9 += 4;
          xnn_packed2planar(&ksum10, w10, out + 40); w10 += 4;
          xnn_packed2planar(&ksum11, w11, out + 44); w11 += 4;
          xnn_packed2planar(&ksum12, w12, out + 48); w12 += 4;
          xnn_packed2planar(&ksum13, w13, out + 52); w13 += 4;
          xnn_packed2planar(&ksum14, w14, out + 56); w14 += 4;
          xnn_packed2planar(&ksum15, w15, out + 60); w15 += 4;

          out += 64;
        }
        float scale0 = math_cvt_fp32_bf16(s0[0]);
        float scale1 = math_cvt_fp32_bf16(s1[0]);
        float scale2 = math_cvt_fp32_bf16(s2[0]);
        float scale3 = math_cvt_fp32_bf16(s3[0]);
        float scale4 = math_cvt_fp32_bf16(s4[0]);
        float scale5 = math_cvt_fp32_bf16(s5[0]);
        float scale6 = math_cvt_fp32_bf16(s6[0]);
        float scale7 = math_cvt_fp32_bf16(s7[0]);
        float scale8 = math_cvt_fp32_bf16(s8[0]);
        float scale9 = math_cvt_fp32_bf16(s9[0]);
        float scale10 = math_cvt_fp32_bf16(s10[0]);
        float scale11 = math_cvt_fp32_bf16(s11[0]);
        float scale12 = math_cvt_fp32_bf16(s12[0]);
        float scale13 = math_cvt_fp32_bf16(s13[0]);
        float scale14 = math_cvt_fp32_bf16(s14[0]);
        float scale15 = math_cvt_fp32_bf16(s15[0]);
        s0 += 1;
        s1 += 1;
        s2 += 1;
        s3 += 1;
        s4 += 1;
        s5 += 1;
        s6 += 1;
        s7 += 1;
        s8 += 1;
        s9 += 1;
        s10 += 1;
        s11 += 1;
        s12 += 1;
        s13 += 1;
        s14 += 1;
        s15 += 1;

        packed_k_scaled_sum[0] -= (float)ksum0 * izp * scale0;
        packed_k_scaled_sum[1] -= (float)ksum1 * izp * scale1;
        packed_k_scaled_sum[2] -= (float)ksum2 * izp * scale2;
        packed_k_scaled_sum[3] -= (float)ksum3 * izp * scale3;
        packed_k_scaled_sum[4] -= (float)ksum4 * izp * scale4;
        packed_k_scaled_sum[5] -= (float)ksum5 * izp * scale5;
        packed_k_scaled_sum[6] -= (float)ksum6 * izp * scale6;
        packed_k_scaled_sum[7] -= (float)ksum7 * izp * scale7;
        packed_k_scaled_sum[8] -= (float)ksum8 * izp * scale8;
        packed_k_scaled_sum[9] -= (float)ksum9 * izp * scale9;
        packed_k_scaled_sum[10] -= (float)ksum10 * izp * scale10;
        packed_k_scaled_sum[11] -= (float)ksum11 * izp * scale11;
        packed_k_scaled_sum[12] -= (float)ksum12 * izp * scale12;
        packed_k_scaled_sum[13] -= (float)ksum13 * izp * scale13;
        packed_k_scaled_sum[14] -= (float)ksum14 * izp * scale14;
        packed_k_scaled_sum[15] -= (float)ksum15 * izp * scale15;

        ((uint16_t*) out)[0] = math_cvt_bf16_fp32(scale0 / 16.0f);
        ((uint16_t*) out)[1] = math_cvt_bf16_fp32(scale1 / 16.0f);
        ((uint16_t*) out)[2] = math_cvt_bf16_fp32(scale2 / 16.0f);
        ((uint16_t*) out)[3] = math_cvt_bf16_fp32(scale3 / 16.0f);
        ((uint16_t*) out)[4] = math_cvt_bf16_fp32(scale4 / 16.0f);
        ((uint16_t*) out)[5] = math_cvt_bf16_fp32(scale5 / 16.0f);
        ((uint16_t*) out)[6] = math_cvt_bf16_fp32(scale6 / 16.0f);
        ((uint16_t*) out)[7] = math_cvt_bf16_fp32(scale7 / 16.0f);
        ((uint16_t*) out)[8] = math_cvt_bf16_fp32(scale8 / 16.0f);
        ((uint16_t*) out)[9] = math_cvt_bf16_fp32(scale9 / 16.0f);
        ((uint16_t*) out)[10] = math_cvt_bf16_fp32(scale10 / 16.0f);
        ((uint16_t*) out)[11] = math_cvt_bf16_fp32(scale11 / 16.0f);
        ((uint16_t*) out)[12] = math_cvt_bf16_fp32(scale12 / 16.0f);
        ((uint16_t*) out)[13] = math_cvt_bf16_fp32(scale13 / 16.0f);
        ((uint16_t*) out)[14] = math_cvt_bf16_fp32(scale14 / 16.0f);
        ((uint16_t*) out)[15] = math_cvt_bf16_fp32(scale15 / 16.0f);

        out += 16 * sizeof(uint16_t);
      }

      if XNN_LIKELY(b != NULL){
        ((uint32_t*) out)[0] = b[0];
        ((uint32_t*) out)[1] = b[1];
        ((uint32_t*) out)[2] = b[2];
        ((uint32_t*) out)[3] = b[3];
        ((uint32_t*) out)[4] = b[4];
        ((uint32_t*) out)[5] = b[5];
        ((uint32_t*) out)[6] = b[6];
        ((uint32_t*) out)[7] = b[7];
        ((uint32_t*) out)[8] = b[8];
        ((uint32_t*) out)[9] = b[9];
        ((uint32_t*) out)[10] = b[10];
        ((uint32_t*) out)[11] = b[11];
        ((uint32_t*) out)[12] = b[12];
        ((uint32_t*) out)[13] = b[13];
        ((uint32_t*) out)[14] = b[14];
        ((uint32_t*) out)[15] = b[15];
        b += 16;
      } else {
        ((uint32_t*) out)[0] = 0;
        ((uint32_t*) out)[1] = 0;
        ((uint32_t*) out)[2] = 0;
        ((uint32_t*) out)[3] = 0;
        ((uint32_t*) out)[4] = 0;
        ((uint32_t*) out)[5] = 0;
        ((uint32_t*) out)[6] = 0;
        ((uint32_t*) out)[7] = 0;
        ((uint32_t*) out)[8] = 0;
        ((uint32_t*) out)[9] = 0;
        ((uint32_t*) out)[10] = 0;
        ((uint32_t*) out)[11] = 0;
        ((uint32_t*) out)[12] = 0;
        ((uint32_t*) out)[13] = 0;
        ((uint32_t*) out)[14] = 0;
        ((uint32_t*) out)[15] = 0;
      }
      out += 16 * sizeof(uint32_t);
      w0 = w15;
      s0 = s15;
    }
  } while (--g != 0);
}
