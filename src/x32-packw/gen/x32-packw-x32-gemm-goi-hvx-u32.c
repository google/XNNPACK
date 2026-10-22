// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x32-packw/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/simd/s32-hvx.h"


void xnn_x32_packw_gemm_goi_ukernel_x32__hvx_u32(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(nr == 32);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const xnn_simd_s32_t vzero = xnn_set1_s32(0);
  int32_t* out = (int32_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  do {
    // NC main loop multiple of 32
    const int32_t* w0 = (const int32_t*) weights;
    size_t n = nc;
    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const xnn_simd_s32_t vb0 = xnn_loadu_s32(b + 0);
        xnn_store_s32(out + 0, vb0);
        b += 32;
      } else {
        xnn_store_s32(out + 0, vzero);
      }
      out += 32;

      const int32_t* w1 = w0 + n_stride;
      const int32_t* w2 = w1 + n_stride;
      const int32_t* w3 = w2 + n_stride;
      const int32_t* w4 = w3 + n_stride;
      const int32_t* w5 = w4 + n_stride;
      const int32_t* w6 = w5 + n_stride;
      const int32_t* w7 = w6 + n_stride;
      const int32_t* w8 = w7 + n_stride;
      const int32_t* w9 = w8 + n_stride;
      const int32_t* w10 = w9 + n_stride;
      const int32_t* w11 = w10 + n_stride;
      const int32_t* w12 = w11 + n_stride;
      const int32_t* w13 = w12 + n_stride;
      const int32_t* w14 = w13 + n_stride;
      const int32_t* w15 = w14 + n_stride;
      const int32_t* w16 = w15 + n_stride;
      const int32_t* w17 = w16 + n_stride;
      const int32_t* w18 = w17 + n_stride;
      const int32_t* w19 = w18 + n_stride;
      const int32_t* w20 = w19 + n_stride;
      const int32_t* w21 = w20 + n_stride;
      const int32_t* w22 = w21 + n_stride;
      const int32_t* w23 = w22 + n_stride;
      const int32_t* w24 = w23 + n_stride;
      const int32_t* w25 = w24 + n_stride;
      const int32_t* w26 = w25 + n_stride;
      const int32_t* w27 = w26 + n_stride;
      const int32_t* w28 = w27 + n_stride;
      const int32_t* w29 = w28 + n_stride;
      const int32_t* w30 = w29 + n_stride;
      const int32_t* w31 = w30 + n_stride;

      // KC main loop multiple of 32x32 using 32x32 HVX butterfly transpose
      size_t k = kc;
      for (; k >= 32; k -= 32) {
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w0);
          w0 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w1);
          w1 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w2);
          w2 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w3);
          w3 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w4);
          w4 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w5);
          w5 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w6);
          w6 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w7);
          w7 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w8);
          w8 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w9);
          w9 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w10);
          w10 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w11);
          w11 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w12);
          w12 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w13);
          w13 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w14);
          w14 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w15);
          w15 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w16);
          w16 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w17);
          w17 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w18);
          w18 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w19);
          w19 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w20);
          w20 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w21);
          w21 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w22);
          w22 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w23);
          w23 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w24);
          w24 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w25);
          w25 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w26);
          w26 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w27);
          w27 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w28);
          w28 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w29);
          w29 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w30);
          w30 += 32;
          const HVX_Vector v5_31 = xnn_loadu_s32(w31);
          w31 += 32;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 0, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 32, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 64, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 96, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 128, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 160, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 192, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 224, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 256, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 288, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 320, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 352, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 384, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 416, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 448, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 480, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 512, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 544, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 576, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 608, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 640, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 672, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 704, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 736, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 768, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 800, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 832, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 864, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 896, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 928, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 960, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 992, Q6_V_hi_W(v0_15));
        }
        out += 1024;
      }

      // KC remainder (1..31)
      for (; k >= 2; k -= 2) {
        const int32_t v0_0 = w0[0];
        const int32_t v0_1 = w0[1];
        w0 += 2;
        const int32_t v1_0 = w1[0];
        const int32_t v1_1 = w1[1];
        w1 += 2;
        const int32_t v2_0 = w2[0];
        const int32_t v2_1 = w2[1];
        w2 += 2;
        const int32_t v3_0 = w3[0];
        const int32_t v3_1 = w3[1];
        w3 += 2;
        const int32_t v4_0 = w4[0];
        const int32_t v4_1 = w4[1];
        w4 += 2;
        const int32_t v5_0 = w5[0];
        const int32_t v5_1 = w5[1];
        w5 += 2;
        const int32_t v6_0 = w6[0];
        const int32_t v6_1 = w6[1];
        w6 += 2;
        const int32_t v7_0 = w7[0];
        const int32_t v7_1 = w7[1];
        w7 += 2;
        const int32_t v8_0 = w8[0];
        const int32_t v8_1 = w8[1];
        w8 += 2;
        const int32_t v9_0 = w9[0];
        const int32_t v9_1 = w9[1];
        w9 += 2;
        const int32_t v10_0 = w10[0];
        const int32_t v10_1 = w10[1];
        w10 += 2;
        const int32_t v11_0 = w11[0];
        const int32_t v11_1 = w11[1];
        w11 += 2;
        const int32_t v12_0 = w12[0];
        const int32_t v12_1 = w12[1];
        w12 += 2;
        const int32_t v13_0 = w13[0];
        const int32_t v13_1 = w13[1];
        w13 += 2;
        const int32_t v14_0 = w14[0];
        const int32_t v14_1 = w14[1];
        w14 += 2;
        const int32_t v15_0 = w15[0];
        const int32_t v15_1 = w15[1];
        w15 += 2;
        const int32_t v16_0 = w16[0];
        const int32_t v16_1 = w16[1];
        w16 += 2;
        const int32_t v17_0 = w17[0];
        const int32_t v17_1 = w17[1];
        w17 += 2;
        const int32_t v18_0 = w18[0];
        const int32_t v18_1 = w18[1];
        w18 += 2;
        const int32_t v19_0 = w19[0];
        const int32_t v19_1 = w19[1];
        w19 += 2;
        const int32_t v20_0 = w20[0];
        const int32_t v20_1 = w20[1];
        w20 += 2;
        const int32_t v21_0 = w21[0];
        const int32_t v21_1 = w21[1];
        w21 += 2;
        const int32_t v22_0 = w22[0];
        const int32_t v22_1 = w22[1];
        w22 += 2;
        const int32_t v23_0 = w23[0];
        const int32_t v23_1 = w23[1];
        w23 += 2;
        const int32_t v24_0 = w24[0];
        const int32_t v24_1 = w24[1];
        w24 += 2;
        const int32_t v25_0 = w25[0];
        const int32_t v25_1 = w25[1];
        w25 += 2;
        const int32_t v26_0 = w26[0];
        const int32_t v26_1 = w26[1];
        w26 += 2;
        const int32_t v27_0 = w27[0];
        const int32_t v27_1 = w27[1];
        w27 += 2;
        const int32_t v28_0 = w28[0];
        const int32_t v28_1 = w28[1];
        w28 += 2;
        const int32_t v29_0 = w29[0];
        const int32_t v29_1 = w29[1];
        w29 += 2;
        const int32_t v30_0 = w30[0];
        const int32_t v30_1 = w30[1];
        w30 += 2;
        const int32_t v31_0 = w31[0];
        const int32_t v31_1 = w31[1];
        w31 += 2;
        out[0] = v0_0;
        out[32] = v0_1;
        out[1] = v1_0;
        out[33] = v1_1;
        out[2] = v2_0;
        out[34] = v2_1;
        out[3] = v3_0;
        out[35] = v3_1;
        out[4] = v4_0;
        out[36] = v4_1;
        out[5] = v5_0;
        out[37] = v5_1;
        out[6] = v6_0;
        out[38] = v6_1;
        out[7] = v7_0;
        out[39] = v7_1;
        out[8] = v8_0;
        out[40] = v8_1;
        out[9] = v9_0;
        out[41] = v9_1;
        out[10] = v10_0;
        out[42] = v10_1;
        out[11] = v11_0;
        out[43] = v11_1;
        out[12] = v12_0;
        out[44] = v12_1;
        out[13] = v13_0;
        out[45] = v13_1;
        out[14] = v14_0;
        out[46] = v14_1;
        out[15] = v15_0;
        out[47] = v15_1;
        out[16] = v16_0;
        out[48] = v16_1;
        out[17] = v17_0;
        out[49] = v17_1;
        out[18] = v18_0;
        out[50] = v18_1;
        out[19] = v19_0;
        out[51] = v19_1;
        out[20] = v20_0;
        out[52] = v20_1;
        out[21] = v21_0;
        out[53] = v21_1;
        out[22] = v22_0;
        out[54] = v22_1;
        out[23] = v23_0;
        out[55] = v23_1;
        out[24] = v24_0;
        out[56] = v24_1;
        out[25] = v25_0;
        out[57] = v25_1;
        out[26] = v26_0;
        out[58] = v26_1;
        out[27] = v27_0;
        out[59] = v27_1;
        out[28] = v28_0;
        out[60] = v28_1;
        out[29] = v29_0;
        out[61] = v29_1;
        out[30] = v30_0;
        out[62] = v30_1;
        out[31] = v31_0;
        out[63] = v31_1;
        out += 64;
      }
      for (; k != 0; --k) {
        const int32_t v0 = *w0++;
        out[0] = v0;
        const int32_t v1 = *w1++;
        out[1] = v1;
        const int32_t v2 = *w2++;
        out[2] = v2;
        const int32_t v3 = *w3++;
        out[3] = v3;
        const int32_t v4 = *w4++;
        out[4] = v4;
        const int32_t v5 = *w5++;
        out[5] = v5;
        const int32_t v6 = *w6++;
        out[6] = v6;
        const int32_t v7 = *w7++;
        out[7] = v7;
        const int32_t v8 = *w8++;
        out[8] = v8;
        const int32_t v9 = *w9++;
        out[9] = v9;
        const int32_t v10 = *w10++;
        out[10] = v10;
        const int32_t v11 = *w11++;
        out[11] = v11;
        const int32_t v12 = *w12++;
        out[12] = v12;
        const int32_t v13 = *w13++;
        out[13] = v13;
        const int32_t v14 = *w14++;
        out[14] = v14;
        const int32_t v15 = *w15++;
        out[15] = v15;
        const int32_t v16 = *w16++;
        out[16] = v16;
        const int32_t v17 = *w17++;
        out[17] = v17;
        const int32_t v18 = *w18++;
        out[18] = v18;
        const int32_t v19 = *w19++;
        out[19] = v19;
        const int32_t v20 = *w20++;
        out[20] = v20;
        const int32_t v21 = *w21++;
        out[21] = v21;
        const int32_t v22 = *w22++;
        out[22] = v22;
        const int32_t v23 = *w23++;
        out[23] = v23;
        const int32_t v24 = *w24++;
        out[24] = v24;
        const int32_t v25 = *w25++;
        out[25] = v25;
        const int32_t v26 = *w26++;
        out[26] = v26;
        const int32_t v27 = *w27++;
        out[27] = v27;
        const int32_t v28 = *w28++;
        out[28] = v28;
        const int32_t v29 = *w29++;
        out[29] = v29;
        const int32_t v30 = *w30++;
        out[30] = v30;
        const int32_t v31 = *w31++;
        out[31] = v31;
        out += 32;
      }
      out = (int32_t*) ((uintptr_t) out + extra_bytes);
      w0 = w31 + n_stride - kc;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *out++ = *b++;
        } while (--nb != 0);
        out += (32 - n);
      } else {
        xnn_store_s32(out + 0, vzero);
        out += 32;
      }

      // NR remainder has less than 32 rows so last row is not loaded
      const int32_t* w1 = w0 + n_stride;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int32_t* w2 = w1 + n_stride;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int32_t* w3 = w2 + n_stride;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int32_t* w4 = w3 + n_stride;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int32_t* w5 = w4 + n_stride;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int32_t* w6 = w5 + n_stride;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int32_t* w7 = w6 + n_stride;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }
      const int32_t* w8 = w7 + n_stride;
      if XNN_UNPREDICTABLE(n <= 8) {
        w8 = w7;
      }
      const int32_t* w9 = w8 + n_stride;
      if XNN_UNPREDICTABLE(n < 10) {
        w9 = w8;
      }
      const int32_t* w10 = w9 + n_stride;
      if XNN_UNPREDICTABLE(n <= 10) {
        w10 = w9;
      }
      const int32_t* w11 = w10 + n_stride;
      if XNN_UNPREDICTABLE(n < 12) {
        w11 = w10;
      }
      const int32_t* w12 = w11 + n_stride;
      if XNN_UNPREDICTABLE(n <= 12) {
        w12 = w11;
      }
      const int32_t* w13 = w12 + n_stride;
      if XNN_UNPREDICTABLE(n < 14) {
        w13 = w12;
      }
      const int32_t* w14 = w13 + n_stride;
      if XNN_UNPREDICTABLE(n <= 14) {
        w14 = w13;
      }
      const int32_t* w15 = w14 + n_stride;
      if XNN_UNPREDICTABLE(n < 16) {
        w15 = w14;
      }
      const int32_t* w16 = w15 + n_stride;
      if XNN_UNPREDICTABLE(n <= 16) {
        w16 = w15;
      }
      const int32_t* w17 = w16 + n_stride;
      if XNN_UNPREDICTABLE(n < 18) {
        w17 = w16;
      }
      const int32_t* w18 = w17 + n_stride;
      if XNN_UNPREDICTABLE(n <= 18) {
        w18 = w17;
      }
      const int32_t* w19 = w18 + n_stride;
      if XNN_UNPREDICTABLE(n < 20) {
        w19 = w18;
      }
      const int32_t* w20 = w19 + n_stride;
      if XNN_UNPREDICTABLE(n <= 20) {
        w20 = w19;
      }
      const int32_t* w21 = w20 + n_stride;
      if XNN_UNPREDICTABLE(n < 22) {
        w21 = w20;
      }
      const int32_t* w22 = w21 + n_stride;
      if XNN_UNPREDICTABLE(n <= 22) {
        w22 = w21;
      }
      const int32_t* w23 = w22 + n_stride;
      if XNN_UNPREDICTABLE(n < 24) {
        w23 = w22;
      }
      const int32_t* w24 = w23 + n_stride;
      if XNN_UNPREDICTABLE(n <= 24) {
        w24 = w23;
      }
      const int32_t* w25 = w24 + n_stride;
      if XNN_UNPREDICTABLE(n < 26) {
        w25 = w24;
      }
      const int32_t* w26 = w25 + n_stride;
      if XNN_UNPREDICTABLE(n <= 26) {
        w26 = w25;
      }
      const int32_t* w27 = w26 + n_stride;
      if XNN_UNPREDICTABLE(n < 28) {
        w27 = w26;
      }
      const int32_t* w28 = w27 + n_stride;
      if XNN_UNPREDICTABLE(n <= 28) {
        w28 = w27;
      }
      const int32_t* w29 = w28 + n_stride;
      if XNN_UNPREDICTABLE(n < 30) {
        w29 = w28;
      }
      const int32_t* w30 = w29 + n_stride;
      if XNN_UNPREDICTABLE(n <= 30) {
        w30 = w29;
      }

      // KC main loop multiple of 32x32 using 32x32 HVX butterfly transpose
      size_t k = kc;
      for (; k >= 32; k -= 32) {
        {
          const HVX_Vector v5_0 = xnn_loadu_s32(w0);
          w0 += 32;
          const HVX_Vector v5_1 = xnn_loadu_s32(w1);
          w1 += 32;
          const HVX_Vector v5_2 = xnn_loadu_s32(w2);
          w2 += 32;
          const HVX_Vector v5_3 = xnn_loadu_s32(w3);
          w3 += 32;
          const HVX_Vector v5_4 = xnn_loadu_s32(w4);
          w4 += 32;
          const HVX_Vector v5_5 = xnn_loadu_s32(w5);
          w5 += 32;
          const HVX_Vector v5_6 = xnn_loadu_s32(w6);
          w6 += 32;
          const HVX_Vector v5_7 = xnn_loadu_s32(w7);
          w7 += 32;
          const HVX_Vector v5_8 = xnn_loadu_s32(w8);
          w8 += 32;
          const HVX_Vector v5_9 = xnn_loadu_s32(w9);
          w9 += 32;
          const HVX_Vector v5_10 = xnn_loadu_s32(w10);
          w10 += 32;
          const HVX_Vector v5_11 = xnn_loadu_s32(w11);
          w11 += 32;
          const HVX_Vector v5_12 = xnn_loadu_s32(w12);
          w12 += 32;
          const HVX_Vector v5_13 = xnn_loadu_s32(w13);
          w13 += 32;
          const HVX_Vector v5_14 = xnn_loadu_s32(w14);
          w14 += 32;
          const HVX_Vector v5_15 = xnn_loadu_s32(w15);
          w15 += 32;
          const HVX_Vector v5_16 = xnn_loadu_s32(w16);
          w16 += 32;
          const HVX_Vector v5_17 = xnn_loadu_s32(w17);
          w17 += 32;
          const HVX_Vector v5_18 = xnn_loadu_s32(w18);
          w18 += 32;
          const HVX_Vector v5_19 = xnn_loadu_s32(w19);
          w19 += 32;
          const HVX_Vector v5_20 = xnn_loadu_s32(w20);
          w20 += 32;
          const HVX_Vector v5_21 = xnn_loadu_s32(w21);
          w21 += 32;
          const HVX_Vector v5_22 = xnn_loadu_s32(w22);
          w22 += 32;
          const HVX_Vector v5_23 = xnn_loadu_s32(w23);
          w23 += 32;
          const HVX_Vector v5_24 = xnn_loadu_s32(w24);
          w24 += 32;
          const HVX_Vector v5_25 = xnn_loadu_s32(w25);
          w25 += 32;
          const HVX_Vector v5_26 = xnn_loadu_s32(w26);
          w26 += 32;
          const HVX_Vector v5_27 = xnn_loadu_s32(w27);
          w27 += 32;
          const HVX_Vector v5_28 = xnn_loadu_s32(w28);
          w28 += 32;
          const HVX_Vector v5_29 = xnn_loadu_s32(w29);
          w29 += 32;
          const HVX_Vector v5_30 = xnn_loadu_s32(w30);
          w30 += 32;
          const HVX_Vector v5_31 = vzero;

          const HVX_VectorPair v4_0 = Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 = Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 = Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 = Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 = Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 = Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 = Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 = Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 = Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 = Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 = Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 = Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 = Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 = Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 = Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 = Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          xnn_store_s32(out + 0, Q6_V_lo_W(v0_0));
          xnn_store_s32(out + 32, Q6_V_hi_W(v0_0));
          xnn_store_s32(out + 64, Q6_V_lo_W(v0_1));
          xnn_store_s32(out + 96, Q6_V_hi_W(v0_1));
          xnn_store_s32(out + 128, Q6_V_lo_W(v0_2));
          xnn_store_s32(out + 160, Q6_V_hi_W(v0_2));
          xnn_store_s32(out + 192, Q6_V_lo_W(v0_3));
          xnn_store_s32(out + 224, Q6_V_hi_W(v0_3));
          xnn_store_s32(out + 256, Q6_V_lo_W(v0_4));
          xnn_store_s32(out + 288, Q6_V_hi_W(v0_4));
          xnn_store_s32(out + 320, Q6_V_lo_W(v0_5));
          xnn_store_s32(out + 352, Q6_V_hi_W(v0_5));
          xnn_store_s32(out + 384, Q6_V_lo_W(v0_6));
          xnn_store_s32(out + 416, Q6_V_hi_W(v0_6));
          xnn_store_s32(out + 448, Q6_V_lo_W(v0_7));
          xnn_store_s32(out + 480, Q6_V_hi_W(v0_7));
          xnn_store_s32(out + 512, Q6_V_lo_W(v0_8));
          xnn_store_s32(out + 544, Q6_V_hi_W(v0_8));
          xnn_store_s32(out + 576, Q6_V_lo_W(v0_9));
          xnn_store_s32(out + 608, Q6_V_hi_W(v0_9));
          xnn_store_s32(out + 640, Q6_V_lo_W(v0_10));
          xnn_store_s32(out + 672, Q6_V_hi_W(v0_10));
          xnn_store_s32(out + 704, Q6_V_lo_W(v0_11));
          xnn_store_s32(out + 736, Q6_V_hi_W(v0_11));
          xnn_store_s32(out + 768, Q6_V_lo_W(v0_12));
          xnn_store_s32(out + 800, Q6_V_hi_W(v0_12));
          xnn_store_s32(out + 832, Q6_V_lo_W(v0_13));
          xnn_store_s32(out + 864, Q6_V_hi_W(v0_13));
          xnn_store_s32(out + 896, Q6_V_lo_W(v0_14));
          xnn_store_s32(out + 928, Q6_V_hi_W(v0_14));
          xnn_store_s32(out + 960, Q6_V_lo_W(v0_15));
          xnn_store_s32(out + 992, Q6_V_hi_W(v0_15));
        }
        out += 1024;
      }

      // KC remainder (1..31)
      for (; k >= 2; k -= 2) {
        const int32_t v0_0 = w0[0];
        const int32_t v0_1 = w0[1];
        w0 += 2;
        const int32_t v1_0 = w1[0];
        const int32_t v1_1 = w1[1];
        w1 += 2;
        const int32_t v2_0 = w2[0];
        const int32_t v2_1 = w2[1];
        w2 += 2;
        const int32_t v3_0 = w3[0];
        const int32_t v3_1 = w3[1];
        w3 += 2;
        const int32_t v4_0 = w4[0];
        const int32_t v4_1 = w4[1];
        w4 += 2;
        const int32_t v5_0 = w5[0];
        const int32_t v5_1 = w5[1];
        w5 += 2;
        const int32_t v6_0 = w6[0];
        const int32_t v6_1 = w6[1];
        w6 += 2;
        const int32_t v7_0 = w7[0];
        const int32_t v7_1 = w7[1];
        w7 += 2;
        const int32_t v8_0 = w8[0];
        const int32_t v8_1 = w8[1];
        w8 += 2;
        const int32_t v9_0 = w9[0];
        const int32_t v9_1 = w9[1];
        w9 += 2;
        const int32_t v10_0 = w10[0];
        const int32_t v10_1 = w10[1];
        w10 += 2;
        const int32_t v11_0 = w11[0];
        const int32_t v11_1 = w11[1];
        w11 += 2;
        const int32_t v12_0 = w12[0];
        const int32_t v12_1 = w12[1];
        w12 += 2;
        const int32_t v13_0 = w13[0];
        const int32_t v13_1 = w13[1];
        w13 += 2;
        const int32_t v14_0 = w14[0];
        const int32_t v14_1 = w14[1];
        w14 += 2;
        const int32_t v15_0 = w15[0];
        const int32_t v15_1 = w15[1];
        w15 += 2;
        const int32_t v16_0 = w16[0];
        const int32_t v16_1 = w16[1];
        w16 += 2;
        const int32_t v17_0 = w17[0];
        const int32_t v17_1 = w17[1];
        w17 += 2;
        const int32_t v18_0 = w18[0];
        const int32_t v18_1 = w18[1];
        w18 += 2;
        const int32_t v19_0 = w19[0];
        const int32_t v19_1 = w19[1];
        w19 += 2;
        const int32_t v20_0 = w20[0];
        const int32_t v20_1 = w20[1];
        w20 += 2;
        const int32_t v21_0 = w21[0];
        const int32_t v21_1 = w21[1];
        w21 += 2;
        const int32_t v22_0 = w22[0];
        const int32_t v22_1 = w22[1];
        w22 += 2;
        const int32_t v23_0 = w23[0];
        const int32_t v23_1 = w23[1];
        w23 += 2;
        const int32_t v24_0 = w24[0];
        const int32_t v24_1 = w24[1];
        w24 += 2;
        const int32_t v25_0 = w25[0];
        const int32_t v25_1 = w25[1];
        w25 += 2;
        const int32_t v26_0 = w26[0];
        const int32_t v26_1 = w26[1];
        w26 += 2;
        const int32_t v27_0 = w27[0];
        const int32_t v27_1 = w27[1];
        w27 += 2;
        const int32_t v28_0 = w28[0];
        const int32_t v28_1 = w28[1];
        w28 += 2;
        const int32_t v29_0 = w29[0];
        const int32_t v29_1 = w29[1];
        w29 += 2;
        const int32_t v30_0 = w30[0];
        const int32_t v30_1 = w30[1];
        w30 += 2;
        out[0] = v0_0;
        out[32] = v0_1;
        out[1] = v1_0;
        out[33] = v1_1;
        out[2] = v2_0;
        out[34] = v2_1;
        out[3] = v3_0;
        out[35] = v3_1;
        out[4] = v4_0;
        out[36] = v4_1;
        out[5] = v5_0;
        out[37] = v5_1;
        out[6] = v6_0;
        out[38] = v6_1;
        out[7] = v7_0;
        out[39] = v7_1;
        out[8] = v8_0;
        out[40] = v8_1;
        out[9] = v9_0;
        out[41] = v9_1;
        out[10] = v10_0;
        out[42] = v10_1;
        out[11] = v11_0;
        out[43] = v11_1;
        out[12] = v12_0;
        out[44] = v12_1;
        out[13] = v13_0;
        out[45] = v13_1;
        out[14] = v14_0;
        out[46] = v14_1;
        out[15] = v15_0;
        out[47] = v15_1;
        out[16] = v16_0;
        out[48] = v16_1;
        out[17] = v17_0;
        out[49] = v17_1;
        out[18] = v18_0;
        out[50] = v18_1;
        out[19] = v19_0;
        out[51] = v19_1;
        out[20] = v20_0;
        out[52] = v20_1;
        out[21] = v21_0;
        out[53] = v21_1;
        out[22] = v22_0;
        out[54] = v22_1;
        out[23] = v23_0;
        out[55] = v23_1;
        out[24] = v24_0;
        out[56] = v24_1;
        out[25] = v25_0;
        out[57] = v25_1;
        out[26] = v26_0;
        out[58] = v26_1;
        out[27] = v27_0;
        out[59] = v27_1;
        out[28] = v28_0;
        out[60] = v28_1;
        out[29] = v29_0;
        out[61] = v29_1;
        out[30] = v30_0;
        out[62] = v30_1;
        out += 64;
      }
      for (; k != 0; --k) {
        const int32_t v0 = *w0++;
        out[0] = v0;
        const int32_t v1 = *w1++;
        out[1] = v1;
        const int32_t v2 = *w2++;
        out[2] = v2;
        const int32_t v3 = *w3++;
        out[3] = v3;
        const int32_t v4 = *w4++;
        out[4] = v4;
        const int32_t v5 = *w5++;
        out[5] = v5;
        const int32_t v6 = *w6++;
        out[6] = v6;
        const int32_t v7 = *w7++;
        out[7] = v7;
        const int32_t v8 = *w8++;
        out[8] = v8;
        const int32_t v9 = *w9++;
        out[9] = v9;
        const int32_t v10 = *w10++;
        out[10] = v10;
        const int32_t v11 = *w11++;
        out[11] = v11;
        const int32_t v12 = *w12++;
        out[12] = v12;
        const int32_t v13 = *w13++;
        out[13] = v13;
        const int32_t v14 = *w14++;
        out[14] = v14;
        const int32_t v15 = *w15++;
        out[15] = v15;
        const int32_t v16 = *w16++;
        out[16] = v16;
        const int32_t v17 = *w17++;
        out[17] = v17;
        const int32_t v18 = *w18++;
        out[18] = v18;
        const int32_t v19 = *w19++;
        out[19] = v19;
        const int32_t v20 = *w20++;
        out[20] = v20;
        const int32_t v21 = *w21++;
        out[21] = v21;
        const int32_t v22 = *w22++;
        out[22] = v22;
        const int32_t v23 = *w23++;
        out[23] = v23;
        const int32_t v24 = *w24++;
        out[24] = v24;
        const int32_t v25 = *w25++;
        out[25] = v25;
        const int32_t v26 = *w26++;
        out[26] = v26;
        const int32_t v27 = *w27++;
        out[27] = v27;
        const int32_t v28 = *w28++;
        out[28] = v28;
        const int32_t v29 = *w29++;
        out[29] = v29;
        const int32_t v30 = *w30++;
        out[30] = v30;
        out += 32;
      }
      out = (int32_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * n_stride;
  } while (--g != 0);
}
