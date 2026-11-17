// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4-hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/simd/s32-hvx.h"
#include "src/xnnpack/unaligned.h"

void xnn_qs8_to_qu8_packw_gemm_goi_ukernel_x128c4__hvx(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t n_stride,
    const int8_t* weights,
    const int32_t* bias,
    const void* scale,
    int8_t* packed_weights,
    size_t extra_bytes,
    const void* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 128);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const int32_t izp = (int32_t) (params
      ? (((const struct xnn_qs8_packw_params*) params)->input_zero_point + 128)
      : 128);
  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector vones = Q6_Vb_vsplat_R(1);
  const HVX_Vector vizp = Q6_V_vsplat_R(izp);

  int8_t* out = packed_weights;
  const int32_t* b = bias;

  do {
    const int8_t* w0 = weights;
    size_t n = nc;
    for (; n >= 128; n -= 128) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        *((HVX_UVector*) (packed_b + 0)) =
            *((const HVX_UVector*) (b + 0));
        *((HVX_UVector*) (packed_b + 32)) =
            *((const HVX_UVector*) (b + 32));
        *((HVX_UVector*) (packed_b + 64)) =
            *((const HVX_UVector*) (b + 64));
        *((HVX_UVector*) (packed_b + 96)) =
            *((const HVX_UVector*) (b + 96));
        b += 128;
      } else {
        *((HVX_UVector*) (packed_b + 0)) = vzero;
        *((HVX_UVector*) (packed_b + 32)) = vzero;
        *((HVX_UVector*) (packed_b + 64)) = vzero;
        *((HVX_UVector*) (packed_b + 96)) = vzero;
      }
      out += 128 * sizeof(int32_t);

      const int8_t* w1 = w0 + n_stride;
      const int8_t* w2 = w1 + n_stride;
      const int8_t* w3 = w2 + n_stride;
      const int8_t* w4 = w3 + n_stride;
      const int8_t* w5 = w4 + n_stride;
      const int8_t* w6 = w5 + n_stride;
      const int8_t* w7 = w6 + n_stride;
      const int8_t* w8 = w7 + n_stride;
      const int8_t* w9 = w8 + n_stride;
      const int8_t* w10 = w9 + n_stride;
      const int8_t* w11 = w10 + n_stride;
      const int8_t* w12 = w11 + n_stride;
      const int8_t* w13 = w12 + n_stride;
      const int8_t* w14 = w13 + n_stride;
      const int8_t* w15 = w14 + n_stride;
      const int8_t* w16 = w15 + n_stride;
      const int8_t* w17 = w16 + n_stride;
      const int8_t* w18 = w17 + n_stride;
      const int8_t* w19 = w18 + n_stride;
      const int8_t* w20 = w19 + n_stride;
      const int8_t* w21 = w20 + n_stride;
      const int8_t* w22 = w21 + n_stride;
      const int8_t* w23 = w22 + n_stride;
      const int8_t* w24 = w23 + n_stride;
      const int8_t* w25 = w24 + n_stride;
      const int8_t* w26 = w25 + n_stride;
      const int8_t* w27 = w26 + n_stride;
      const int8_t* w28 = w27 + n_stride;
      const int8_t* w29 = w28 + n_stride;
      const int8_t* w30 = w29 + n_stride;
      const int8_t* w31 = w30 + n_stride;
      const int8_t* w32 = w31 + n_stride;
      const int8_t* w33 = w32 + n_stride;
      const int8_t* w34 = w33 + n_stride;
      const int8_t* w35 = w34 + n_stride;
      const int8_t* w36 = w35 + n_stride;
      const int8_t* w37 = w36 + n_stride;
      const int8_t* w38 = w37 + n_stride;
      const int8_t* w39 = w38 + n_stride;
      const int8_t* w40 = w39 + n_stride;
      const int8_t* w41 = w40 + n_stride;
      const int8_t* w42 = w41 + n_stride;
      const int8_t* w43 = w42 + n_stride;
      const int8_t* w44 = w43 + n_stride;
      const int8_t* w45 = w44 + n_stride;
      const int8_t* w46 = w45 + n_stride;
      const int8_t* w47 = w46 + n_stride;
      const int8_t* w48 = w47 + n_stride;
      const int8_t* w49 = w48 + n_stride;
      const int8_t* w50 = w49 + n_stride;
      const int8_t* w51 = w50 + n_stride;
      const int8_t* w52 = w51 + n_stride;
      const int8_t* w53 = w52 + n_stride;
      const int8_t* w54 = w53 + n_stride;
      const int8_t* w55 = w54 + n_stride;
      const int8_t* w56 = w55 + n_stride;
      const int8_t* w57 = w56 + n_stride;
      const int8_t* w58 = w57 + n_stride;
      const int8_t* w59 = w58 + n_stride;
      const int8_t* w60 = w59 + n_stride;
      const int8_t* w61 = w60 + n_stride;
      const int8_t* w62 = w61 + n_stride;
      const int8_t* w63 = w62 + n_stride;
      const int8_t* w64 = w63 + n_stride;
      const int8_t* w65 = w64 + n_stride;
      const int8_t* w66 = w65 + n_stride;
      const int8_t* w67 = w66 + n_stride;
      const int8_t* w68 = w67 + n_stride;
      const int8_t* w69 = w68 + n_stride;
      const int8_t* w70 = w69 + n_stride;
      const int8_t* w71 = w70 + n_stride;
      const int8_t* w72 = w71 + n_stride;
      const int8_t* w73 = w72 + n_stride;
      const int8_t* w74 = w73 + n_stride;
      const int8_t* w75 = w74 + n_stride;
      const int8_t* w76 = w75 + n_stride;
      const int8_t* w77 = w76 + n_stride;
      const int8_t* w78 = w77 + n_stride;
      const int8_t* w79 = w78 + n_stride;
      const int8_t* w80 = w79 + n_stride;
      const int8_t* w81 = w80 + n_stride;
      const int8_t* w82 = w81 + n_stride;
      const int8_t* w83 = w82 + n_stride;
      const int8_t* w84 = w83 + n_stride;
      const int8_t* w85 = w84 + n_stride;
      const int8_t* w86 = w85 + n_stride;
      const int8_t* w87 = w86 + n_stride;
      const int8_t* w88 = w87 + n_stride;
      const int8_t* w89 = w88 + n_stride;
      const int8_t* w90 = w89 + n_stride;
      const int8_t* w91 = w90 + n_stride;
      const int8_t* w92 = w91 + n_stride;
      const int8_t* w93 = w92 + n_stride;
      const int8_t* w94 = w93 + n_stride;
      const int8_t* w95 = w94 + n_stride;
      const int8_t* w96 = w95 + n_stride;
      const int8_t* w97 = w96 + n_stride;
      const int8_t* w98 = w97 + n_stride;
      const int8_t* w99 = w98 + n_stride;
      const int8_t* w100 = w99 + n_stride;
      const int8_t* w101 = w100 + n_stride;
      const int8_t* w102 = w101 + n_stride;
      const int8_t* w103 = w102 + n_stride;
      const int8_t* w104 = w103 + n_stride;
      const int8_t* w105 = w104 + n_stride;
      const int8_t* w106 = w105 + n_stride;
      const int8_t* w107 = w106 + n_stride;
      const int8_t* w108 = w107 + n_stride;
      const int8_t* w109 = w108 + n_stride;
      const int8_t* w110 = w109 + n_stride;
      const int8_t* w111 = w110 + n_stride;
      const int8_t* w112 = w111 + n_stride;
      const int8_t* w113 = w112 + n_stride;
      const int8_t* w114 = w113 + n_stride;
      const int8_t* w115 = w114 + n_stride;
      const int8_t* w116 = w115 + n_stride;
      const int8_t* w117 = w116 + n_stride;
      const int8_t* w118 = w117 + n_stride;
      const int8_t* w119 = w118 + n_stride;
      const int8_t* w120 = w119 + n_stride;
      const int8_t* w121 = w120 + n_stride;
      const int8_t* w122 = w121 + n_stride;
      const int8_t* w123 = w122 + n_stride;
      const int8_t* w124 = w123 + n_stride;
      const int8_t* w125 = w124 + n_stride;
      const int8_t* w126 = w125 + n_stride;
      const int8_t* w127 = w126 + n_stride;
      HVX_Vector vacc0_0 = vzero;
      HVX_Vector vacc0_1 = vzero;
      HVX_Vector vacc1_0 = vzero;
      HVX_Vector vacc1_1 = vzero;
      HVX_Vector vacc2_0 = vzero;
      HVX_Vector vacc2_1 = vzero;
      HVX_Vector vacc3_0 = vzero;
      HVX_Vector vacc3_1 = vzero;

      size_t k = kc;
      for (; k >= 128; k -= 128) {
        {
          const HVX_Vector v5_0 = *((const HVX_UVector*) w0);
          w0 += 128;
          const HVX_Vector v5_1 = *((const HVX_UVector*) w1);
          w1 += 128;
          const HVX_Vector v5_2 = *((const HVX_UVector*) w2);
          w2 += 128;
          const HVX_Vector v5_3 = *((const HVX_UVector*) w3);
          w3 += 128;
          const HVX_Vector v5_4 = *((const HVX_UVector*) w4);
          w4 += 128;
          const HVX_Vector v5_5 = *((const HVX_UVector*) w5);
          w5 += 128;
          const HVX_Vector v5_6 = *((const HVX_UVector*) w6);
          w6 += 128;
          const HVX_Vector v5_7 = *((const HVX_UVector*) w7);
          w7 += 128;
          const HVX_Vector v5_8 = *((const HVX_UVector*) w8);
          w8 += 128;
          const HVX_Vector v5_9 = *((const HVX_UVector*) w9);
          w9 += 128;
          const HVX_Vector v5_10 = *((const HVX_UVector*) w10);
          w10 += 128;
          const HVX_Vector v5_11 = *((const HVX_UVector*) w11);
          w11 += 128;
          const HVX_Vector v5_12 = *((const HVX_UVector*) w12);
          w12 += 128;
          const HVX_Vector v5_13 = *((const HVX_UVector*) w13);
          w13 += 128;
          const HVX_Vector v5_14 = *((const HVX_UVector*) w14);
          w14 += 128;
          const HVX_Vector v5_15 = *((const HVX_UVector*) w15);
          w15 += 128;
          const HVX_Vector v5_16 = *((const HVX_UVector*) w16);
          w16 += 128;
          const HVX_Vector v5_17 = *((const HVX_UVector*) w17);
          w17 += 128;
          const HVX_Vector v5_18 = *((const HVX_UVector*) w18);
          w18 += 128;
          const HVX_Vector v5_19 = *((const HVX_UVector*) w19);
          w19 += 128;
          const HVX_Vector v5_20 = *((const HVX_UVector*) w20);
          w20 += 128;
          const HVX_Vector v5_21 = *((const HVX_UVector*) w21);
          w21 += 128;
          const HVX_Vector v5_22 = *((const HVX_UVector*) w22);
          w22 += 128;
          const HVX_Vector v5_23 = *((const HVX_UVector*) w23);
          w23 += 128;
          const HVX_Vector v5_24 = *((const HVX_UVector*) w24);
          w24 += 128;
          const HVX_Vector v5_25 = *((const HVX_UVector*) w25);
          w25 += 128;
          const HVX_Vector v5_26 = *((const HVX_UVector*) w26);
          w26 += 128;
          const HVX_Vector v5_27 = *((const HVX_UVector*) w27);
          w27 += 128;
          const HVX_Vector v5_28 = *((const HVX_UVector*) w28);
          w28 += 128;
          const HVX_Vector v5_29 = *((const HVX_UVector*) w29);
          w29 += 128;
          const HVX_Vector v5_30 = *((const HVX_UVector*) w30);
          w30 += 128;
          const HVX_Vector v5_31 = *((const HVX_UVector*) w31);
          w31 += 128;

          const HVX_VectorPair v4_0 =
              Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 =
              Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 =
              Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 =
              Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 =
              Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 =
              Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 =
              Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 =
              Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 =
              Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 =
              Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 =
              Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 =
              Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 =
              Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 =
              Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 =
              Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 =
              Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          const HVX_Vector vt_lo_0 = Q6_V_lo_W(v0_0);
          const HVX_Vector vt_hi_0 = Q6_V_hi_W(v0_0);
          *((HVX_UVector*) (out + 0)) =
              vt_lo_0;
          *((HVX_UVector*) (out + 512)) =
              vt_hi_0;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_0, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_0, vones);
          const HVX_Vector vt_lo_1 = Q6_V_lo_W(v0_1);
          const HVX_Vector vt_hi_1 = Q6_V_hi_W(v0_1);
          *((HVX_UVector*) (out + 1024)) =
              vt_lo_1;
          *((HVX_UVector*) (out + 1536)) =
              vt_hi_1;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_1, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_1, vones);
          const HVX_Vector vt_lo_2 = Q6_V_lo_W(v0_2);
          const HVX_Vector vt_hi_2 = Q6_V_hi_W(v0_2);
          *((HVX_UVector*) (out + 2048)) =
              vt_lo_2;
          *((HVX_UVector*) (out + 2560)) =
              vt_hi_2;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_2, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_2, vones);
          const HVX_Vector vt_lo_3 = Q6_V_lo_W(v0_3);
          const HVX_Vector vt_hi_3 = Q6_V_hi_W(v0_3);
          *((HVX_UVector*) (out + 3072)) =
              vt_lo_3;
          *((HVX_UVector*) (out + 3584)) =
              vt_hi_3;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_3, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_3, vones);
          const HVX_Vector vt_lo_4 = Q6_V_lo_W(v0_4);
          const HVX_Vector vt_hi_4 = Q6_V_hi_W(v0_4);
          *((HVX_UVector*) (out + 4096)) =
              vt_lo_4;
          *((HVX_UVector*) (out + 4608)) =
              vt_hi_4;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_4, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_4, vones);
          const HVX_Vector vt_lo_5 = Q6_V_lo_W(v0_5);
          const HVX_Vector vt_hi_5 = Q6_V_hi_W(v0_5);
          *((HVX_UVector*) (out + 5120)) =
              vt_lo_5;
          *((HVX_UVector*) (out + 5632)) =
              vt_hi_5;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_5, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_5, vones);
          const HVX_Vector vt_lo_6 = Q6_V_lo_W(v0_6);
          const HVX_Vector vt_hi_6 = Q6_V_hi_W(v0_6);
          *((HVX_UVector*) (out + 6144)) =
              vt_lo_6;
          *((HVX_UVector*) (out + 6656)) =
              vt_hi_6;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_6, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_6, vones);
          const HVX_Vector vt_lo_7 = Q6_V_lo_W(v0_7);
          const HVX_Vector vt_hi_7 = Q6_V_hi_W(v0_7);
          *((HVX_UVector*) (out + 7168)) =
              vt_lo_7;
          *((HVX_UVector*) (out + 7680)) =
              vt_hi_7;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_7, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_7, vones);
          const HVX_Vector vt_lo_8 = Q6_V_lo_W(v0_8);
          const HVX_Vector vt_hi_8 = Q6_V_hi_W(v0_8);
          *((HVX_UVector*) (out + 8192)) =
              vt_lo_8;
          *((HVX_UVector*) (out + 8704)) =
              vt_hi_8;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_8, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_8, vones);
          const HVX_Vector vt_lo_9 = Q6_V_lo_W(v0_9);
          const HVX_Vector vt_hi_9 = Q6_V_hi_W(v0_9);
          *((HVX_UVector*) (out + 9216)) =
              vt_lo_9;
          *((HVX_UVector*) (out + 9728)) =
              vt_hi_9;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_9, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_9, vones);
          const HVX_Vector vt_lo_10 = Q6_V_lo_W(v0_10);
          const HVX_Vector vt_hi_10 = Q6_V_hi_W(v0_10);
          *((HVX_UVector*) (out + 10240)) =
              vt_lo_10;
          *((HVX_UVector*) (out + 10752)) =
              vt_hi_10;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_10, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_10, vones);
          const HVX_Vector vt_lo_11 = Q6_V_lo_W(v0_11);
          const HVX_Vector vt_hi_11 = Q6_V_hi_W(v0_11);
          *((HVX_UVector*) (out + 11264)) =
              vt_lo_11;
          *((HVX_UVector*) (out + 11776)) =
              vt_hi_11;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_11, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_11, vones);
          const HVX_Vector vt_lo_12 = Q6_V_lo_W(v0_12);
          const HVX_Vector vt_hi_12 = Q6_V_hi_W(v0_12);
          *((HVX_UVector*) (out + 12288)) =
              vt_lo_12;
          *((HVX_UVector*) (out + 12800)) =
              vt_hi_12;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_12, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_12, vones);
          const HVX_Vector vt_lo_13 = Q6_V_lo_W(v0_13);
          const HVX_Vector vt_hi_13 = Q6_V_hi_W(v0_13);
          *((HVX_UVector*) (out + 13312)) =
              vt_lo_13;
          *((HVX_UVector*) (out + 13824)) =
              vt_hi_13;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_13, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_13, vones);
          const HVX_Vector vt_lo_14 = Q6_V_lo_W(v0_14);
          const HVX_Vector vt_hi_14 = Q6_V_hi_W(v0_14);
          *((HVX_UVector*) (out + 14336)) =
              vt_lo_14;
          *((HVX_UVector*) (out + 14848)) =
              vt_hi_14;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_14, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_14, vones);
          const HVX_Vector vt_lo_15 = Q6_V_lo_W(v0_15);
          const HVX_Vector vt_hi_15 = Q6_V_hi_W(v0_15);
          *((HVX_UVector*) (out + 15360)) =
              vt_lo_15;
          *((HVX_UVector*) (out + 15872)) =
              vt_hi_15;
          vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, vt_lo_15, vones);
          vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, vt_hi_15, vones);
        }
        {
          const HVX_Vector v5_0 = *((const HVX_UVector*) w32);
          w32 += 128;
          const HVX_Vector v5_1 = *((const HVX_UVector*) w33);
          w33 += 128;
          const HVX_Vector v5_2 = *((const HVX_UVector*) w34);
          w34 += 128;
          const HVX_Vector v5_3 = *((const HVX_UVector*) w35);
          w35 += 128;
          const HVX_Vector v5_4 = *((const HVX_UVector*) w36);
          w36 += 128;
          const HVX_Vector v5_5 = *((const HVX_UVector*) w37);
          w37 += 128;
          const HVX_Vector v5_6 = *((const HVX_UVector*) w38);
          w38 += 128;
          const HVX_Vector v5_7 = *((const HVX_UVector*) w39);
          w39 += 128;
          const HVX_Vector v5_8 = *((const HVX_UVector*) w40);
          w40 += 128;
          const HVX_Vector v5_9 = *((const HVX_UVector*) w41);
          w41 += 128;
          const HVX_Vector v5_10 = *((const HVX_UVector*) w42);
          w42 += 128;
          const HVX_Vector v5_11 = *((const HVX_UVector*) w43);
          w43 += 128;
          const HVX_Vector v5_12 = *((const HVX_UVector*) w44);
          w44 += 128;
          const HVX_Vector v5_13 = *((const HVX_UVector*) w45);
          w45 += 128;
          const HVX_Vector v5_14 = *((const HVX_UVector*) w46);
          w46 += 128;
          const HVX_Vector v5_15 = *((const HVX_UVector*) w47);
          w47 += 128;
          const HVX_Vector v5_16 = *((const HVX_UVector*) w48);
          w48 += 128;
          const HVX_Vector v5_17 = *((const HVX_UVector*) w49);
          w49 += 128;
          const HVX_Vector v5_18 = *((const HVX_UVector*) w50);
          w50 += 128;
          const HVX_Vector v5_19 = *((const HVX_UVector*) w51);
          w51 += 128;
          const HVX_Vector v5_20 = *((const HVX_UVector*) w52);
          w52 += 128;
          const HVX_Vector v5_21 = *((const HVX_UVector*) w53);
          w53 += 128;
          const HVX_Vector v5_22 = *((const HVX_UVector*) w54);
          w54 += 128;
          const HVX_Vector v5_23 = *((const HVX_UVector*) w55);
          w55 += 128;
          const HVX_Vector v5_24 = *((const HVX_UVector*) w56);
          w56 += 128;
          const HVX_Vector v5_25 = *((const HVX_UVector*) w57);
          w57 += 128;
          const HVX_Vector v5_26 = *((const HVX_UVector*) w58);
          w58 += 128;
          const HVX_Vector v5_27 = *((const HVX_UVector*) w59);
          w59 += 128;
          const HVX_Vector v5_28 = *((const HVX_UVector*) w60);
          w60 += 128;
          const HVX_Vector v5_29 = *((const HVX_UVector*) w61);
          w61 += 128;
          const HVX_Vector v5_30 = *((const HVX_UVector*) w62);
          w62 += 128;
          const HVX_Vector v5_31 = *((const HVX_UVector*) w63);
          w63 += 128;

          const HVX_VectorPair v4_0 =
              Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 =
              Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 =
              Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 =
              Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 =
              Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 =
              Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 =
              Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 =
              Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 =
              Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 =
              Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 =
              Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 =
              Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 =
              Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 =
              Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 =
              Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 =
              Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          const HVX_Vector vt_lo_0 = Q6_V_lo_W(v0_0);
          const HVX_Vector vt_hi_0 = Q6_V_hi_W(v0_0);
          *((HVX_UVector*) (out + 128)) =
              vt_lo_0;
          *((HVX_UVector*) (out + 640)) =
              vt_hi_0;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_0, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_0, vones);
          const HVX_Vector vt_lo_1 = Q6_V_lo_W(v0_1);
          const HVX_Vector vt_hi_1 = Q6_V_hi_W(v0_1);
          *((HVX_UVector*) (out + 1152)) =
              vt_lo_1;
          *((HVX_UVector*) (out + 1664)) =
              vt_hi_1;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_1, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_1, vones);
          const HVX_Vector vt_lo_2 = Q6_V_lo_W(v0_2);
          const HVX_Vector vt_hi_2 = Q6_V_hi_W(v0_2);
          *((HVX_UVector*) (out + 2176)) =
              vt_lo_2;
          *((HVX_UVector*) (out + 2688)) =
              vt_hi_2;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_2, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_2, vones);
          const HVX_Vector vt_lo_3 = Q6_V_lo_W(v0_3);
          const HVX_Vector vt_hi_3 = Q6_V_hi_W(v0_3);
          *((HVX_UVector*) (out + 3200)) =
              vt_lo_3;
          *((HVX_UVector*) (out + 3712)) =
              vt_hi_3;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_3, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_3, vones);
          const HVX_Vector vt_lo_4 = Q6_V_lo_W(v0_4);
          const HVX_Vector vt_hi_4 = Q6_V_hi_W(v0_4);
          *((HVX_UVector*) (out + 4224)) =
              vt_lo_4;
          *((HVX_UVector*) (out + 4736)) =
              vt_hi_4;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_4, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_4, vones);
          const HVX_Vector vt_lo_5 = Q6_V_lo_W(v0_5);
          const HVX_Vector vt_hi_5 = Q6_V_hi_W(v0_5);
          *((HVX_UVector*) (out + 5248)) =
              vt_lo_5;
          *((HVX_UVector*) (out + 5760)) =
              vt_hi_5;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_5, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_5, vones);
          const HVX_Vector vt_lo_6 = Q6_V_lo_W(v0_6);
          const HVX_Vector vt_hi_6 = Q6_V_hi_W(v0_6);
          *((HVX_UVector*) (out + 6272)) =
              vt_lo_6;
          *((HVX_UVector*) (out + 6784)) =
              vt_hi_6;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_6, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_6, vones);
          const HVX_Vector vt_lo_7 = Q6_V_lo_W(v0_7);
          const HVX_Vector vt_hi_7 = Q6_V_hi_W(v0_7);
          *((HVX_UVector*) (out + 7296)) =
              vt_lo_7;
          *((HVX_UVector*) (out + 7808)) =
              vt_hi_7;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_7, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_7, vones);
          const HVX_Vector vt_lo_8 = Q6_V_lo_W(v0_8);
          const HVX_Vector vt_hi_8 = Q6_V_hi_W(v0_8);
          *((HVX_UVector*) (out + 8320)) =
              vt_lo_8;
          *((HVX_UVector*) (out + 8832)) =
              vt_hi_8;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_8, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_8, vones);
          const HVX_Vector vt_lo_9 = Q6_V_lo_W(v0_9);
          const HVX_Vector vt_hi_9 = Q6_V_hi_W(v0_9);
          *((HVX_UVector*) (out + 9344)) =
              vt_lo_9;
          *((HVX_UVector*) (out + 9856)) =
              vt_hi_9;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_9, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_9, vones);
          const HVX_Vector vt_lo_10 = Q6_V_lo_W(v0_10);
          const HVX_Vector vt_hi_10 = Q6_V_hi_W(v0_10);
          *((HVX_UVector*) (out + 10368)) =
              vt_lo_10;
          *((HVX_UVector*) (out + 10880)) =
              vt_hi_10;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_10, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_10, vones);
          const HVX_Vector vt_lo_11 = Q6_V_lo_W(v0_11);
          const HVX_Vector vt_hi_11 = Q6_V_hi_W(v0_11);
          *((HVX_UVector*) (out + 11392)) =
              vt_lo_11;
          *((HVX_UVector*) (out + 11904)) =
              vt_hi_11;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_11, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_11, vones);
          const HVX_Vector vt_lo_12 = Q6_V_lo_W(v0_12);
          const HVX_Vector vt_hi_12 = Q6_V_hi_W(v0_12);
          *((HVX_UVector*) (out + 12416)) =
              vt_lo_12;
          *((HVX_UVector*) (out + 12928)) =
              vt_hi_12;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_12, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_12, vones);
          const HVX_Vector vt_lo_13 = Q6_V_lo_W(v0_13);
          const HVX_Vector vt_hi_13 = Q6_V_hi_W(v0_13);
          *((HVX_UVector*) (out + 13440)) =
              vt_lo_13;
          *((HVX_UVector*) (out + 13952)) =
              vt_hi_13;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_13, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_13, vones);
          const HVX_Vector vt_lo_14 = Q6_V_lo_W(v0_14);
          const HVX_Vector vt_hi_14 = Q6_V_hi_W(v0_14);
          *((HVX_UVector*) (out + 14464)) =
              vt_lo_14;
          *((HVX_UVector*) (out + 14976)) =
              vt_hi_14;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_14, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_14, vones);
          const HVX_Vector vt_lo_15 = Q6_V_lo_W(v0_15);
          const HVX_Vector vt_hi_15 = Q6_V_hi_W(v0_15);
          *((HVX_UVector*) (out + 15488)) =
              vt_lo_15;
          *((HVX_UVector*) (out + 16000)) =
              vt_hi_15;
          vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, vt_lo_15, vones);
          vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, vt_hi_15, vones);
        }
        {
          const HVX_Vector v5_0 = *((const HVX_UVector*) w64);
          w64 += 128;
          const HVX_Vector v5_1 = *((const HVX_UVector*) w65);
          w65 += 128;
          const HVX_Vector v5_2 = *((const HVX_UVector*) w66);
          w66 += 128;
          const HVX_Vector v5_3 = *((const HVX_UVector*) w67);
          w67 += 128;
          const HVX_Vector v5_4 = *((const HVX_UVector*) w68);
          w68 += 128;
          const HVX_Vector v5_5 = *((const HVX_UVector*) w69);
          w69 += 128;
          const HVX_Vector v5_6 = *((const HVX_UVector*) w70);
          w70 += 128;
          const HVX_Vector v5_7 = *((const HVX_UVector*) w71);
          w71 += 128;
          const HVX_Vector v5_8 = *((const HVX_UVector*) w72);
          w72 += 128;
          const HVX_Vector v5_9 = *((const HVX_UVector*) w73);
          w73 += 128;
          const HVX_Vector v5_10 = *((const HVX_UVector*) w74);
          w74 += 128;
          const HVX_Vector v5_11 = *((const HVX_UVector*) w75);
          w75 += 128;
          const HVX_Vector v5_12 = *((const HVX_UVector*) w76);
          w76 += 128;
          const HVX_Vector v5_13 = *((const HVX_UVector*) w77);
          w77 += 128;
          const HVX_Vector v5_14 = *((const HVX_UVector*) w78);
          w78 += 128;
          const HVX_Vector v5_15 = *((const HVX_UVector*) w79);
          w79 += 128;
          const HVX_Vector v5_16 = *((const HVX_UVector*) w80);
          w80 += 128;
          const HVX_Vector v5_17 = *((const HVX_UVector*) w81);
          w81 += 128;
          const HVX_Vector v5_18 = *((const HVX_UVector*) w82);
          w82 += 128;
          const HVX_Vector v5_19 = *((const HVX_UVector*) w83);
          w83 += 128;
          const HVX_Vector v5_20 = *((const HVX_UVector*) w84);
          w84 += 128;
          const HVX_Vector v5_21 = *((const HVX_UVector*) w85);
          w85 += 128;
          const HVX_Vector v5_22 = *((const HVX_UVector*) w86);
          w86 += 128;
          const HVX_Vector v5_23 = *((const HVX_UVector*) w87);
          w87 += 128;
          const HVX_Vector v5_24 = *((const HVX_UVector*) w88);
          w88 += 128;
          const HVX_Vector v5_25 = *((const HVX_UVector*) w89);
          w89 += 128;
          const HVX_Vector v5_26 = *((const HVX_UVector*) w90);
          w90 += 128;
          const HVX_Vector v5_27 = *((const HVX_UVector*) w91);
          w91 += 128;
          const HVX_Vector v5_28 = *((const HVX_UVector*) w92);
          w92 += 128;
          const HVX_Vector v5_29 = *((const HVX_UVector*) w93);
          w93 += 128;
          const HVX_Vector v5_30 = *((const HVX_UVector*) w94);
          w94 += 128;
          const HVX_Vector v5_31 = *((const HVX_UVector*) w95);
          w95 += 128;

          const HVX_VectorPair v4_0 =
              Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 =
              Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 =
              Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 =
              Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 =
              Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 =
              Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 =
              Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 =
              Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 =
              Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 =
              Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 =
              Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 =
              Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 =
              Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 =
              Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 =
              Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 =
              Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          const HVX_Vector vt_lo_0 = Q6_V_lo_W(v0_0);
          const HVX_Vector vt_hi_0 = Q6_V_hi_W(v0_0);
          *((HVX_UVector*) (out + 256)) =
              vt_lo_0;
          *((HVX_UVector*) (out + 768)) =
              vt_hi_0;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_0, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_0, vones);
          const HVX_Vector vt_lo_1 = Q6_V_lo_W(v0_1);
          const HVX_Vector vt_hi_1 = Q6_V_hi_W(v0_1);
          *((HVX_UVector*) (out + 1280)) =
              vt_lo_1;
          *((HVX_UVector*) (out + 1792)) =
              vt_hi_1;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_1, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_1, vones);
          const HVX_Vector vt_lo_2 = Q6_V_lo_W(v0_2);
          const HVX_Vector vt_hi_2 = Q6_V_hi_W(v0_2);
          *((HVX_UVector*) (out + 2304)) =
              vt_lo_2;
          *((HVX_UVector*) (out + 2816)) =
              vt_hi_2;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_2, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_2, vones);
          const HVX_Vector vt_lo_3 = Q6_V_lo_W(v0_3);
          const HVX_Vector vt_hi_3 = Q6_V_hi_W(v0_3);
          *((HVX_UVector*) (out + 3328)) =
              vt_lo_3;
          *((HVX_UVector*) (out + 3840)) =
              vt_hi_3;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_3, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_3, vones);
          const HVX_Vector vt_lo_4 = Q6_V_lo_W(v0_4);
          const HVX_Vector vt_hi_4 = Q6_V_hi_W(v0_4);
          *((HVX_UVector*) (out + 4352)) =
              vt_lo_4;
          *((HVX_UVector*) (out + 4864)) =
              vt_hi_4;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_4, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_4, vones);
          const HVX_Vector vt_lo_5 = Q6_V_lo_W(v0_5);
          const HVX_Vector vt_hi_5 = Q6_V_hi_W(v0_5);
          *((HVX_UVector*) (out + 5376)) =
              vt_lo_5;
          *((HVX_UVector*) (out + 5888)) =
              vt_hi_5;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_5, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_5, vones);
          const HVX_Vector vt_lo_6 = Q6_V_lo_W(v0_6);
          const HVX_Vector vt_hi_6 = Q6_V_hi_W(v0_6);
          *((HVX_UVector*) (out + 6400)) =
              vt_lo_6;
          *((HVX_UVector*) (out + 6912)) =
              vt_hi_6;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_6, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_6, vones);
          const HVX_Vector vt_lo_7 = Q6_V_lo_W(v0_7);
          const HVX_Vector vt_hi_7 = Q6_V_hi_W(v0_7);
          *((HVX_UVector*) (out + 7424)) =
              vt_lo_7;
          *((HVX_UVector*) (out + 7936)) =
              vt_hi_7;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_7, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_7, vones);
          const HVX_Vector vt_lo_8 = Q6_V_lo_W(v0_8);
          const HVX_Vector vt_hi_8 = Q6_V_hi_W(v0_8);
          *((HVX_UVector*) (out + 8448)) =
              vt_lo_8;
          *((HVX_UVector*) (out + 8960)) =
              vt_hi_8;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_8, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_8, vones);
          const HVX_Vector vt_lo_9 = Q6_V_lo_W(v0_9);
          const HVX_Vector vt_hi_9 = Q6_V_hi_W(v0_9);
          *((HVX_UVector*) (out + 9472)) =
              vt_lo_9;
          *((HVX_UVector*) (out + 9984)) =
              vt_hi_9;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_9, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_9, vones);
          const HVX_Vector vt_lo_10 = Q6_V_lo_W(v0_10);
          const HVX_Vector vt_hi_10 = Q6_V_hi_W(v0_10);
          *((HVX_UVector*) (out + 10496)) =
              vt_lo_10;
          *((HVX_UVector*) (out + 11008)) =
              vt_hi_10;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_10, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_10, vones);
          const HVX_Vector vt_lo_11 = Q6_V_lo_W(v0_11);
          const HVX_Vector vt_hi_11 = Q6_V_hi_W(v0_11);
          *((HVX_UVector*) (out + 11520)) =
              vt_lo_11;
          *((HVX_UVector*) (out + 12032)) =
              vt_hi_11;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_11, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_11, vones);
          const HVX_Vector vt_lo_12 = Q6_V_lo_W(v0_12);
          const HVX_Vector vt_hi_12 = Q6_V_hi_W(v0_12);
          *((HVX_UVector*) (out + 12544)) =
              vt_lo_12;
          *((HVX_UVector*) (out + 13056)) =
              vt_hi_12;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_12, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_12, vones);
          const HVX_Vector vt_lo_13 = Q6_V_lo_W(v0_13);
          const HVX_Vector vt_hi_13 = Q6_V_hi_W(v0_13);
          *((HVX_UVector*) (out + 13568)) =
              vt_lo_13;
          *((HVX_UVector*) (out + 14080)) =
              vt_hi_13;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_13, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_13, vones);
          const HVX_Vector vt_lo_14 = Q6_V_lo_W(v0_14);
          const HVX_Vector vt_hi_14 = Q6_V_hi_W(v0_14);
          *((HVX_UVector*) (out + 14592)) =
              vt_lo_14;
          *((HVX_UVector*) (out + 15104)) =
              vt_hi_14;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_14, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_14, vones);
          const HVX_Vector vt_lo_15 = Q6_V_lo_W(v0_15);
          const HVX_Vector vt_hi_15 = Q6_V_hi_W(v0_15);
          *((HVX_UVector*) (out + 15616)) =
              vt_lo_15;
          *((HVX_UVector*) (out + 16128)) =
              vt_hi_15;
          vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, vt_lo_15, vones);
          vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, vt_hi_15, vones);
        }
        {
          const HVX_Vector v5_0 = *((const HVX_UVector*) w96);
          w96 += 128;
          const HVX_Vector v5_1 = *((const HVX_UVector*) w97);
          w97 += 128;
          const HVX_Vector v5_2 = *((const HVX_UVector*) w98);
          w98 += 128;
          const HVX_Vector v5_3 = *((const HVX_UVector*) w99);
          w99 += 128;
          const HVX_Vector v5_4 = *((const HVX_UVector*) w100);
          w100 += 128;
          const HVX_Vector v5_5 = *((const HVX_UVector*) w101);
          w101 += 128;
          const HVX_Vector v5_6 = *((const HVX_UVector*) w102);
          w102 += 128;
          const HVX_Vector v5_7 = *((const HVX_UVector*) w103);
          w103 += 128;
          const HVX_Vector v5_8 = *((const HVX_UVector*) w104);
          w104 += 128;
          const HVX_Vector v5_9 = *((const HVX_UVector*) w105);
          w105 += 128;
          const HVX_Vector v5_10 = *((const HVX_UVector*) w106);
          w106 += 128;
          const HVX_Vector v5_11 = *((const HVX_UVector*) w107);
          w107 += 128;
          const HVX_Vector v5_12 = *((const HVX_UVector*) w108);
          w108 += 128;
          const HVX_Vector v5_13 = *((const HVX_UVector*) w109);
          w109 += 128;
          const HVX_Vector v5_14 = *((const HVX_UVector*) w110);
          w110 += 128;
          const HVX_Vector v5_15 = *((const HVX_UVector*) w111);
          w111 += 128;
          const HVX_Vector v5_16 = *((const HVX_UVector*) w112);
          w112 += 128;
          const HVX_Vector v5_17 = *((const HVX_UVector*) w113);
          w113 += 128;
          const HVX_Vector v5_18 = *((const HVX_UVector*) w114);
          w114 += 128;
          const HVX_Vector v5_19 = *((const HVX_UVector*) w115);
          w115 += 128;
          const HVX_Vector v5_20 = *((const HVX_UVector*) w116);
          w116 += 128;
          const HVX_Vector v5_21 = *((const HVX_UVector*) w117);
          w117 += 128;
          const HVX_Vector v5_22 = *((const HVX_UVector*) w118);
          w118 += 128;
          const HVX_Vector v5_23 = *((const HVX_UVector*) w119);
          w119 += 128;
          const HVX_Vector v5_24 = *((const HVX_UVector*) w120);
          w120 += 128;
          const HVX_Vector v5_25 = *((const HVX_UVector*) w121);
          w121 += 128;
          const HVX_Vector v5_26 = *((const HVX_UVector*) w122);
          w122 += 128;
          const HVX_Vector v5_27 = *((const HVX_UVector*) w123);
          w123 += 128;
          const HVX_Vector v5_28 = *((const HVX_UVector*) w124);
          w124 += 128;
          const HVX_Vector v5_29 = *((const HVX_UVector*) w125);
          w125 += 128;
          const HVX_Vector v5_30 = *((const HVX_UVector*) w126);
          w126 += 128;
          const HVX_Vector v5_31 = *((const HVX_UVector*) w127);
          w127 += 128;

          const HVX_VectorPair v4_0 =
              Q6_W_vshuff_VVR(v5_1, v5_0, -4);
          const HVX_VectorPair v4_1 =
              Q6_W_vshuff_VVR(v5_3, v5_2, -4);
          const HVX_VectorPair v4_2 =
              Q6_W_vshuff_VVR(v5_5, v5_4, -4);
          const HVX_VectorPair v4_3 =
              Q6_W_vshuff_VVR(v5_7, v5_6, -4);
          const HVX_VectorPair v4_4 =
              Q6_W_vshuff_VVR(v5_9, v5_8, -4);
          const HVX_VectorPair v4_5 =
              Q6_W_vshuff_VVR(v5_11, v5_10, -4);
          const HVX_VectorPair v4_6 =
              Q6_W_vshuff_VVR(v5_13, v5_12, -4);
          const HVX_VectorPair v4_7 =
              Q6_W_vshuff_VVR(v5_15, v5_14, -4);
          const HVX_VectorPair v4_8 =
              Q6_W_vshuff_VVR(v5_17, v5_16, -4);
          const HVX_VectorPair v4_9 =
              Q6_W_vshuff_VVR(v5_19, v5_18, -4);
          const HVX_VectorPair v4_10 =
              Q6_W_vshuff_VVR(v5_21, v5_20, -4);
          const HVX_VectorPair v4_11 =
              Q6_W_vshuff_VVR(v5_23, v5_22, -4);
          const HVX_VectorPair v4_12 =
              Q6_W_vshuff_VVR(v5_25, v5_24, -4);
          const HVX_VectorPair v4_13 =
              Q6_W_vshuff_VVR(v5_27, v5_26, -4);
          const HVX_VectorPair v4_14 =
              Q6_W_vshuff_VVR(v5_29, v5_28, -4);
          const HVX_VectorPair v4_15 =
              Q6_W_vshuff_VVR(v5_31, v5_30, -4);

          const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
          const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
          const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
          const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
          const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
          const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
          const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
          const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
          const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
          const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
          const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
          const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
          const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
          const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
          const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
          const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

          const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
          const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
          const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
          const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
          const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
          const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
          const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
          const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
          const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
          const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
          const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
          const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
          const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
          const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
          const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
          const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

          const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
          const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
          const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
          const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
          const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
          const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
          const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
          const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
          const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
          const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
          const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
          const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
          const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
          const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
          const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
          const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

          const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
          const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
          const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
          const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
          const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
          const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
          const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
          const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
          const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
          const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
          const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
          const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
          const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
          const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
          const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(
              Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
          const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(
              Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

          const HVX_Vector vt_lo_0 = Q6_V_lo_W(v0_0);
          const HVX_Vector vt_hi_0 = Q6_V_hi_W(v0_0);
          *((HVX_UVector*) (out + 384)) =
              vt_lo_0;
          *((HVX_UVector*) (out + 896)) =
              vt_hi_0;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_0, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_0, vones);
          const HVX_Vector vt_lo_1 = Q6_V_lo_W(v0_1);
          const HVX_Vector vt_hi_1 = Q6_V_hi_W(v0_1);
          *((HVX_UVector*) (out + 1408)) =
              vt_lo_1;
          *((HVX_UVector*) (out + 1920)) =
              vt_hi_1;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_1, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_1, vones);
          const HVX_Vector vt_lo_2 = Q6_V_lo_W(v0_2);
          const HVX_Vector vt_hi_2 = Q6_V_hi_W(v0_2);
          *((HVX_UVector*) (out + 2432)) =
              vt_lo_2;
          *((HVX_UVector*) (out + 2944)) =
              vt_hi_2;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_2, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_2, vones);
          const HVX_Vector vt_lo_3 = Q6_V_lo_W(v0_3);
          const HVX_Vector vt_hi_3 = Q6_V_hi_W(v0_3);
          *((HVX_UVector*) (out + 3456)) =
              vt_lo_3;
          *((HVX_UVector*) (out + 3968)) =
              vt_hi_3;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_3, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_3, vones);
          const HVX_Vector vt_lo_4 = Q6_V_lo_W(v0_4);
          const HVX_Vector vt_hi_4 = Q6_V_hi_W(v0_4);
          *((HVX_UVector*) (out + 4480)) =
              vt_lo_4;
          *((HVX_UVector*) (out + 4992)) =
              vt_hi_4;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_4, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_4, vones);
          const HVX_Vector vt_lo_5 = Q6_V_lo_W(v0_5);
          const HVX_Vector vt_hi_5 = Q6_V_hi_W(v0_5);
          *((HVX_UVector*) (out + 5504)) =
              vt_lo_5;
          *((HVX_UVector*) (out + 6016)) =
              vt_hi_5;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_5, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_5, vones);
          const HVX_Vector vt_lo_6 = Q6_V_lo_W(v0_6);
          const HVX_Vector vt_hi_6 = Q6_V_hi_W(v0_6);
          *((HVX_UVector*) (out + 6528)) =
              vt_lo_6;
          *((HVX_UVector*) (out + 7040)) =
              vt_hi_6;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_6, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_6, vones);
          const HVX_Vector vt_lo_7 = Q6_V_lo_W(v0_7);
          const HVX_Vector vt_hi_7 = Q6_V_hi_W(v0_7);
          *((HVX_UVector*) (out + 7552)) =
              vt_lo_7;
          *((HVX_UVector*) (out + 8064)) =
              vt_hi_7;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_7, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_7, vones);
          const HVX_Vector vt_lo_8 = Q6_V_lo_W(v0_8);
          const HVX_Vector vt_hi_8 = Q6_V_hi_W(v0_8);
          *((HVX_UVector*) (out + 8576)) =
              vt_lo_8;
          *((HVX_UVector*) (out + 9088)) =
              vt_hi_8;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_8, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_8, vones);
          const HVX_Vector vt_lo_9 = Q6_V_lo_W(v0_9);
          const HVX_Vector vt_hi_9 = Q6_V_hi_W(v0_9);
          *((HVX_UVector*) (out + 9600)) =
              vt_lo_9;
          *((HVX_UVector*) (out + 10112)) =
              vt_hi_9;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_9, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_9, vones);
          const HVX_Vector vt_lo_10 = Q6_V_lo_W(v0_10);
          const HVX_Vector vt_hi_10 = Q6_V_hi_W(v0_10);
          *((HVX_UVector*) (out + 10624)) =
              vt_lo_10;
          *((HVX_UVector*) (out + 11136)) =
              vt_hi_10;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_10, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_10, vones);
          const HVX_Vector vt_lo_11 = Q6_V_lo_W(v0_11);
          const HVX_Vector vt_hi_11 = Q6_V_hi_W(v0_11);
          *((HVX_UVector*) (out + 11648)) =
              vt_lo_11;
          *((HVX_UVector*) (out + 12160)) =
              vt_hi_11;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_11, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_11, vones);
          const HVX_Vector vt_lo_12 = Q6_V_lo_W(v0_12);
          const HVX_Vector vt_hi_12 = Q6_V_hi_W(v0_12);
          *((HVX_UVector*) (out + 12672)) =
              vt_lo_12;
          *((HVX_UVector*) (out + 13184)) =
              vt_hi_12;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_12, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_12, vones);
          const HVX_Vector vt_lo_13 = Q6_V_lo_W(v0_13);
          const HVX_Vector vt_hi_13 = Q6_V_hi_W(v0_13);
          *((HVX_UVector*) (out + 13696)) =
              vt_lo_13;
          *((HVX_UVector*) (out + 14208)) =
              vt_hi_13;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_13, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_13, vones);
          const HVX_Vector vt_lo_14 = Q6_V_lo_W(v0_14);
          const HVX_Vector vt_hi_14 = Q6_V_hi_W(v0_14);
          *((HVX_UVector*) (out + 14720)) =
              vt_lo_14;
          *((HVX_UVector*) (out + 15232)) =
              vt_hi_14;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_14, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_14, vones);
          const HVX_Vector vt_lo_15 = Q6_V_lo_W(v0_15);
          const HVX_Vector vt_hi_15 = Q6_V_hi_W(v0_15);
          *((HVX_UVector*) (out + 15744)) =
              vt_lo_15;
          *((HVX_UVector*) (out + 16256)) =
              vt_hi_15;
          vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, vt_lo_15, vones);
          vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, vt_hi_15, vones);
        }
        out += 16384;
      }

      for (; k >= 4; k -= 4) {
        unaligned_store_u32(out + 0, unaligned_load_u32(w0));
        w0 += 4;
        unaligned_store_u32(out + 4, unaligned_load_u32(w1));
        w1 += 4;
        unaligned_store_u32(out + 8, unaligned_load_u32(w2));
        w2 += 4;
        unaligned_store_u32(out + 12, unaligned_load_u32(w3));
        w3 += 4;
        unaligned_store_u32(out + 16, unaligned_load_u32(w4));
        w4 += 4;
        unaligned_store_u32(out + 20, unaligned_load_u32(w5));
        w5 += 4;
        unaligned_store_u32(out + 24, unaligned_load_u32(w6));
        w6 += 4;
        unaligned_store_u32(out + 28, unaligned_load_u32(w7));
        w7 += 4;
        unaligned_store_u32(out + 32, unaligned_load_u32(w8));
        w8 += 4;
        unaligned_store_u32(out + 36, unaligned_load_u32(w9));
        w9 += 4;
        unaligned_store_u32(out + 40, unaligned_load_u32(w10));
        w10 += 4;
        unaligned_store_u32(out + 44, unaligned_load_u32(w11));
        w11 += 4;
        unaligned_store_u32(out + 48, unaligned_load_u32(w12));
        w12 += 4;
        unaligned_store_u32(out + 52, unaligned_load_u32(w13));
        w13 += 4;
        unaligned_store_u32(out + 56, unaligned_load_u32(w14));
        w14 += 4;
        unaligned_store_u32(out + 60, unaligned_load_u32(w15));
        w15 += 4;
        unaligned_store_u32(out + 64, unaligned_load_u32(w16));
        w16 += 4;
        unaligned_store_u32(out + 68, unaligned_load_u32(w17));
        w17 += 4;
        unaligned_store_u32(out + 72, unaligned_load_u32(w18));
        w18 += 4;
        unaligned_store_u32(out + 76, unaligned_load_u32(w19));
        w19 += 4;
        unaligned_store_u32(out + 80, unaligned_load_u32(w20));
        w20 += 4;
        unaligned_store_u32(out + 84, unaligned_load_u32(w21));
        w21 += 4;
        unaligned_store_u32(out + 88, unaligned_load_u32(w22));
        w22 += 4;
        unaligned_store_u32(out + 92, unaligned_load_u32(w23));
        w23 += 4;
        unaligned_store_u32(out + 96, unaligned_load_u32(w24));
        w24 += 4;
        unaligned_store_u32(out + 100, unaligned_load_u32(w25));
        w25 += 4;
        unaligned_store_u32(out + 104, unaligned_load_u32(w26));
        w26 += 4;
        unaligned_store_u32(out + 108, unaligned_load_u32(w27));
        w27 += 4;
        unaligned_store_u32(out + 112, unaligned_load_u32(w28));
        w28 += 4;
        unaligned_store_u32(out + 116, unaligned_load_u32(w29));
        w29 += 4;
        unaligned_store_u32(out + 120, unaligned_load_u32(w30));
        w30 += 4;
        unaligned_store_u32(out + 124, unaligned_load_u32(w31));
        w31 += 4;
        unaligned_store_u32(out + 128, unaligned_load_u32(w32));
        w32 += 4;
        unaligned_store_u32(out + 132, unaligned_load_u32(w33));
        w33 += 4;
        unaligned_store_u32(out + 136, unaligned_load_u32(w34));
        w34 += 4;
        unaligned_store_u32(out + 140, unaligned_load_u32(w35));
        w35 += 4;
        unaligned_store_u32(out + 144, unaligned_load_u32(w36));
        w36 += 4;
        unaligned_store_u32(out + 148, unaligned_load_u32(w37));
        w37 += 4;
        unaligned_store_u32(out + 152, unaligned_load_u32(w38));
        w38 += 4;
        unaligned_store_u32(out + 156, unaligned_load_u32(w39));
        w39 += 4;
        unaligned_store_u32(out + 160, unaligned_load_u32(w40));
        w40 += 4;
        unaligned_store_u32(out + 164, unaligned_load_u32(w41));
        w41 += 4;
        unaligned_store_u32(out + 168, unaligned_load_u32(w42));
        w42 += 4;
        unaligned_store_u32(out + 172, unaligned_load_u32(w43));
        w43 += 4;
        unaligned_store_u32(out + 176, unaligned_load_u32(w44));
        w44 += 4;
        unaligned_store_u32(out + 180, unaligned_load_u32(w45));
        w45 += 4;
        unaligned_store_u32(out + 184, unaligned_load_u32(w46));
        w46 += 4;
        unaligned_store_u32(out + 188, unaligned_load_u32(w47));
        w47 += 4;
        unaligned_store_u32(out + 192, unaligned_load_u32(w48));
        w48 += 4;
        unaligned_store_u32(out + 196, unaligned_load_u32(w49));
        w49 += 4;
        unaligned_store_u32(out + 200, unaligned_load_u32(w50));
        w50 += 4;
        unaligned_store_u32(out + 204, unaligned_load_u32(w51));
        w51 += 4;
        unaligned_store_u32(out + 208, unaligned_load_u32(w52));
        w52 += 4;
        unaligned_store_u32(out + 212, unaligned_load_u32(w53));
        w53 += 4;
        unaligned_store_u32(out + 216, unaligned_load_u32(w54));
        w54 += 4;
        unaligned_store_u32(out + 220, unaligned_load_u32(w55));
        w55 += 4;
        unaligned_store_u32(out + 224, unaligned_load_u32(w56));
        w56 += 4;
        unaligned_store_u32(out + 228, unaligned_load_u32(w57));
        w57 += 4;
        unaligned_store_u32(out + 232, unaligned_load_u32(w58));
        w58 += 4;
        unaligned_store_u32(out + 236, unaligned_load_u32(w59));
        w59 += 4;
        unaligned_store_u32(out + 240, unaligned_load_u32(w60));
        w60 += 4;
        unaligned_store_u32(out + 244, unaligned_load_u32(w61));
        w61 += 4;
        unaligned_store_u32(out + 248, unaligned_load_u32(w62));
        w62 += 4;
        unaligned_store_u32(out + 252, unaligned_load_u32(w63));
        w63 += 4;
        unaligned_store_u32(out + 256, unaligned_load_u32(w64));
        w64 += 4;
        unaligned_store_u32(out + 260, unaligned_load_u32(w65));
        w65 += 4;
        unaligned_store_u32(out + 264, unaligned_load_u32(w66));
        w66 += 4;
        unaligned_store_u32(out + 268, unaligned_load_u32(w67));
        w67 += 4;
        unaligned_store_u32(out + 272, unaligned_load_u32(w68));
        w68 += 4;
        unaligned_store_u32(out + 276, unaligned_load_u32(w69));
        w69 += 4;
        unaligned_store_u32(out + 280, unaligned_load_u32(w70));
        w70 += 4;
        unaligned_store_u32(out + 284, unaligned_load_u32(w71));
        w71 += 4;
        unaligned_store_u32(out + 288, unaligned_load_u32(w72));
        w72 += 4;
        unaligned_store_u32(out + 292, unaligned_load_u32(w73));
        w73 += 4;
        unaligned_store_u32(out + 296, unaligned_load_u32(w74));
        w74 += 4;
        unaligned_store_u32(out + 300, unaligned_load_u32(w75));
        w75 += 4;
        unaligned_store_u32(out + 304, unaligned_load_u32(w76));
        w76 += 4;
        unaligned_store_u32(out + 308, unaligned_load_u32(w77));
        w77 += 4;
        unaligned_store_u32(out + 312, unaligned_load_u32(w78));
        w78 += 4;
        unaligned_store_u32(out + 316, unaligned_load_u32(w79));
        w79 += 4;
        unaligned_store_u32(out + 320, unaligned_load_u32(w80));
        w80 += 4;
        unaligned_store_u32(out + 324, unaligned_load_u32(w81));
        w81 += 4;
        unaligned_store_u32(out + 328, unaligned_load_u32(w82));
        w82 += 4;
        unaligned_store_u32(out + 332, unaligned_load_u32(w83));
        w83 += 4;
        unaligned_store_u32(out + 336, unaligned_load_u32(w84));
        w84 += 4;
        unaligned_store_u32(out + 340, unaligned_load_u32(w85));
        w85 += 4;
        unaligned_store_u32(out + 344, unaligned_load_u32(w86));
        w86 += 4;
        unaligned_store_u32(out + 348, unaligned_load_u32(w87));
        w87 += 4;
        unaligned_store_u32(out + 352, unaligned_load_u32(w88));
        w88 += 4;
        unaligned_store_u32(out + 356, unaligned_load_u32(w89));
        w89 += 4;
        unaligned_store_u32(out + 360, unaligned_load_u32(w90));
        w90 += 4;
        unaligned_store_u32(out + 364, unaligned_load_u32(w91));
        w91 += 4;
        unaligned_store_u32(out + 368, unaligned_load_u32(w92));
        w92 += 4;
        unaligned_store_u32(out + 372, unaligned_load_u32(w93));
        w93 += 4;
        unaligned_store_u32(out + 376, unaligned_load_u32(w94));
        w94 += 4;
        unaligned_store_u32(out + 380, unaligned_load_u32(w95));
        w95 += 4;
        unaligned_store_u32(out + 384, unaligned_load_u32(w96));
        w96 += 4;
        unaligned_store_u32(out + 388, unaligned_load_u32(w97));
        w97 += 4;
        unaligned_store_u32(out + 392, unaligned_load_u32(w98));
        w98 += 4;
        unaligned_store_u32(out + 396, unaligned_load_u32(w99));
        w99 += 4;
        unaligned_store_u32(out + 400, unaligned_load_u32(w100));
        w100 += 4;
        unaligned_store_u32(out + 404, unaligned_load_u32(w101));
        w101 += 4;
        unaligned_store_u32(out + 408, unaligned_load_u32(w102));
        w102 += 4;
        unaligned_store_u32(out + 412, unaligned_load_u32(w103));
        w103 += 4;
        unaligned_store_u32(out + 416, unaligned_load_u32(w104));
        w104 += 4;
        unaligned_store_u32(out + 420, unaligned_load_u32(w105));
        w105 += 4;
        unaligned_store_u32(out + 424, unaligned_load_u32(w106));
        w106 += 4;
        unaligned_store_u32(out + 428, unaligned_load_u32(w107));
        w107 += 4;
        unaligned_store_u32(out + 432, unaligned_load_u32(w108));
        w108 += 4;
        unaligned_store_u32(out + 436, unaligned_load_u32(w109));
        w109 += 4;
        unaligned_store_u32(out + 440, unaligned_load_u32(w110));
        w110 += 4;
        unaligned_store_u32(out + 444, unaligned_load_u32(w111));
        w111 += 4;
        unaligned_store_u32(out + 448, unaligned_load_u32(w112));
        w112 += 4;
        unaligned_store_u32(out + 452, unaligned_load_u32(w113));
        w113 += 4;
        unaligned_store_u32(out + 456, unaligned_load_u32(w114));
        w114 += 4;
        unaligned_store_u32(out + 460, unaligned_load_u32(w115));
        w115 += 4;
        unaligned_store_u32(out + 464, unaligned_load_u32(w116));
        w116 += 4;
        unaligned_store_u32(out + 468, unaligned_load_u32(w117));
        w117 += 4;
        unaligned_store_u32(out + 472, unaligned_load_u32(w118));
        w118 += 4;
        unaligned_store_u32(out + 476, unaligned_load_u32(w119));
        w119 += 4;
        unaligned_store_u32(out + 480, unaligned_load_u32(w120));
        w120 += 4;
        unaligned_store_u32(out + 484, unaligned_load_u32(w121));
        w121 += 4;
        unaligned_store_u32(out + 488, unaligned_load_u32(w122));
        w122 += 4;
        unaligned_store_u32(out + 492, unaligned_load_u32(w123));
        w123 += 4;
        unaligned_store_u32(out + 496, unaligned_load_u32(w124));
        w124 += 4;
        unaligned_store_u32(out + 500, unaligned_load_u32(w125));
        w125 += 4;
        unaligned_store_u32(out + 504, unaligned_load_u32(w126));
        w126 += 4;
        unaligned_store_u32(out + 508, unaligned_load_u32(w127));
        w127 += 4;
        vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc0_0, *((const HVX_UVector*) (out + 0)), vones);
        vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc1_0, *((const HVX_UVector*) (out + 128)), vones);
        vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc2_0, *((const HVX_UVector*) (out + 256)), vones);
        vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc3_0, *((const HVX_UVector*) (out + 384)), vones);
        out += 512;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        {
          const uint32_t b0 = (uint8_t) w0[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w0[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w0[2] : 0;
          unaligned_store_u32(out + 0, b0 | (b1 << 8) | (b2 << 16));
          w0 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w1[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w1[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w1[2] : 0;
          unaligned_store_u32(out + 4, b0 | (b1 << 8) | (b2 << 16));
          w1 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w2[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w2[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w2[2] : 0;
          unaligned_store_u32(out + 8, b0 | (b1 << 8) | (b2 << 16));
          w2 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w3[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w3[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w3[2] : 0;
          unaligned_store_u32(out + 12, b0 | (b1 << 8) | (b2 << 16));
          w3 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w4[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w4[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w4[2] : 0;
          unaligned_store_u32(out + 16, b0 | (b1 << 8) | (b2 << 16));
          w4 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w5[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w5[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w5[2] : 0;
          unaligned_store_u32(out + 20, b0 | (b1 << 8) | (b2 << 16));
          w5 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w6[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w6[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w6[2] : 0;
          unaligned_store_u32(out + 24, b0 | (b1 << 8) | (b2 << 16));
          w6 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w7[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w7[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w7[2] : 0;
          unaligned_store_u32(out + 28, b0 | (b1 << 8) | (b2 << 16));
          w7 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w8[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w8[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w8[2] : 0;
          unaligned_store_u32(out + 32, b0 | (b1 << 8) | (b2 << 16));
          w8 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w9[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w9[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w9[2] : 0;
          unaligned_store_u32(out + 36, b0 | (b1 << 8) | (b2 << 16));
          w9 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w10[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w10[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w10[2] : 0;
          unaligned_store_u32(out + 40, b0 | (b1 << 8) | (b2 << 16));
          w10 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w11[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w11[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w11[2] : 0;
          unaligned_store_u32(out + 44, b0 | (b1 << 8) | (b2 << 16));
          w11 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w12[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w12[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w12[2] : 0;
          unaligned_store_u32(out + 48, b0 | (b1 << 8) | (b2 << 16));
          w12 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w13[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w13[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w13[2] : 0;
          unaligned_store_u32(out + 52, b0 | (b1 << 8) | (b2 << 16));
          w13 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w14[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w14[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w14[2] : 0;
          unaligned_store_u32(out + 56, b0 | (b1 << 8) | (b2 << 16));
          w14 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w15[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w15[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w15[2] : 0;
          unaligned_store_u32(out + 60, b0 | (b1 << 8) | (b2 << 16));
          w15 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w16[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w16[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w16[2] : 0;
          unaligned_store_u32(out + 64, b0 | (b1 << 8) | (b2 << 16));
          w16 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w17[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w17[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w17[2] : 0;
          unaligned_store_u32(out + 68, b0 | (b1 << 8) | (b2 << 16));
          w17 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w18[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w18[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w18[2] : 0;
          unaligned_store_u32(out + 72, b0 | (b1 << 8) | (b2 << 16));
          w18 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w19[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w19[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w19[2] : 0;
          unaligned_store_u32(out + 76, b0 | (b1 << 8) | (b2 << 16));
          w19 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w20[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w20[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w20[2] : 0;
          unaligned_store_u32(out + 80, b0 | (b1 << 8) | (b2 << 16));
          w20 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w21[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w21[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w21[2] : 0;
          unaligned_store_u32(out + 84, b0 | (b1 << 8) | (b2 << 16));
          w21 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w22[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w22[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w22[2] : 0;
          unaligned_store_u32(out + 88, b0 | (b1 << 8) | (b2 << 16));
          w22 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w23[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w23[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w23[2] : 0;
          unaligned_store_u32(out + 92, b0 | (b1 << 8) | (b2 << 16));
          w23 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w24[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w24[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w24[2] : 0;
          unaligned_store_u32(out + 96, b0 | (b1 << 8) | (b2 << 16));
          w24 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w25[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w25[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w25[2] : 0;
          unaligned_store_u32(out + 100, b0 | (b1 << 8) | (b2 << 16));
          w25 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w26[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w26[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w26[2] : 0;
          unaligned_store_u32(out + 104, b0 | (b1 << 8) | (b2 << 16));
          w26 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w27[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w27[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w27[2] : 0;
          unaligned_store_u32(out + 108, b0 | (b1 << 8) | (b2 << 16));
          w27 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w28[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w28[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w28[2] : 0;
          unaligned_store_u32(out + 112, b0 | (b1 << 8) | (b2 << 16));
          w28 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w29[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w29[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w29[2] : 0;
          unaligned_store_u32(out + 116, b0 | (b1 << 8) | (b2 << 16));
          w29 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w30[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w30[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w30[2] : 0;
          unaligned_store_u32(out + 120, b0 | (b1 << 8) | (b2 << 16));
          w30 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w31[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w31[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w31[2] : 0;
          unaligned_store_u32(out + 124, b0 | (b1 << 8) | (b2 << 16));
          w31 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w32[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w32[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w32[2] : 0;
          unaligned_store_u32(out + 128, b0 | (b1 << 8) | (b2 << 16));
          w32 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w33[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w33[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w33[2] : 0;
          unaligned_store_u32(out + 132, b0 | (b1 << 8) | (b2 << 16));
          w33 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w34[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w34[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w34[2] : 0;
          unaligned_store_u32(out + 136, b0 | (b1 << 8) | (b2 << 16));
          w34 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w35[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w35[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w35[2] : 0;
          unaligned_store_u32(out + 140, b0 | (b1 << 8) | (b2 << 16));
          w35 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w36[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w36[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w36[2] : 0;
          unaligned_store_u32(out + 144, b0 | (b1 << 8) | (b2 << 16));
          w36 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w37[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w37[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w37[2] : 0;
          unaligned_store_u32(out + 148, b0 | (b1 << 8) | (b2 << 16));
          w37 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w38[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w38[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w38[2] : 0;
          unaligned_store_u32(out + 152, b0 | (b1 << 8) | (b2 << 16));
          w38 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w39[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w39[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w39[2] : 0;
          unaligned_store_u32(out + 156, b0 | (b1 << 8) | (b2 << 16));
          w39 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w40[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w40[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w40[2] : 0;
          unaligned_store_u32(out + 160, b0 | (b1 << 8) | (b2 << 16));
          w40 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w41[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w41[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w41[2] : 0;
          unaligned_store_u32(out + 164, b0 | (b1 << 8) | (b2 << 16));
          w41 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w42[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w42[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w42[2] : 0;
          unaligned_store_u32(out + 168, b0 | (b1 << 8) | (b2 << 16));
          w42 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w43[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w43[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w43[2] : 0;
          unaligned_store_u32(out + 172, b0 | (b1 << 8) | (b2 << 16));
          w43 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w44[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w44[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w44[2] : 0;
          unaligned_store_u32(out + 176, b0 | (b1 << 8) | (b2 << 16));
          w44 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w45[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w45[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w45[2] : 0;
          unaligned_store_u32(out + 180, b0 | (b1 << 8) | (b2 << 16));
          w45 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w46[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w46[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w46[2] : 0;
          unaligned_store_u32(out + 184, b0 | (b1 << 8) | (b2 << 16));
          w46 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w47[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w47[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w47[2] : 0;
          unaligned_store_u32(out + 188, b0 | (b1 << 8) | (b2 << 16));
          w47 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w48[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w48[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w48[2] : 0;
          unaligned_store_u32(out + 192, b0 | (b1 << 8) | (b2 << 16));
          w48 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w49[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w49[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w49[2] : 0;
          unaligned_store_u32(out + 196, b0 | (b1 << 8) | (b2 << 16));
          w49 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w50[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w50[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w50[2] : 0;
          unaligned_store_u32(out + 200, b0 | (b1 << 8) | (b2 << 16));
          w50 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w51[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w51[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w51[2] : 0;
          unaligned_store_u32(out + 204, b0 | (b1 << 8) | (b2 << 16));
          w51 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w52[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w52[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w52[2] : 0;
          unaligned_store_u32(out + 208, b0 | (b1 << 8) | (b2 << 16));
          w52 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w53[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w53[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w53[2] : 0;
          unaligned_store_u32(out + 212, b0 | (b1 << 8) | (b2 << 16));
          w53 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w54[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w54[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w54[2] : 0;
          unaligned_store_u32(out + 216, b0 | (b1 << 8) | (b2 << 16));
          w54 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w55[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w55[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w55[2] : 0;
          unaligned_store_u32(out + 220, b0 | (b1 << 8) | (b2 << 16));
          w55 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w56[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w56[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w56[2] : 0;
          unaligned_store_u32(out + 224, b0 | (b1 << 8) | (b2 << 16));
          w56 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w57[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w57[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w57[2] : 0;
          unaligned_store_u32(out + 228, b0 | (b1 << 8) | (b2 << 16));
          w57 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w58[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w58[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w58[2] : 0;
          unaligned_store_u32(out + 232, b0 | (b1 << 8) | (b2 << 16));
          w58 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w59[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w59[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w59[2] : 0;
          unaligned_store_u32(out + 236, b0 | (b1 << 8) | (b2 << 16));
          w59 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w60[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w60[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w60[2] : 0;
          unaligned_store_u32(out + 240, b0 | (b1 << 8) | (b2 << 16));
          w60 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w61[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w61[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w61[2] : 0;
          unaligned_store_u32(out + 244, b0 | (b1 << 8) | (b2 << 16));
          w61 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w62[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w62[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w62[2] : 0;
          unaligned_store_u32(out + 248, b0 | (b1 << 8) | (b2 << 16));
          w62 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w63[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w63[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w63[2] : 0;
          unaligned_store_u32(out + 252, b0 | (b1 << 8) | (b2 << 16));
          w63 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w64[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w64[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w64[2] : 0;
          unaligned_store_u32(out + 256, b0 | (b1 << 8) | (b2 << 16));
          w64 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w65[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w65[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w65[2] : 0;
          unaligned_store_u32(out + 260, b0 | (b1 << 8) | (b2 << 16));
          w65 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w66[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w66[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w66[2] : 0;
          unaligned_store_u32(out + 264, b0 | (b1 << 8) | (b2 << 16));
          w66 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w67[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w67[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w67[2] : 0;
          unaligned_store_u32(out + 268, b0 | (b1 << 8) | (b2 << 16));
          w67 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w68[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w68[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w68[2] : 0;
          unaligned_store_u32(out + 272, b0 | (b1 << 8) | (b2 << 16));
          w68 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w69[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w69[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w69[2] : 0;
          unaligned_store_u32(out + 276, b0 | (b1 << 8) | (b2 << 16));
          w69 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w70[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w70[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w70[2] : 0;
          unaligned_store_u32(out + 280, b0 | (b1 << 8) | (b2 << 16));
          w70 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w71[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w71[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w71[2] : 0;
          unaligned_store_u32(out + 284, b0 | (b1 << 8) | (b2 << 16));
          w71 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w72[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w72[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w72[2] : 0;
          unaligned_store_u32(out + 288, b0 | (b1 << 8) | (b2 << 16));
          w72 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w73[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w73[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w73[2] : 0;
          unaligned_store_u32(out + 292, b0 | (b1 << 8) | (b2 << 16));
          w73 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w74[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w74[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w74[2] : 0;
          unaligned_store_u32(out + 296, b0 | (b1 << 8) | (b2 << 16));
          w74 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w75[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w75[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w75[2] : 0;
          unaligned_store_u32(out + 300, b0 | (b1 << 8) | (b2 << 16));
          w75 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w76[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w76[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w76[2] : 0;
          unaligned_store_u32(out + 304, b0 | (b1 << 8) | (b2 << 16));
          w76 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w77[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w77[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w77[2] : 0;
          unaligned_store_u32(out + 308, b0 | (b1 << 8) | (b2 << 16));
          w77 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w78[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w78[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w78[2] : 0;
          unaligned_store_u32(out + 312, b0 | (b1 << 8) | (b2 << 16));
          w78 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w79[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w79[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w79[2] : 0;
          unaligned_store_u32(out + 316, b0 | (b1 << 8) | (b2 << 16));
          w79 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w80[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w80[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w80[2] : 0;
          unaligned_store_u32(out + 320, b0 | (b1 << 8) | (b2 << 16));
          w80 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w81[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w81[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w81[2] : 0;
          unaligned_store_u32(out + 324, b0 | (b1 << 8) | (b2 << 16));
          w81 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w82[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w82[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w82[2] : 0;
          unaligned_store_u32(out + 328, b0 | (b1 << 8) | (b2 << 16));
          w82 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w83[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w83[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w83[2] : 0;
          unaligned_store_u32(out + 332, b0 | (b1 << 8) | (b2 << 16));
          w83 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w84[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w84[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w84[2] : 0;
          unaligned_store_u32(out + 336, b0 | (b1 << 8) | (b2 << 16));
          w84 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w85[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w85[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w85[2] : 0;
          unaligned_store_u32(out + 340, b0 | (b1 << 8) | (b2 << 16));
          w85 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w86[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w86[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w86[2] : 0;
          unaligned_store_u32(out + 344, b0 | (b1 << 8) | (b2 << 16));
          w86 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w87[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w87[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w87[2] : 0;
          unaligned_store_u32(out + 348, b0 | (b1 << 8) | (b2 << 16));
          w87 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w88[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w88[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w88[2] : 0;
          unaligned_store_u32(out + 352, b0 | (b1 << 8) | (b2 << 16));
          w88 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w89[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w89[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w89[2] : 0;
          unaligned_store_u32(out + 356, b0 | (b1 << 8) | (b2 << 16));
          w89 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w90[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w90[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w90[2] : 0;
          unaligned_store_u32(out + 360, b0 | (b1 << 8) | (b2 << 16));
          w90 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w91[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w91[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w91[2] : 0;
          unaligned_store_u32(out + 364, b0 | (b1 << 8) | (b2 << 16));
          w91 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w92[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w92[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w92[2] : 0;
          unaligned_store_u32(out + 368, b0 | (b1 << 8) | (b2 << 16));
          w92 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w93[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w93[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w93[2] : 0;
          unaligned_store_u32(out + 372, b0 | (b1 << 8) | (b2 << 16));
          w93 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w94[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w94[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w94[2] : 0;
          unaligned_store_u32(out + 376, b0 | (b1 << 8) | (b2 << 16));
          w94 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w95[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w95[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w95[2] : 0;
          unaligned_store_u32(out + 380, b0 | (b1 << 8) | (b2 << 16));
          w95 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w96[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w96[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w96[2] : 0;
          unaligned_store_u32(out + 384, b0 | (b1 << 8) | (b2 << 16));
          w96 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w97[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w97[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w97[2] : 0;
          unaligned_store_u32(out + 388, b0 | (b1 << 8) | (b2 << 16));
          w97 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w98[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w98[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w98[2] : 0;
          unaligned_store_u32(out + 392, b0 | (b1 << 8) | (b2 << 16));
          w98 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w99[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w99[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w99[2] : 0;
          unaligned_store_u32(out + 396, b0 | (b1 << 8) | (b2 << 16));
          w99 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w100[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w100[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w100[2] : 0;
          unaligned_store_u32(out + 400, b0 | (b1 << 8) | (b2 << 16));
          w100 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w101[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w101[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w101[2] : 0;
          unaligned_store_u32(out + 404, b0 | (b1 << 8) | (b2 << 16));
          w101 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w102[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w102[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w102[2] : 0;
          unaligned_store_u32(out + 408, b0 | (b1 << 8) | (b2 << 16));
          w102 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w103[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w103[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w103[2] : 0;
          unaligned_store_u32(out + 412, b0 | (b1 << 8) | (b2 << 16));
          w103 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w104[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w104[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w104[2] : 0;
          unaligned_store_u32(out + 416, b0 | (b1 << 8) | (b2 << 16));
          w104 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w105[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w105[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w105[2] : 0;
          unaligned_store_u32(out + 420, b0 | (b1 << 8) | (b2 << 16));
          w105 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w106[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w106[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w106[2] : 0;
          unaligned_store_u32(out + 424, b0 | (b1 << 8) | (b2 << 16));
          w106 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w107[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w107[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w107[2] : 0;
          unaligned_store_u32(out + 428, b0 | (b1 << 8) | (b2 << 16));
          w107 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w108[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w108[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w108[2] : 0;
          unaligned_store_u32(out + 432, b0 | (b1 << 8) | (b2 << 16));
          w108 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w109[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w109[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w109[2] : 0;
          unaligned_store_u32(out + 436, b0 | (b1 << 8) | (b2 << 16));
          w109 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w110[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w110[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w110[2] : 0;
          unaligned_store_u32(out + 440, b0 | (b1 << 8) | (b2 << 16));
          w110 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w111[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w111[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w111[2] : 0;
          unaligned_store_u32(out + 444, b0 | (b1 << 8) | (b2 << 16));
          w111 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w112[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w112[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w112[2] : 0;
          unaligned_store_u32(out + 448, b0 | (b1 << 8) | (b2 << 16));
          w112 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w113[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w113[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w113[2] : 0;
          unaligned_store_u32(out + 452, b0 | (b1 << 8) | (b2 << 16));
          w113 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w114[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w114[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w114[2] : 0;
          unaligned_store_u32(out + 456, b0 | (b1 << 8) | (b2 << 16));
          w114 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w115[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w115[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w115[2] : 0;
          unaligned_store_u32(out + 460, b0 | (b1 << 8) | (b2 << 16));
          w115 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w116[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w116[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w116[2] : 0;
          unaligned_store_u32(out + 464, b0 | (b1 << 8) | (b2 << 16));
          w116 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w117[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w117[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w117[2] : 0;
          unaligned_store_u32(out + 468, b0 | (b1 << 8) | (b2 << 16));
          w117 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w118[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w118[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w118[2] : 0;
          unaligned_store_u32(out + 472, b0 | (b1 << 8) | (b2 << 16));
          w118 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w119[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w119[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w119[2] : 0;
          unaligned_store_u32(out + 476, b0 | (b1 << 8) | (b2 << 16));
          w119 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w120[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w120[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w120[2] : 0;
          unaligned_store_u32(out + 480, b0 | (b1 << 8) | (b2 << 16));
          w120 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w121[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w121[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w121[2] : 0;
          unaligned_store_u32(out + 484, b0 | (b1 << 8) | (b2 << 16));
          w121 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w122[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w122[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w122[2] : 0;
          unaligned_store_u32(out + 488, b0 | (b1 << 8) | (b2 << 16));
          w122 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w123[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w123[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w123[2] : 0;
          unaligned_store_u32(out + 492, b0 | (b1 << 8) | (b2 << 16));
          w123 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w124[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w124[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w124[2] : 0;
          unaligned_store_u32(out + 496, b0 | (b1 << 8) | (b2 << 16));
          w124 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w125[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w125[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w125[2] : 0;
          unaligned_store_u32(out + 500, b0 | (b1 << 8) | (b2 << 16));
          w125 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w126[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w126[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w126[2] : 0;
          unaligned_store_u32(out + 504, b0 | (b1 << 8) | (b2 << 16));
          w126 += k;
        }
        {
          const uint32_t b0 = (uint8_t) w127[0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w127[1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w127[2] : 0;
          unaligned_store_u32(out + 508, b0 | (b1 << 8) | (b2 << 16));
          w127 += k;
        }
        vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc0_0, *((const HVX_UVector*) (out + 0)), vones);
        vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc1_0, *((const HVX_UVector*) (out + 128)), vones);
        vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc2_0, *((const HVX_UVector*) (out + 256)), vones);
        vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(
            vacc3_0, *((const HVX_UVector*) (out + 384)), vones);
        out += 512;
      }

      {
        const HVX_Vector vksum = Q6_Vw_vadd_VwVw(vacc0_0, vacc0_1);
        const HVX_Vector vpb = *((const HVX_UVector*) (packed_b + 0));
        *((HVX_UVector*) (packed_b + 0)) =
            Q6_Vw_vsub_VwVw(vpb, xnn_mul_s32(vksum, vizp));
      }
      {
        const HVX_Vector vksum = Q6_Vw_vadd_VwVw(vacc1_0, vacc1_1);
        const HVX_Vector vpb = *((const HVX_UVector*) (packed_b + 32));
        *((HVX_UVector*) (packed_b + 32)) =
            Q6_Vw_vsub_VwVw(vpb, xnn_mul_s32(vksum, vizp));
      }
      {
        const HVX_Vector vksum = Q6_Vw_vadd_VwVw(vacc2_0, vacc2_1);
        const HVX_Vector vpb = *((const HVX_UVector*) (packed_b + 64));
        *((HVX_UVector*) (packed_b + 64)) =
            Q6_Vw_vsub_VwVw(vpb, xnn_mul_s32(vksum, vizp));
      }
      {
        const HVX_Vector vksum = Q6_Vw_vadd_VwVw(vacc3_0, vacc3_1);
        const HVX_Vector vpb = *((const HVX_UVector*) (packed_b + 96));
        *((HVX_UVector*) (packed_b + 96)) =
            Q6_Vw_vsub_VwVw(vpb, xnn_mul_s32(vksum, vizp));
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w127 + n_stride - kc;
    }

    if XNN_UNLIKELY(n != 0) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        memcpy(packed_b, b, n * sizeof(int32_t));
        b += n;
      } else {
        memset(packed_b, 0, n * sizeof(int32_t));
      }
      out += 128 * sizeof(int32_t);

      const int8_t* w_ptr[128];
      w_ptr[0] = w0;
      for (size_t i = 1; i < n; ++i) {
        w_ptr[i] = w_ptr[i - 1] + n_stride;
      }

      HVX_Vector vacc_even[4];
      HVX_Vector vacc_odd[4];
      vacc_even[0] = vzero;
      vacc_odd[0] = vzero;
      vacc_even[1] = vzero;
      vacc_odd[1] = vzero;
      vacc_even[2] = vzero;
      vacc_odd[2] = vzero;
      vacc_even[3] = vzero;
      vacc_odd[3] = vzero;

      size_t k = kc;
      for (; k >= 128; k -= 128) {
        for (size_t t = 0; t < 4; ++t) {
          const size_t n_off = t * 32;
          if (n_off >= n) {
            for (size_t kk = 0; kk < 32; ++kk) {
              *((HVX_UVector*) (out + kk * 512 + n_off * 4)) = vzero;
            }
          } else {
            HVX_Vector v5_0 = vzero;
            if (n_off + 0 < n) {
              v5_0 = *((const HVX_UVector*) w_ptr[n_off + 0]);
              w_ptr[n_off + 0] += 128;
            }
            HVX_Vector v5_1 = vzero;
            if (n_off + 1 < n) {
              v5_1 = *((const HVX_UVector*) w_ptr[n_off + 1]);
              w_ptr[n_off + 1] += 128;
            }
            HVX_Vector v5_2 = vzero;
            if (n_off + 2 < n) {
              v5_2 = *((const HVX_UVector*) w_ptr[n_off + 2]);
              w_ptr[n_off + 2] += 128;
            }
            HVX_Vector v5_3 = vzero;
            if (n_off + 3 < n) {
              v5_3 = *((const HVX_UVector*) w_ptr[n_off + 3]);
              w_ptr[n_off + 3] += 128;
            }
            HVX_Vector v5_4 = vzero;
            if (n_off + 4 < n) {
              v5_4 = *((const HVX_UVector*) w_ptr[n_off + 4]);
              w_ptr[n_off + 4] += 128;
            }
            HVX_Vector v5_5 = vzero;
            if (n_off + 5 < n) {
              v5_5 = *((const HVX_UVector*) w_ptr[n_off + 5]);
              w_ptr[n_off + 5] += 128;
            }
            HVX_Vector v5_6 = vzero;
            if (n_off + 6 < n) {
              v5_6 = *((const HVX_UVector*) w_ptr[n_off + 6]);
              w_ptr[n_off + 6] += 128;
            }
            HVX_Vector v5_7 = vzero;
            if (n_off + 7 < n) {
              v5_7 = *((const HVX_UVector*) w_ptr[n_off + 7]);
              w_ptr[n_off + 7] += 128;
            }
            HVX_Vector v5_8 = vzero;
            if (n_off + 8 < n) {
              v5_8 = *((const HVX_UVector*) w_ptr[n_off + 8]);
              w_ptr[n_off + 8] += 128;
            }
            HVX_Vector v5_9 = vzero;
            if (n_off + 9 < n) {
              v5_9 = *((const HVX_UVector*) w_ptr[n_off + 9]);
              w_ptr[n_off + 9] += 128;
            }
            HVX_Vector v5_10 = vzero;
            if (n_off + 10 < n) {
              v5_10 = *((const HVX_UVector*) w_ptr[n_off + 10]);
              w_ptr[n_off + 10] += 128;
            }
            HVX_Vector v5_11 = vzero;
            if (n_off + 11 < n) {
              v5_11 = *((const HVX_UVector*) w_ptr[n_off + 11]);
              w_ptr[n_off + 11] += 128;
            }
            HVX_Vector v5_12 = vzero;
            if (n_off + 12 < n) {
              v5_12 = *((const HVX_UVector*) w_ptr[n_off + 12]);
              w_ptr[n_off + 12] += 128;
            }
            HVX_Vector v5_13 = vzero;
            if (n_off + 13 < n) {
              v5_13 = *((const HVX_UVector*) w_ptr[n_off + 13]);
              w_ptr[n_off + 13] += 128;
            }
            HVX_Vector v5_14 = vzero;
            if (n_off + 14 < n) {
              v5_14 = *((const HVX_UVector*) w_ptr[n_off + 14]);
              w_ptr[n_off + 14] += 128;
            }
            HVX_Vector v5_15 = vzero;
            if (n_off + 15 < n) {
              v5_15 = *((const HVX_UVector*) w_ptr[n_off + 15]);
              w_ptr[n_off + 15] += 128;
            }
            HVX_Vector v5_16 = vzero;
            if (n_off + 16 < n) {
              v5_16 = *((const HVX_UVector*) w_ptr[n_off + 16]);
              w_ptr[n_off + 16] += 128;
            }
            HVX_Vector v5_17 = vzero;
            if (n_off + 17 < n) {
              v5_17 = *((const HVX_UVector*) w_ptr[n_off + 17]);
              w_ptr[n_off + 17] += 128;
            }
            HVX_Vector v5_18 = vzero;
            if (n_off + 18 < n) {
              v5_18 = *((const HVX_UVector*) w_ptr[n_off + 18]);
              w_ptr[n_off + 18] += 128;
            }
            HVX_Vector v5_19 = vzero;
            if (n_off + 19 < n) {
              v5_19 = *((const HVX_UVector*) w_ptr[n_off + 19]);
              w_ptr[n_off + 19] += 128;
            }
            HVX_Vector v5_20 = vzero;
            if (n_off + 20 < n) {
              v5_20 = *((const HVX_UVector*) w_ptr[n_off + 20]);
              w_ptr[n_off + 20] += 128;
            }
            HVX_Vector v5_21 = vzero;
            if (n_off + 21 < n) {
              v5_21 = *((const HVX_UVector*) w_ptr[n_off + 21]);
              w_ptr[n_off + 21] += 128;
            }
            HVX_Vector v5_22 = vzero;
            if (n_off + 22 < n) {
              v5_22 = *((const HVX_UVector*) w_ptr[n_off + 22]);
              w_ptr[n_off + 22] += 128;
            }
            HVX_Vector v5_23 = vzero;
            if (n_off + 23 < n) {
              v5_23 = *((const HVX_UVector*) w_ptr[n_off + 23]);
              w_ptr[n_off + 23] += 128;
            }
            HVX_Vector v5_24 = vzero;
            if (n_off + 24 < n) {
              v5_24 = *((const HVX_UVector*) w_ptr[n_off + 24]);
              w_ptr[n_off + 24] += 128;
            }
            HVX_Vector v5_25 = vzero;
            if (n_off + 25 < n) {
              v5_25 = *((const HVX_UVector*) w_ptr[n_off + 25]);
              w_ptr[n_off + 25] += 128;
            }
            HVX_Vector v5_26 = vzero;
            if (n_off + 26 < n) {
              v5_26 = *((const HVX_UVector*) w_ptr[n_off + 26]);
              w_ptr[n_off + 26] += 128;
            }
            HVX_Vector v5_27 = vzero;
            if (n_off + 27 < n) {
              v5_27 = *((const HVX_UVector*) w_ptr[n_off + 27]);
              w_ptr[n_off + 27] += 128;
            }
            HVX_Vector v5_28 = vzero;
            if (n_off + 28 < n) {
              v5_28 = *((const HVX_UVector*) w_ptr[n_off + 28]);
              w_ptr[n_off + 28] += 128;
            }
            HVX_Vector v5_29 = vzero;
            if (n_off + 29 < n) {
              v5_29 = *((const HVX_UVector*) w_ptr[n_off + 29]);
              w_ptr[n_off + 29] += 128;
            }
            HVX_Vector v5_30 = vzero;
            if (n_off + 30 < n) {
              v5_30 = *((const HVX_UVector*) w_ptr[n_off + 30]);
              w_ptr[n_off + 30] += 128;
            }
            HVX_Vector v5_31 = vzero;
            if (n_off + 31 < n) {
              v5_31 = *((const HVX_UVector*) w_ptr[n_off + 31]);
              w_ptr[n_off + 31] += 128;
            }

            const HVX_VectorPair v4_0 =
                Q6_W_vshuff_VVR(v5_1, v5_0, -4);
            const HVX_VectorPair v4_1 =
                Q6_W_vshuff_VVR(v5_3, v5_2, -4);
            const HVX_VectorPair v4_2 =
                Q6_W_vshuff_VVR(v5_5, v5_4, -4);
            const HVX_VectorPair v4_3 =
                Q6_W_vshuff_VVR(v5_7, v5_6, -4);
            const HVX_VectorPair v4_4 =
                Q6_W_vshuff_VVR(v5_9, v5_8, -4);
            const HVX_VectorPair v4_5 =
                Q6_W_vshuff_VVR(v5_11, v5_10, -4);
            const HVX_VectorPair v4_6 =
                Q6_W_vshuff_VVR(v5_13, v5_12, -4);
            const HVX_VectorPair v4_7 =
                Q6_W_vshuff_VVR(v5_15, v5_14, -4);
            const HVX_VectorPair v4_8 =
                Q6_W_vshuff_VVR(v5_17, v5_16, -4);
            const HVX_VectorPair v4_9 =
                Q6_W_vshuff_VVR(v5_19, v5_18, -4);
            const HVX_VectorPair v4_10 =
                Q6_W_vshuff_VVR(v5_21, v5_20, -4);
            const HVX_VectorPair v4_11 =
                Q6_W_vshuff_VVR(v5_23, v5_22, -4);
            const HVX_VectorPair v4_12 =
                Q6_W_vshuff_VVR(v5_25, v5_24, -4);
            const HVX_VectorPair v4_13 =
                Q6_W_vshuff_VVR(v5_27, v5_26, -4);
            const HVX_VectorPair v4_14 =
                Q6_W_vshuff_VVR(v5_29, v5_28, -4);
            const HVX_VectorPair v4_15 =
                Q6_W_vshuff_VVR(v5_31, v5_30, -4);

            const HVX_VectorPair v3_0 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_1), Q6_V_lo_W(v4_0), -8);
            const HVX_VectorPair v3_1 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_1), Q6_V_hi_W(v4_0), -8);
            const HVX_VectorPair v3_2 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_3), Q6_V_lo_W(v4_2), -8);
            const HVX_VectorPair v3_3 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_3), Q6_V_hi_W(v4_2), -8);
            const HVX_VectorPair v3_4 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_5), Q6_V_lo_W(v4_4), -8);
            const HVX_VectorPair v3_5 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_5), Q6_V_hi_W(v4_4), -8);
            const HVX_VectorPair v3_6 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_7), Q6_V_lo_W(v4_6), -8);
            const HVX_VectorPair v3_7 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_7), Q6_V_hi_W(v4_6), -8);
            const HVX_VectorPair v3_8 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_9), Q6_V_lo_W(v4_8), -8);
            const HVX_VectorPair v3_9 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_9), Q6_V_hi_W(v4_8), -8);
            const HVX_VectorPair v3_10 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_11), Q6_V_lo_W(v4_10), -8);
            const HVX_VectorPair v3_11 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_11), Q6_V_hi_W(v4_10), -8);
            const HVX_VectorPair v3_12 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_13), Q6_V_lo_W(v4_12), -8);
            const HVX_VectorPair v3_13 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_13), Q6_V_hi_W(v4_12), -8);
            const HVX_VectorPair v3_14 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v4_15), Q6_V_lo_W(v4_14), -8);
            const HVX_VectorPair v3_15 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v4_15), Q6_V_hi_W(v4_14), -8);

            const HVX_VectorPair v2_0 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_2), Q6_V_lo_W(v3_0), -16);
            const HVX_VectorPair v2_1 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_2), Q6_V_hi_W(v3_0), -16);
            const HVX_VectorPair v2_2 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_3), Q6_V_lo_W(v3_1), -16);
            const HVX_VectorPair v2_3 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_3), Q6_V_hi_W(v3_1), -16);
            const HVX_VectorPair v2_4 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_6), Q6_V_lo_W(v3_4), -16);
            const HVX_VectorPair v2_5 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_6), Q6_V_hi_W(v3_4), -16);
            const HVX_VectorPair v2_6 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_7), Q6_V_lo_W(v3_5), -16);
            const HVX_VectorPair v2_7 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_7), Q6_V_hi_W(v3_5), -16);
            const HVX_VectorPair v2_8 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_10), Q6_V_lo_W(v3_8), -16);
            const HVX_VectorPair v2_9 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_10), Q6_V_hi_W(v3_8), -16);
            const HVX_VectorPair v2_10 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_11), Q6_V_lo_W(v3_9), -16);
            const HVX_VectorPair v2_11 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_11), Q6_V_hi_W(v3_9), -16);
            const HVX_VectorPair v2_12 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_14), Q6_V_lo_W(v3_12), -16);
            const HVX_VectorPair v2_13 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_14), Q6_V_hi_W(v3_12), -16);
            const HVX_VectorPair v2_14 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v3_15), Q6_V_lo_W(v3_13), -16);
            const HVX_VectorPair v2_15 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v3_15), Q6_V_hi_W(v3_13), -16);

            const HVX_VectorPair v1_0 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_4), Q6_V_lo_W(v2_0), -32);
            const HVX_VectorPair v1_1 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_4), Q6_V_hi_W(v2_0), -32);
            const HVX_VectorPair v1_2 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_5), Q6_V_lo_W(v2_1), -32);
            const HVX_VectorPair v1_3 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_5), Q6_V_hi_W(v2_1), -32);
            const HVX_VectorPair v1_4 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_6), Q6_V_lo_W(v2_2), -32);
            const HVX_VectorPair v1_5 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_6), Q6_V_hi_W(v2_2), -32);
            const HVX_VectorPair v1_6 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_7), Q6_V_lo_W(v2_3), -32);
            const HVX_VectorPair v1_7 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_7), Q6_V_hi_W(v2_3), -32);
            const HVX_VectorPair v1_8 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_12), Q6_V_lo_W(v2_8), -32);
            const HVX_VectorPair v1_9 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_12), Q6_V_hi_W(v2_8), -32);
            const HVX_VectorPair v1_10 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_13), Q6_V_lo_W(v2_9), -32);
            const HVX_VectorPair v1_11 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_13), Q6_V_hi_W(v2_9), -32);
            const HVX_VectorPair v1_12 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_14), Q6_V_lo_W(v2_10), -32);
            const HVX_VectorPair v1_13 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_14), Q6_V_hi_W(v2_10), -32);
            const HVX_VectorPair v1_14 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v2_15), Q6_V_lo_W(v2_11), -32);
            const HVX_VectorPair v1_15 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v2_15), Q6_V_hi_W(v2_11), -32);

            const HVX_VectorPair v0_0 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_8), Q6_V_lo_W(v1_0), -64);
            const HVX_VectorPair v0_1 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_8), Q6_V_hi_W(v1_0), -64);
            const HVX_VectorPair v0_2 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_9), Q6_V_lo_W(v1_1), -64);
            const HVX_VectorPair v0_3 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_9), Q6_V_hi_W(v1_1), -64);
            const HVX_VectorPair v0_4 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_10), Q6_V_lo_W(v1_2), -64);
            const HVX_VectorPair v0_5 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_10), Q6_V_hi_W(v1_2), -64);
            const HVX_VectorPair v0_6 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_11), Q6_V_lo_W(v1_3), -64);
            const HVX_VectorPair v0_7 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_11), Q6_V_hi_W(v1_3), -64);
            const HVX_VectorPair v0_8 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_12), Q6_V_lo_W(v1_4), -64);
            const HVX_VectorPair v0_9 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_12), Q6_V_hi_W(v1_4), -64);
            const HVX_VectorPair v0_10 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_13), Q6_V_lo_W(v1_5), -64);
            const HVX_VectorPair v0_11 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_13), Q6_V_hi_W(v1_5), -64);
            const HVX_VectorPair v0_12 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_14), Q6_V_lo_W(v1_6), -64);
            const HVX_VectorPair v0_13 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_14), Q6_V_hi_W(v1_6), -64);
            const HVX_VectorPair v0_14 = Q6_W_vshuff_VVR(
                Q6_V_lo_W(v1_15), Q6_V_lo_W(v1_7), -64);
            const HVX_VectorPair v0_15 = Q6_W_vshuff_VVR(
                Q6_V_hi_W(v1_15), Q6_V_hi_W(v1_7), -64);

            const HVX_Vector vt_lo_0 = Q6_V_lo_W(v0_0);
            const HVX_Vector vt_hi_0 = Q6_V_hi_W(v0_0);
            *((HVX_UVector*) (out + 0 + n_off * 4)) =
                vt_lo_0;
            *((HVX_UVector*) (out + 512 + n_off * 4)) =
                vt_hi_0;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_0, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_0, vones);
            const HVX_Vector vt_lo_1 = Q6_V_lo_W(v0_1);
            const HVX_Vector vt_hi_1 = Q6_V_hi_W(v0_1);
            *((HVX_UVector*) (out + 1024 + n_off * 4)) =
                vt_lo_1;
            *((HVX_UVector*) (out + 1536 + n_off * 4)) =
                vt_hi_1;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_1, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_1, vones);
            const HVX_Vector vt_lo_2 = Q6_V_lo_W(v0_2);
            const HVX_Vector vt_hi_2 = Q6_V_hi_W(v0_2);
            *((HVX_UVector*) (out + 2048 + n_off * 4)) =
                vt_lo_2;
            *((HVX_UVector*) (out + 2560 + n_off * 4)) =
                vt_hi_2;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_2, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_2, vones);
            const HVX_Vector vt_lo_3 = Q6_V_lo_W(v0_3);
            const HVX_Vector vt_hi_3 = Q6_V_hi_W(v0_3);
            *((HVX_UVector*) (out + 3072 + n_off * 4)) =
                vt_lo_3;
            *((HVX_UVector*) (out + 3584 + n_off * 4)) =
                vt_hi_3;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_3, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_3, vones);
            const HVX_Vector vt_lo_4 = Q6_V_lo_W(v0_4);
            const HVX_Vector vt_hi_4 = Q6_V_hi_W(v0_4);
            *((HVX_UVector*) (out + 4096 + n_off * 4)) =
                vt_lo_4;
            *((HVX_UVector*) (out + 4608 + n_off * 4)) =
                vt_hi_4;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_4, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_4, vones);
            const HVX_Vector vt_lo_5 = Q6_V_lo_W(v0_5);
            const HVX_Vector vt_hi_5 = Q6_V_hi_W(v0_5);
            *((HVX_UVector*) (out + 5120 + n_off * 4)) =
                vt_lo_5;
            *((HVX_UVector*) (out + 5632 + n_off * 4)) =
                vt_hi_5;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_5, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_5, vones);
            const HVX_Vector vt_lo_6 = Q6_V_lo_W(v0_6);
            const HVX_Vector vt_hi_6 = Q6_V_hi_W(v0_6);
            *((HVX_UVector*) (out + 6144 + n_off * 4)) =
                vt_lo_6;
            *((HVX_UVector*) (out + 6656 + n_off * 4)) =
                vt_hi_6;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_6, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_6, vones);
            const HVX_Vector vt_lo_7 = Q6_V_lo_W(v0_7);
            const HVX_Vector vt_hi_7 = Q6_V_hi_W(v0_7);
            *((HVX_UVector*) (out + 7168 + n_off * 4)) =
                vt_lo_7;
            *((HVX_UVector*) (out + 7680 + n_off * 4)) =
                vt_hi_7;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_7, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_7, vones);
            const HVX_Vector vt_lo_8 = Q6_V_lo_W(v0_8);
            const HVX_Vector vt_hi_8 = Q6_V_hi_W(v0_8);
            *((HVX_UVector*) (out + 8192 + n_off * 4)) =
                vt_lo_8;
            *((HVX_UVector*) (out + 8704 + n_off * 4)) =
                vt_hi_8;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_8, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_8, vones);
            const HVX_Vector vt_lo_9 = Q6_V_lo_W(v0_9);
            const HVX_Vector vt_hi_9 = Q6_V_hi_W(v0_9);
            *((HVX_UVector*) (out + 9216 + n_off * 4)) =
                vt_lo_9;
            *((HVX_UVector*) (out + 9728 + n_off * 4)) =
                vt_hi_9;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_9, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_9, vones);
            const HVX_Vector vt_lo_10 = Q6_V_lo_W(v0_10);
            const HVX_Vector vt_hi_10 = Q6_V_hi_W(v0_10);
            *((HVX_UVector*) (out + 10240 + n_off * 4)) =
                vt_lo_10;
            *((HVX_UVector*) (out + 10752 + n_off * 4)) =
                vt_hi_10;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_10, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_10, vones);
            const HVX_Vector vt_lo_11 = Q6_V_lo_W(v0_11);
            const HVX_Vector vt_hi_11 = Q6_V_hi_W(v0_11);
            *((HVX_UVector*) (out + 11264 + n_off * 4)) =
                vt_lo_11;
            *((HVX_UVector*) (out + 11776 + n_off * 4)) =
                vt_hi_11;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_11, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_11, vones);
            const HVX_Vector vt_lo_12 = Q6_V_lo_W(v0_12);
            const HVX_Vector vt_hi_12 = Q6_V_hi_W(v0_12);
            *((HVX_UVector*) (out + 12288 + n_off * 4)) =
                vt_lo_12;
            *((HVX_UVector*) (out + 12800 + n_off * 4)) =
                vt_hi_12;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_12, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_12, vones);
            const HVX_Vector vt_lo_13 = Q6_V_lo_W(v0_13);
            const HVX_Vector vt_hi_13 = Q6_V_hi_W(v0_13);
            *((HVX_UVector*) (out + 13312 + n_off * 4)) =
                vt_lo_13;
            *((HVX_UVector*) (out + 13824 + n_off * 4)) =
                vt_hi_13;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_13, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_13, vones);
            const HVX_Vector vt_lo_14 = Q6_V_lo_W(v0_14);
            const HVX_Vector vt_hi_14 = Q6_V_hi_W(v0_14);
            *((HVX_UVector*) (out + 14336 + n_off * 4)) =
                vt_lo_14;
            *((HVX_UVector*) (out + 14848 + n_off * 4)) =
                vt_hi_14;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_14, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_14, vones);
            const HVX_Vector vt_lo_15 = Q6_V_lo_W(v0_15);
            const HVX_Vector vt_hi_15 = Q6_V_hi_W(v0_15);
            *((HVX_UVector*) (out + 15360 + n_off * 4)) =
                vt_lo_15;
            *((HVX_UVector*) (out + 15872 + n_off * 4)) =
                vt_hi_15;
            vacc_even[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_even[t], vt_lo_15, vones);
            vacc_odd[t] =
                Q6_Vw_vrmpyacc_VwVbVb(vacc_odd[t], vt_hi_15, vones);
          }
        }
        out += 16384;
      }

      for (; k >= 4; k -= 4) {
        size_t i = 0;
        for (; i < n; ++i) {
          unaligned_store_u32(out + i * 4, unaligned_load_u32(w_ptr[i]));
          w_ptr[i] += 4;
        }
        for (; i < 128; ++i) {
          unaligned_store_u32(out + i * 4, 0);
        }
        vacc_even[0] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[0], *((const HVX_UVector*) (out + 0)),
            vones);
        vacc_even[1] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[1], *((const HVX_UVector*) (out + 128)),
            vones);
        vacc_even[2] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[2], *((const HVX_UVector*) (out + 256)),
            vones);
        vacc_even[3] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[3], *((const HVX_UVector*) (out + 384)),
            vones);
        out += 512;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        size_t i = 0;
        for (; i < n; ++i) {
          const uint32_t b0 = (uint8_t) w_ptr[i][0];
          const uint32_t b1 = k > 1 ? (uint32_t) (uint8_t) w_ptr[i][1] : 0;
          const uint32_t b2 = k > 2 ? (uint32_t) (uint8_t) w_ptr[i][2] : 0;
          unaligned_store_u32(out + i * 4, b0 | (b1 << 8) | (b2 << 16));
          w_ptr[i] += k;
        }
        for (; i < 128; ++i) {
          unaligned_store_u32(out + i * 4, 0);
        }
        vacc_even[0] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[0], *((const HVX_UVector*) (out + 0)),
            vones);
        vacc_even[1] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[1], *((const HVX_UVector*) (out + 128)),
            vones);
        vacc_even[2] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[2], *((const HVX_UVector*) (out + 256)),
            vones);
        vacc_even[3] = Q6_Vw_vrmpyacc_VwVbVb(
            vacc_even[3], *((const HVX_UVector*) (out + 384)),
            vones);
        out += 512;
      }

      for (size_t t = 0; t < 4; ++t) {
        const size_t n_off = t * 32;
        if (n_off >= n) {
          break;
        }
        const HVX_Vector vadj =
            xnn_mul_s32(Q6_Vw_vadd_VwVw(vacc_even[t], vacc_odd[t]), vizp);
        if (n - n_off >= 32) {
          const HVX_Vector vpb = *((const HVX_UVector*) (packed_b + n_off));
          *((HVX_UVector*) (packed_b + n_off)) = Q6_Vw_vsub_VwVw(vpb, vadj);
        } else {
          XNN_ALIGN(128) int32_t adj_buf[32];
          *((HVX_Vector*) adj_buf) = vadj;
          const size_t rem = n - n_off;
          for (size_t r = 0; r < rem; ++r) {
            packed_b[n_off + r] =
                (int32_t) ((uint32_t) packed_b[n_off + r] -
                           (uint32_t) adj_buf[r]);
          }
        }
      }

      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }
    weights += nc * n_stride;
  } while (--g != 0);
}
