// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x8-packw/c4-gio-hvx.c.in
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
#include "src/xnnpack/simd/s8-hvx.h"
#include "src/xnnpack/simd/s32-hvx.h"

void xnn_qs8_to_qu8_packw_gemm_gio_ukernel_x128c4__hvx(
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

      HVX_Vector vacc0_0 = vzero;
      HVX_Vector vacc0_1 = vzero;
      HVX_Vector vacc1_0 = vzero;
      HVX_Vector vacc1_1 = vzero;
      HVX_Vector vacc2_0 = vzero;
      HVX_Vector vacc2_1 = vzero;
      HVX_Vector vacc3_0 = vzero;
      HVX_Vector vacc3_1 = vzero;

      const int8_t* w_k = w0;
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        const HVX_Vector r0 = xnn_loadu_s8(w_k);
        const HVX_Vector r1 = xnn_loadu_s8(w_k + k_stride);
        const HVX_Vector r2 = xnn_loadu_s8(w_k + 2 * k_stride);
        const HVX_Vector r3 = xnn_loadu_s8(w_k + 3 * k_stride);
        const HVX_Vector r4 = xnn_loadu_s8(w_k + 4 * k_stride);
        const HVX_Vector r5 = xnn_loadu_s8(w_k + 5 * k_stride);
        const HVX_Vector r6 = xnn_loadu_s8(w_k + 6 * k_stride);
        const HVX_Vector r7 = xnn_loadu_s8(w_k + 7 * k_stride);
        w_k += 8 * k_stride;

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -1);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(r3, r2, -1);
        const HVX_VectorPair q0_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -2);
        const HVX_VectorPair q0_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -2);

        const HVX_VectorPair p45 = Q6_W_vshuff_VVR(r5, r4, -1);
        const HVX_VectorPair p67 = Q6_W_vshuff_VVR(r7, r6, -1);
        const HVX_VectorPair q1_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p67), Q6_V_lo_W(p45), -2);
        const HVX_VectorPair q1_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p67), Q6_V_hi_W(p45), -2);

        *((HVX_UVector*) (out + 0)) = Q6_V_lo_W(q0_lo);
        *((HVX_UVector*) (out + 128)) = Q6_V_hi_W(q0_lo);
        *((HVX_UVector*) (out + 256)) = Q6_V_lo_W(q0_hi);
        *((HVX_UVector*) (out + 384)) = Q6_V_hi_W(q0_hi);
        *((HVX_UVector*) (out + 512)) = Q6_V_lo_W(q1_lo);
        *((HVX_UVector*) (out + 640)) = Q6_V_hi_W(q1_lo);
        *((HVX_UVector*) (out + 768)) = Q6_V_lo_W(q1_hi);
        *((HVX_UVector*) (out + 896)) = Q6_V_hi_W(q1_hi);
        out += 1024;

        vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, Q6_V_lo_W(q0_lo), vones);
        vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, Q6_V_hi_W(q0_lo), vones);
        vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, Q6_V_lo_W(q0_hi), vones);
        vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, Q6_V_hi_W(q0_hi), vones);
        vacc0_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_1, Q6_V_lo_W(q1_lo), vones);
        vacc1_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_1, Q6_V_hi_W(q1_lo), vones);
        vacc2_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_1, Q6_V_lo_W(q1_hi), vones);
        vacc3_1 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_1, Q6_V_hi_W(q1_hi), vones);
      }
      for (; k >= 4; k -= 4) {
        const HVX_Vector r0 = xnn_loadu_s8(w_k);
        const HVX_Vector r1 = xnn_loadu_s8(w_k + k_stride);
        const HVX_Vector r2 = xnn_loadu_s8(w_k + 2 * k_stride);
        const HVX_Vector r3 = xnn_loadu_s8(w_k + 3 * k_stride);
        w_k += 4 * k_stride;

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -1);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(r3, r2, -1);
        const HVX_VectorPair q0_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -2);
        const HVX_VectorPair q0_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -2);

        *((HVX_UVector*) (out + 0)) = Q6_V_lo_W(q0_lo);
        *((HVX_UVector*) (out + 128)) = Q6_V_hi_W(q0_lo);
        *((HVX_UVector*) (out + 256)) = Q6_V_lo_W(q0_hi);
        *((HVX_UVector*) (out + 384)) = Q6_V_hi_W(q0_hi);
        out += 512;

        vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, Q6_V_lo_W(q0_lo), vones);
        vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, Q6_V_hi_W(q0_lo), vones);
        vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, Q6_V_lo_W(q0_hi), vones);
        vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, Q6_V_hi_W(q0_hi), vones);
      }
      if (k != 0) {
        assert(k >= 1 && k <= 3);
        const HVX_Vector r0 = xnn_loadu_s8(w_k);
        const HVX_Vector r1 = k > 1 ? xnn_loadu_s8(w_k + k_stride) : vzero;
        const HVX_Vector r2 =
            k > 2 ? xnn_loadu_s8(w_k + 2 * k_stride) : vzero;
        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -1);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vzero, r2, -1);
        const HVX_VectorPair q0_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -2);
        const HVX_VectorPair q0_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -2);

        *((HVX_UVector*) (out + 0)) = Q6_V_lo_W(q0_lo);
        *((HVX_UVector*) (out + 128)) = Q6_V_hi_W(q0_lo);
        *((HVX_UVector*) (out + 256)) = Q6_V_lo_W(q0_hi);
        *((HVX_UVector*) (out + 384)) = Q6_V_hi_W(q0_hi);
        out += 512;

        vacc0_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc0_0, Q6_V_lo_W(q0_lo), vones);
        vacc1_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc1_0, Q6_V_hi_W(q0_lo), vones);
        vacc2_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc2_0, Q6_V_lo_W(q0_hi), vones);
        vacc3_0 = Q6_Vw_vrmpyacc_VwVbVb(vacc3_0, Q6_V_hi_W(q0_hi), vones);
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
      w0 += 128;
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

      const HVX_VectorPred qn = Q6_Q_vsetq_R(n);
      HVX_Vector vacc[4];
      vacc[0] = vzero;
      vacc[1] = vzero;
      vacc[2] = vzero;
      vacc[3] = vzero;

      const int8_t* w_k = w0;
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const HVX_Vector r0 = Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k, n));
        const HVX_Vector r1 =
            Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k + k_stride, n));
        const HVX_Vector r2 =
            Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k + 2 * k_stride, n));
        const HVX_Vector r3 =
            Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k + 3 * k_stride, n));
        w_k += 4 * k_stride;

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -1);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(r3, r2, -1);
        const HVX_VectorPair q0_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -2);
        *((HVX_UVector*) (out + 0)) = Q6_V_lo_W(q0_lo);
        *((HVX_UVector*) (out + 128)) = Q6_V_hi_W(q0_lo);
        vacc[0] = Q6_Vw_vrmpyacc_VwVbVb(vacc[0], Q6_V_lo_W(q0_lo), vones);
        vacc[1] = Q6_Vw_vrmpyacc_VwVbVb(vacc[1], Q6_V_hi_W(q0_lo), vones);
        const HVX_VectorPair q0_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -2);
        *((HVX_UVector*) (out + 256)) = Q6_V_lo_W(q0_hi);
        *((HVX_UVector*) (out + 384)) = Q6_V_hi_W(q0_hi);
        vacc[2] = Q6_Vw_vrmpyacc_VwVbVb(vacc[2], Q6_V_lo_W(q0_hi), vones);
        vacc[3] = Q6_Vw_vrmpyacc_VwVbVb(vacc[3], Q6_V_hi_W(q0_hi), vones);
        out += 512;
      }

      if (k != 0) {
        assert(k >= 1 && k <= 3);
        const HVX_Vector r0 = Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k, n));
        const HVX_Vector r1 =
            k > 1 ? Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k + k_stride, n))
                  : vzero;
        const HVX_Vector r2 =
            k > 2
                ? Q6_V_vand_QV(qn, xnn_load_tail_safe_s8(w_k + 2 * k_stride, n))
                : vzero;

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(r1, r0, -1);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vzero, r2, -1);
        const HVX_VectorPair q0_lo =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -2);
        *((HVX_UVector*) (out + 0)) = Q6_V_lo_W(q0_lo);
        *((HVX_UVector*) (out + 128)) = Q6_V_hi_W(q0_lo);
        vacc[0] = Q6_Vw_vrmpyacc_VwVbVb(vacc[0], Q6_V_lo_W(q0_lo), vones);
        vacc[1] = Q6_Vw_vrmpyacc_VwVbVb(vacc[1], Q6_V_hi_W(q0_lo), vones);
        const HVX_VectorPair q0_hi =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -2);
        *((HVX_UVector*) (out + 256)) = Q6_V_lo_W(q0_hi);
        *((HVX_UVector*) (out + 384)) = Q6_V_hi_W(q0_hi);
        vacc[2] = Q6_Vw_vrmpyacc_VwVbVb(vacc[2], Q6_V_lo_W(q0_hi), vones);
        vacc[3] = Q6_Vw_vrmpyacc_VwVbVb(vacc[3], Q6_V_hi_W(q0_hi), vones);
        out += 512;
      }

      for (size_t t = 0; t < 4; ++t) {
        const size_t n_off = t * 32;
        if (n_off >= n) {
          break;
        }
        const HVX_Vector vadj = xnn_mul_s32(vacc[t], vizp);
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
    weights += kc * k_stride;
  } while (--g != 0);
}
