// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/c2-hvx.c.in
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

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"

// TODO(b/574286664): Support qc4w and qc8w weights (f16-qc4w-gemm and
// f16-qc8w-gemm) by unpacking/dequantizing 4-bit and 8-bit weights to FP16
// in HVX using the same KR=2 interleaved layout.

static XNN_INLINE void l2fetch_2d(
    const void* p,
    uint32_t stride,
    uint32_t width,
    uint32_t height) {
  const uint64_t control = ((uint64_t) stride << 32) |
                           ((uint64_t) width << 16) |
                           (uint64_t) height;
  __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(control));
}

void xnn_f16_gemm_minmax_ukernel_4x64c2__hvx(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params) {
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t min_bits = *((const uint16_t*) &params->scalar.min);
  const uint16_t max_bits = *((const uint16_t*) &params->scalar.max);
  const HVX_Vector v_min = Q6_Vh_vsplat_R(min_bits);
  const HVX_Vector v_max = Q6_Vh_vsplat_R(max_bits);
  const HVX_Vector v_one = Q6_Vh_vsplat_R(0x3c00);

  const uint8_t* a0 = (const uint8_t*) a;
  xnn_float16* c0 = c;
  const uint8_t* a1 = a0 + a_stride;
  xnn_float16* c1 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = a1 + a_stride;
  xnn_float16* c2 = (xnn_float16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = a2 + a_stride;
  xnn_float16* c3 = (xnn_float16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const HVX_Vector* restrict w_vec = (const HVX_Vector*) w;
  const uint32_t k_pairs = (uint32_t) ((kc + 2) >> 2);
  const int a_unaligned =
      ((((uintptr_t) a) | a_stride) & (sizeof(uint32_t) - 1)) != 0;

  do {
    l2fetch_2d(w_vec, 256, 256, k_pairs + 1);
    const HVX_Vector v_bias_raw = *w_vec++;
    HVX_Vector vacc0_0 = Q6_V_vzero();
    HVX_Vector vacc0_1 = Q6_V_vzero();
    HVX_Vector vacc1_0 = Q6_V_vzero();
    HVX_Vector vacc1_1 = Q6_V_vzero();
    HVX_Vector vacc2_0 = Q6_V_vzero();
    HVX_Vector vacc2_1 = Q6_V_vzero();
    HVX_Vector vacc3_0 = Q6_V_vzero();
    HVX_Vector vacc3_1 = Q6_V_vzero();

    size_t k = kc;
    if XNN_UNLIKELY(a_unaligned) {
      for (; k >= 2 * sizeof(xnn_float16); k -= 2 * sizeof(xnn_float16)) {
        const HVX_Vector va0 = Q6_V_vsplat_R(unaligned_load_u32(a0));
        a0 += sizeof(uint32_t);
        const HVX_Vector va1 = Q6_V_vsplat_R(unaligned_load_u32(a1));
        a1 += sizeof(uint32_t);
        const HVX_Vector va2 = Q6_V_vsplat_R(unaligned_load_u32(a2));
        a2 += sizeof(uint32_t);
        const HVX_Vector va3 = Q6_V_vsplat_R(unaligned_load_u32(a3));
        a3 += sizeof(uint32_t);
        const HVX_Vector vb0 = *w_vec++;
        const HVX_Vector vb1 = *w_vec++;

        const HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(va0, vb0);
        const HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(va0, vb1);
        vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
        const HVX_VectorPair p10 = Q6_Wqf32_vmpy_VhfVhf(va1, vb0);
        const HVX_VectorPair p11 = Q6_Wqf32_vmpy_VhfVhf(va1, vb1);
        vacc1_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc1_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p10), Q6_V_hi_W(p10)));
        vacc1_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc1_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p11), Q6_V_hi_W(p11)));
        const HVX_VectorPair p20 = Q6_Wqf32_vmpy_VhfVhf(va2, vb0);
        const HVX_VectorPair p21 = Q6_Wqf32_vmpy_VhfVhf(va2, vb1);
        vacc2_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc2_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p20), Q6_V_hi_W(p20)));
        vacc2_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc2_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p21), Q6_V_hi_W(p21)));
        const HVX_VectorPair p30 = Q6_Wqf32_vmpy_VhfVhf(va3, vb0);
        const HVX_VectorPair p31 = Q6_Wqf32_vmpy_VhfVhf(va3, vb1);
        vacc3_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc3_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p30), Q6_V_hi_W(p30)));
        vacc3_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc3_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p31), Q6_V_hi_W(p31)));
      }
    } else {
      for (; k >= 2 * sizeof(xnn_float16); k -= 2 * sizeof(xnn_float16)) {
        const HVX_Vector va0 = Q6_V_vsplat_R(*((const uint32_t*) a0));
        a0 += sizeof(uint32_t);
        const HVX_Vector va1 = Q6_V_vsplat_R(*((const uint32_t*) a1));
        a1 += sizeof(uint32_t);
        const HVX_Vector va2 = Q6_V_vsplat_R(*((const uint32_t*) a2));
        a2 += sizeof(uint32_t);
        const HVX_Vector va3 = Q6_V_vsplat_R(*((const uint32_t*) a3));
        a3 += sizeof(uint32_t);
        const HVX_Vector vb0 = *w_vec++;
        const HVX_Vector vb1 = *w_vec++;

        const HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(va0, vb0);
        const HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(va0, vb1);
        vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
        vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc0_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
        const HVX_VectorPair p10 = Q6_Wqf32_vmpy_VhfVhf(va1, vb0);
        const HVX_VectorPair p11 = Q6_Wqf32_vmpy_VhfVhf(va1, vb1);
        vacc1_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc1_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p10), Q6_V_hi_W(p10)));
        vacc1_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc1_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p11), Q6_V_hi_W(p11)));
        const HVX_VectorPair p20 = Q6_Wqf32_vmpy_VhfVhf(va2, vb0);
        const HVX_VectorPair p21 = Q6_Wqf32_vmpy_VhfVhf(va2, vb1);
        vacc2_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc2_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p20), Q6_V_hi_W(p20)));
        vacc2_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc2_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p21), Q6_V_hi_W(p21)));
        const HVX_VectorPair p30 = Q6_Wqf32_vmpy_VhfVhf(va3, vb0);
        const HVX_VectorPair p31 = Q6_Wqf32_vmpy_VhfVhf(va3, vb1);
        vacc3_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc3_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p30), Q6_V_hi_W(p30)));
        vacc3_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
            vacc3_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p31), Q6_V_hi_W(p31)));
      }
    }
    if XNN_UNLIKELY(k != 0) {
      const uint16_t a0_tail = *((const uint16_t*) a0);
      a0 += sizeof(uint16_t);
      const HVX_Vector va0 = Q6_V_vsplat_R((uint32_t) a0_tail);
      const uint16_t a1_tail = *((const uint16_t*) a1);
      a1 += sizeof(uint16_t);
      const HVX_Vector va1 = Q6_V_vsplat_R((uint32_t) a1_tail);
      const uint16_t a2_tail = *((const uint16_t*) a2);
      a2 += sizeof(uint16_t);
      const HVX_Vector va2 = Q6_V_vsplat_R((uint32_t) a2_tail);
      const uint16_t a3_tail = *((const uint16_t*) a3);
      a3 += sizeof(uint16_t);
      const HVX_Vector va3 = Q6_V_vsplat_R((uint32_t) a3_tail);
      const HVX_Vector vb0 = *w_vec++;
      const HVX_Vector vb1 = *w_vec++;

      const HVX_VectorPair p00 = Q6_Wqf32_vmpy_VhfVhf(va0, vb0);
      const HVX_VectorPair p01 = Q6_Wqf32_vmpy_VhfVhf(va0, vb1);
      vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc0_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p00), Q6_V_hi_W(p00)));
      vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc0_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p01), Q6_V_hi_W(p01)));
      const HVX_VectorPair p10 = Q6_Wqf32_vmpy_VhfVhf(va1, vb0);
      const HVX_VectorPair p11 = Q6_Wqf32_vmpy_VhfVhf(va1, vb1);
      vacc1_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc1_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p10), Q6_V_hi_W(p10)));
      vacc1_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc1_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p11), Q6_V_hi_W(p11)));
      const HVX_VectorPair p20 = Q6_Wqf32_vmpy_VhfVhf(va2, vb0);
      const HVX_VectorPair p21 = Q6_Wqf32_vmpy_VhfVhf(va2, vb1);
      vacc2_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc2_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p20), Q6_V_hi_W(p20)));
      vacc2_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc2_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p21), Q6_V_hi_W(p21)));
      const HVX_VectorPair p30 = Q6_Wqf32_vmpy_VhfVhf(va3, vb0);
      const HVX_VectorPair p31 = Q6_Wqf32_vmpy_VhfVhf(va3, vb1);
      vacc3_0 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc3_0, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p30), Q6_V_hi_W(p30)));
      vacc3_1 = Q6_Vqf32_vadd_Vqf32Vqf32(
          vacc3_1, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p31), Q6_V_hi_W(p31)));
    }

    const HVX_Vector v_bias_shuff = Q6_Vh_vshuff_Vh(v_bias_raw);
    const HVX_VectorPair b_pair = Q6_Wqf32_vmpy_VhfVhf(v_bias_shuff, v_one);
    vacc0_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0_0, Q6_V_lo_W(b_pair));
    vacc0_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0_1, Q6_V_hi_W(b_pair));
    HVX_Vector vout0 = Q6_Vh_vdeal_Vh(
        Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vacc0_1, vacc0_0)));
    vout0 = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vout0, v_min), v_max);
    vacc1_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1_0, Q6_V_lo_W(b_pair));
    vacc1_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1_1, Q6_V_hi_W(b_pair));
    HVX_Vector vout1 = Q6_Vh_vdeal_Vh(
        Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vacc1_1, vacc1_0)));
    vout1 = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vout1, v_min), v_max);
    vacc2_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2_0, Q6_V_lo_W(b_pair));
    vacc2_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc2_1, Q6_V_hi_W(b_pair));
    HVX_Vector vout2 = Q6_Vh_vdeal_Vh(
        Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vacc2_1, vacc2_0)));
    vout2 = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vout2, v_min), v_max);
    vacc3_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3_0, Q6_V_lo_W(b_pair));
    vacc3_1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc3_1, Q6_V_hi_W(b_pair));
    HVX_Vector vout3 = Q6_Vh_vdeal_Vh(
        Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vacc3_1, vacc3_0)));
    vout3 = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vout3, v_min), v_max);

    if XNN_LIKELY(nc >= 64) {
      *((HVX_UVector*) c0) = vout0;
      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);
      a0 -= kc;
      *((HVX_UVector*) c1) = vout1;
      c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);
      a1 -= kc;
      *((HVX_UVector*) c2) = vout2;
      c2 = (xnn_float16*) ((uintptr_t) c2 + cn_stride);
      a2 -= kc;
      *((HVX_UVector*) c3) = vout3;
      c3 = (xnn_float16*) ((uintptr_t) c3 + cn_stride);
      a3 -= kc;
      nc -= 64;
    } else {
      Q6_V_vstu_variable(c0, nc * sizeof(xnn_float16), vout0);
      Q6_V_vstu_variable(c1, nc * sizeof(xnn_float16), vout1);
      Q6_V_vstu_variable(c2, nc * sizeof(xnn_float16), vout2);
      Q6_V_vstu_variable(c3, nc * sizeof(xnn_float16), vout3);
      nc = 0;
    }
  } while (nc != 0);
}
