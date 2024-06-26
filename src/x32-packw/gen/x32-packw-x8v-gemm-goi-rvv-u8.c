// Auto-generated file. Do not edit!
//   Template: src/x32-packw/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <riscv_vector.h>

#include "xnnpack/packw.h"

void xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u8(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
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
  assert(nr == __riscv_vsetvlmax_e32m8());
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  uint32_t* out = packed_weights;
  const uint32_t* b = bias;
  size_t kc_bstride = kc * 4;

  do {
    const uint32_t* w0 = weights;
    size_t n = nc;
    // NC main loop: process multiple of NR
    for (;n >= nr; n -= nr) {
      // Pack nr bias at begining of tile
      size_t vlmax = __riscv_vsetvlmax_e32m8();
      vuint32m8_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m8(b, vlmax); b += nr;
      } else {
        v_bias = __riscv_vmv_v_x_u32m8(0, vlmax);
      }
      __riscv_vse32_v_u32m8(out, v_bias, vlmax); out += nr;

      uint32_t* out0 = out;
      size_t k = kc;
      // vlsseg8, LMUL must <= 1
      vlmax = __riscv_vsetvlmax_e32m1();
      // Pack 8 x nr weights
      for (; k >= 8; k -= 8) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        uint32_t* out4 = out3 + nr;
        uint32_t* out5 = out4 + nr;
        uint32_t* out6 = out5 + nr;
        uint32_t* out7 = out6 + nr;
        // When vlsseg8, LMUL is contraint to 1. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint32m1x8_t v_w_m1x8 = __riscv_vlsseg8e32_v_u32m1x8(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;
          vuint32m1_t v_w0 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 0);
          __riscv_vse32_v_u32m1(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m1_t v_w1 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 1);
          __riscv_vse32_v_u32m1(out1, v_w1, vlmax); out1 += vlmax;
          vuint32m1_t v_w2 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 2);
          __riscv_vse32_v_u32m1(out2, v_w2, vlmax); out2 += vlmax;
          vuint32m1_t v_w3 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 3);
          __riscv_vse32_v_u32m1(out3, v_w3, vlmax); out3 += vlmax;
          vuint32m1_t v_w4 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 4);
          __riscv_vse32_v_u32m1(out4, v_w4, vlmax); out4 += vlmax;
          vuint32m1_t v_w5 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 5);
          __riscv_vse32_v_u32m1(out5, v_w5, vlmax); out5 += vlmax;
          vuint32m1_t v_w6 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 6);
          __riscv_vse32_v_u32m1(out6, v_w6, vlmax); out6 += vlmax;
          vuint32m1_t v_w7 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 7);
          __riscv_vse32_v_u32m1(out7, v_w7, vlmax); out7 += vlmax;
          remaining_n -= vlmax;
        } while(remaining_n > 0);
        out0 = out7;
        w0 += 8;
      }
      // vlsseg4, LMUL must <= 2
      vlmax = __riscv_vsetvlmax_e32m2();
      // Pack 4 x nr weights
      for (; k >= 4; k -= 4) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        // When vlsseg4, LMUL is contraint to 2. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint32m2x4_t v_w_m2x4 = __riscv_vlsseg4e32_v_u32m2x4(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;
          vuint32m2_t v_w0 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 0);
          __riscv_vse32_v_u32m2(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m2_t v_w1 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 1);
          __riscv_vse32_v_u32m2(out1, v_w1, vlmax); out1 += vlmax;
          vuint32m2_t v_w2 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 2);
          __riscv_vse32_v_u32m2(out2, v_w2, vlmax); out2 += vlmax;
          vuint32m2_t v_w3 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 3);
          __riscv_vse32_v_u32m2(out3, v_w3, vlmax); out3 += vlmax;
          remaining_n -= vlmax;
        } while(remaining_n > 0);
        out0 = out3;
        w0 += 4;
      }
      vlmax = __riscv_vsetvlmax_e32m8();
      // Pack nr weights
      for (; k >= 1; k -= 1) {
        vuint32m8_t v_w = __riscv_vlse32_v_u32m8(w0, kc_bstride, vlmax);
        __riscv_vse32_v_u32m8(out0, v_w, vlmax);
        out0 += vlmax;
        w0 += 1;
      }
      out = (uint32_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc;
    }
    // NC remainder: process n < NR
    if (n > 0) {
      // Pack nr bias at begining of tile
      size_t vl = __riscv_vsetvl_e32m8(n);
      vuint32m8_t v_bias;
      if XNN_LIKELY(b != NULL) {
        v_bias = __riscv_vle32_v_u32m8(b, vl); b += vl;
      } else {
        v_bias = __riscv_vmv_v_x_u32m8(0, vl);
      }
      __riscv_vse32_v_u32m8(out, v_bias, vl); out += nr;

      size_t vlmax;
      uint32_t* out0 = out;
      size_t k = kc;
      // vlsseg8, LMUL must <= 1
      vlmax = __riscv_vsetvlmax_e32m1();
      // Pack 8 x n weights
      for (; k >= 8; k -= 8) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        uint32_t* out4 = out3 + nr;
        uint32_t* out5 = out4 + nr;
        uint32_t* out6 = out5 + nr;
        uint32_t* out7 = out6 + nr;
        // When vlsseg8, LMUL is contraint to 1. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        unsigned char remaining_blocks = 8;
        size_t remaining_n = n;
        do {
          size_t vl;
          if XNN_LIKELY(remaining_n >= vlmax) {
            vl = vlmax;
          } else {
            vl = __riscv_vsetvl_e32m1(remaining_n);
          }
          vuint32m1x8_t v_w_m1x8 = __riscv_vlsseg8e32_v_u32m1x8(w_ptr, kc_bstride, vl);
          w_ptr += kc * vl;
          vuint32m1_t v_w0 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 0);
          __riscv_vse32_v_u32m1(out0, v_w0, vl); out0 += vlmax;
          vuint32m1_t v_w1 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 1);
          __riscv_vse32_v_u32m1(out1, v_w1, vl); out1 += vlmax;
          vuint32m1_t v_w2 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 2);
          __riscv_vse32_v_u32m1(out2, v_w2, vl); out2 += vlmax;
          vuint32m1_t v_w3 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 3);
          __riscv_vse32_v_u32m1(out3, v_w3, vl); out3 += vlmax;
          vuint32m1_t v_w4 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 4);
          __riscv_vse32_v_u32m1(out4, v_w4, vl); out4 += vlmax;
          vuint32m1_t v_w5 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 5);
          __riscv_vse32_v_u32m1(out5, v_w5, vl); out5 += vlmax;
          vuint32m1_t v_w6 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 6);
          __riscv_vse32_v_u32m1(out6, v_w6, vl); out6 += vlmax;
          vuint32m1_t v_w7 = __riscv_vget_v_u32m1x8_u32m1(v_w_m1x8, 7);
          __riscv_vse32_v_u32m1(out7, v_w7, vl); out7 += vlmax;
          remaining_n -= vl;
          remaining_blocks--;
        } while(remaining_n > 0);
        out0 = out7 + remaining_blocks * vlmax;
        w0 += 8;
      }
      // vlsseg4, LMUL must <= 2
      vlmax = __riscv_vsetvlmax_e32m2();
      // Pack 4 x n weights
      for (; k >= 4; k -= 4) {
        uint32_t* out1 = out0 + nr;
        uint32_t* out2 = out1 + nr;
        uint32_t* out3 = out2 + nr;
        // When vlsseg4, LMUL is contraint to 2. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        unsigned char remaining_blocks = 4;
        size_t remaining_n = n;
        do {
          size_t vl;
          if XNN_LIKELY(remaining_n >= vlmax) {
            vl = vlmax;
          } else {
            vl = __riscv_vsetvl_e32m2(remaining_n);
          }
          vuint32m2x4_t v_w_m2x4 = __riscv_vlsseg4e32_v_u32m2x4(w_ptr, kc_bstride, vl);
          w_ptr += kc * vl;
          vuint32m2_t v_w0 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 0);
          __riscv_vse32_v_u32m2(out0, v_w0, vl); out0 += vlmax;
          vuint32m2_t v_w1 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 1);
          __riscv_vse32_v_u32m2(out1, v_w1, vl); out1 += vlmax;
          vuint32m2_t v_w2 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 2);
          __riscv_vse32_v_u32m2(out2, v_w2, vl); out2 += vlmax;
          vuint32m2_t v_w3 = __riscv_vget_v_u32m2x4_u32m2(v_w_m2x4, 3);
          __riscv_vse32_v_u32m2(out3, v_w3, vl); out3 += vlmax;
          remaining_n -= vl;
          remaining_blocks--;
        } while(remaining_n > 0);
        out0 = out3 + remaining_blocks * vlmax;
        w0 += 4;
      }
      vlmax = __riscv_vsetvlmax_e32m8();
      vl = __riscv_vsetvl_e32m8(n);
      // Pack n weights
      for (; k >= 1; k -= 1) {
        vuint32m8_t v_w = __riscv_vlse32_v_u32m8(w0, kc_bstride, vl);
        __riscv_vse32_v_u32m8(out0, v_w, vl);
        out0 += vlmax;
        w0 += 1;
      }
      out = (uint32_t*) ((uintptr_t) out0 + extra_bytes);
      w0 += (nr - 1) * kc; 
    }
    weights += nc * kc;
  } while (--g != 0);
}
