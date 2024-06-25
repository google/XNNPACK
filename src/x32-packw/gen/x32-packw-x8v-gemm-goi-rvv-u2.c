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

void xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u2(
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
      // vlsseg2, LMUL must <= 4
      vlmax = __riscv_vsetvlmax_e32m4();
      // Pack 2 x nr weights
      for (; k >= 2; k -= 2) {
        uint32_t* out1 = out0 + nr;
        // When vlsseg2, LMUL is contraint to 4. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        size_t remaining_n = nr;
        do {
          vuint32m4x2_t v_w_m4x2 = __riscv_vlsseg2e32_v_u32m4x2(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vlmax;
          vuint32m4_t v_w0 = __riscv_vget_v_u32m4x2_u32m4(v_w_m4x2, 0);
          __riscv_vse32_v_u32m4(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m4_t v_w1 = __riscv_vget_v_u32m4x2_u32m4(v_w_m4x2, 1);
          __riscv_vse32_v_u32m4(out1, v_w1, vlmax); out1 += vlmax;
          remaining_n -= vlmax;
        } while(remaining_n > 0);
        out0 = out1;
        w0 += 2;
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
      // vlsseg2, LMUL must <= 4
      vlmax = __riscv_vsetvlmax_e32m4();
      // Pack 2 x n weights
       for (; k >= 2; k -= 2) {
        uint32_t* out1 = out0 + nr;
        // When vlsseg2, LMUL is contraint to 4. We need to use multiple of load & store.
        const uint32_t* w_ptr = w0;
        unsigned char remaining_blocks = 2;
        size_t remaining_n = n;
        do {
          if XNN_LIKELY(remaining_n >= vlmax) {
            vl = vlmax;
          } else {
            vl = __riscv_vsetvl_e32m4(remaining_n);
          }
          vuint32m4x2_t v_w_m4x2 = __riscv_vlsseg2e32_v_u32m4x2(w_ptr, kc_bstride, vlmax);
          w_ptr += kc * vl;
          vuint32m4_t v_w0 = __riscv_vget_v_u32m4x2_u32m4(v_w_m4x2, 0);
          __riscv_vse32_v_u32m4(out0, v_w0, vlmax); out0 += vlmax;
          vuint32m4_t v_w1 = __riscv_vget_v_u32m4x2_u32m4(v_w_m4x2, 1);
          __riscv_vse32_v_u32m4(out1, v_w1, vlmax); out1 += vlmax;
          remaining_n -= vl;
          remaining_blocks--;
        } while(remaining_n > 0);
        out0 = out1 + remaining_blocks * vlmax;
        w0 += 2;
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
