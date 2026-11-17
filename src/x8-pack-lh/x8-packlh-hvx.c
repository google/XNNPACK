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
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

static XNN_INLINE HVX_Vector xnn_load_tail_u8_xor80(const int8_t* src,
                                                    size_t rem,
                                                    HVX_VectorPred qtail,
                                                    HVX_Vector v0x80,
                                                    HVX_Vector vzero) {
  assert(rem > 0 && rem < 128);
  const uintptr_t addr = (uintptr_t) src;
  const HVX_Vector* aligned_src = (const HVX_Vector*) (addr & ~(uintptr_t) 127);
  const size_t off = (size_t) (addr & 127u);
  const HVX_Vector v0 = aligned_src[0];
  const HVX_Vector v1 = (off + rem > 128) ? aligned_src[1] : v0;
  const HVX_Vector vraw = Q6_V_valign_VVR(v1, v0, addr);
  return Q6_V_vmux_QVV(qtail, Q6_V_vxor_VV(vraw, v0x80), vzero);
}

size_t xnn_x8_pack_lh_size__hvx(size_t m, size_t k, size_t mr_packed,
                                size_t kr, size_t sr) {
  (void) kr;
  (void) sr;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  }
  assert(mr_packed == 32);
  assert(kr == 4);
  assert(sr == 1);
  const size_t num_m_tiles = divide_round_up(m, mr_packed);
  const size_t num_k_tiles = divide_round_up(k, 32);
  return num_m_tiles * num_k_tiles * (mr_packed * 32);
}

size_t xnn_x8_pack_lh_offset__hvx(size_t m, size_t k, size_t mr_packed,
                                  size_t kr, size_t sr) {
  (void) kr;
  (void) sr;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  }
  assert(mr_packed == 32);
  assert(kr == 4);
  assert(sr == 1);
  assert(m % mr_packed == 0);
  const size_t num_k_tiles = divide_round_up(k, 32);
  return (m / mr_packed) * num_k_tiles * (mr_packed * 32);
}

void xnn_x8_pack_lh_ukernel__hvx(size_t m, size_t k, size_t mr_packed,
                                 size_t kr, size_t sr, size_t m_idx_start,
                                 const int8_t* XNN_RESTRICT lhs,
                                 size_t lhs_stride,
                                 void* XNN_RESTRICT lhs_packed) {
  (void) m_idx_start;
  assert(m_idx_start == 0);
  if (m == 0) {
    return;
  }
  if (mr_packed == 1) {
    if (lhs_stride == k * sizeof(int8_t)) {
      memcpy(lhs_packed, lhs, m * k * sizeof(int8_t));
    } else {
      uint8_t* dst = (uint8_t*) lhs_packed;
      const uint8_t* src = (const uint8_t*) lhs;
      for (size_t i = 0; i < m; ++i) {
        memcpy(dst + i * k, src + i * lhs_stride, k * sizeof(int8_t));
      }
    }
    return;
  }

  assert(mr_packed == 32);
  assert(kr == 4);
  assert(sr == 1);

  const size_t num_k_tiles = divide_round_up(k, 32);
  const size_t tail_k = k & 127u;
  const HVX_VectorPred qtail = Q6_Q_vsetq_R(tail_k);
  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector v0x80 = Q6_Vb_vsplat_R(-128);
  const bool aligned_dst = (((uintptr_t) lhs_packed & 127u) == 0u);
  HVX_Vector* XNN_RESTRICT packed_vec = (HVX_Vector*) lhs_packed;
  HVX_UVector* XNN_RESTRICT packed_uvec = (HVX_UVector*) lhs_packed;

  for (size_t i_m = 0; i_m < m; i_m += 32) {
    const size_t tile_m = min(m - i_m, (size_t) 32);
    const int8_t* tile_lhs =
        (const int8_t*) ((uintptr_t) lhs + i_m * lhs_stride);

    for (size_t g = 0; g < 8; ++g) {
      const size_t r_base = g * 4;
      if (r_base >= tile_m) {
        if XNN_LIKELY(aligned_dst) {
          for (size_t t = 0; t < num_k_tiles; ++t) {
            packed_vec[t * 8 + g] = vzero;
          }
        } else {
          for (size_t t = 0; t < num_k_tiles; ++t) {
            packed_uvec[t * 8 + g] = vzero;
          }
        }
        continue;
      }

      const int8_t* r0 = tile_lhs + (r_base + 0) * lhs_stride;
      const int8_t* r1 =
          (r_base + 1 < tile_m) ? (tile_lhs + (r_base + 1) * lhs_stride) : NULL;
      const int8_t* r2 =
          (r_base + 2 < tile_m) ? (tile_lhs + (r_base + 2) * lhs_stride) : NULL;
      const int8_t* r3 =
          (r_base + 3 < tile_m) ? (tile_lhs + (r_base + 3) * lhs_stride) : NULL;

      size_t rem_k = k;
      size_t t = 0;
      for (; rem_k >= 128; rem_k -= 128, t += 4) {
        const HVX_Vector vr0 = Q6_V_vxor_VV(*((const HVX_UVector*) r0), v0x80);
        r0 += 128;
        HVX_Vector vr1 = vzero;
        if (r1 != NULL) {
          vr1 = Q6_V_vxor_VV(*((const HVX_UVector*) r1), v0x80);
          r1 += 128;
        }
        HVX_Vector vr2 = vzero;
        if (r2 != NULL) {
          vr2 = Q6_V_vxor_VV(*((const HVX_UVector*) r2), v0x80);
          r2 += 128;
        }
        HVX_Vector vr3 = vzero;
        if (r3 != NULL) {
          vr3 = Q6_V_vxor_VV(*((const HVX_UVector*) r3), v0x80);
          r3 += 128;
        }

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(vr1, vr0, -32);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vr3, vr2, -32);
        const HVX_VectorPair q01 =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
        const HVX_VectorPair q23 =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
        if XNN_LIKELY(aligned_dst) {
          packed_vec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
          packed_vec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
          packed_vec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
          packed_vec[(t + 3) * 8 + g] = Q6_V_hi_W(q23);
        } else {
          packed_uvec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
          packed_uvec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
          packed_uvec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
          packed_uvec[(t + 3) * 8 + g] = Q6_V_hi_W(q23);
        }
      }

      if (rem_k != 0) {
        const HVX_Vector vr0 =
            xnn_load_tail_u8_xor80(r0, rem_k, qtail, v0x80, vzero);
        const HVX_Vector vr1 =
            (r1 != NULL) ? xnn_load_tail_u8_xor80(r1, rem_k, qtail, v0x80, vzero)
                         : vzero;
        const HVX_Vector vr2 =
            (r2 != NULL) ? xnn_load_tail_u8_xor80(r2, rem_k, qtail, v0x80, vzero)
                         : vzero;
        const HVX_Vector vr3 =
            (r3 != NULL) ? xnn_load_tail_u8_xor80(r3, rem_k, qtail, v0x80, vzero)
                         : vzero;

        const HVX_VectorPair p01 = Q6_W_vshuff_VVR(vr1, vr0, -32);
        const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vr3, vr2, -32);
        const HVX_VectorPair q01 =
            Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
        const HVX_VectorPair q23 =
            Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
        const size_t rem_tiles = (rem_k + 31) >> 5;
        if XNN_LIKELY(aligned_dst) {
          packed_vec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
          if (rem_tiles > 1) {
            packed_vec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
          }
          if (rem_tiles > 2) {
            packed_vec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
          }
          if (rem_tiles > 3) {
            packed_vec[(t + 3) * 8 + g] = Q6_V_hi_W(q23);
          }
        } else {
          packed_uvec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
          if (rem_tiles > 1) {
            packed_uvec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
          }
          if (rem_tiles > 2) {
            packed_uvec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
          }
          if (rem_tiles > 3) {
            packed_uvec[(t + 3) * 8 + g] = Q6_V_hi_W(q23);
          }
        }
      }
    }

    packed_vec += num_k_tiles * 8;
    packed_uvec += num_k_tiles * 8;
  }
}
