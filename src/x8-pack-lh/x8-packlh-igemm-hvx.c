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
#include "src/xnnpack/unaligned.h"

static XNN_INLINE const int8_t* xnn_igemm_row_ptr(const void* p,
                                                  size_t a_offset,
                                                  const void* zero) {
  return (p == zero) ? (const int8_t*) p
                     : (const int8_t*) ((uintptr_t) p + a_offset);
}

static XNN_INLINE HVX_Vector xnn_igemm_load_tail_u8_xor80(
    const int8_t* src, size_t rem, HVX_VectorPred qtail, HVX_Vector v0x80,
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

static XNN_INLINE uint32_t xnn_load_kc3_xor80(const int8_t* p) {
  if XNN_LIKELY(((uintptr_t) p & 4095u) <= 4092u) {
    return (unaligned_load_u32(p) & 0x00FFFFFFu) ^ 0x00808080u;
  }
  return (((uint32_t) (uint8_t) p[0]) | ((uint32_t) (uint8_t) p[1] << 8) |
          ((uint32_t) (uint8_t) p[2] << 16)) ^
         0x00808080u;
}

static XNN_INLINE uint32_t xnn_load_kc_le4_xor80(const int8_t* p, size_t kc) {
  switch (kc) {
    case 4:
      return unaligned_load_u32(p) ^ 0x80808080u;
    case 3:
      return xnn_load_kc3_xor80(p);
    case 2:
      return ((uint32_t) unaligned_load_u16(p)) ^ 0x00008080u;
    default:
      assert(kc == 1);
      return (uint32_t) ((uint8_t) p[0] ^ 0x80u);
  }
}

static XNN_INLINE HVX_Vector xnn_pack_8x32_to_32b_splat(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, uint32_t w4,
    uint32_t w5, uint32_t w6, uint32_t w7) {
  const HVX_Vector d01 = Q6_V_lo_W(
      Q6_W_vshuff_VVR(Q6_V_vsplat_R(w1), Q6_V_vsplat_R(w0), -4));
  const HVX_Vector d23 = Q6_V_lo_W(
      Q6_W_vshuff_VVR(Q6_V_vsplat_R(w3), Q6_V_vsplat_R(w2), -4));
  const HVX_Vector d45 = Q6_V_lo_W(
      Q6_W_vshuff_VVR(Q6_V_vsplat_R(w5), Q6_V_vsplat_R(w4), -4));
  const HVX_Vector d67 = Q6_V_lo_W(
      Q6_W_vshuff_VVR(Q6_V_vsplat_R(w7), Q6_V_vsplat_R(w6), -4));
  const HVX_Vector q0123 = Q6_V_lo_W(Q6_W_vshuff_VVR(d23, d01, -8));
  const HVX_Vector q4567 = Q6_V_lo_W(Q6_W_vshuff_VVR(d67, d45, -8));
  return Q6_V_lo_W(Q6_W_vshuff_VVR(q4567, q0123, -16));
}

size_t xnn_x8_pack_lh_size__igemm_hvx(size_t m, size_t kc, size_t ks,
                                      size_t mr_packed, size_t kr, size_t sr) {
  assert(kr == 4);
  assert(sr == 1);
  const size_t total_k = ks * round_up_po2(kc, kr * sr);
  if (mr_packed == 1) {
    return m * total_k;
  }
  assert(mr_packed == 32);
  const size_t num_m_tiles = divide_round_up(m, mr_packed);
  const size_t num_k_tiles = divide_round_up(total_k, 32);
  return num_m_tiles * num_k_tiles * (mr_packed * 32);
}

size_t xnn_x8_pack_lh_offset__igemm_hvx(size_t m, size_t kc, size_t ks,
                                        size_t mr_packed, size_t kr,
                                        size_t sr) {
  assert(kr == 4);
  assert(sr == 1);
  const size_t total_k = ks * round_up_po2(kc, kr * sr);
  if (mr_packed == 1) {
    return m * total_k;
  }
  assert(mr_packed == 32);
  assert(m % mr_packed == 0);
  const size_t num_k_tiles = divide_round_up(total_k, 32);
  return (m / mr_packed) * num_k_tiles * (mr_packed * 32);
}

void xnn_x8_pack_lh_ukernel__igemm_hvx(size_t m, size_t kc, size_t ks,
                                       size_t mr_packed, size_t kr, size_t sr,
                                       const void** XNN_RESTRICT a,
                                       size_t a_offset, const void* zero,
                                       void* XNN_RESTRICT lhs_packed) {
  assert(kr == 4);
  assert(sr == 1);
  if (m == 0 || ks == 0 || kc == 0) {
    return;
  }

  const size_t kc_aligned = round_up_po2(kc, 4);
  const size_t total_k = ks * kc_aligned;

  if (mr_packed == 1) {
    uint8_t* dst = (uint8_t*) lhs_packed;
    for (size_t row = 0; row < m; ++row) {
      for (size_t p = 0; p < ks; ++p) {
        const int8_t* src =
            xnn_igemm_row_ptr(a[row * ks + p], a_offset, zero);
        memcpy(dst, src, kc);
        if (kc < kc_aligned) {
          memset(dst + kc, 0x80, kc_aligned - kc);
        }
        dst += kc_aligned;
      }
    }
    return;
  }

  assert(mr_packed == 32);
  const size_t num_k_tiles = divide_round_up(total_k, 32);
  const size_t tail_kc = kc & 127u;
  const HVX_VectorPred qtail_kc = Q6_Q_vsetq_R(tail_kc);
  const HVX_Vector vzero = Q6_V_vzero();
  const HVX_Vector v0x80 = Q6_Vb_vsplat_R(-128);
  const bool aligned_dst = (((uintptr_t) lhs_packed & 127u) == 0u);
  HVX_Vector* XNN_RESTRICT packed_vec = (HVX_Vector*) lhs_packed;
  HVX_UVector* XNN_RESTRICT packed_uvec = (HVX_UVector*) lhs_packed;

  for (size_t i_m = 0; i_m < m; i_m += 32) {
    const size_t tile_m = min(m - i_m, (size_t) 32);
    const void** tile_a = a + i_m * ks;

    if (ks == 1) {
      // Fast path 1: 1x1 convs (`ks == 1`) weave 4 rows directly via HVX.
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
        const int8_t* r0 =
            xnn_igemm_row_ptr(tile_a[r_base + 0], a_offset, zero);
        const int8_t* r1 =
            (r_base + 1 < tile_m)
                ? xnn_igemm_row_ptr(tile_a[r_base + 1], a_offset, zero)
                : NULL;
        const int8_t* r2 =
            (r_base + 2 < tile_m)
                ? xnn_igemm_row_ptr(tile_a[r_base + 2], a_offset, zero)
                : NULL;
        const int8_t* r3 =
            (r_base + 3 < tile_m)
                ? xnn_igemm_row_ptr(tile_a[r_base + 3], a_offset, zero)
                : NULL;

        size_t rem_k = kc;
        size_t t = 0;
        for (; rem_k >= 128; rem_k -= 128, t += 4) {
          const HVX_Vector vr0 =
              Q6_V_vxor_VV(*((const HVX_UVector*) r0), v0x80);
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
          const HVX_Vector vr0 = xnn_igemm_load_tail_u8_xor80(
              r0, rem_k, qtail_kc, v0x80, vzero);
          const HVX_Vector vr1 =
              (r1 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                 r1, rem_k, qtail_kc, v0x80, vzero)
                           : vzero;
          const HVX_Vector vr2 =
              (r2 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                 r2, rem_k, qtail_kc, v0x80, vzero)
                           : vzero;
          const HVX_Vector vr3 =
              (r3 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                 r3, rem_k, qtail_kc, v0x80, vzero)
                           : vzero;

          const HVX_VectorPair p01 = Q6_W_vshuff_VVR(vr1, vr0, -32);
          const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vr3, vr2, -32);
          const HVX_VectorPair q01 =
              Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
          const HVX_VectorPair q23 =
              Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
          const size_t rem_tiles = num_k_tiles - t;
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
    } else if (kc <= 4) {
      // Fast path 2: Shallow input convs (`kc <= 4`, e.g. RGB `kc == 3`).
      // Assemble 8 `ks` steps x 4 aligned bytes = 32 bytes per row directly in
      // HVX registers via `vsplat` + `vshuff` (zero stack buffers or L1D snoops).
      for (size_t t = 0; t < num_k_tiles; ++t) {
        const size_t p_start = t * 8;
        const size_t p_count = min(ks - p_start, (size_t) 8);
        const void** t_a = tile_a + p_start * 32;

        for (size_t g = 0; g < 8; ++g) {
          const size_t r_base = g * 4;
          if (r_base >= tile_m) {
            if XNN_LIKELY(aligned_dst) {
              packed_vec[t * 8 + g] = vzero;
            } else {
              packed_uvec[t * 8 + g] = vzero;
            }
            continue;
          }

          const size_t active_rows = min(tile_m - r_base, (size_t) 4);
          HVX_Vector vr[4] = {vzero, vzero, vzero, vzero};

          if (kc == 3 && p_count == 8) {
            for (size_t sub = 0; sub < active_rows; ++sub) {
              const void** r_a = t_a + r_base + sub;
              const uint32_t w0 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[0 * 32], a_offset, zero));
              const uint32_t w1 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[1 * 32], a_offset, zero));
              const uint32_t w2 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[2 * 32], a_offset, zero));
              const uint32_t w3 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[3 * 32], a_offset, zero));
              const uint32_t w4 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[4 * 32], a_offset, zero));
              const uint32_t w5 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[5 * 32], a_offset, zero));
              const uint32_t w6 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[6 * 32], a_offset, zero));
              const uint32_t w7 = xnn_load_kc3_xor80(
                  xnn_igemm_row_ptr(r_a[7 * 32], a_offset, zero));
              vr[sub] =
                  xnn_pack_8x32_to_32b_splat(w0, w1, w2, w3, w4, w5, w6, w7);
            }
          } else {
            for (size_t sub = 0; sub < active_rows; ++sub) {
              const void** r_a = t_a + r_base + sub;
              uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
              if (kc == 3) {
                for (size_t dp = 0; dp < p_count; ++dp) {
                  w[dp] = xnn_load_kc3_xor80(
                      xnn_igemm_row_ptr(r_a[dp * 32], a_offset, zero));
                }
              } else {
                for (size_t dp = 0; dp < p_count; ++dp) {
                  w[dp] = xnn_load_kc_le4_xor80(
                      xnn_igemm_row_ptr(r_a[dp * 32], a_offset, zero), kc);
                }
              }
              vr[sub] = xnn_pack_8x32_to_32b_splat(w[0], w[1], w[2], w[3], w[4],
                                                   w[5], w[6], w[7]);
            }
          }

          const HVX_Vector v01 = Q6_V_lo_W(Q6_W_vshuff_VVR(vr[1], vr[0], -32));
          const HVX_Vector v23 = Q6_V_lo_W(Q6_W_vshuff_VVR(vr[3], vr[2], -32));
          const HVX_Vector vout = Q6_V_lo_W(Q6_W_vshuff_VVR(v23, v01, -64));
          if XNN_LIKELY(aligned_dst) {
            packed_vec[t * 8 + g] = vout;
          } else {
            packed_uvec[t * 8 + g] = vout;
          }
        }
      }
    } else if ((kc & 31) == 0) {
      // Fast path 3: `kc` is a multiple of 32 (e.g. `kc = 32, 64, 128, 256,
      // 512`), so each `ks` step aligns on a 32-byte K-tile boundary.
      const size_t tiles_per_p = kc >> 5;
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
        for (size_t p = 0; p < ks; ++p) {
          const void** p_a = tile_a + p * 32 + r_base;
          const int8_t* r0 = xnn_igemm_row_ptr(p_a[0], a_offset, zero);
          const int8_t* r1 =
              (r_base + 1 < tile_m)
                  ? xnn_igemm_row_ptr(p_a[1], a_offset, zero)
                  : NULL;
          const int8_t* r2 =
              (r_base + 2 < tile_m)
                  ? xnn_igemm_row_ptr(p_a[2], a_offset, zero)
                  : NULL;
          const int8_t* r3 =
              (r_base + 3 < tile_m)
                  ? xnn_igemm_row_ptr(p_a[3], a_offset, zero)
                  : NULL;

          size_t rem_k = kc;
          size_t t = p * tiles_per_p;
          for (; rem_k >= 128; rem_k -= 128, t += 4) {
            const HVX_Vector vr0 =
                Q6_V_vxor_VV(*((const HVX_UVector*) r0), v0x80);
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
            const HVX_Vector vr0 = xnn_igemm_load_tail_u8_xor80(
                r0, rem_k, qtail_kc, v0x80, vzero);
            const HVX_Vector vr1 =
                (r1 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                   r1, rem_k, qtail_kc, v0x80, vzero)
                             : vzero;
            const HVX_Vector vr2 =
                (r2 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                   r2, rem_k, qtail_kc, v0x80, vzero)
                             : vzero;
            const HVX_Vector vr3 =
                (r3 != NULL) ? xnn_igemm_load_tail_u8_xor80(
                                   r3, rem_k, qtail_kc, v0x80, vzero)
                             : vzero;

            const HVX_VectorPair p01 = Q6_W_vshuff_VVR(vr1, vr0, -32);
            const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vr3, vr2, -32);
            const HVX_VectorPair q01 =
                Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
            const HVX_VectorPair q23 =
                Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
            const size_t rem_tiles = rem_k >> 5;
            if XNN_LIKELY(aligned_dst) {
              packed_vec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
              if (rem_tiles > 1) {
                packed_vec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
              }
              if (rem_tiles > 2) {
                packed_vec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
              }
            } else {
              packed_uvec[(t + 0) * 8 + g] = Q6_V_lo_W(q01);
              if (rem_tiles > 1) {
                packed_uvec[(t + 1) * 8 + g] = Q6_V_hi_W(q01);
              }
              if (rem_tiles > 2) {
                packed_uvec[(t + 2) * 8 + g] = Q6_V_lo_W(q23);
              }
            }
          }
        }
      }
    } else {
      // General path: stream 128 flattened K-bytes at a time for 4 rows, then
      // weave via HVX `vshuff`. Handles arbitrarily large `ks * kc_aligned`
      // without heap allocation or `vstu_variable`.
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
        const size_t active_rows = min(tile_m - r_base, (size_t) 4);
        size_t cur_p = 0;
        size_t cur_off = 0;
        size_t t = 0;
        for (size_t flat_pos = 0; flat_pos < total_k;
             flat_pos += 128, t += 4) {
          const size_t chunk_len = min(total_k - flat_pos, (size_t) 128);
          XNN_ALIGN(128) uint8_t rbuf[4][128];

          size_t next_p = cur_p;
          size_t next_off = cur_off;
          for (size_t sub = 0; sub < active_rows; ++sub) {
            const size_t row = r_base + sub;
            size_t p = cur_p;
            size_t off = cur_off;
            size_t dst_pos = 0;
            while (dst_pos < chunk_len) {
              const size_t step_rem = kc_aligned - off;
              const size_t take = min(chunk_len - dst_pos, step_rem);
              const int8_t* src_base =
                  xnn_igemm_row_ptr(tile_a[p * 32 + row], a_offset, zero);
              size_t pos = off;
              const size_t end_pos = off + take;
              if (pos < kc) {
                const size_t valid_end = min(end_pos, kc);
                const size_t valid_cnt = valid_end - pos;
                memcpy(&rbuf[sub][dst_pos], src_base + pos, valid_cnt);
                dst_pos += valid_cnt;
                pos = valid_end;
              }
              if (pos < end_pos) {
                const size_t pad_cnt = end_pos - pos;
                memset(&rbuf[sub][dst_pos], 0x80, pad_cnt);
                dst_pos += pad_cnt;
              }
              off = end_pos;
              if (off == kc_aligned) {
                off = 0;
                ++p;
              }
            }
            if (chunk_len < 128) {
              memset(&rbuf[sub][chunk_len], 0x80, 128 - chunk_len);
            }
            next_p = p;
            next_off = off;
          }
          cur_p = next_p;
          cur_off = next_off;

          const HVX_Vector vr0 =
              Q6_V_vxor_VV(*((const HVX_Vector*) rbuf[0]), v0x80);
          const HVX_Vector vr1 =
              (active_rows > 1)
                  ? Q6_V_vxor_VV(*((const HVX_Vector*) rbuf[1]), v0x80)
                  : vzero;
          const HVX_Vector vr2 =
              (active_rows > 2)
                  ? Q6_V_vxor_VV(*((const HVX_Vector*) rbuf[2]), v0x80)
                  : vzero;
          const HVX_Vector vr3 =
              (active_rows > 3)
                  ? Q6_V_vxor_VV(*((const HVX_Vector*) rbuf[3]), v0x80)
                  : vzero;

          const HVX_VectorPair p01 = Q6_W_vshuff_VVR(vr1, vr0, -32);
          const HVX_VectorPair p23 = Q6_W_vshuff_VVR(vr3, vr2, -32);
          const HVX_VectorPair q01 =
              Q6_W_vshuff_VVR(Q6_V_lo_W(p23), Q6_V_lo_W(p01), -64);
          const HVX_VectorPair q23 =
              Q6_W_vshuff_VVR(Q6_V_hi_W(p23), Q6_V_hi_W(p01), -64);
          const size_t rem_tiles = (chunk_len + 31) >> 5;
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
    }

    packed_vec += num_k_tiles * 8;
    packed_uvec += num_k_tiles * 8;
  }
}
