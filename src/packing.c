// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <fp16/fp16.h>

#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/unaligned.h>

#if XNN_ARCH_ARM64
#include <arm_neon.h>
#endif

void xnn_pack_f32_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = fp16_ieee_from_fp32_value(b[nr_block_start + nr_block_offset]);
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = fp16_ieee_from_fp32_value(k[(nr_block_start + nr_block_offset) * kc + kc_idx]);
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (int32_t) kv;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*) packed_weights)[kr_block_offset] = kv;
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

#if XNN_ARCH_ARM64
inline void xnn_pack_qs8_qc4w_gemm_bl_goi_w_nr8_kr4(
  size_t g,
  size_t nc,
  size_t kc,
  size_t bl,
  const uint8_t* k,
  const float* b,
  const float* scale,
  void* packed_weights,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  // Specialized packing routine for these params
  const size_t nr = 8;
  const size_t kr = 4;
  const size_t sr = 1;

  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(k != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8);
  assert(bl != 0);
  assert(round_up_po2(kc, kr) % bl == 0); // must be round number of blocks inside a column
  assert(bl % kr == 0); // must be round number of kr
  assert(bl <= round_up_po2(kc, kr));
  assert(2 * kr <= bl); // must be at least two kr to avoid back-to-back empty_bytes

  // TODO add support for this
  assert(nc % nr == 0);

  const int32_t izp = (int32_t) params->input_zero_point;
  const size_t num_blocks = round_up_po2(kc, kr) / bl;

  const int32_t shift_const = 16 * 4;
  const uint64_t and_mask = 0xF0F0F0F00F0F0F0F;
  const uint64_t shifts = 0x0000FCFC04040000;
  const uint64_t tbl_keys = 0x0501040007030602;
  const uint64_t eor_mask = 0x0000000088888888;
  const uint64_t and_mask2 = 0xFFFFFFFF00000000;

  const uint8x8_t v_zero = vdup_n_u8(0x0);
  const uint8x8_t v_and_mask = vcreate_u8(and_mask);
  const int8x8_t v_shifts = vcreate_s8(shifts);
  const uint8x8_t v_tbl_keys = vcreate_u8(tbl_keys);
  const uint8x8_t v_eor_mask = vcreate_u8(eor_mask);
  const uint8x8_t v_and_mask2 = vcreate_u8(and_mask2);

  do { // g loop
    size_t nr_block_start = 0;
    do { // NC loop
      float* packed_b = (float*) packed_weights;
      packed_weights = (void*) ((uintptr_t) packed_weights + /*ksums=*/nr * sizeof(float));

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, kr * 2); kr_block_start += kr * 2) {
        assert(kr_block_start == round_down_po2(kr_block_start, kr));
        const size_t block_index = kr_block_start / bl;

        // Better than spills?
        int32_t ksum[/*nr=*/8];
        float ksum_muls[/*nr=*/8];  // Multipliers for ksum
        float scales[/*nr=*/8];
        const uint8_t* ip[/*nr=*/8];
        uint8_t* op[/*nr=*/8];

        // TODO simplify/simdify this
        for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
          ksum[nr_block_offset] = 0;
          // TODO prefetch next cache-line in the same row
          ip[nr_block_offset] =
            (const uint8_t*) ((uintptr_t) k + (((nr_block_start + nr_block_offset) * kc + kr_block_start) >> 1));
          op[nr_block_offset] = (uint8_t*)((uintptr_t) packed_weights + nr_block_offset * kr);  // 2 nibbles
          scales[nr_block_offset] = scale[(nr_block_start + nr_block_offset) * num_blocks + block_index];
          ksum_muls[nr_block_offset] = -1 * (float) izp * 16 * scales[nr_block_offset];
        }

        uint8x8_t w0x0 = vld1_u8(ip[0]);
        uint8x8_t w0x1 = vld1_u8(ip[1]);
        uint8x8_t w0x2 = vld1_u8(ip[2]);
        uint8x8_t w0x3 = vld1_u8(ip[3]);
        uint8x8_t w0x4 = vld1_u8(ip[4]);
        uint8x8_t w0x5 = vld1_u8(ip[5]);
        uint8x8_t w0x6 = vld1_u8(ip[6]);
        uint8x8_t w0x7 = vld1_u8(ip[7]);

        uint8x8_t w1x0 = vext_u8(v_zero, w0x0, 4);
        uint8x8_t w1x1 = vext_u8(v_zero, w0x1, 4);
        uint8x8_t w1x2 = vext_u8(v_zero, w0x2, 4);
        uint8x8_t w1x3 = vext_u8(v_zero, w0x3, 4);
        uint8x8_t w1x4 = vext_u8(v_zero, w0x4, 4);
        uint8x8_t w1x5 = vext_u8(v_zero, w0x5, 4);
        uint8x8_t w1x6 = vext_u8(v_zero, w0x6, 4);
        uint8x8_t w1x7 = vext_u8(v_zero, w0x7, 4);

        uint8x8_t w2x0 = vext_u8(w1x0, w0x0, 4);
        uint8x8_t w2x1 = vext_u8(w1x1, w0x1, 4);
        uint8x8_t w2x2 = vext_u8(w1x2, w0x2, 4);
        uint8x8_t w2x3 = vext_u8(w1x3, w0x3, 4);
        uint8x8_t w2x4 = vext_u8(w1x4, w0x4, 4);
        uint8x8_t w2x5 = vext_u8(w1x5, w0x5, 4);
        uint8x8_t w2x6 = vext_u8(w1x6, w0x6, 4);
        uint8x8_t w2x7 = vext_u8(w1x7, w0x7, 4);

        uint8x8_t w3x0 = vand_u8(w2x0, v_and_mask);
        uint8x8_t w3x1 = vand_u8(w2x1, v_and_mask);
        uint8x8_t w3x2 = vand_u8(w2x2, v_and_mask);
        uint8x8_t w3x3 = vand_u8(w2x3, v_and_mask);
        uint8x8_t w3x4 = vand_u8(w2x4, v_and_mask);
        uint8x8_t w3x5 = vand_u8(w2x5, v_and_mask);
        uint8x8_t w3x6 = vand_u8(w2x6, v_and_mask);
        uint8x8_t w3x7 = vand_u8(w2x7, v_and_mask);

        uint8x8_t w4x0 = vshl_u8(w3x0, v_shifts);
        uint8x8_t w4x1 = vshl_u8(w3x1, v_shifts);
        uint8x8_t w4x2 = vshl_u8(w3x2, v_shifts);
        uint8x8_t w4x3 = vshl_u8(w3x3, v_shifts);
        uint8x8_t w4x4 = vshl_u8(w3x4, v_shifts);
        uint8x8_t w4x5 = vshl_u8(w3x5, v_shifts);
        uint8x8_t w4x6 = vshl_u8(w3x6, v_shifts);
        uint8x8_t w4x7 = vshl_u8(w3x7, v_shifts);

        uint8x8_t w5x0 = vtbl1_u8(w4x0, v_tbl_keys);
        uint8x8_t w5x1 = vtbl1_u8(w4x1, v_tbl_keys);
        uint8x8_t w5x2 = vtbl1_u8(w4x2, v_tbl_keys);
        uint8x8_t w5x3 = vtbl1_u8(w4x3, v_tbl_keys);
        uint8x8_t w5x4 = vtbl1_u8(w4x4, v_tbl_keys);
        uint8x8_t w5x5 = vtbl1_u8(w4x5, v_tbl_keys);
        uint8x8_t w5x6 = vtbl1_u8(w4x6, v_tbl_keys);
        uint8x8_t w5x7 = vtbl1_u8(w4x7, v_tbl_keys);

        uint8x8_t w51x0 = vshr_n_u8(w5x0, 4);
        uint8x8_t w51x1 = vshr_n_u8(w5x1, 4);
        uint8x8_t w51x2 = vshr_n_u8(w5x2, 4);
        uint8x8_t w51x3 = vshr_n_u8(w5x3, 4);
        uint8x8_t w51x4 = vshr_n_u8(w5x4, 4);
        uint8x8_t w51x5 = vshr_n_u8(w5x5, 4);
        uint8x8_t w51x6 = vshr_n_u8(w5x6, 4);
        uint8x8_t w51x7 = vshr_n_u8(w5x7, 4);

        uint8x8_t w52x0 = vand_u8(w5x0, v_and_mask2);
        uint8x8_t w52x1 = vand_u8(w5x1, v_and_mask2);
        uint8x8_t w52x2 = vand_u8(w5x2, v_and_mask2);
        uint8x8_t w52x3 = vand_u8(w5x3, v_and_mask2);
        uint8x8_t w52x4 = vand_u8(w5x4, v_and_mask2);
        uint8x8_t w52x5 = vand_u8(w5x5, v_and_mask2);
        uint8x8_t w52x6 = vand_u8(w5x6, v_and_mask2);
        uint8x8_t w52x7 = vand_u8(w5x7, v_and_mask2);

        uint8x8_t w53x0 = vadd_u8(w51x0, w52x0);
        uint8x8_t w53x1 = vadd_u8(w51x1, w52x1);
        uint8x8_t w53x2 = vadd_u8(w51x2, w52x2);
        uint8x8_t w53x3 = vadd_u8(w51x3, w52x3);
        uint8x8_t w53x4 = vadd_u8(w51x4, w52x4);
        uint8x8_t w53x5 = vadd_u8(w51x5, w52x5);
        uint8x8_t w53x6 = vadd_u8(w51x6, w52x6);
        uint8x8_t w53x7 = vadd_u8(w51x7, w52x7);

        // TODO better way to do vksum?
        ksum[0] = (int32_t) (uint16_t) vaddlv_u8(w53x0);
        ksum[1] = (int32_t) (uint16_t) vaddlv_u8(w53x1);
        ksum[2] = (int32_t) (uint16_t) vaddlv_u8(w53x2);
        ksum[3] = (int32_t) (uint16_t) vaddlv_u8(w53x3);
        ksum[4] = (int32_t) (uint16_t) vaddlv_u8(w53x4);
        ksum[5] = (int32_t) (uint16_t) vaddlv_u8(w53x5);
        ksum[6] = (int32_t) (uint16_t) vaddlv_u8(w53x6);
        ksum[7] = (int32_t) (uint16_t) vaddlv_u8(w53x7);

        float32x4x2_t w_packed_b = vld1q_f32_x2(packed_b);
        int32x4x2_t w_ksum_s32 = vld1q_s32_x2(ksum);
        int32x4_t w_shift_const = vdupq_n_s32(shift_const);
        float32x4x2_t w_ksum;
        w_ksum.val[0] = vcvtq_f32_s32(vsubq_s32(w_ksum_s32.val[0], w_shift_const));
        w_ksum.val[1] = vcvtq_f32_s32(vsubq_s32(w_ksum_s32.val[1], w_shift_const));
        float32x4x2_t w_ksum_muls = vld1q_f32_x2(ksum_muls);
        w_packed_b.val[0] = vfmaq_f32(w_packed_b.val[0], w_ksum.val[0], w_ksum_muls.val[0]);
        w_packed_b.val[1] = vfmaq_f32(w_packed_b.val[1], w_ksum.val[1], w_ksum_muls.val[1]);
        vst1q_f32_x2(packed_b, w_packed_b);

        uint8x8_t w6x0 = vext_u8(w5x0, v_zero, 4);
        uint8x8_t w6x1 = vext_u8(w5x1, v_zero, 4);
        uint8x8_t w6x2 = vext_u8(w5x2, v_zero, 4);
        uint8x8_t w6x3 = vext_u8(w5x3, v_zero, 4);
        uint8x8_t w6x4 = vext_u8(w5x4, v_zero, 4);
        uint8x8_t w6x5 = vext_u8(w5x5, v_zero, 4);
        uint8x8_t w6x6 = vext_u8(w5x6, v_zero, 4);
        uint8x8_t w6x7 = vext_u8(w5x7, v_zero, 4);

        uint8x8_t w7x0 = vadd_u8(w5x0, w6x0);
        uint8x8_t w7x1 = vadd_u8(w5x1, w6x1);
        uint8x8_t w7x2 = vadd_u8(w5x2, w6x2);
        uint8x8_t w7x3 = vadd_u8(w5x3, w6x3);
        uint8x8_t w7x4 = vadd_u8(w5x4, w6x4);
        uint8x8_t w7x5 = vadd_u8(w5x5, w6x5);
        uint8x8_t w7x6 = vadd_u8(w5x6, w6x6);
        uint8x8_t w7x7 = vadd_u8(w5x7, w6x7);

        uint8x8_t w8x0 = veor_u8(w7x0, v_eor_mask);
        uint8x8_t w8x1 = veor_u8(w7x1, v_eor_mask);
        uint8x8_t w8x2 = veor_u8(w7x2, v_eor_mask);
        uint8x8_t w8x3 = veor_u8(w7x3, v_eor_mask);
        uint8x8_t w8x4 = veor_u8(w7x4, v_eor_mask);
        uint8x8_t w8x5 = veor_u8(w7x5, v_eor_mask);
        uint8x8_t w8x6 = veor_u8(w7x6, v_eor_mask);
        uint8x8_t w8x7 = veor_u8(w7x7, v_eor_mask);

        vst1_u8(op[0], w8x0);
        vst1_u8(op[1], w8x1);
        vst1_u8(op[2], w8x2);
        vst1_u8(op[3], w8x3);
        vst1_u8(op[4], w8x4);
        vst1_u8(op[5], w8x5);
        vst1_u8(op[6], w8x6);
        vst1_u8(op[7], w8x7);

        packed_weights = (uint8_t*) packed_weights + nr * kr;  // kr * 2 nibbles

        if (((2 * kr) + kr_block_start) % bl == 0) {
          float32x4x2_t w_scales = vld1q_f32_x2(scales);
          vst1q_f32_x2(packed_weights, w_scales);
          packed_weights = (void*) ((uintptr_t) packed_weights + /*extra_bytes_bl=*/ nr * sizeof(float));
        }
      }

      float32x4x2_t w_bias;
      if XNN_LIKELY(b != NULL) {
        const float* bias = (float*)((uintptr_t) b + nr_block_start * sizeof(float));
        w_bias = vld1q_f32_x2(bias);
      } else {
        w_bias.val[0] = vdupq_n_f32(0.0);
        w_bias.val[1] = vdupq_n_f32(0.0);
      }
      vst1q_f32_x2(packed_weights, w_bias);
      packed_weights = (void*) ((uintptr_t) packed_weights + /*extra_bytes_n=*/nr * sizeof(float));
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}
#endif  // XNN_ARCH_ARM64

bool can_use_xnn_pack_qs8_qc4w_gemm_bl_goi_w_nr8_kr4(
  size_t nc, size_t nr, size_t kr, size_t sr, size_t extra_bytes_bl, size_t extra_bytes_n)
{
#if XNN_ARCH_ARM64
  return nr == 8 && sr == 1 && kr == 4 && nc % nr == 0 && extra_bytes_bl == nr * sizeof(float) && extra_bytes_n == nr * sizeof(float);
#else
  return false;
#endif  // XNN_ARCH_ARM64
}

void xnn_pack_qs8_qc4w_gemm_bl_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* k,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8);
  assert(bl != 0);
  assert(sr == 1); // TODO - Add support for sr != 1 for blockwise
  assert(round_up_po2(kc, kr) % bl == 0); // must be round number of blocks inside a column
  assert(bl % kr == 0); // must be round number of kr
  assert(bl <= round_up_po2(kc, kr));
  assert(2 * kr <= bl); // must be at least two kr to avoid back-to-back empty_bytes

  #if XNN_ARCH_ARM64
  if (can_use_xnn_pack_qs8_qc4w_gemm_bl_goi_w_nr8_kr4(nc, nr, kr, sr, extra_bytes_bl, extra_bytes_n)) {
    xnn_pack_qs8_qc4w_gemm_bl_goi_w_nr8_kr4(
      g, nc, kc, bl, k, bias, scale, packed_weights, params);
    return;
  }
  #endif  // XNN_ARCH_ARM64


  const size_t skr = sr * kr;
  const size_t num_blocks = round_up_po2(kc, skr) / bl;
  const int32_t izp = (int32_t) params->input_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      packed_weights = (float*) packed_weights + nr_block_size;
      packed_weights = (float*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*) packed_weights)[kr_block_offset] = kv;
          }

          size_t block_index = kr_block_start / bl;
          size_t scale_index = (nr_block_start + nr_block_offset) * num_blocks + block_index;
          unaligned_indexed_store_f32(packed_b, nr_block_offset, unaligned_indexed_load_f32(packed_b, nr_block_offset) - (float) ksum * izp * scale[scale_index] * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        if (((2 * kr) + kr_block_start) % bl == 0) {
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_bl);
        }

        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes_n);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(bias != NULL) {
      bias += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_qc4w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            const size_t k_offset = kc_idx * k_stride + (nr_block_start + nr_block_offset);
            const size_t kh_offset = (kc_idx + kr) * k_stride + (nr_block_start + nr_block_offset);
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*) packed_weights)[kr_block_offset] = kv;
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp * 16);
          packed_weights = (uint8_t*) packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_qs8w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

// qs4 packs 2 columns into 2 rows.
// kc can be odd.  assume k values in a row are padded to a byte boundary
void xnn_pack_f32_qc4w_gemm_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const void* k,  // 4 bit values
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  kc = (kc + 1) >> 1;
  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = ((const uint8_t*) k)[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k = (const uint8_t*) k + nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_xw_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
              ksum += (uint32_t) kv;
              ((int16_t*) packed_weights)[kr_block_offset] = (int16_t) kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int16_t*) packed_weights + kr;
        }
        packed_weights = (int16_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0.0f;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[kc_idx * k_stride + nr_block_start + nr_block_offset];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = UINT16_C(0);
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[kc_idx * k_stride + nr_block_start + nr_block_offset];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = fp16_ieee_from_fp32_value(b[nr_block_start + nr_block_offset]);
        }
      } else {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = 0.0f;
        }
      }
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = fp16_ieee_from_fp32_value(k[kc_idx * k_stride + nr_block_start + nr_block_offset]);
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const uint8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (int32_t) kv;
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (uint8_t*) packed_weights + kr;
        }
        packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (uint32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (uint32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          uint32_t ksum = 0;
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (uint32_t) kv;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_qs8w_gemm_gio_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const int8_t* k,
  const float* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const int32_t* b = (const int32_t*) bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (int32_t*) packed_weights + 1;
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (int32_t*) packed_weights + 1;
        } while (--n != 0);
      }
      packed_weights = (int32_t*) packed_weights + (nr - nr_block_size);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              const int8_t kv = k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_weights = (int8_t*) packed_weights + kr;
        }
        packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = fp16_ieee_from_fp32_value(b[nr_block_start + nr_block_offset]);
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                packed_weights[kr_block_offset] = fp16_ieee_from_fp32_value(k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx]);
              }
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) ks * (int32_t) kc * izp * (int32_t) params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            int32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const uint8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (int32_t) kv;
                ((uint8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (uint8_t*) packed_weights + kr;
          }
          packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t) params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            uint32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const int8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (uint32_t) kv;
                ((int8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (int8_t*) packed_weights + kr;
          }
          packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_conv_goki_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t) params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
            uint32_t ksum = 0;
            for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
              const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
              if (kc_idx < kc) {
                const int8_t kv = k[((nr_block_start + nr_block_offset) * ks + ki) * kc + kc_idx];
                ksum += (uint32_t) kv;
                ((int8_t*) packed_weights)[kr_block_offset] = kv;
              }
            }
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
            packed_weights = (int8_t*) packed_weights + kr;
          }
          packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (float*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          packed_weights[nr_block_offset] = fp16_ieee_from_fp32_value(b[nr_block_start + nr_block_offset]);
        }
      }
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = fp16_ieee_from_fp32_value(k[ki * g * nc + (nr_block_start + nr_block_offset)]);
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t bzp = (int32_t) ks * izp * (int32_t) params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, bzp);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const uint8_t kv = k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((uint8_t*) packed_weights)[nr_block_offset * kr] = kv;
            unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - (int32_t) kv * izp);
          }
          packed_weights = (uint8_t*) packed_weights + nr * kr;
        }
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void pack_qs8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  int32_t zero_point_offset,
  const struct xnn_qs8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const uint32_t izp = (uint32_t) params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*) packed_weights;
      if XNN_LIKELY(b != NULL) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = nr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr; sr_block_offset++) {
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1); nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const int8_t kv = k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((int8_t*) packed_weights)[nr_block_offset * kr] = kv;
            unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - (uint32_t) kv * izp);
          }
          packed_weights = (int8_t*) packed_weights + nr * kr;
        }
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  return pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                             extra_bytes, /*zero_point_offset=*/0, params);
}

void xnn_pack_qs8_to_qu8_conv_kgo_w(
  size_t g,
  size_t nc,
  size_t ks,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  return pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                             extra_bytes, /*zero_point_offset=*/128, params);
}

void xnn_pack_f32_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != NULL) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_f16_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != NULL) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = b[nr_block_start + nr_block_offset];
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          if XNN_LIKELY(b != NULL) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              packed_weights[nr_block_offset] = fp16_ieee_from_fp32_value(b[nr_block_start + nr_block_offset]);
            }
          }
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      packed_weights[kr_block_offset] = fp16_ieee_from_fp32_value(k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx]);
                    }
                  }
                  packed_weights += kr;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void pack_qs8_deconv_goki_w(
  size_t groups,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  int32_t zero_point_offset,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  assert(groups != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t) params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < groups; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          int32_t* packed_b = (int32_t*) packed_weights;
          if XNN_LIKELY(b != 0) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              unaligned_store_s32(packed_weights, b[nr_block_start + nr_block_offset]);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            }
          } else {
            size_t n = nr_block_size;
            do {
              unaligned_store_s32(packed_weights, 0);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            } while (--n != 0);
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  uint32_t ksum = 0;
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      const int8_t kv = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                      ksum += (uint32_t) kv;
                      ((int8_t*) packed_weights)[kr_block_offset] = kv;
                    }
                  }
                  unaligned_indexed_store_u32(packed_b, nr_block_offset, unaligned_indexed_load_u32(packed_b, nr_block_offset) - ksum * izp);
                  packed_weights = (int8_t*) packed_weights + kr;
                }
                packed_weights = (int8_t*) packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  return pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                                packed_weights, extra_bytes, /*zero_point_offset=*/0, subconv_params, params);
}

void xnn_pack_qs8_to_qu8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qs8_packing_params* params)
{
  return pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                                packed_weights, extra_bytes, /*zero_point_offset=*/128, subconv_params, params);
}

void xnn_pack_qu8_deconv_goki_w(
  size_t g,
  size_t nc,
  size_t kh,
  size_t kw,
  size_t kc,
  size_t sh,
  size_t sw,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t extra_bytes,
  struct subconvolution_params* subconv_params,
  const struct xnn_qu8_packing_params* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t kzp = (int32_t) params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        const int32_t bzp = (int32_t) divide_round_up(kh - oy, sh) * (int32_t) divide_round_up(kw - ox, sw) * (int32_t) kc * izp * kzp;
        for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          int32_t* packed_b = (int32_t*) packed_weights;
          if XNN_LIKELY(b != 0) {
            for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
              unaligned_store_s32(packed_weights, bzp + b[nr_block_start + nr_block_offset]);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            }
          } else {
            size_t n = nr_block_size;
            do {
              unaligned_store_s32(packed_weights, bzp);
              packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
            } while (--n != 0);
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + (nr - nr_block_size) * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
                for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
                  int32_t ksum = 0;
                  for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
                    const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
                    if (kc_idx < kc) {
                      const uint8_t kv = k[(((nr_block_start + nr_block_offset) * kh + ky) * kw + kx) * kc + kc_idx];
                      ksum += (int32_t) kv;
                      ((uint8_t*) packed_weights)[kr_block_offset] = kv;
                    }
                  }
                  unaligned_indexed_store_s32(packed_b, nr_block_offset, unaligned_indexed_load_s32(packed_b, nr_block_offset) - ksum * izp);
                  packed_weights = (uint8_t*) packed_weights + kr;
                }
                packed_weights = (uint8_t*) packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  }
}

// Helper function to advance x and y indices.
inline static void advance_x_y(size_t h, size_t* x, size_t* y) {
  if (++*y == h) {
    *y = 0;
    ++*x;
  }
}

void xnn_pack_f32_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_to_f16_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = fp16_ieee_from_fp32_value(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = fp16_ieee_from_fp32_value(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}


void xnn_pack_qu8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t boff = (int32_t) h * (int32_t) w * izp * (int32_t) params->kernel_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qs8_dwconv_ghw_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const uint32_t izp = (uint32_t) params->input_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
      }
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
      }
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  float* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const float kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (float*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint16_t* k,
  const uint16_t* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = b[cr_block_start + cr_block_offset];
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0.0f;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_to_f16_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const float* k,
  const float* b,
  const void* scale,
  uint16_t* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = fp16_ieee_from_fp32_value(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0;
        } while (--n != 0);
      }
      packed_weights += channel_tile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          *packed_weights++ = fp16_ieee_from_fp32_value(b[cr_block_start + cr_block_offset]);
        }
      } else {
        size_t n = cr_block_size;
        do {
          *packed_weights++ = 0;
        } while (--n != 0);
      }
      packed_weights += channel_subtile - cr_block_size;

      x = processed_x;
      y = processed_y;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights += doz(first_pass_tile, kernel_size) * cr_block_size;
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = fp16_ieee_from_fp32_value(k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
          *packed_weights++ = kv;
        }
        packed_weights += channel_tile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_tile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint16_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *packed_weights++ = fp16_ieee_from_fp32_value(kv);
        }
        packed_weights += channel_subtile - cr_block_size;
        if (++y == h) {
          y = 0;
          x++;
        }
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights += (last_pass_tile - kernel_size) * channel_subtile;
      packed_weights = (uint16_t*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qu8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const uint8_t* k,
  const int32_t* b,
  const void* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qu8_packing_params* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const int32_t izp = (int32_t) params->input_zero_point;
  const int32_t boff = (int32_t) h * (int32_t) w * izp * (int32_t) params->kernel_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, boff + b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, boff);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_s32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_s32(packed_b, cr_block_offset) - (int32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const uint8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((uint8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(uint8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(uint8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_qs8_dwconv_hwg_w(
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t h,
  size_t w,
  size_t c,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  const int8_t* k,
  const int32_t* b,
  const float* scale,
  void* packed_weights,
  size_t per_tile_extra_bytes,
  size_t per_subtile_extra_bytes,
  const struct xnn_qs8_packing_params* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);
  size_t kernel_size = h * w;
  if (middle_pass_tile == 0) {
    // Uni-pass DWCONV.
    assert(last_pass_tile == 0);
  } else {
    // Multi-pass DWCONV.
    assert(kernel_size > first_pass_tile);
  }

  const uint32_t izp = (uint32_t) params->input_zero_point;
  // Stores the x and y index that should be processed next.
  size_t processed_x = 0;
  size_t processed_y = 0;
  size_t x = 0;
  size_t y = 0;
  // First and middle pass packs in sizes of channel_tile to tiled_c, then in sizes of channel_subtile.
  const size_t tiled_c = round_down_po2(round_up_po2(c, channel_round), channel_tile);

  // Pack in blocks of channel_tile, then in blocks of channel_subtile.
  {
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }

      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
      }
    }

    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      int32_t* packed_b = (int32_t*) packed_weights;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      if XNN_LIKELY(b != NULL) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          unaligned_store_s32(packed_weights, b[cr_block_start + cr_block_offset]);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        }
      } else {
        size_t n = cr_block_size;
        do {
          unaligned_store_s32(packed_weights, 0);
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int32_t));
        } while (--n != 0);
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int32_t));

      // Biases need to be offset by all kernel values.
      for (size_t x = 0; x < w; x++) {
        for (size_t y = 0; y < h; y++) {
          for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
            const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
            unaligned_indexed_store_u32(packed_b, cr_block_offset,
                                        unaligned_indexed_load_u32(packed_b, cr_block_offset) - (uint32_t) kv * izp);
          }
        }
      }

      x = 0;
      y = 0;
      // kernel_size can be less than the first_pass_tile, in this case, pack up
      // to the smaller of the two.
      for (size_t i = 0; i < min(first_pass_tile, kernel_size); i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // And make sure to skip weights if kernel_size < first_pass_tile.
      packed_weights = (void*) ((uintptr_t) packed_weights + doz(first_pass_tile, kernel_size) * cr_block_size);
      // If unipass and QC8, we need to pack extra bytes for scale values here.
      if (middle_pass_tile == 0) {
        packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
      }
    }
  }

  if (kernel_size <= first_pass_tile) {
    return;
  }

  kernel_size -= first_pass_tile;

  processed_x = x;
  processed_y = y;

  // Middle pass. (kernel_size / middle_pass_tile) blocks, within each block is
  // middle_pass_tile * cr weights.
  for (; kernel_size > last_pass_tile; kernel_size -= middle_pass_tile) {
    assert(kernel_size >= middle_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t j = 0; j < middle_pass_tile; j++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
    }
    processed_x = x;
    processed_y = y;
  }

  // Last pass.
  {
    assert(kernel_size <= last_pass_tile);
    size_t cr_block_start = 0;
    for (; cr_block_start < round_down_po2(c, channel_tile); cr_block_start += channel_tile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_tile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_tile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_tile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_tile_extra_bytes);
    }
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      // Last pass does not pack to rounded c, since it handles remainder.
      x = processed_x;
      y = processed_y;
      const size_t cr_block_size = min(c - cr_block_start, channel_subtile);
      for (size_t i = 0; i < kernel_size; i++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
          const int8_t kv = k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          *((int8_t*) packed_weights) = kv;
          packed_weights = (void*) ((uintptr_t) packed_weights + sizeof(int8_t));
        }
        packed_weights = (void*) ((uintptr_t) packed_weights + (channel_subtile - cr_block_size) * sizeof(int8_t));
        advance_x_y(h, &x, &y);
      }
      // Pad so that we can always read last_pass_tile weights in the last pass.
      packed_weights = (void*) ((uintptr_t) packed_weights + (last_pass_tile - kernel_size) * channel_subtile);
      packed_weights = (void*) ((uintptr_t) packed_weights + per_subtile_extra_bytes);
    }
  }
}

void xnn_pack_f32_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const float* k,
  float* packed_weights,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

void xnn_pack_f16_gemminc_goi_w(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint16_t* k,
  uint16_t* packed_weights,
  const void* params)
{
  assert(g != 0);
  assert(nr >= sr);
  assert(k != NULL);
  assert(packed_weights != NULL);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr); kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr; kr_block_offset++) {
            const size_t kc_idx = round_down_po2(kr_block_start, skr) + ((kr_block_start + kr_block_offset + nr_block_offset * kr) & (skr - 1));
            if (kc_idx < kc) {
              packed_weights[kr_block_offset] = k[(nr_block_start + nr_block_offset) * kc + kc_idx];
            }
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
    }
    k += nc * kc;
  } while (--g != 0);
}

void xnn_pack_f32_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = 0.0f;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nr;
    }
  }
}

void xnn_pack_f32_to_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const float* k,
  const float* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = fp16_ieee_from_fp32_value(b[min(nr_block_offset, nr_block_size - 1)]);
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = fp16_ieee_from_fp32_value(k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c]);
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nr;
    }
  }
}

void xnn_pack_f16_dconv_oki_w(
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kh,
  size_t kw,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY(b != NULL) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
        *packed_weights++ = b[min(nr_block_offset, nr_block_size - 1)];
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr; nr_block_offset++) {
            *packed_weights++ = k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) * kh + ky) * kw + kx) * kc + c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nr;
    }
  }
}

void xnn_pack_f32_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0.0f;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[g * kernel_size + i];
    }
  }
}

void xnn_pack_f32_to_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = fp16_ieee_from_fp32_value(*b++);
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = fp16_ieee_from_fp32_value(k[g * kernel_size + i]);
    }
  }
}

void xnn_pack_f16_chw_dwconv_ghw_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[g * kernel_size + i];
    }
  }
}

void xnn_pack_f32_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0.0f;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[i * groups + g];
    }
  }
}

void xnn_pack_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const uint16_t* k,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = *b++;
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = k[i * groups + g];
    }
  }
}

void xnn_pack_f32_to_f16_chw_dwconv_hwg_w(
  size_t kernel_size,
  size_t groups,
  const float* k,
  const float* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(k != NULL);
  assert(packed_weights != NULL);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY(b != NULL) {
      *packed_weights = fp16_ieee_from_fp32_value(*b++);
    } else {
      *packed_weights = 0;
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = fp16_ieee_from_fp32_value(k[i * groups + g]);
    }
  }
}


void xnn_pack_f32_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  float* packed_weights,
  const void* params)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = 0.0f;
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const uint16_t* s,
  const uint16_t* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = b[cr_block_start + cr_block_offset];
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f32_to_f16_vmulcaddc_w(
  size_t c,
  size_t cr,
  const float* s,
  const float* b,
  uint16_t* packed_weights,
  const void* params)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
      *packed_weights++ = fp16_ieee_from_fp32_value(s[cr_block_start + cr_block_offset]);
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY(b != NULL) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size; cr_block_offset++) {
        *packed_weights++ = fp16_ieee_from_fp32_value(b[cr_block_start + cr_block_offset]);
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = 0;
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_pack_f32_prelu_w(
  size_t c,
  const float* s,
  float* packed_weights)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  memcpy(packed_weights, s, c * sizeof(float));
}

void xnn_pack_f16_prelu_w(
  size_t c,
  const uint16_t* s,
  uint16_t* packed_weights)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  memcpy(packed_weights, s, c * sizeof(uint16_t));
}

void xnn_pack_f32_to_f16_prelu_w(
  size_t c,
  const float* s,
  uint16_t* packed_weights)
{
  assert(s != NULL);
  assert(packed_weights != NULL);

  do {
    *packed_weights++ = fp16_ieee_from_fp32_value(*s++);
  } while (--c != 0);
}

void xnn_analyze_f32_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const float* kernel,
  struct xnn_spmm_packing_params* params)
{
  assert(kernel != NULL);
  assert(params != NULL);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      const size_t row2_nonzero = (size_t) (kernel[(oc + 2) * group_input_channels + ic] != 0.0f);
      const size_t row3_nonzero = (size_t) (kernel[(oc + 3) * group_input_channels + ic] != 0.0f);
      num_nonzeroes += row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 += (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4); oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2); oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes += (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

void xnn_analyze_f16_spmm_w(
  size_t group_output_channels,
  size_t group_input_channels,
  const uint16_t* kernel,
  struct xnn_spmm_packing_params* params)
{
  assert(kernel != NULL);
  assert(params != NULL);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0);
      const size_t row2_nonzero = (size_t) (kernel[(oc + 2) * group_input_channels + ic] != 0);
      const size_t row3_nonzero = (size_t) (kernel[(oc + 3) * group_input_channels + ic] != 0);
      num_nonzeroes += row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 += (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4); oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0);
      const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2); oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes += (size_t) (kernel[oc * group_input_channels + ic] != 0);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

enum xnn_status xnn_pack_f32_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  float* nonzero_values,
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != NULL) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = bias[ocb + oco];
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = 0.0f;
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != NULL) {
      *nonzero_values++ = bias[oc];
    } else {
      *nonzero_values++ = 0.0f;
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0.0f) {
        *nonzero_values++ = weight;
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}


enum xnn_status xnn_pack_f32_to_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const float* kernel,
  const float* bias,
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  uint16_t* nonzero_values,  // fp16 values
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != NULL) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = fp16_ieee_from_fp32_value(bias[ocb + oco]);
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = 0;
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = fp16_ieee_from_fp32_value(kernel[(ocb + oco) * group_input_channels + ic]);
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != NULL) {
      *nonzero_values++ = fp16_ieee_from_fp32_value(bias[oc]);
    } else {
      *nonzero_values++ = 0;
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0.0f) {
        *nonzero_values++ = fp16_ieee_from_fp32_value(weight);
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

enum xnn_status xnn_pack_f16_spmm_w(
  size_t group_output_channels,
  size_t output_channels_block_size,
  size_t group_input_channels,
  const uint16_t* kernel,  // fp16 values
  const uint16_t* bias,  // fp16 values
  int32_t* input_channel_diffs,
  uint32_t* output_channel_nonzeros,
  uint16_t* nonzero_values,  // fp16 values
  size_t* first_input_channel)
{
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
    if XNN_LIKELY(bias != NULL) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = bias[ocb + oco];
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = 0;
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
    if XNN_LIKELY(bias != NULL) {
      *nonzero_values++ = bias[oc];
    } else {
      *nonzero_values++ = 0;
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0) {
        *nonzero_values++ = weight;
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
          if (diff != (int64_t) (int32_t) diff) {
            xnn_log_error("failed to convert kernel to sparse representation: "
              "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t) diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(uint16_t);
    if (diff != (int64_t) (int32_t) diff) {
      xnn_log_error("failed to convert kernel to sparse representation: "
        "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t) diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}
