// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x4-packw/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/packw.h"


static int8_t sign_extend_int4(int8_t value) {
  return (value ^ 0x8) - 8;
}

void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x16c8__scalar(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t n_stride,
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
  assert(nr == 16);
  assert(kr == 8);
  assert(sr == 1);
  assert(k != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = 1 * 8;
  const uint32_t izp = (uint32_t) params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t) params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, 16);
      int32_t* packed_b = (int32_t*) packed_weights;
      size_t n = 0;
      if XNN_LIKELY(b != NULL) {
        while (n < nr_block_size) {
          packed_b[n] = b[n + nr_block_start];
          ++n;
        }
      }
      while (n < 16) {
        packed_b[n++] = 0;
      }
      packed_weights = (int32_t*) packed_weights + 16;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr * 2); kr_block_start += 8 * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size; nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin = round_down_po2(kr_block_start, skr) + ((kr_block_start + nr_block_offset * 8) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < 8; kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset = (nr_block_start + nr_block_offset) * n_stride + kc_idx;
            const size_t kh_offset = k_offset + 8;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + 8) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*) packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4) : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + 8) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4) : (k[kh_offset >> 1] & 0xF));
              }
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ksum += kv_lo + kv_hi - 2 * kernel_zero_point;  // subtract 2 zero points
              ((uint8_t*) packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp * 16;
          packed_weights = (uint8_t*) packed_weights + 8;  // 8 * 2 nibbles
        }
        packed_weights = (uint8_t*) packed_weights + (16 - nr_block_size) * 8;  // skip NR remainder
      }
      packed_weights = (void*) ((uintptr_t) packed_weights + extra_bytes);
      nr_block_start += 16;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE(b != NULL) {
      b += nc;
    }
  } while (--g != 0);
}
