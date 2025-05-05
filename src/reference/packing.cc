// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/unaligned.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

#endif  // XNN_ENABLE_KLEIDIAI

struct unaligned_int32_t {
  char value[sizeof(int32_t)];

  XNN_INLINE unaligned_int32_t(int32_t v) { memcpy(value, &v, sizeof(v)); }

  XNN_INLINE operator int32_t() const {
    int32_t v;
    memcpy(&v, value, sizeof(v));
    return v;
  }
};

template <typename Src, typename Dst>
void copy_bias(const Src* b, size_t b_offset, size_t n, Dst* packed_b) {
  if (b) {
    std::copy_n(b + b_offset, n, packed_b);
  } else {
    std::fill_n(packed_b, n, static_cast<Dst>(0));
  }
}

template <typename Src, typename Dst>
void copy_bias(const Src* b, size_t b_offset, size_t n, Dst* packed_b,
               Src zero_point) {
  if (b) {
    for (size_t i = 0; i < n; ++i) {
      *packed_b++ = zero_point + b[b_offset + i];
    }
  } else {
    std::fill_n(packed_b, n, zero_point);
  }
}

template <typename Src, typename Dst>
int32_t copy_n_and_sum(const Src* src, size_t n, Dst* dst) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    const auto v = *src++;
    sum += (int32_t)v;
    *dst++ = v;
  }
  return sum;
}

extern "C" {

void xnn_pack_f32_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, const float* k,
                             const float* b, const void* scale,
                             float* packed_weights, size_t extra_bytes,
                             const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          if (kc_begin < kc_end) {
            std::copy_n(&k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                        kc_end - kc_begin, packed_weights);
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (float*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_bf16_f32_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                  size_t kr, size_t sr, const xnn_bfloat16* k,
                                  const float* bias, const void* scale,
                                  void* packed_weights, size_t extra_bytes,
                                  const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      float* packed_weights_float = (float*)packed_weights;
      copy_bias(bias, nr_block_start, nr_block_size, packed_weights_float);
      packed_weights = (void*)((uintptr_t)packed_weights + nr * sizeof(float));

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          xnn_bfloat16* end = (xnn_bfloat16*)packed_weights + kr;
          if (kc_begin < kc_end) {
            std::copy_n(&k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                        kc_end - kc_begin, (xnn_bfloat16*)packed_weights);
            packed_weights = (xnn_bfloat16*)packed_weights + kc_end - kc_begin;
          }
          std::fill((xnn_bfloat16*)packed_weights, end, xnn_bfloat16(0.0f));
          packed_weights = end;
        }
        packed_weights = (void*)((uintptr_t)packed_weights +
                                 (nr - nr_block_size) * kr * sizeof(uint16_t));
      }
      packed_weights = (void*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (bias != nullptr) {
      bias += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, const uint16_t* k,
                             const uint16_t* b, const void* scale,
                             uint16_t* packed_weights, size_t extra_bytes,
                             const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          if (kc_begin < kc_end) {
            std::copy_n(&k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                        kc_end - kc_begin, packed_weights);
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (uint16_t*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                    size_t kr, size_t sr, const float* k,
                                    const float* b, const void* scale,
                                    xnn_float16* packed_weights,
                                    size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          if (kc_begin < kc_end) {
            std::copy_n(&k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                        kc_end - kc_begin, packed_weights);
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (xnn_float16*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, const uint8_t* k,
                             const int32_t* b, const void* scale,
                             void* packed_weights, size_t extra_bytes,
                             const struct xnn_qu8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t bzp = (int32_t)kc * izp * (int32_t)params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b, bzp);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          uint8_t* end = (uint8_t*)packed_weights + kr;
          if (kc_begin < kc_end) {
            int32_t ksum = copy_n_and_sum(
                &k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                kc_end - kc_begin, (uint8_t*)packed_weights);
            packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
            packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          }
          std::fill((uint8_t*)packed_weights, end, params->kernel_zero_point);
          packed_weights = end;
        }
        packed_weights = (uint8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, const int8_t* k,
                             const int32_t* b, const float* scale,
                             void* packed_weights, size_t extra_bytes,
                             const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          int8_t* end = (int8_t*)packed_weights + kr;
          if (kc_begin < kc_end) {
            uint32_t ksum = copy_n_and_sum(
                &k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                kc_end - kc_begin, (int8_t*)packed_weights);
            packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
            packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          }
          std::fill((int8_t*)packed_weights, end, INT8_C(0));
          packed_weights = end;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_goi_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const int8_t* k, const int32_t* b, const float* scale, void* packed_weights,
    size_t extra_bytes, const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          int8_t* end = (int8_t*)packed_weights + kr;
          if (kc_begin < kc_end) {
            uint32_t ksum = copy_n_and_sum(
                &k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                kc_end - kc_begin, (int8_t*)packed_weights);
            packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
            packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          }
          std::fill((int8_t*)packed_weights, end, INT8_C(0));
          packed_weights = end;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

static int8_t sign_extend_int4(int8_t value) { return (value ^ 0x8) - 8; }

void xnn_pack_qs8_qc4w_gemm_goi_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*)packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ksum += kv_lo + kv_hi -
                      2 * kernel_zero_point;  // subtract 2 zero points
              ((uint8_t*)packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_b[nr_block_offset] =
              packed_b[nr_block_offset] - ksum * izp * 16;
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// Packs the weights so as to minimize register usage in kernels.
// For example:
// 0 1
// 2 3
// 4 5
// 6 7
// 8 9
// A B
// C D
// E F
//
// is packed for a Mx8c4 microkernel as:
// (row sums) 1 5 9 13 17 21 2 29 | (packed weights) 08 19 00 00 | 2A 3B 00 00 |
// 4C 5D 00 | 6E 7F 00 00 The row sums are packed first. In contrast to planar
// packing which packs the weights from the same channel side by side, so
// position + kr. The register bytes parameter is needed so that we know the
// offset between each weight's load.
void xnn_pack_qs8_qc4w_gemm_goi_w_non_planar(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t register_bytes, const uint8_t* k, const int32_t* b,
    const float* scale, void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;
  int row_offset = register_bytes / kr;
  do {
    size_t nr_block_start = 0;
    do {
      size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      if (b) {
        for (size_t i = 0; i < nr_block_size; ++i) {
          packed_b[i] = b[nr_block_start + i] * 16;
        }
      } else {
        std::fill_n(packed_b, nr_block_size, 0);
      }
      packed_weights = (int32_t*)packed_weights + nr;

      size_t num_k_blocks = round_up_po2(kc, skr * 1);
      for (size_t kr_block_start = 0; kr_block_start < num_k_blocks;
           kr_block_start += kr * 1) {
        void* pw = packed_weights;
        for (size_t nr_block_offset_ = 0; nr_block_offset_ < nr_block_size;
             nr_block_offset_ += row_offset * 2) {
          for (size_t inner_nr_block_offset = 0;
               inner_nr_block_offset < row_offset; inner_nr_block_offset += 1) {
            size_t actual_nr_block_offset =
                inner_nr_block_offset + nr_block_offset_;
            int32_t ksum_lo = 0;
            int32_t ksum_hi = 0;
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + actual_nr_block_offset * kr) & (skr - 1));
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const size_t kc_idx = kc_begin + kr_block_offset;
              const size_t k_offset =
                  (nr_block_start + actual_nr_block_offset) * kc + kc_idx;
              const size_t kh_offset = k_offset + kc * row_offset;
              if (kernel_zero_point == 0) {
                int8_t kv_lo = kernel_zero_point;
                if ((nr_block_start + actual_nr_block_offset) < nc) {
                  if (kc_idx < kc) {
                    kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                            : (k[k_offset >> 1] & 0xF));
                  }
                }
                int8_t kv_hi = kernel_zero_point;
                if ((nr_block_start + actual_nr_block_offset + row_offset) < nc) {
                  if (kc_idx < kc) {
                    kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                             : (k[kh_offset >> 1] & 0xF));
                  }
                }
                // Pack and flip the sign bit.
                const int8_t kv = (kv_lo | (kv_hi << 4));
                kv_lo = sign_extend_int4(kv_lo);
                kv_hi = sign_extend_int4(kv_hi);
                ksum_lo += kv_lo;
                ksum_hi += kv_hi;
                ((uint8_t*)pw)[kr_block_offset] = kv;
              } else {
                uint8_t kv_lo = kernel_zero_point;
                if ((nr_block_start + actual_nr_block_offset) < nc) {
                  if (kc_idx < kc) {
                    kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                            : (k[k_offset >> 1] & 0xF));
                  }
                }
                uint8_t kv_hi = kernel_zero_point;
                if ((nr_block_start + actual_nr_block_offset + row_offset) < nc) {
                  if (kc_idx < kc) {
                    kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                             : (k[kh_offset >> 1] & 0xF));
                  }
                }
                // Pack and flip the sign bit.
                const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
                ksum_lo += kv_lo - kernel_zero_point;
                ksum_hi += kv_hi - kernel_zero_point;
                ((uint8_t*)pw)[kr_block_offset] = kv;
              }
            }
            packed_b[actual_nr_block_offset] =
                packed_b[actual_nr_block_offset] - ksum_lo * izp * 16;
            packed_b[actual_nr_block_offset + row_offset] =
                packed_b[actual_nr_block_offset + row_offset] -
                ksum_hi * izp * 16;
            pw = (uint8_t*)pw + kr;  // kr * 2 nibbles
          }
        }
        packed_weights =
            (uint8_t*)packed_weights + (nr)*kr / 2;  // skip NR remainder
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_scalar(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  xnn_pack_qs8_qc4w_gemm_goi_w_non_planar(g, nc, kc, nr, kr, sr,
                                          /*register_bytes=*/1, k, b, scale,
                                          packed_weights, extra_bytes, params);
}

void xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  xnn_pack_qs8_qc4w_gemm_goi_w_non_planar(g, nc, kc, nr, kr, sr,
                                          /*register_bytes=*/16, k, b, scale,
                                          packed_weights, extra_bytes, params);
}

void xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  xnn_pack_qs8_qc4w_gemm_goi_w_non_planar(g, nc, kc, nr, kr, sr,
                                          /*register_bytes=*/64, k, b, scale,
                                          packed_weights, extra_bytes, params);
}

// Same as qc4w but unsigned 4 bit output
// Applies kv ^ 0x88 to convert int4 to uint4
// Does not multiply bias by 16
void xnn_pack_qs8_qc4uw_gemm_goi_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*)packed_weights)[kr_block_offset] =
                  kv ^ 0x88;  // Convert to uint4
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ksum += kv_lo + kv_hi -
                      2 * kernel_zero_point;  // subtract 2 zero points
              ((uint8_t*)packed_weights)[kr_block_offset] =
                  kv ^ 0x88;  // Convert to uint4
            }
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_qb4w_gemm_goi_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t bl,         // blocksize
    const uint8_t* k,  // kernel
    const float* bias, const xnn_bfloat16* scale, void* packed_weights,
    size_t extra_bytes_bl,  // extra bytes per block
    size_t extra_bytes_n,   // extra bytes per n
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);
  assert(bias == nullptr);  // Not used here. Must be updated outside.

  const size_t skr = sr * kr;

  // Constraints for blocksize
  // These need to be reevaluated in the future.
  assert(bl != 0);
  assert(round_up_po2(kc, skr) % bl ==
         0);              // must be round number of blocks inside a column
  assert(bl % skr == 0);  // must be round number of kr * sr
  assert(bl <= round_up_po2(kc, skr));  // must not be larger than K
  assert(2 * skr <=
         bl);  // must be at least two skr to avoid back-to-back extra_bytes

  const size_t num_blocks = round_up_po2(kc, skr) / bl;
  const int32_t izp = (int32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;

  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      float* packed_b = (float*)packed_weights;
      std::fill_n(packed_b, nr, 0.0f);
      packed_weights = (float*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                (nr_block_start + nr_block_offset) * kc + kc_idx;
            const size_t kh_offset = k_offset + kr;
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*)packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = 8;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = 8;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ((uint8_t*)packed_weights)[kr_block_offset] = kv;
            }
          }

          size_t block_index = kr_block_start / bl;
          size_t scale_index =
              (nr_block_start + nr_block_offset) * num_blocks + block_index;
          unaligned_indexed_store_f32(
              packed_b, nr_block_offset,
              unaligned_indexed_load_f32(packed_b, nr_block_offset) -
                  (float)ksum * izp *
                      xnn_bfloat16_to_float(scale[scale_index]));
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        if (((2 * kr) + kr_block_start) % bl == 0) {
          packed_weights = (void*)((uintptr_t)packed_weights + extra_bytes_bl);
        }

        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*)((uintptr_t)packed_weights + extra_bytes_n);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
  } while (--g != 0);
}

void xnn_pack_qs8_qb4w_gemm_gio_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride,
    size_t bl,         // block size
    const uint8_t* k,  // kernel
    const float* bias,
    const xnn_bfloat16* scale,  // block scales (bf16 format)
    void* packed_weights,
    size_t extra_bytes_bl,  // extra bytes per block
    size_t extra_bytes_n,   // extra bytes per n
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8);
  assert(bias == nullptr);  // Not used here. Must be updated outside.

  const size_t skr = sr * kr;

  // Constraints for blocksize
  // These need to be reevaluated in the future.
  assert(bl != 0);
  assert(round_up_po2(kc, skr) % bl ==
         0);              // must be round number of blocks inside a column
  assert(bl % skr == 0);  // must be round number of kr * sr
  assert(bl <= round_up_po2(kc, skr));  // must not be larger than K
  assert(2 * skr <=
         bl);  // must be at least two skr to avoid back-to-back extra_bytes

  const size_t num_blocks = round_up_po2(kc, skr) / bl;
  const int32_t izp = (int32_t)params->input_zero_point;

  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      int32_t* packed_b = (int32_t*)packed_weights;
      std::fill_n(packed_b, nr, 0);
      packed_weights = (float*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                (nr_block_start + nr_block_offset + kc_idx * k_stride);
            const size_t kh_offset = k_offset + (kr * k_stride);
            uint8_t kv_lo = 8;
            if (kc_idx < kc) {
              kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                      : (k[k_offset >> 1] & 0xF));
            }
            uint8_t kv_hi = 8;
            if ((kc_idx + kr) < kc) {
              kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                       : (k[kh_offset >> 1] & 0xF));
            }
            ksum += kv_lo + kv_hi - 16;  // subtract 2 zero points (8)
            const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
            ((uint8_t*)packed_weights)[kr_block_offset] = kv;
          }

          size_t block_index = kr_block_start / bl;
          size_t scale_index =
              (nr_block_start + nr_block_offset) * num_blocks + block_index;
          unaligned_indexed_store_f32(
              packed_b, nr_block_offset,
              unaligned_indexed_load_f32(packed_b, nr_block_offset) -
                  (float)ksum * izp *
                      xnn_bfloat16_to_float(scale[scale_index]));
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        if (((2 * kr) + kr_block_start) % bl == 0) {
          packed_weights = (void*)((uintptr_t)packed_weights + extra_bytes_bl);
        }

        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights = (void*)((uintptr_t)packed_weights + extra_bytes_n);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
  } while (--g != 0);
}

void xnn_pack_qs8_qc4w_gemm_gio_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride, const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                kc_idx * k_stride + (nr_block_start + nr_block_offset);
            const size_t kh_offset =
                (kc_idx + kr) * k_stride + (nr_block_start + nr_block_offset);
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*)packed_weights)[kr_block_offset] = kv;
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              ksum += kv_lo + kv_hi -
                      2 * kernel_zero_point;  // subtract 2 zero points
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ((uint8_t*)packed_weights)[kr_block_offset] = kv;
            }
          }
          packed_b[nr_block_offset] =
              packed_b[nr_block_offset] - ksum * izp * 16;
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// Same as qc4w but unsigned 4 bit output
// Applies kv ^ 0x88 to convert int4 to uint4
// Does not multiply bias by 16
void xnn_pack_qs8_qc4uw_gemm_gio_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride, const uint8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_qc4w_packing_params* params) {
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr >= sr);
  assert(kr >= 1 && kr <= 16);
  assert(sr >= 1 && sr <= 16);
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  assert(params != nullptr);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  const uint32_t kernel_zero_point = (uint32_t)params->kernel_zero_point;
  do {
    size_t nr_block_start = 0;
    do {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0;
           kr_block_start < round_up_po2(kc, skr * 2);
           kr_block_start += kr * 2) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const size_t k_offset =
                kc_idx * k_stride + (nr_block_start + nr_block_offset);
            const size_t kh_offset =
                (kc_idx + kr) * k_stride + (nr_block_start + nr_block_offset);
            if (kernel_zero_point == 0) {
              int8_t kv_lo = 0;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              int8_t kv_hi = 0;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              const int8_t kv = (kv_lo | (kv_hi << 4));
              kv_lo = sign_extend_int4(kv_lo);
              kv_hi = sign_extend_int4(kv_hi);
              ksum += kv_lo + kv_hi;
              ((int8_t*)packed_weights)[kr_block_offset] =
                  kv ^ 0x88;  // Convert to uint4
            } else {
              uint8_t kv_lo = kernel_zero_point;
              if (kc_idx < kc) {
                kv_lo = ((k_offset & 1) ? (k[k_offset >> 1] >> 4)
                                        : (k[k_offset >> 1] & 0xF));
              }
              uint8_t kv_hi = kernel_zero_point;
              if ((kc_idx + kr) < kc) {
                kv_hi = ((kh_offset & 1) ? (k[kh_offset >> 1] >> 4)
                                         : (k[kh_offset >> 1] & 0xF));
              }
              ksum += kv_lo + kv_hi -
                      2 * kernel_zero_point;  // subtract 2 zero points
              const uint8_t kv = (kv_lo | (kv_hi << 4)) ^ 0x88;
              ((uint8_t*)packed_weights)[kr_block_offset] =
                  kv ^ 0x88;  // Convert to uint4
            }
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          packed_weights = (uint8_t*)packed_weights + kr;  // kr * 2 nibbles
        }
        packed_weights = (uint8_t*)packed_weights +
                         (nr - nr_block_size) * kr;  // skip NR remainder
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
      nr_block_start += nr;
    } while (nr_block_start < nc);
    k += nc * kc;  // kc * 2 nibbles
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_qs8w_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                  size_t kr, size_t sr, const int8_t* k,
                                  const float* bias, const float* scale,
                                  void* packed_weights, size_t extra_bytes,
                                  const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t* b = (const int32_t*)bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          int8_t* end = (int8_t*)packed_weights + kr;
          if (kc_begin < kc_end) {
            std::copy_n(&k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                        kc_end - kc_begin, (int8_t*)packed_weights);
            packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
          }
          std::fill((int8_t*)packed_weights, end, INT8_C(0));
          packed_weights = end;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

// qs4 packs 2 columns into 2 rows.
// kc can be odd.  assume k values in a row are padded to a byte boundary
void xnn_pack_f32_qc4w_gemm_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                  size_t kr, size_t sr,
                                  const void* k,  // 4 bit values
                                  const float* bias, const float* scale,
                                  void* packed_weights, size_t extra_bytes,
                                  const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  kc = (kc + 1) >> 1;
  const int32_t* b = (const int32_t*)bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          const size_t kc_end = std::min(kc, kc_begin + kr);
          uint8_t* end = (uint8_t*)packed_weights + kr;
          if (kc_begin < kc_end) {
            std::copy_n(
                &((const uint8_t*)
                      k)[(nr_block_start + nr_block_offset) * kc + kc_begin],
                kc_end - kc_begin, (uint8_t*)packed_weights);
            packed_weights = (uint8_t*)packed_weights + kc_end - kc_begin;
          }
          std::fill((uint8_t*)packed_weights, end, UINT8_C(0));
          packed_weights = end;
        }
        packed_weights = (uint8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k = (const uint8_t*)k + nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, size_t k_stride,
                             const float* k, const float* b, const void* scale,
                             float* packed_weights, size_t extra_bytes,
                             const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      // Special case for trivial packings.
      if (skr == 1) {
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start++) {
          const size_t kc_idx = round_down_po2(kr_block_start, skr);
          if (kc_idx < kc) {
            std::copy_n(&k[kc_idx * k_stride + nr_block_start], nr_block_size,
                        packed_weights);
          }
          packed_weights += nr;
        }

      } else {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const size_t kc_idx = kc_begin + kr_block_offset;
              packed_weights[kr_block_offset] =
                  kc_idx < kc
                      ? k[kc_idx * k_stride + nr_block_start + nr_block_offset]
                      : 0.0f;
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (float*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_bf16_f32_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                                  size_t kr, size_t sr, size_t k_stride,
                                  const xnn_bfloat16* k, const float* b,
                                  const void* scale, void* packed_weights,
                                  size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, (float*)packed_weights);
      packed_weights = (float*)packed_weights + nr;

      // Special case for trivial packings.
      if (skr == 1) {
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start++) {
          const size_t kc_idx = round_down_po2(kr_block_start, skr);
          if (kc_idx < kc) {
            std::copy_n(&k[kc_idx * k_stride + nr_block_start], nr_block_size,
                        (xnn_bfloat16*)packed_weights);
          }
          packed_weights = (xnn_bfloat16*)packed_weights + nr;
        }

      } else {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const size_t kc_idx = kc_begin + kr_block_offset;
              ((xnn_bfloat16*)packed_weights)[kr_block_offset] =
                  kc_idx < kc
                      ? k[kc_idx * k_stride + nr_block_start + nr_block_offset]
                      : static_cast<xnn_bfloat16>(0.0f);
            }
            packed_weights = (xnn_bfloat16*)packed_weights + kr;
          }
          packed_weights =
              (xnn_bfloat16*)packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (float*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, size_t k_stride,
                             const uint16_t* k, const uint16_t* b,
                             const void* scale, uint16_t* packed_weights,
                             size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      // Special case for trivial packings.
      if (skr == 1) {
        for (size_t kr_block_start = 0; kr_block_start < kc; kr_block_start++) {
          const size_t kc_idx = round_down_po2(kr_block_start, skr);
          if (kc_idx < kc) {
            std::copy_n(&k[kc_idx * k_stride + nr_block_start], nr_block_size,
                        packed_weights);
          }
          packed_weights += nr;
        }

      } else {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            for (size_t kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const size_t kc_idx = kc_begin + kr_block_offset;
              packed_weights[kr_block_offset] =
                  kc_idx < kc
                      ? k[kc_idx * k_stride + nr_block_start + nr_block_offset]
                      : UINT16_C(0);
            }
            packed_weights += kr;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (uint16_t*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                                    size_t kr, size_t sr, size_t k_stride,
                                    const float* k, const float* b,
                                    const void* scale,
                                    xnn_float16* packed_weights,
                                    size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            packed_weights[kr_block_offset] =
                kc_idx < kc
                    ? k[kc_idx * k_stride + nr_block_start + nr_block_offset]
                    : 0.0f;
          }
          packed_weights += kr;
        }
        packed_weights += (nr - nr_block_size) * kr;
      }
      packed_weights = (xnn_float16*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, size_t k_stride,
                             const uint8_t* k, const int32_t* b,
                             const void* scale, void* packed_weights,
                             size_t extra_bytes,
                             const struct xnn_qu8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t bzp = (int32_t)kc * izp * (int32_t)params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b, bzp);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          int32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            if (kc_idx < kc) {
              const uint8_t kv =
                  k[kc_idx * k_stride + (nr_block_start + nr_block_offset)];
              ksum += (int32_t)kv;
              ((uint8_t*)packed_weights)[kr_block_offset] = kv;
            } else {
              ((uint8_t*)packed_weights)[kr_block_offset] =
                  params->kernel_zero_point;
            }
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          packed_weights = (uint8_t*)packed_weights + kr;
        }
        packed_weights = (uint8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_gemm_gio_w(
    size_t g, size_t nc, size_t kc, size_t nr, size_t kr, size_t sr,
    size_t k_stride, const int8_t* k, const int32_t* b, const float* scale,
    void* packed_weights, size_t extra_bytes,
    const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (uint32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          uint32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const int8_t kv =
                kc_idx < kc
                    ? k[kc_idx * k_stride + (nr_block_start + nr_block_offset)]
                    : INT8_C(0);
            ksum += (uint32_t)kv;
            ((int8_t*)packed_weights)[kr_block_offset] = kv;
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          packed_weights = (int8_t*)packed_weights + kr;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                             size_t kr, size_t sr, size_t k_stride,
                             const int8_t* k, const int32_t* b,
                             const float* scale, void* packed_weights,
                             size_t extra_bytes,
                             const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (uint32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          uint32_t ksum = 0;
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const int8_t kv =
                kc_idx < kc
                    ? k[kc_idx * k_stride + (nr_block_start + nr_block_offset)]
                    : INT8_C(0);
            ksum += (uint32_t)kv;
            ((int8_t*)packed_weights)[kr_block_offset] = kv;
          }
          packed_b[nr_block_offset] = packed_b[nr_block_offset] - ksum * izp;
          packed_weights = (int8_t*)packed_weights + kr;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void pack_weights_and_biases(uint32_t flags,                                 //
                             const struct xnn_gemm_config* gemm_config,      //
                             size_t input_channels,                          //
                             size_t output_channels,                         //
                             size_t groups,                                  //
                             size_t unused_block_size,                       //
                             size_t weights_stride,                          //
                             xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio_w,  //
                             xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi_w,  //
                             const void* accumulator_init,                   //
                             const void* weights,                            //
                             xnn_init_scale_params_fn init_extra_data0_fn,   //
                             const void* extra_data0,                        //
                             size_t extra_data0_element_size,                //
                             xnn_init_scale_params_fn init_extra_data1_fn,   //
                             const void* extra_data1,                        //
                             size_t extra_data1_element_size,                //
                             void* packed_weights_ptr,                       //
                             size_t extra_bytes,                             //
                             const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t n_stride = round_up(output_channels, nr);
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    pack_gemm_gio_w(groups, output_channels, input_channels, nr, kr, sr,
                    output_channels, weights, accumulator_init,
                    /*scale=*/nullptr, packed_weights_ptr, nr * extra_bytes,
                    params);
  } else {
    pack_gemm_goi_w(groups, output_channels, input_channels, nr, kr, sr,
                    weights, accumulator_init, /*scale=*/nullptr,
                    packed_weights_ptr, nr * extra_bytes, params);
  }
  if (extra_data1 != nullptr) {
    assert(init_extra_data1_fn != nullptr);

    for (size_t group = 0; group < groups; group++) {
      void* packed_group_ptr = (void*)((char*)packed_weights_ptr +
                                       group * n_stride * weights_stride);
      void* weights = (void*)((uintptr_t)packed_group_ptr +
                              nr * (weights_stride - extra_bytes));
      void* extra_data_ptr =
          (void*)((uintptr_t)extra_data1 +
                  extra_data1_element_size * output_channels * group);
      init_extra_data1_fn(output_channels, nr, nr * weights_stride,
                          extra_data_ptr, weights);
    }
  }

  if (extra_data0 != nullptr) {
    assert(init_extra_data0_fn != nullptr);
    for (size_t group = 0; group < groups; group++) {
      void* packed_group_ptr = (void*)((char*)packed_weights_ptr +
                                       group * n_stride * weights_stride);
      void* weights = (void*)((uintptr_t)packed_group_ptr +
                              nr * (weights_stride - extra_bytes));
      if (extra_data1 != nullptr) {
        weights = (void*)((uintptr_t)weights + nr * sizeof(float));
      }
      void* extra_data_ptr =
          (void*)((uintptr_t)extra_data0 +
                  extra_data0_element_size * output_channels * group);
      init_extra_data0_fn(output_channels, nr, nr * weights_stride,
                          extra_data_ptr, weights);
    }
  }
}

size_t xnn_packed_stride_qs8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k,
    size_t unused_block_size, size_t k_stride, size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qs8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes =
      extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, unused_block_size, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      unused_block_size, weights_stride,
      (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_qs8_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_qs8_gemm_goi_w, accumulator_init,
      weights, init_extra_data0_fn, extra_data0, extra_data0_element_size,
      init_extra_data1_fn, extra_data1, extra_data1_element_size,
      packed_weights_ptr, extra_bytes, params);
}

size_t xnn_packed_stride_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k,
    size_t unused_block_size, size_t k_stride, size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qs4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes =
      extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, unused_block_size, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      unused_block_size, weights_stride,
      (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_qs8_qc4w_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_qs8_qc4w_gemm_goi_w,
      accumulator_init, weights, init_extra_data0_fn, extra_data0,
      extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, extra_bytes, params);
}

size_t xnn_packed_stride_qb4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k, size_t block_size,
    size_t k_stride, size_t extra_bytes) {
  const size_t planes = gemm_config->planes;
  size_t input_channels = round_up_po2(k, planes);

  size_t block_scale_bytes = 0;
  size_t num_blocks = 0;
  const bool block_wise = (block_size != 0);
  if (block_wise) {
    num_blocks = input_channels / block_size;
    block_scale_bytes += num_blocks * sizeof(uint16_t);
  }

  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes + block_scale_bytes;
}

void xnn_pack_qb4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  const size_t extra_bytes_bl = sizeof(uint16_t);
  const size_t extra_bytes_n = sizeof(uint32_t);
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    xnn_pack_qs8_qb4w_gemm_gio_w(
        /*g=*/groups,
        /*nc=*/output_channels,
        /*kc=*/input_channels,
        /*nr=*/nr,
        /*kr=*/kr,
        /*sr=*/sr,
        /*k_stride=*/k_stride,
        /*bl=*/block_size,
        /*kernel=*/(const uint8_t*)weights,
        /*bias=*/nullptr,
        /*scale=*/(const xnn_bfloat16*)extra_data1,
        /*packed_weights=*/packed_weights_ptr,
        /*extra_bytes_bl=*/nr * extra_bytes_bl,
        /*extra_bytes_n=*/nr * extra_bytes_n,
        /*params*/ (const struct xnn_qs8_qc4w_packing_params*)params);
  } else {
    xnn_pack_qs8_qb4w_gemm_goi_w(
        /*g=*/groups,
        /*nc=*/output_channels,
        /*kc=*/input_channels,
        /*nr=*/nr,
        /*kr=*/kr,
        /*sr=*/sr,
        /*bl=*/block_size,
        /*kernel=*/(const uint8_t*)weights,
        /*bias=*/nullptr,
        /*scale=*/(const xnn_bfloat16*)extra_data1,
        /*packed_weights=*/packed_weights_ptr,
        /*extra_bytes_bl=*/nr * extra_bytes_bl,
        /*extra_bytes_n=*/nr * extra_bytes_n,
        /*params*/ (const struct xnn_qs8_qc4w_packing_params*)params);
  }

  // fill in kernel scales
  const size_t num_blocks = input_channels / block_size;
  const size_t weights_stride = xnn_packed_stride_qb4_weights_and_biases(
      gemm_config, input_channels, block_size, k_stride, extra_bytes_n);
  void* weights_start =
      (void*)((uintptr_t)packed_weights_ptr +
              nr * (sizeof(float) + (block_size * sizeof(int8_t) / 2)));

  const size_t block_stride = /*weights*/ block_size / 2 + sizeof(uint16_t);
  xnn_init_blockwise_scale_bf16_params(
      output_channels, nr, nr * weights_stride,
      /*num_blocks=*/num_blocks,
      /*block_stride=*/gemm_config->nr * block_stride,
      (const xnn_bfloat16*)extra_data1, weights_start);

  // fill in bias if not null
  if (accumulator_init != nullptr) {
    weights_start = (void*)((uintptr_t)packed_weights_ptr +
                            gemm_config->nr * (weights_stride - sizeof(float)));
    xnn_init_qs8_qc8w_scale_fp32_params(
        output_channels, gemm_config->nr, gemm_config->nr * weights_stride,
        (const float*)accumulator_init, weights_start);
  }
}

size_t xnn_packed_stride_qu8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t unused_k,
    size_t unused_block_size, size_t k_stride, size_t extra_bytes) {
  const size_t bias_element_size = sizeof(int32_t);
  const size_t log2_filter_element_size = XNN_LOG2_SIZEOF_INT8_T;
  return (k_stride << log2_filter_element_size) + bias_element_size +
         extra_bytes;
}

void xnn_pack_qu8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const size_t extra_bytes =
      extra_data0_element_size + extra_data1_element_size;
  const size_t weights_stride = xnn_packed_stride_qs8_weights_and_biases(
      gemm_config, input_channels, unused_block_size, k_stride, extra_bytes);
  return pack_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      unused_block_size, weights_stride,
      (xnn_packw_gemm_gio_ukernel_fn)xnn_pack_qu8_gemm_gio_w,
      (xnn_packw_gemm_goi_ukernel_fn)xnn_pack_qu8_gemm_goi_w, accumulator_init,
      weights, init_extra_data0_fn, extra_data0, extra_data0_element_size,
      init_extra_data1_fn, extra_data1, extra_data1_element_size,
      packed_weights_ptr, extra_bytes, params);
}

#if XNN_ENABLE_KLEIDIAI
size_t xnn_packed_stride_kai_qs4_weights_and_biases_sme(
    const struct xnn_gemm_config* gemm_config, size_t k, size_t unused_k_stride,
    size_t unused_block_size,  //
    size_t extra_bytes) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = gemm_config->nr;
  const uint32_t sr = gemm_config->nr;
  size_t ret_val =
      kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
          k, nr, kr, sr) /
      kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nr);
  return ret_val;
}

void xnn_pack_kai_qs4_weights_and_biases_sme(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const struct xnn_qs8_qc4w_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params);

  struct kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon_params kai_params;
  kai_params.lhs_zero_point = xnn_params->input_zero_point;
  kai_params.rhs_zero_point = xnn_params->kernel_zero_point;

  bool free_accumulator_init = false;
  if (extra_data0 == nullptr) {
    extra_data0 = calloc(output_channels, sizeof(float));
    free_accumulator_init = true;
  }
  kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(
      groups, output_channels, input_channels, nr, kr, sr,
      /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
      /*bias=*/reinterpret_cast<const float*>(extra_data0),
      /*scale=*/reinterpret_cast<const float*>(extra_data1),
      /*rhs_packed=*/packed_weights_ptr,
      /*extra_bytes=*/0, &kai_params);
  if (free_accumulator_init) {
    free((void*)extra_data0);
  }
}

size_t xnn_packed_stride_kai_qs4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k,
    size_t unused_block_size, size_t unused_k_stride, size_t extra_bytes) {
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  return kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(k, /*nr=*/1,
                                                                  kr, sr);
}

void xnn_pack_kai_qs4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const struct xnn_qs8_qc4w_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params);

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // Repack the packing params.
    struct kai_rhs_pack_kxn_qsi4cxp_qs4cxs1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;

    kai_run_rhs_pack_kxn_qsi4cxp_qs4cxs1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
  } else {
    // Repack the packing params.
    struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;

    kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
  }
}

#if XNN_ENABLE_KLEIDIAI
size_t xnn_packed_stride_kai_qs8_qc8w_weights_and_biases_sme2(
    const struct xnn_gemm_config* gemm_config, size_t k,
    size_t unused_block_size, size_t unused_k_stride, size_t extra_bytes) {
  size_t ret_val =
      kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
          k) /
      kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme();
  return ret_val;
}

void transpose_weights_x8(const int8_t* in, int8_t* out, size_t height,
                          size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      out[j * height + i] = in[i * width + j];
    }
  }
}

void xnn_pack_kai_qs8_qc8w_weights_and_biases_sme2(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t rhs_stride = output_channels * sizeof(int8_t);

  // Some packing kernels assume that the bias is non-null. Allocate a zero
  // initialized array as a workaround if bias is null.
  bool free_accumulator_init = false;
  if (accumulator_init == NULL) {
    accumulator_init = calloc(output_channels, sizeof(int32_t));
    free_accumulator_init = true;
  }
  const struct xnn_qs8_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_packing_params*>(params);
  struct kai_rhs_pack_qsi8cx_params kai_params;
  kai_params.lhs_zero_point = xnn_params->input_zero_point;
  kai_params.scale_multiplier = 1.f;
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/weights,
        /*bias=*/accumulator_init,
        /*scale=*/extra_data1,
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
  } else {
    // Transpose the weights until the transpose packing function is ready.
    int8_t* tmp_data =
        (int8_t*)malloc(input_channels * output_channels * sizeof(int8_t));
    transpose_weights_x8((const int8_t*)weights, tmp_data, output_channels,
                         input_channels);
    kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/tmp_data,
        /*bias=*/accumulator_init,
        /*scale=*/extra_data0,
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
    free(tmp_data);
  }
  if (free_accumulator_init) {
    free((void*)accumulator_init);
  }
}
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_packed_stride_kai_f16_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k,
    size_t unused_block_size, size_t unused_k_stride, size_t extra_bytes) {
  size_t ret_val =
      kai_get_rhs_packed_stride_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(k) /
      kai_get_n_step_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme();
  return ret_val;
}

void transpose_weights_x16(const xnn_float16* in, xnn_float16* out,
                           size_t height, size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      out[j * height + i] = in[i * width + j];
    }
  }
}

size_t xnn_packed_stride_kai_qs8_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k,
    size_t unused_block_size, size_t unused_k_stride, size_t extra_bytes) {
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  return kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(k, /*nr=*/1,
                                                                    kr, sr);
}

void xnn_pack_kai_qs8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const struct xnn_qs8_qc8w_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_qc8w_packing_params*>(params);

  // Repack the packing params.
  struct kai_rhs_pack_qsi8cx_params kai_params;
  kai_params.lhs_zero_point = xnn_params->input_zero_point;
  kai_params.scale_multiplier = xnn_params->scale_multiplier;

  const size_t weights_group_stride =
      sizeof(int8_t) * input_channels * output_channels;
  const size_t n_stride = round_up(output_channels, nr);
  const size_t packed_weights_group_stride =
      n_stride * xnn_packed_stride_kai_qs8_weights_and_biases(
                     gemm_config, input_channels, unused_block_size, k_stride,
                     extra_data0_element_size + extra_data1_element_size);

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    for (size_t group = 0; group < groups; group++) {
      kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(
          /*groups=*/1, output_channels, input_channels, nr, kr, sr,
          /*rhs=*/
          reinterpret_cast<const int8_t*>((uintptr_t)weights +
                                          group * weights_group_stride),
          /*bias=*/
          extra_data0 ? reinterpret_cast<const float*>(extra_data0) +
                            group * output_channels
                      : NULL,
          /*scale=*/
          extra_data1 ? reinterpret_cast<const float*>(extra_data1) +
                            group * output_channels
                      : NULL,
          /*rhs_packed=*/
          (void*)((uintptr_t)packed_weights_ptr +
                  group * packed_weights_group_stride),
          /*extra_bytes=*/0, &kai_params);
    }
  } else {
    for (size_t group = 0; group < groups; group++) {
      kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
          /*groups=*/1, output_channels, input_channels, nr, kr, sr,
          /*rhs=*/
          reinterpret_cast<const int8_t*>((uintptr_t)weights +
                                          group * weights_group_stride),
          /*bias=*/
          extra_data0 ? reinterpret_cast<const float*>(extra_data0) +
                            group * output_channels
                      : NULL,
          /*scale=*/
          extra_data1 ? reinterpret_cast<const float*>(extra_data1) +
                            group * output_channels
                      : NULL,
          /*rhs_packed=*/
          (void*)((uintptr_t)packed_weights_ptr +
                  group * packed_weights_group_stride),
          /*extra_bytes=*/0, &kai_params);
    }
  }
}

size_t xnn_packed_stride_kai_f32_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k,
    size_t unused_block_size, size_t unused_k_stride, size_t extra_bytes) {
  size_t ret_val =
      kai_get_rhs_packed_stride_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(k) /
      kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();
  return ret_val;
}

void xnn_pack_kai_f16_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  assert(extra_data0 == nullptr);
  assert(extra_data1 == nullptr);
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t rhs_stride = output_channels * sizeof(xnn_float16);

  // Some packing kernels assume that the bias is non-null. Allocate a zero
  // initialized array as a workaround if bias is null.
  bool free_accumulator_init = false;
  if (accumulator_init == NULL) {
    accumulator_init = calloc(output_channels, sizeof(xnn_float16));
    free_accumulator_init = true;
  }
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/weights,
        /*bias=*/accumulator_init,
        /*scale=*/extra_data1,
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/extra_data0_element_size + extra_data1_element_size,
        NULL);
  } else {
    // Transpose the weights until the transpose packing function is ready.
    xnn_float16* tmp_data = (xnn_float16*)malloc(
        input_channels * output_channels * sizeof(xnn_float16));
    transpose_weights_x16((const xnn_float16*)weights, tmp_data,
                          output_channels, input_channels);
    kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/tmp_data,
        /*bias=*/accumulator_init,
        /*scale=*/extra_data1,
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/extra_data0_element_size + extra_data1_element_size,
        NULL);
    free(tmp_data);
  }
  if (free_accumulator_init) {
    free((void*)accumulator_init);
  }
}

void transpose_weights(const float* in, float* out, size_t height,
                       size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      out[j * height + i] = in[i * width + j];
    }
  }
}

void xnn_pack_kai_f32_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t unused_block_size, size_t k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  assert(extra_data0 == nullptr);
  assert(extra_data1 == nullptr);
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t rhs_stride = output_channels * sizeof(float);

  // Some packing kernels assume that the bias is non-null. Allocate a zero
  // initialized array as a workaround if bias is null.
  bool free_accumulator_init = false;
  if (accumulator_init == NULL) {
    accumulator_init = calloc(output_channels, sizeof(float));
    free_accumulator_init = true;
  }
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*bias=*/reinterpret_cast<const float*>(accumulator_init),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/extra_data0_element_size + extra_data1_element_size,
        NULL);
  } else {
    // Transpose the weights until the transpose packing function is ready.
    float* tmp_data =
        (float*)malloc(input_channels * output_channels * sizeof(float));
    transpose_weights((const float*)weights, tmp_data, output_channels,
                      input_channels);
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        groups, output_channels, input_channels, nr, kr, sr, rhs_stride,
        /*rhs=*/reinterpret_cast<const uint8_t*>(tmp_data),
        /*bias=*/reinterpret_cast<const float*>(accumulator_init),
        /*scale=*/reinterpret_cast<const float*>(extra_data1),
        /*rhs_packed=*/packed_weights_ptr,
        /*extra_bytes=*/extra_data0_element_size + extra_data1_element_size,
        NULL);
    free(tmp_data);
  }
  if (free_accumulator_init) {
    free((void*)accumulator_init);
  }
}

size_t xnn_packed_stride_kai_qb4_weights_and_biases(
    const struct xnn_gemm_config* gemm_config, size_t k, size_t block_size,
    size_t unused_k_stride, size_t extra_bytes) {
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const uint32_t nr = gemm_config->nr;

  // We want the weight stride with nr = 1, but kleidi enforces a constraint
  // where nr % 4 == 0. So instead we give nr to get the nr-scaled stride, and
  // divide by nr to scaled down the stride.
  const size_t nr_scaled_packed_stride =
      kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
          k, nr, kr, sr, block_size, kai_datatype::kai_dt_bf16);

  return nr_scaled_packed_stride / nr;
}

void xnn_pack_kai_qb4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t unused_k_stride, const void* accumulator_init,
    const void* weights, xnn_init_scale_params_fn init_extra_data0_fn,
    const void* extra_data0, size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const struct xnn_qs8_qc4w_packing_params* xnn_params =
      reinterpret_cast<const struct xnn_qs8_qc4w_packing_params*>(params);

  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    struct kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;
    kai_params.scale_dt = kai_datatype::kai_dt_bf16;
    size_t rhs_stride = (output_channels + 1) / 2;
    size_t blocks_per_row = (input_channels + block_size - 1) / block_size;
    kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*bl=*/block_size,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*rhs_stride=*/rhs_stride,
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const uint16_t*>(extra_data1),
        /*scale_stride=*/blocks_per_row * sizeof(uint16_t),
        /*rhs_packed*/ packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
  } else {
    // Repack the packing params.
    struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params kai_params;
    kai_params.lhs_zero_point = xnn_params->input_zero_point;
    kai_params.rhs_zero_point = xnn_params->kernel_zero_point;
    kai_params.scale_dt = kai_datatype::kai_dt_bf16;
    size_t rhs_stride = (input_channels + 1) / 2;
    size_t blocks_per_row = (input_channels + block_size - 1) / block_size;
    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        groups, output_channels, input_channels, nr, kr, sr,
        /*bl=*/block_size,
        /*rhs=*/reinterpret_cast<const uint8_t*>(weights),
        /*rhs_stride=*/rhs_stride,
        /*bias=*/reinterpret_cast<const float*>(extra_data0),
        /*scale=*/reinterpret_cast<const uint16_t*>(extra_data1),
        /*scale_stride=*/blocks_per_row * sizeof(uint16_t),
        /*rhs_packed*/ packed_weights_ptr,
        /*extra_bytes=*/0, &kai_params);
  }

  // init bias
  const size_t weights_stride = xnn_packed_stride_kai_qb4_weights_and_biases(
      gemm_config, input_channels, block_size, unused_k_stride, 0);
  if (accumulator_init != NULL) {
    void* weights_start =
        (void*)((uintptr_t)packed_weights_ptr +
                nr * (sizeof(float) + (block_size * sizeof(int8_t) / 2)));
    weights_start = (void*)((uintptr_t)packed_weights_ptr +
                            nr * (weights_stride - sizeof(float)));
    xnn_init_qs8_qc8w_scale_fp32_params(
        output_channels, nr, nr * weights_stride,
        (const float*)accumulator_init, weights_start);
  }
}
#endif  // XNN_ENABLE_KLEIDIAI

void xnn_pack_f32_qs8w_gemm_gio_w(size_t g, size_t nc, size_t kc, size_t nr,
                                  size_t kr, size_t sr, size_t k_stride,
                                  const int8_t* k, const float* bias,
                                  const float* scale, void* packed_weights,
                                  size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t* b = (const int32_t*)bias;
  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights = (int32_t*)packed_weights + nr;

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          const size_t kc_begin =
              round_down_po2(kr_block_start, skr) +
              ((kr_block_start + nr_block_offset * kr) & (skr - 1));
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_idx = kc_begin + kr_block_offset;
            const int8_t kv =
                kc_idx < kc
                    ? k[kc_idx * k_stride + (nr_block_start + nr_block_offset)]
                    : INT8_C(0);
            ((int8_t*)packed_weights)[kr_block_offset] = kv;
          }
          packed_weights = (int8_t*)packed_weights + kr;
        }
        packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc * kc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_goki_w(size_t g, size_t nc, size_t ks, size_t kc,
                              size_t nr, size_t kr, size_t sr, const float* k,
                              const float* b, const void* scale,
                              float* packed_weights, size_t extra_bytes,
                              const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            float* end = packed_weights + kr;
            if (kc_begin < kc_end) {
              std::copy_n(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, packed_weights);
              packed_weights += kc_end - kc_begin;
            }
            std::fill(packed_weights, end, 0.0f);
            packed_weights = end;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (float*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f16_conv_goki_w(size_t g, size_t nc, size_t ks, size_t kc,
                              size_t nr, size_t kr, size_t sr,
                              const uint16_t* k, const uint16_t* b,
                              const void* scale, uint16_t* packed_weights,
                              size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            uint16_t* end = packed_weights + kr;
            if (kc_begin < kc_end) {
              std::copy_n(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, packed_weights);
              packed_weights += kc_end - kc_begin;
            }
            std::fill(packed_weights, end, UINT16_C(0));
            packed_weights = end;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (uint16_t*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_to_f16_conv_goki_w(size_t g, size_t nc, size_t ks, size_t kc,
                                     size_t nr, size_t kr, size_t sr,
                                     const float* k, const float* b,
                                     const void* scale,
                                     xnn_float16* packed_weights,
                                     size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            xnn_float16* end = packed_weights + kr;
            if (kc_begin < kc_end) {
              std::copy_n(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, packed_weights);
              packed_weights += kc_end - kc_begin;
            }
            std::fill(packed_weights, end, 0.0f);
            packed_weights = end;
          }
          packed_weights += (nr - nr_block_size) * kr;
        }
      }
      packed_weights = (xnn_float16*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qu8_conv_goki_w(size_t g, size_t nc, size_t ks, size_t kc,
                              size_t nr, size_t kr, size_t sr, const uint8_t* k,
                              const int32_t* b, const void* scale,
                              void* packed_weights, size_t extra_bytes,
                              const struct xnn_qu8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t bzp =
      (int32_t)ks * (int32_t)kc * izp * (int32_t)params->kernel_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b, bzp);
      packed_weights =
          (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            uint8_t* end = (uint8_t*)packed_weights + kr;
            if (kc_begin < kc_end) {
              int32_t ksum = copy_n_and_sum(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, (uint8_t*)packed_weights);
              packed_weights = (uint8_t*)packed_weights + kc_end - kc_begin;
              packed_b[nr_block_offset] =
                  packed_b[nr_block_offset] - ksum * izp;
            }
            std::fill((uint8_t*)packed_weights, end, params->kernel_zero_point);
            packed_weights = end;
          }
          packed_weights = (uint8_t*)packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_to_qu8_conv_goki_w(
    size_t g, size_t nc, size_t ks, size_t kc, size_t nr, size_t kr, size_t sr,
    const int8_t* k, const int32_t* b, const float* scale, void* packed_weights,
    size_t extra_bytes, const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t)params->input_zero_point + 128;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights =
          (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            int8_t* end = (int8_t*)packed_weights + kr;
            if (kc_begin < kc_end) {
              uint32_t ksum = copy_n_and_sum(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, (int8_t*)packed_weights);
              packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
              packed_b[nr_block_offset] =
                  packed_b[nr_block_offset] - ksum * izp;
            }
            std::fill((int8_t*)packed_weights, end, INT8_C(0));
            packed_weights = end;
          }
          packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_qs8_conv_goki_w(size_t g, size_t nc, size_t ks, size_t kc,
                              size_t nr, size_t kr, size_t sr, const int8_t* k,
                              const int32_t* b, const float* scale,
                              void* packed_weights, size_t extra_bytes,
                              const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (int32_t)params->input_zero_point;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights =
          (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
             kr_block_start += kr) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            int8_t* end = (int8_t*)packed_weights + kr;
            if (kc_begin < kc_end) {
              uint32_t ksum = copy_n_and_sum(
                  &k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                     kc_begin],
                  kc_end - kc_begin, (int8_t*)packed_weights);
              packed_weights = (int8_t*)packed_weights + kc_end - kc_begin;
              packed_b[nr_block_offset] =
                  packed_b[nr_block_offset] - ksum * izp;
            }
            std::fill((int8_t*)packed_weights, end, INT8_C(0));
            packed_weights = end;
          }
          packed_weights = (int8_t*)packed_weights + (nr - nr_block_size) * kr;
        }
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += ks * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  } while (--g != 0);
}

void xnn_pack_f32_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr,
                             size_t kr, size_t sr, const float* k,
                             const float* b, const void* scale,
                             float* packed_weights, size_t extra_bytes,
                             const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr;
             sr_block_offset++) {
          // TODO: Is there a more precise zeroing we could do here?
          std::fill_n(packed_weights, nr * kr, 0.0f);
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1);
               nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] =
                k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (float*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f16_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr,
                             size_t kr, size_t sr, const uint16_t* k,
                             const uint16_t* b, const void* scale,
                             uint16_t* packed_weights, size_t extra_bytes,
                             const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr;
             sr_block_offset++) {
          // TODO: Is there a more precise zeroing we could do here?
          std::fill_n(packed_weights, nr * kr, UINT16_C(0));
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1);
               nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] =
                k[ki * g * nc + (nr_block_start + nr_block_offset)];
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (uint16_t*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr,
                                    size_t kr, size_t sr, const float* k,
                                    const float* b, const void* scale,
                                    xnn_float16* packed_weights,
                                    size_t extra_bytes, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      copy_bias(b, nr_block_start, nr_block_size, packed_weights);
      packed_weights += nr;

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr;
             sr_block_offset++) {
          // TODO: Is there a more precise zeroing we could do here?
          std::fill_n(packed_weights, nr * kr, static_cast<xnn_float16>(0.0f));
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1);
               nr_block_offset < nr_block_size; nr_block_offset += sr) {
            packed_weights[nr_block_offset * kr] = xnn_float16_from_float(
                k[ki * g * nc + (nr_block_start + nr_block_offset)]);
          }
          packed_weights += nr * kr;
        }
      }
      packed_weights = (xnn_float16*)((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qu8_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr,
                             size_t kr, size_t sr, const uint8_t* k,
                             const int32_t* b, const void* scale,
                             void* packed_weights, size_t extra_bytes,
                             const struct xnn_qu8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t bzp = (int32_t)ks * izp * (int32_t)params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b, bzp);
      packed_weights =
          (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr;
             sr_block_offset++) {
          // TODO: Is there a more precise zeroing we could do here?
          std::fill_n((uint8_t*)packed_weights, nr * kr,
                      params->kernel_zero_point);
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1);
               nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const uint8_t kv =
                k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((uint8_t*)packed_weights)[nr_block_offset * kr] = kv;
            packed_b[nr_block_offset] =
                packed_b[nr_block_offset] - (int32_t)kv * izp;
          }
          packed_weights = (uint8_t*)packed_weights + nr * kr;
        }
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void pack_qs8_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr, size_t kr,
                         size_t sr, const int8_t* k, const int32_t* b,
                         const float* scale, void* packed_weights,
                         size_t extra_bytes, int32_t zero_point_offset,
                         const struct xnn_qs8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const uint32_t izp = (uint32_t)params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < g; i++) {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);
      unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
      copy_bias(b, nr_block_start, nr_block_size, packed_b);
      packed_weights =
          (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));

      for (size_t ki = 0; ki < ks; ki++) {
        for (size_t sr_block_offset = 0; sr_block_offset < sr;
             sr_block_offset++) {
          // TODO: Is there a more precise zeroing we could do here?
          std::fill_n((int8_t*)packed_weights, nr * kr, INT8_C(0));
          for (size_t nr_block_offset = (-sr_block_offset) & (sr - 1);
               nr_block_offset < nr_block_size; nr_block_offset += sr) {
            const int8_t kv =
                k[ki * g * nc + (nr_block_start + nr_block_offset)];
            ((int8_t*)packed_weights)[nr_block_offset * kr] = kv;
            packed_b[nr_block_offset] =
                packed_b[nr_block_offset] - (uint32_t)kv * izp;
          }
          packed_weights = (int8_t*)packed_weights + nr * kr;
        }
      }
      packed_weights =
          reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
    }
    k += nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_conv_kgo_w(size_t g, size_t nc, size_t ks, size_t nr,
                             size_t kr, size_t sr, const int8_t* k,
                             const int32_t* b, const float* scale,
                             void* packed_weights, size_t extra_bytes,
                             const struct xnn_qs8_packing_params* params) {
  pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                      extra_bytes, /*zero_point_offset=*/0, params);
}

void xnn_pack_qs8_to_qu8_conv_kgo_w(
    size_t g, size_t nc, size_t ks, size_t nr, size_t kr, size_t sr,
    const int8_t* k, const int32_t* b, const float* scale, void* packed_weights,
    size_t extra_bytes, const struct xnn_qs8_packing_params* params) {
  pack_qs8_conv_kgo_w(g, nc, ks, nr, kr, sr, k, b, scale, packed_weights,
                      extra_bytes, /*zero_point_offset=*/128, params);
}

void xnn_pack_f32_deconv_goki_w(size_t g, size_t nc, size_t kh, size_t kw,
                                size_t kc, size_t sh, size_t sw, size_t nr,
                                size_t kr, size_t sr, const float* k,
                                const float* b, const void* scale,
                                float* packed_weights, size_t extra_bytes,
                                struct subconvolution_params* subconv_params,
                                const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc;
             nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          copy_bias(b, nr_block_start, nr_block_size, packed_weights);
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0;
                   kr_block_start < round_up_po2(kc, skr);
                   kr_block_start += kr) {
                for (size_t nr_block_offset = 0;
                     nr_block_offset < nr_block_size; nr_block_offset++) {
                  const size_t kc_begin =
                      round_down_po2(kr_block_start, skr) +
                      ((kr_block_start + nr_block_offset * kr) & (skr - 1));
                  const size_t kc_end = std::min(kc, kc_begin + kr);
                  float* end = packed_weights + kr;
                  if (kc_begin < kc_end) {
                    std::copy_n(
                        &k[(((nr_block_start + nr_block_offset) * kh + ky) *
                                kw +
                            kx) *
                               kc +
                           kc_begin],
                        kc_end - kc_begin, packed_weights);
                    packed_weights += kc_end - kc_begin;
                  }
                  std::fill(packed_weights, end, 0.0f);
                  packed_weights = end;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights =
              reinterpret_cast<float*>((uintptr_t)packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f16_deconv_goki_w(size_t g, size_t nc, size_t kh, size_t kw,
                                size_t kc, size_t sh, size_t sw, size_t nr,
                                size_t kr, size_t sr, const uint16_t* k,
                                const uint16_t* b, const void* scale,
                                uint16_t* packed_weights, size_t extra_bytes,
                                struct subconvolution_params* subconv_params,
                                const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc;
             nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          copy_bias(b, nr_block_start, nr_block_size, packed_weights);
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0;
                   kr_block_start < round_up_po2(kc, skr);
                   kr_block_start += kr) {
                for (size_t nr_block_offset = 0;
                     nr_block_offset < nr_block_size; nr_block_offset++) {
                  const size_t kc_begin =
                      round_down_po2(kr_block_start, skr) +
                      ((kr_block_start + nr_block_offset * kr) & (skr - 1));
                  const size_t kc_end = std::min(kc, kc_begin + kr);
                  uint16_t* end = packed_weights + kr;
                  if (kc_begin < kc_end) {
                    std::copy_n(
                        &k[(((nr_block_start + nr_block_offset) * kh + ky) *
                                kw +
                            kx) *
                               kc +
                           kc_begin],
                        kc_end - kc_begin, packed_weights);
                    packed_weights += kc_end - kc_begin;
                  }
                  std::fill(packed_weights, end, UINT16_C(0));
                  packed_weights = end;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<uint16_t*>(
              (uintptr_t)packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_f32_to_f16_deconv_goki_w(
    size_t g, size_t nc, size_t kh, size_t kw, size_t kc, size_t sh, size_t sw,
    size_t nr, size_t kr, size_t sr, const float* k, const float* b,
    const void* scale, xnn_float16* packed_weights, size_t extra_bytes,
    struct subconvolution_params* subconv_params, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc;
             nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          copy_bias(b, nr_block_start, nr_block_size, packed_weights);
          packed_weights += nr;
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0;
                   kr_block_start < round_up_po2(kc, skr);
                   kr_block_start += kr) {
                for (size_t nr_block_offset = 0;
                     nr_block_offset < nr_block_size; nr_block_offset++) {
                  const size_t kc_begin =
                      round_down_po2(kr_block_start, skr) +
                      ((kr_block_start + nr_block_offset * kr) & (skr - 1));
                  const size_t kc_end = std::min(kc, kc_begin + kr);
                  xnn_float16* end = packed_weights + kr;
                  if (kc_begin < kc_end) {
                    std::copy_n(
                        &k[(((nr_block_start + nr_block_offset) * kh + ky) *
                                kw +
                            kx) *
                               kc +
                           kc_begin],
                        kc_end - kc_begin, packed_weights);
                    packed_weights += kc_end - kc_begin;
                  }
                  std::fill(packed_weights, end,
                            static_cast<xnn_float16>(0.0f));
                  packed_weights = end;
                }
                packed_weights += (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights = reinterpret_cast<xnn_float16*>(
              (uintptr_t)packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void pack_qs8_deconv_goki_w(size_t groups, size_t nc, size_t kh, size_t kw,
                            size_t kc, size_t sh, size_t sw, size_t nr,
                            size_t kr, size_t sr, const int8_t* k,
                            const int32_t* b, const float* scale,
                            void* packed_weights, size_t extra_bytes,
                            int32_t zero_point_offset,
                            struct subconvolution_params* subconv_params,
                            const struct xnn_qs8_packing_params* params) {
  assert(groups != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const uint32_t izp = (uint32_t)params->input_zero_point + zero_point_offset;
  for (size_t i = 0; i < groups; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        for (size_t nr_block_start = 0; nr_block_start < nc;
             nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
          copy_bias(b, nr_block_start, nr_block_size, packed_b);
          packed_weights =
              (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0;
                   kr_block_start < round_up_po2(kc, skr);
                   kr_block_start += kr) {
                for (size_t nr_block_offset = 0;
                     nr_block_offset < nr_block_size; nr_block_offset++) {
                  const size_t kc_begin =
                      round_down_po2(kr_block_start, skr) +
                      ((kr_block_start + nr_block_offset * kr) & (skr - 1));
                  const size_t kc_end = std::min(kc, kc_begin + kr);
                  int8_t* end = (int8_t*)packed_weights + kr;
                  if (kc_begin < kc_end) {
                    uint32_t ksum = copy_n_and_sum(
                        &k[(((nr_block_start + nr_block_offset) * kh + ky) *
                                kw +
                            kx) *
                               kc +
                           kc_begin],
                        kc_end - kc_begin, (int8_t*)packed_weights);
                    packed_b[nr_block_offset] =
                        packed_b[nr_block_offset] - ksum * izp;
                    packed_weights =
                        (int8_t*)packed_weights + kc_end - kc_begin;
                  }
                  std::fill((int8_t*)packed_weights, end, INT8_C(0));
                  packed_weights = end;
                }
                packed_weights =
                    (int8_t*)packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights =
              reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nc;
    }
  }
}

void xnn_pack_qs8_deconv_goki_w(size_t g, size_t nc, size_t kh, size_t kw,
                                size_t kc, size_t sh, size_t sw, size_t nr,
                                size_t kr, size_t sr, const int8_t* k,
                                const int32_t* b, const float* scale,
                                void* packed_weights, size_t extra_bytes,
                                struct subconvolution_params* subconv_params,
                                const struct xnn_qs8_packing_params* params) {
  pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                         packed_weights, extra_bytes, /*zero_point_offset=*/0,
                         subconv_params, params);
}

void xnn_pack_qs8_to_qu8_deconv_goki_w(
    size_t g, size_t nc, size_t kh, size_t kw, size_t kc, size_t sh, size_t sw,
    size_t nr, size_t kr, size_t sr, const int8_t* k, const int32_t* b,
    const float* scale, void* packed_weights, size_t extra_bytes,
    struct subconvolution_params* subconv_params,
    const struct xnn_qs8_packing_params* params) {
  pack_qs8_deconv_goki_w(g, nc, kh, kw, kc, sh, sw, nr, kr, sr, k, b, scale,
                         packed_weights, extra_bytes, /*zero_point_offset=*/128,
                         subconv_params, params);
}

void xnn_pack_qu8_deconv_goki_w(size_t g, size_t nc, size_t kh, size_t kw,
                                size_t kc, size_t sh, size_t sw, size_t nr,
                                size_t kr, size_t sr, const uint8_t* k,
                                const int32_t* b, const void* scale,
                                void* packed_weights, size_t extra_bytes,
                                struct subconvolution_params* subconv_params,
                                const struct xnn_qu8_packing_params* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t kzp = (int32_t)params->kernel_zero_point;
  for (size_t i = 0; i < g; i++) {
    for (size_t oy = 0; oy < sh; oy++) {
      for (size_t ox = 0; ox < sw; ox++) {
        if (i == 0) {
          (*subconv_params++).weights = packed_weights;
        }
        const int32_t bzp = (int32_t)divide_round_up(kh - oy, sh) *
                            (int32_t)divide_round_up(kw - ox, sw) *
                            (int32_t)kc * izp * kzp;
        for (size_t nr_block_start = 0; nr_block_start < nc;
             nr_block_start += nr) {
          const size_t nr_block_size = min(nc - nr_block_start, nr);
          unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
          copy_bias(b, nr_block_start, nr_block_size, packed_b, bzp);
          packed_weights =
              (void*)((uintptr_t)packed_weights + nr * sizeof(int32_t));
          for (size_t ky = oy; ky < kh; ky += sh) {
            for (size_t kx = ox; kx < kw; kx += sw) {
              for (size_t kr_block_start = 0;
                   kr_block_start < round_up_po2(kc, skr);
                   kr_block_start += kr) {
                for (size_t nr_block_offset = 0;
                     nr_block_offset < nr_block_size; nr_block_offset++) {
                  const size_t kc_begin =
                      round_down_po2(kr_block_start, skr) +
                      ((kr_block_start + nr_block_offset * kr) & (skr - 1));
                  const size_t kc_end = std::min(kc, kc_begin + kr);
                  uint8_t* end = (uint8_t*)packed_weights + kr;
                  if (kc_begin < kc_end) {
                    int32_t ksum = copy_n_and_sum(
                        &k[(((nr_block_start + nr_block_offset) * kh + ky) *
                                kw +
                            kx) *
                               kc +
                           kc_begin],
                        kc_end - kc_begin, (uint8_t*)packed_weights);
                    packed_b[nr_block_offset] =
                        packed_b[nr_block_offset] - ksum * izp;
                    packed_weights =
                        (uint8_t*)packed_weights + kc_end - kc_begin;
                  }
                  std::fill((uint8_t*)packed_weights, end,
                            params->kernel_zero_point);
                  packed_weights = end;
                }
                packed_weights =
                    (uint8_t*)packed_weights + (nr - nr_block_size) * kr;
              }
            }
          }
          packed_weights =
              reinterpret_cast<void*>((uintptr_t)packed_weights + extra_bytes);
        }
      }
    }
    k += kh * kw * kc * nc;
    if XNN_UNPREDICTABLE (b != nullptr) {
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

void xnn_pack_f32_dwconv_ghw_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const float* k,
                               const float* b, const void* scale,
                               float* packed_weights,
                               size_t per_tile_extra_bytes,
                               const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  assert(kernel_size <= primary_tile);

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const float kv =
            k[((cr_block_start + cr_block_offset) * h + y) * w + x];
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                0.0f);
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_f16_dwconv_ghw_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const uint16_t* k,
                               const uint16_t* b, const void* scale,
                               uint16_t* packed_weights,
                               size_t per_tile_extra_bytes,
                               const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const uint16_t kv =
            k[((cr_block_start + cr_block_offset) * h + y) * w + x];
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                UINT16_C(0));
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_f32_to_f16_dwconv_ghw_w(size_t primary_tile, size_t h, size_t w,
                                      size_t c, size_t channel_tile,
                                      const float* k, const float* b,
                                      const void* scale,
                                      xnn_float16* packed_weights,
                                      size_t per_tile_extra_bytes,
                                      const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const xnn_float16 kv = xnn_float16_from_float(
            k[((cr_block_start + cr_block_offset) * h + y) * w + x]);
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                static_cast<xnn_float16>(0.0f));
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_qu8_dwconv_ghw_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const uint8_t* k,
                               const int32_t* b, const void* scale,
                               void* packed_weights,
                               size_t per_tile_extra_bytes,
                               const struct xnn_qu8_packing_params* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t boff =
      (int32_t)h * (int32_t)w * izp * (int32_t)params->kernel_zero_point;
  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_b, boff);
    packed_weights =
        (void*)((uintptr_t)packed_weights + channel_tile * sizeof(int32_t));

    // Biases need to be offset by all kernel values.
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          unaligned_indexed_store_s32(
              packed_b, cr_block_offset,
              unaligned_indexed_load_s32(packed_b, cr_block_offset) -
                  (int32_t)kv * izp);
        }
      }
    }

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const uint8_t kv =
            k[((cr_block_start + cr_block_offset) * h + y) * w + x];
        *((uint8_t*)packed_weights) = kv;
        packed_weights = (void*)((uintptr_t)packed_weights + sizeof(uint8_t));
      }
      packed_weights =
          (void*)((uintptr_t)packed_weights +
                  (channel_tile - cr_block_size) * sizeof(uint8_t));
      advance_x_y(h, &x, &y);
    }
    std::fill_n((uint8_t*)packed_weights,
                (primary_tile - kernel_size) * channel_tile,
                params->kernel_zero_point);
    packed_weights = (void*)((uintptr_t)packed_weights +
                             (primary_tile - kernel_size) * cr_block_size);
  }
}

void xnn_pack_qs8_dwconv_ghw_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const int8_t* k,
                               const int32_t* b, const float* scale,
                               void* packed_weights,
                               size_t per_tile_extra_bytes,
                               const struct xnn_qs8_packing_params* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;
  const uint32_t izp = (uint32_t)params->input_zero_point;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_b);
    packed_weights =
        (void*)((uintptr_t)packed_weights + channel_tile * sizeof(int32_t));

    // Biases need to be offset by all kernel values.
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const int8_t kv =
              k[((cr_block_start + cr_block_offset) * h + y) * w + x];
          packed_b[cr_block_offset] =
              packed_b[cr_block_offset] - (uint32_t)kv * izp;
        }
      }
    }

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const int8_t kv =
            k[((cr_block_start + cr_block_offset) * h + y) * w + x];
        *((int8_t*)packed_weights) = kv;
        packed_weights = (void*)((uintptr_t)packed_weights + sizeof(int8_t));
      }
      std::fill_n((int8_t*)packed_weights, channel_tile - cr_block_size,
                  INT8_C(0));
      packed_weights = (void*)((uintptr_t)packed_weights +
                               (channel_tile - cr_block_size) * sizeof(int8_t));
      advance_x_y(h, &x, &y);
    }
    std::fill_n((int8_t*)packed_weights,
                (primary_tile - kernel_size) * channel_tile, INT8_C(0));
    packed_weights = (void*)((uintptr_t)packed_weights +
                             (primary_tile - kernel_size) * cr_block_size);
    // We need to pack extra bytes for scale values here.
    packed_weights = (void*)((uintptr_t)packed_weights + per_tile_extra_bytes);
  }
}

void xnn_pack_f32_dwconv_hwg_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const float* k,
                               const float* b, const void* scale,
                               float* packed_weights,
                               size_t per_tile_extra_bytes,
                               const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const float kv =
            k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                0.0f);
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_f16_dwconv_hwg_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const uint16_t* k,
                               const uint16_t* b, const void* scale,
                               uint16_t* packed_weights,
                               size_t per_tile_extra_bytes,
                               const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const uint16_t kv =
            k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                UINT16_C(0));
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_f32_to_f16_dwconv_hwg_w(size_t primary_tile, size_t h, size_t w,
                                      size_t c, size_t channel_tile,
                                      const float* k, const float* b,
                                      const void* scale,
                                      xnn_float16* packed_weights,
                                      size_t per_tile_extra_bytes,
                                      const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_weights);
    packed_weights += channel_tile;

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const xnn_float16 kv = xnn_float16_from_float(
            k[(y * w + x) * c + (cr_block_start + cr_block_offset)]);
        *packed_weights++ = kv;
      }
      packed_weights += channel_tile - cr_block_size;
      advance_x_y(h, &x, &y);
    }
    std::fill_n(packed_weights, (primary_tile - kernel_size) * channel_tile,
                xnn_float16_zero());
    packed_weights += (primary_tile - kernel_size) * cr_block_size;
  }
}

void xnn_pack_qu8_dwconv_hwg_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const uint8_t* k,
                               const int32_t* b, const void* scale,
                               void* packed_weights,
                               size_t per_tile_extra_bytes,
                               const struct xnn_qu8_packing_params* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  const int32_t izp = (int32_t)params->input_zero_point;
  const int32_t boff =
      (int32_t)h * (int32_t)w * izp * (int32_t)params->kernel_zero_point;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_b, boff);
    packed_weights =
        (void*)((uintptr_t)packed_weights + channel_tile * sizeof(int32_t));

    // Biases need to be offset by all kernel values.
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const uint8_t kv =
              k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          unaligned_indexed_store_s32(
              packed_b, cr_block_offset,
              unaligned_indexed_load_s32(packed_b, cr_block_offset) -
                  (int32_t)kv * izp);
        }
      }
    }

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const uint8_t kv =
            k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
        *((uint8_t*)packed_weights) = kv;
        packed_weights = (void*)((uintptr_t)packed_weights + sizeof(uint8_t));
      }
      packed_weights =
          (void*)((uintptr_t)packed_weights +
                  (channel_tile - cr_block_size) * sizeof(uint8_t));
      advance_x_y(h, &x, &y);
    }
    std::fill_n((uint8_t*)packed_weights,
                (primary_tile - kernel_size) * channel_tile,
                params->kernel_zero_point);
    packed_weights = (void*)((uintptr_t)packed_weights +
                             (primary_tile - kernel_size) * cr_block_size);
  }
}

void xnn_pack_qs8_dwconv_hwg_w(size_t primary_tile, size_t h, size_t w,
                               size_t c, size_t channel_tile, const int8_t* k,
                               const int32_t* b, const float* scale,
                               void* packed_weights,
                               size_t per_tile_extra_bytes,
                               const struct xnn_qs8_packing_params* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);
  size_t kernel_size = h * w;

  const uint32_t izp = (uint32_t)params->input_zero_point;

  for (size_t cr_block_start = 0; cr_block_start < c;
       cr_block_start += channel_tile) {
    unaligned_int32_t* packed_b = (unaligned_int32_t*)packed_weights;
    const size_t cr_block_size = min(c - cr_block_start, channel_tile);
    copy_bias(b, cr_block_start, cr_block_size, packed_b);
    packed_weights =
        (void*)((uintptr_t)packed_weights + channel_tile * sizeof(int32_t));

    // Biases need to be offset by all kernel values.
    for (size_t x = 0; x < w; x++) {
      for (size_t y = 0; y < h; y++) {
        for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          const int8_t kv =
              k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
          packed_b[cr_block_offset] =
              packed_b[cr_block_offset] - (uint32_t)kv * izp;
        }
      }
    }

    // Stores the x and y index that should be processed next.
    size_t x = 0;
    size_t y = 0;
    for (size_t i = 0; i < kernel_size; i++) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        const int8_t kv =
            k[(y * w + x) * c + (cr_block_start + cr_block_offset)];
        *((int8_t*)packed_weights) = kv;
        packed_weights = (void*)((uintptr_t)packed_weights + sizeof(int8_t));
      }
      packed_weights = (void*)((uintptr_t)packed_weights +
                               (channel_tile - cr_block_size) * sizeof(int8_t));
      advance_x_y(h, &x, &y);
    }
    std::fill_n((int8_t*)packed_weights,
                (primary_tile - kernel_size) * channel_tile, INT8_C(0));
    packed_weights = (void*)((uintptr_t)packed_weights +
                             (primary_tile - kernel_size) * cr_block_size);
    // We need to pack extra bytes for scale values here.
    packed_weights = (void*)((uintptr_t)packed_weights + per_tile_extra_bytes);
  }
}

void xnn_pack_f32_gemminc_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                size_t kr, size_t sr, const float* k,
                                float* packed_weights, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            if (kc_begin < kc_end) {
              std::copy_n(
                  &k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                  kc_end - kc_begin, packed_weights);
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

void xnn_pack_f16_gemminc_goi_w(size_t g, size_t nc, size_t kc, size_t nr,
                                size_t kr, size_t sr, const uint16_t* k,
                                uint16_t* packed_weights, const void* params) {
  assert(g != 0);
  assert(nr >= sr);
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  const size_t skr = sr * kr;
  do {
    for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
      const size_t nr_block_size = min(nc - nr_block_start, nr);

      for (size_t kr_block_start = 0; kr_block_start < round_up_po2(kc, skr);
           kr_block_start += kr) {
        for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            const size_t kc_begin =
                round_down_po2(kr_block_start, skr) +
                ((kr_block_start + nr_block_offset * kr) & (skr - 1));
            const size_t kc_end = std::min(kc, kc_begin + kr);
            if (kc_begin < kc_end) {
              std::copy_n(
                  &k[(nr_block_start + nr_block_offset) * kc + kc_begin],
                  kc_end - kc_begin, packed_weights);
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

void xnn_pack_f32_dconv_oki_w(size_t nc, size_t kc, size_t nr, size_t kh,
                              size_t kw, const float* k, const float* b,
                              float* packed_weights, const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY (b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr;
           nr_block_offset++) {
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
          for (size_t nr_block_offset = 0; nr_block_offset < nr;
               nr_block_offset++) {
            *packed_weights++ =
                k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) *
                        kh +
                    ky) *
                       kw +
                   kx) *
                      kc +
                  c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f32_to_f16_dconv_oki_w(size_t nc, size_t kc, size_t nr, size_t kh,
                                     size_t kw, const float* k, const float* b,
                                     xnn_float16* packed_weights,
                                     const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY (b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr;
           nr_block_offset++) {
        *packed_weights++ =
            xnn_float16_from_float(b[min(nr_block_offset, nr_block_size - 1)]);
      }
    } else {
      size_t n = nr;
      do {
        *packed_weights++ = xnn_float16_zero();
      } while (--n != 0);
    }

    for (size_t kx = 0; kx < kw; kx++) {
      for (size_t c = 0; c < kc; c++) {
        for (size_t ky = 0; ky < kh; ky++) {
          for (size_t nr_block_offset = 0; nr_block_offset < nr;
               nr_block_offset++) {
            *packed_weights++ = xnn_float16_from_float(
                k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) *
                        kh +
                    ky) *
                       kw +
                   kx) *
                      kc +
                  c]);
          }
        }
      }
    }
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f16_dconv_oki_w(size_t nc, size_t kc, size_t nr, size_t kh,
                              size_t kw, const uint16_t* k, const uint16_t* b,
                              uint16_t* packed_weights, const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
    const size_t nr_block_size = min(nc - nr_block_start, nr);
    if XNN_LIKELY (b != nullptr) {
      for (size_t nr_block_offset = 0; nr_block_offset < nr;
           nr_block_offset++) {
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
          for (size_t nr_block_offset = 0; nr_block_offset < nr;
               nr_block_offset++) {
            *packed_weights++ =
                k[(((nr_block_start + min(nr_block_offset, nr_block_size - 1)) *
                        kh +
                    ky) *
                       kw +
                   kx) *
                      kc +
                  c];
          }
        }
      }
    }
    if XNN_UNPREDICTABLE (b != nullptr) {
      b += nr;
    }
  }
}

void xnn_pack_f32_chw_dwconv_ghw_w(size_t kernel_size, size_t groups,
                                   const float* k, const float* b,
                                   float* packed_weights, const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
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

void xnn_pack_f32_to_f16_chw_dwconv_ghw_w(size_t kernel_size, size_t groups,
                                          const float* k, const float* b,
                                          xnn_float16* packed_weights,
                                          const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
      *packed_weights = xnn_float16_from_float(*b++);
    } else {
      *packed_weights = xnn_float16_zero();
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = xnn_float16_from_float(k[g * kernel_size + i]);
    }
  }
}

void xnn_pack_f16_chw_dwconv_ghw_w(size_t kernel_size, size_t groups,
                                   const uint16_t* k, const uint16_t* b,
                                   uint16_t* packed_weights,
                                   const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
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

void xnn_pack_f32_chw_dwconv_hwg_w(size_t kernel_size, size_t groups,
                                   const float* k, const float* b,
                                   float* packed_weights, const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
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

void xnn_pack_f16_chw_dwconv_hwg_w(size_t kernel_size, size_t groups,
                                   const uint16_t* k, const uint16_t* b,
                                   uint16_t* packed_weights,
                                   const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
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

void xnn_pack_f32_to_f16_chw_dwconv_hwg_w(size_t kernel_size, size_t groups,
                                          const float* k, const float* b,
                                          xnn_float16* packed_weights,
                                          const void* params) {
  assert(k != nullptr);
  assert(packed_weights != nullptr);

  for (size_t g = 0; g < groups; g++) {
    if XNN_LIKELY (b != nullptr) {
      *packed_weights = xnn_float16_from_float(*b++);
    } else {
      *packed_weights = xnn_float16_zero();
    }
    packed_weights += 1;
    for (size_t i = 0; i < kernel_size; i++) {
      *packed_weights++ = xnn_float16_from_float(k[i * groups + g]);
    }
  }
}

void xnn_pack_f32_vmulcaddc_w(size_t c, size_t cr, const float* s,
                              const float* b, float* packed_weights,
                              const void* params) {
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY (b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
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

void xnn_pack_f16_vmulcaddc_w(size_t c, size_t cr, const uint16_t* s,
                              const uint16_t* b, uint16_t* packed_weights,
                              const void* params) {
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *packed_weights++ = s[cr_block_start + cr_block_offset];
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY (b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
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

void xnn_pack_f32_to_f16_vmulcaddc_w(size_t c, size_t cr, const float* s,
                                     const float* b,
                                     xnn_float16* packed_weights,
                                     const void* params) {
  assert(s != nullptr);
  assert(packed_weights != nullptr);

  for (size_t cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
    const size_t cr_block_size = min(c - cr_block_start, cr);
    for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
         cr_block_offset++) {
      *packed_weights++ =
          xnn_float16_from_float(s[cr_block_start + cr_block_offset]);
    }
    packed_weights += cr - cr_block_size;
    if XNN_LIKELY (b != nullptr) {
      for (size_t cr_block_offset = 0; cr_block_offset < cr_block_size;
           cr_block_offset++) {
        *packed_weights++ =
            xnn_float16_from_float(b[cr_block_start + cr_block_offset]);
      }
    } else {
      size_t n = cr_block_size;
      do {
        *packed_weights++ = xnn_float16_zero();
      } while (--n != 0);
    }
    packed_weights += cr - cr_block_size;
  }
}

void xnn_analyze_f32_spmm_w(size_t group_output_channels,
                            size_t group_input_channels, const float* kernel,
                            struct xnn_spmm_packing_params* params) {
  assert(kernel != nullptr);
  assert(params != nullptr);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero =
          (size_t)(kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero =
          (size_t)(kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      const size_t row2_nonzero =
          (size_t)(kernel[(oc + 2) * group_input_channels + ic] != 0.0f);
      const size_t row3_nonzero =
          (size_t)(kernel[(oc + 3) * group_input_channels + ic] != 0.0f);
      num_nonzeroes +=
          row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 +=
          (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 +=
          (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4);
       oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero =
          (size_t)(kernel[oc * group_input_channels + ic] != 0.0f);
      const size_t row1_nonzero =
          (size_t)(kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2);
       oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes += (size_t)(kernel[oc * group_input_channels + ic] != 0.0f);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

void xnn_analyze_f16_spmm_w(size_t group_output_channels,
                            size_t group_input_channels,
                            const xnn_float16* kernel,
                            struct xnn_spmm_packing_params* params) {
  assert(kernel != nullptr);
  assert(params != nullptr);

  // Count number of non-zero values.
  size_t num_nonzeroes = 0;
  size_t num_nonzero_blocks2 = 0;
  size_t num_nonzero_blocks4 = 0;
  for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero =
          (size_t)!xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
      const size_t row1_nonzero = (size_t)!xnn_float16_is_zero(
          kernel[(oc + 1) * group_input_channels + ic]);
      const size_t row2_nonzero = (size_t)!xnn_float16_is_zero(
          kernel[(oc + 2) * group_input_channels + ic]);
      const size_t row3_nonzero = (size_t)!xnn_float16_is_zero(
          kernel[(oc + 3) * group_input_channels + ic]);
      num_nonzeroes +=
          row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
      num_nonzero_blocks2 +=
          (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
      num_nonzero_blocks4 +=
          (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
    }
  }
  const size_t num_block4_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 4);
       oc < round_down_po2(group_output_channels, 2); oc += 2) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const size_t row0_nonzero =
          (size_t)!xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
      const size_t row1_nonzero = (size_t)!xnn_float16_is_zero(
          kernel[(oc + 1) * group_input_channels + ic]);
      num_nonzeroes += row0_nonzero + row1_nonzero;
      num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
    }
  }
  const size_t num_block2_nonzeroes = num_nonzeroes;
  for (size_t oc = round_down_po2(group_output_channels, 2);
       oc < group_output_channels; oc++) {
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      num_nonzeroes +=
          (size_t)!xnn_float16_is_zero(kernel[oc * group_input_channels + ic]);
    }
  }
  params->num_nonzeroes = num_nonzeroes;
  params->num_nonzero_blocks2 = num_nonzero_blocks2;
  params->num_nonzero_blocks4 = num_nonzero_blocks4;
  params->num_block2_nonzeroes = num_block2_nonzeroes;
  params->num_block4_nonzeroes = num_block4_nonzeroes;
}

enum xnn_status xnn_pack_f32_spmm_w(
    size_t group_output_channels, size_t output_channels_block_size,
    size_t group_input_channels, const float* kernel, const float* bias,
    int32_t* input_channel_diffs, uint32_t* output_channel_nonzeros,
    float* nonzero_values, size_t* first_input_channel) {
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0;
       ocb < round_down_po2(group_output_channels, output_channels_block_size);
       ocb += output_channels_block_size) {
    if XNN_LIKELY (bias != nullptr) {
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
        is_nonzero_block |=
            (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(float);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc =
           round_down_po2(group_output_channels, output_channels_block_size);
       oc < group_output_channels; oc++) {
    if XNN_LIKELY (bias != nullptr) {
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
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(float);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input
  // channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t)((uint64_t)first_ic - (uint64_t)last_ic) *
                         (int64_t)sizeof(float);
    if (diff != (int64_t)(int32_t)diff) {
      xnn_log_error(
          "failed to convert kernel to sparse representation: "
          "scaled difference in input channels exceeds int32_t range");
      return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t)diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

enum xnn_status xnn_pack_f32_to_f16_spmm_w(
    size_t group_output_channels, size_t output_channels_block_size,
    size_t group_input_channels, const float* kernel, const float* bias,
    int32_t* input_channel_diffs, uint32_t* output_channel_nonzeros,
    xnn_float16* nonzero_values,  // fp16 values
    size_t* first_input_channel) {
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0;
       ocb < round_down_po2(group_output_channels, output_channels_block_size);
       ocb += output_channels_block_size) {
    if XNN_LIKELY (bias != nullptr) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_from_float(bias[ocb + oco]);
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_zero();
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |=
            (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = xnn_float16_from_float(
              kernel[(ocb + oco) * group_input_channels + ic]);
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(uint16_t);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc =
           round_down_po2(group_output_channels, output_channels_block_size);
       oc < group_output_channels; oc++) {
    if XNN_LIKELY (bias != nullptr) {
      *nonzero_values++ = xnn_float16_from_float(bias[oc]);
    } else {
      *nonzero_values++ = xnn_float16_zero();
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const float weight = kernel[oc * group_input_channels + ic];
      if (weight != 0.0f) {
        *nonzero_values++ = xnn_float16_from_float(weight);
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(uint16_t);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input
  // channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t)((uint64_t)first_ic - (uint64_t)last_ic) *
                         (int64_t)sizeof(uint16_t);
    if (diff != (int64_t)(int32_t)diff) {
      xnn_log_error(
          "failed to convert kernel to sparse representation: "
          "scaled difference in input channels exceeds int32_t range");
      return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t)diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

enum xnn_status xnn_pack_f16_spmm_w(size_t group_output_channels,
                                    size_t output_channels_block_size,
                                    size_t group_input_channels,
                                    const xnn_float16* kernel,  // fp16 values
                                    const xnn_float16* bias,    // fp16 values
                                    int32_t* input_channel_diffs,
                                    uint32_t* output_channel_nonzeros,
                                    xnn_float16* nonzero_values,  // fp16 values
                                    size_t* first_input_channel) {
  size_t first_ic = 0, last_ic = 0;
  bool first_nonzero = true;
  for (size_t ocb = 0;
       ocb < round_down_po2(group_output_channels, output_channels_block_size);
       ocb += output_channels_block_size) {
    if XNN_LIKELY (bias != nullptr) {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = bias[ocb + oco];
      }
    } else {
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        *nonzero_values++ = xnn_float16_zero();
      }
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      bool is_nonzero_block = false;
      for (size_t oco = 0; oco < output_channels_block_size; oco++) {
        is_nonzero_block |= !xnn_float16_is_zero(
            kernel[(ocb + oco) * group_input_channels + ic]);
      }
      if (is_nonzero_block) {
        for (size_t oco = 0; oco < output_channels_block_size; oco++) {
          *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
        }
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(uint16_t);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  for (size_t oc =
           round_down_po2(group_output_channels, output_channels_block_size);
       oc < group_output_channels; oc++) {
    if XNN_LIKELY (bias != nullptr) {
      *nonzero_values++ = bias[oc];
    } else {
      *nonzero_values++ = xnn_float16_zero();
    }
    for (size_t ic = 0; ic < group_input_channels; ic++) {
      const xnn_float16 weight = kernel[oc * group_input_channels + ic];
      if (!xnn_float16_is_zero(weight)) {
        *nonzero_values++ = weight;
        if (first_nonzero) {
          first_ic = ic;
        } else {
          const int64_t diff = (int64_t)((uint64_t)ic - (uint64_t)last_ic) *
                               (int64_t)sizeof(uint16_t);
          if (diff != (int64_t)(int32_t)diff) {
            xnn_log_error(
                "failed to convert kernel to sparse representation: "
                "scaled difference in input channels exceeds int32_t range");
            return xnn_status_unsupported_parameter;
          }
          *input_channel_diffs++ = (int32_t)diff;
        }
        first_nonzero = false;
        last_ic = ic;
        *output_channel_nonzeros += 1;
      }
    }
    output_channel_nonzeros += 1;
  }
  // If there are any non-zero elements, we have to return to the initial input
  // channel.
  if (!first_nonzero) {
    const int64_t diff = (int64_t)((uint64_t)first_ic - (uint64_t)last_ic) *
                         (int64_t)sizeof(uint16_t);
    if (diff != (int64_t)(int32_t)diff) {
      xnn_log_error(
          "failed to convert kernel to sparse representation: "
          "scaled difference in input channels exceeds int32_t range");
      return xnn_status_unsupported_parameter;
    }
    *input_channel_diffs++ = (int32_t)diff;
  }
  *first_input_channel = first_ic;
  return xnn_status_success;
}

}  // extern "C"
