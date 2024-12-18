// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_SRC_XNNPACK_PACKQ_H
#define __XNNPACK_SRC_XNNPACK_PACKQ_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/math.h"

#ifdef __cplusplus
extern "C" {
#endif

// These functions have been adapted from KleidiAI's
// `kai_run_lhs_quant_pack_qai8dxp_f32` as a reference scalar implementation.

inline static size_t k_roundedup(size_t k, size_t kr, size_t sr) {
  // Round up k to be a multiple of 32.
  size_t kai_k_multiple_of = 32;
  return round_up(k, kai_k_multiple_of);
}

inline static size_t lhs_packed_stride(size_t k, size_t mr_packed, size_t kr,
                                       size_t sr) {
  const size_t k_internal = k_roundedup(k, kr, sr);

  assert((k_internal % 2) == 0);

  // Assuming the same sizeof() for kai_num_bytes_per_offset and
  // kai_num_bytes_per_multiplier
  static const size_t kai_num_bytes_per_multiplier = sizeof(float);
  static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

  return mr_packed * (k_internal * sizeof(int8_t) +
                      kai_num_bytes_per_multiplier + kai_num_bytes_per_offset);
}

XNN_INLINE static size_t xnn_x8_packq_f32qp8_packed_offset(
    size_t m_idx, size_t k, size_t mr_packed, size_t kr, size_t sr) {
  // It always points to the beginning of the row
  return (m_idx / mr_packed) * lhs_packed_stride(k, mr_packed, kr, sr);
}

XNN_INLINE static size_t xnn_x8_packq_f32qp8_packed_size(size_t m, size_t k,
                                                         size_t mr_packed,
                                                         size_t kr, size_t sr) {
  const size_t num_rows = round_up(m, mr_packed) / mr_packed;

  return num_rows * lhs_packed_stride(k, mr_packed, kr, sr);
}

XNN_INLINE static size_t xnn_x8_packq_f32qp8_gemm_packed_size(
    const struct xnn_gemm_config* gemm_config, size_t m, size_t k) {
  const uint32_t mr_packed = m == 1                   ? 1
                             : gemm_config->mr_packed ? gemm_config->mr_packed
                                                      : gemm_config->mr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  return xnn_x8_packq_f32qp8_packed_size(m, k, mr_packed, kr, sr);
}

XNN_INLINE static int8_t xnn_x8_packq_f32qp8_get_quantized(
    size_t m_idx, size_t k_idx, const int8_t* lhs_packed, size_t k,
    size_t mr_packed, size_t kr, size_t sr) {
  const int32_t k_block_len = (int32_t)(kr / sr);
  const size_t dst_x = (m_idx % mr_packed);
  const size_t packed_offset =
      xnn_x8_packq_f32qp8_packed_offset(m_idx, k, mr_packed, kr, sr);
  const int8_t* dst_ptr = lhs_packed + packed_offset + dst_x * k_block_len;

  dst_ptr += (k_idx / k_block_len) * (mr_packed - 1) * k_block_len;
  dst_ptr += k_idx;

  return *dst_ptr;
}

XNN_INLINE static float xnn_x8_packq_f32qp8_get_recip_scale(
    size_t m_idx, const int8_t* lhs_packed, size_t k,
    size_t mr_packed, size_t kr, size_t sr) {
  const size_t k_internal = k_roundedup(k, kr, sr);
  const size_t dst_x = (m_idx % mr_packed);
  const size_t packed_offset =
      xnn_x8_packq_f32qp8_packed_offset(m_idx, k, mr_packed, kr, sr);

  // Get the quantization parameters.
  const int8_t* dst_ptr = lhs_packed + packed_offset + mr_packed * k_internal;
  dst_ptr += dst_x * sizeof(int32_t);
  dst_ptr += mr_packed * sizeof(float);
  const float recip_scale = *(const float*)dst_ptr;
  return recip_scale;
}

XNN_INLINE static float xnn_x8_packq_f32qp8_get_neg_nudged_zp(
    size_t m_idx, const int8_t* lhs_packed, size_t k,
    size_t mr_packed, size_t kr, size_t sr) {
  const size_t k_internal = k_roundedup(k, kr, sr);
  const size_t dst_x = (m_idx % mr_packed);
  const size_t packed_offset =
      xnn_x8_packq_f32qp8_packed_offset(m_idx, k, mr_packed, kr, sr);

  // Get the quantization parameters.
  const int8_t* dst_ptr = lhs_packed + packed_offset + mr_packed * k_internal;
  dst_ptr += dst_x * sizeof(int32_t);
  const int32_t neg_nudged_zero_point = *(const int32_t*)dst_ptr;
  return neg_nudged_zero_point;
}

XNN_INLINE static float xnn_x8_packq_f32qp8_get_dequantized(
    size_t m_idx, size_t k_idx, const int8_t* lhs_packed, size_t k,
    size_t mr_packed, size_t kr, size_t sr) {
  const size_t k_internal = k_roundedup(k, kr, sr);
  const int32_t k_block_len = (int32_t)(kr / sr);
  const size_t dst_x = (m_idx % mr_packed);
  const size_t packed_offset =
      xnn_x8_packq_f32qp8_packed_offset(m_idx, k, mr_packed, kr, sr);
  const int8_t* dst_ptr = lhs_packed + packed_offset + dst_x * k_block_len;

  // Get the quantized value.
  dst_ptr += (k_idx / k_block_len) * (mr_packed - 1) * k_block_len;
  dst_ptr += k_idx;
  const int32_t val = *dst_ptr;

  // Get the quantization parameters.
  dst_ptr = lhs_packed + packed_offset + mr_packed * k_internal;
  dst_ptr += dst_x * sizeof(int32_t);
  const int32_t neg_nudged_zero_point = *(const int32_t*)dst_ptr;
  dst_ptr += mr_packed * sizeof(float);
  const float recip_scale = *(const float*)dst_ptr;

  return (val + neg_nudged_zero_point) * recip_scale;
}

#define XNN_UKERNEL(arch_flags, ukernel, unroll)                              \
  XNN_INTERNAL void ukernel(size_t m, size_t k, size_t mr_packed, size_t kr,  \
                            size_t sr, size_t m_idx_start,                    \
                            const float* XNN_RESTRICT lhs, size_t lhs_stride, \
                            void* XNN_RESTRICT lhs_packed);

#include "x8-packq/x8-packq.h"

#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __XNNPACK_SRC_XNNPACK_PACKQ_H
