// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_SRC_XNNPACK_PACKQ_H
#define __XNNPACK_SRC_XNNPACK_PACKQ_H

#include <assert.h>
#include <stddef.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>

#ifdef __cplusplus
extern "C" {
#endif

// These functions have been adapted from KleidiAI's
// `kai_run_lhs_quant_pack_qai8dxp_f32` as a reference scalar implementation.

inline static size_t kai_k_roundedup(size_t k, size_t kr, size_t sr) {
  // Since we pack a float and int32 value at the end of the row,
  // we must make sure that k is a multiple of 4 for memory alignment.
  size_t kr_sr_roundedup4 = round_up(kr * sr, 4);
  return round_up(k, kr_sr_roundedup4);
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr,
                                           size_t sr) {
  const size_t k_internal = kai_k_roundedup(k, kr, sr);

  assert((k_internal % 2) == 0);

  static const size_t kai_num_bytes_per_multiplier = sizeof(float);
  static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

  return mr * (k_internal * sizeof(int8_t) + kai_num_bytes_per_multiplier +
               kai_num_bytes_per_offset);
}

XNN_INLINE size_t xnn_x8_packq_f32qp8_packed_offset(size_t m_idx, size_t k,
                                                    size_t mr, size_t kr,
                                                    size_t sr) {
  // It always points to the beginning of the row
  return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

XNN_INLINE size_t xnn_x8_packq_f32qp8_packed_size(size_t m, size_t k, size_t mr,
                                                  size_t kr, size_t sr) {
    const size_t num_rows = round_up(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, sr);
}

#define DECLARE_X8_PACKQ_UKERNEL_FUNCTION(fn_name)                            \
  XNN_INTERNAL void fn_name(size_t m, size_t k, size_t mr, size_t kr,         \
                            size_t sr, size_t m_idx_start,                    \
                            const float* XNN_RESTRICT lhs, size_t lhs_stride, \
                            void* XNN_RESTRICT lhs_packed);

DECLARE_X8_PACKQ_UKERNEL_FUNCTION(xnn_x8_packq_f32qp8_ukernel__scalar_u1)

#if XNN_ENABLE_KLEIDIAI
DECLARE_X8_PACKQ_UKERNEL_FUNCTION(xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2)
#endif  // XNN_ENABLE_KLEIDIAI

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __XNNPACK_SRC_XNNPACK_PACKQ_H
