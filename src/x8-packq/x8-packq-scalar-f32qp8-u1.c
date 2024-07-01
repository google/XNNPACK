// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/packq.h"

// These functions have been adapted from KleidiAI's
// `kai_run_lhs_quant_pack_qai8dxp_f32` as a reference scalar implementation.

void xnn_x8_packq_f32qp8_ukernel__scalar_u1(size_t m, size_t k, size_t mr,
                                            size_t kr, size_t sr,
                                            size_t m_idx_start,
                                            const float* XNN_RESTRICT lhs,
                                            size_t lhs_stride,
                                            void* XNN_RESTRICT lhs_packed) {
  assert((kr % sr) == 0);

  // Assuming the same sizeof() for kai_num_bytes_per_offset and
  // kai_num_bytes_per_multiplier
  static const size_t num_bytes_per_multiplier = sizeof(float);
  static const size_t num_bytes_per_offset = sizeof(int32_t);
  assert(num_bytes_per_offset == num_bytes_per_multiplier);

  if (m == 0) {
    return;
  }

  const size_t num_rows = m;

  const float* src_ptr = lhs;

  const size_t dst_stride = lhs_packed_stride(k, mr, kr, sr);
  const size_t k_internal = k_roundedup(k, kr, sr);
  const int32_t k_block_len = (int32_t)(kr / sr);

  for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
    float max0 = 0.0f;
    float min0 = 0.0f;

    // Find min/max for each channel
    for (int32_t k_idx = 0; k_idx < (int32_t)k; ++k_idx) {
      const float src0_0 = *(src_ptr + (size_t)k_idx);
      max0 = math_max_f32(src0_0, max0);
      min0 = math_min_f32(src0_0, min0);
    }

    // Maximum/minimum int8 values
    const float qmin = (float)INT8_MIN;
    const float qmax = (float)INT8_MAX;

    const float scale0 = min0 == max0 ? 1.F : (qmax - qmin) / (max0 - min0);

    // Reciprocal to quantize
    const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;

    const float descaled_min0 = min0 * scale0;
    const float descaled_max0 = max0 * scale0;

    const float zero_point_from_min_error0 = qmin + descaled_min0;
    const float zero_point_from_max_error0 = qmax + descaled_max0;

    float zero_point0 =
        zero_point_from_min_error0 + zero_point_from_max_error0 > 0
            ? qmin - descaled_min0
            : qmax - descaled_max0;

    zero_point0 = math_max_f32(zero_point0, qmin);
    zero_point0 = math_min_f32(zero_point0, qmax);

    // Round to nearest integer
    const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);

    const size_t dst_x = ((row_idx + m_idx_start) % mr);

    uint8_t* dst_ptr =
        (uint8_t*)lhs_packed + dst_x * k_block_len * sizeof(int8_t);

    // Quantize the channels
    for (int32_t k_idx = 0; k_idx < (int32_t)k_internal; k_idx += k_block_len) {
      for (size_t k_block_idx = 0; k_block_idx < (size_t)k_block_len;
           ++k_block_idx) {
        // Clamp at the last valid k-index
        const size_t k_idx_start = min((size_t)k_idx + k_block_idx, k - 1);

        const float src0_0 = *(src_ptr + k_idx_start);

        // Scale the values
        int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

        v0_s32 = v0_s32 + nudged_zero_point0;
        v0_s32 = math_max_s32(v0_s32, INT8_MIN);
        v0_s32 = math_min_s32(v0_s32, INT8_MAX);
        *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
        dst_ptr += sizeof(int8_t);
      }
      dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
    }

    dst_ptr = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

    dst_ptr += dst_x * num_bytes_per_offset;

    // LHS offset at the beginning of the row
    *((int32_t*)(dst_ptr)) = -nudged_zero_point0;

    dst_ptr += mr * num_bytes_per_multiplier;

    // Store the scale quantization params
    *((float*)(dst_ptr)) = recip_scale0;

    src_ptr += (lhs_stride / sizeof(float));

    // Move to the next row if we have interleaved all Mr rows
    if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
      lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
    }
  }
}
