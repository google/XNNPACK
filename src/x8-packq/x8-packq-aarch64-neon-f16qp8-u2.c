// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#endif  // XNN_ENABLE_KLEIDIAI

// Wraps KleidiAI's FP16 `qai8dxp` LHS quantize-and-pack microkernel.
void xnn_x8_packq_f16qp8_ukernel__aarch64_neon_u2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const void* XNN_RESTRICT lhs, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_lhs_quant_pack_qai8dxp_f16_neon(m, k, mr, kr, sr, m_idx_start, lhs,
                                          lhs_stride, lhs_packed);
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  if (m == 0) {
    return;
  }
  assert(kr >= sr && kr % sr == 0);
  if (kr < sr) {
    return;
  }
  const size_t k_block_len = kr / sr;
  const size_t k_internal = round_up_po2(k, 32);
  const size_t dst_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);

  for (size_t row_idx = 0; row_idx < m; ++row_idx) {
    const uint16_t* src_ptr =
        (const uint16_t*)((const uint8_t*)lhs + row_idx * lhs_stride);
    size_t k_idx = 0;
    float32x4_t vmax0 = vdupq_n_f32(-FLT_MAX);
    float32x4_t vmin0 = vdupq_n_f32(FLT_MAX);
    for (; k_idx + 8 <= k; k_idx += 8) {
      const uint16x8_t vsrc16 = vld1q_u16(src_ptr + k_idx);
      const float32x4_t src0 =
          vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc16)));
      const float32x4_t src1 =
          vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc16)));
      vmax0 = vmaxq_f32(vmax0, vmaxq_f32(src0, src1));
      vmin0 = vminq_f32(vmin0, vminq_f32(src0, src1));
    }
    float max0 = vmaxvq_f32(vmax0);
    float min0 = vminvq_f32(vmin0);
    for (; k_idx < k; ++k_idx) {
      const float src0 =
          xnn_float16_to_float(xnn_float16_from_bits(src_ptr[k_idx]));
      max0 = src0 > max0 ? src0 : max0;
      min0 = src0 < min0 ? src0 : min0;
    }

    const float qmin = (float)INT8_MIN;
    const float qmax = (float)INT8_MAX;
    const float rmin0 = math_min_f32(0.0f, min0);
    const float rmax0 = math_max_f32(0.0f, max0);
    const float scale0 =
        rmin0 == rmax0 ? 1.0f : (qmax - qmin) / (rmax0 - rmin0);
    const float recip_scale0 = scale0 != 0.0f ? 1.0f / scale0 : 0.0f;

    const float descaled_min0 = rmin0 * scale0;
    const float descaled_max0 = rmax0 * scale0;
    const float zero_point_from_min_error0 = qmin + descaled_min0;
    const float zero_point_from_max_error0 = qmax + descaled_max0;
    float zero_point0 =
        zero_point_from_min_error0 + zero_point_from_max_error0 > 0.0f
            ? qmin - descaled_min0
            : qmax - descaled_max0;
    zero_point0 = math_max_f32(zero_point0, qmin);
    zero_point0 = math_min_f32(zero_point0, qmax);
    const int32_t nudged_zero_point0 = (int32_t)lrintf(zero_point0);
    const int16x8_t vnzp_s16 = vdupq_n_s16((int16_t)nudged_zero_point0);

    const size_t dst_x = (row_idx + m_idx_start) % mr;
    uint8_t* dst_ptr = (uint8_t*)lhs_packed + dst_x * k_block_len;

    if (mr == 1) {
      size_t k_pos = 0;
      for (; k_pos + 16 <= k; k_pos += 16) {
        const uint16x8_t vsrc01 = vld1q_u16(src_ptr + k_pos);
        const uint16x8_t vsrc23 = vld1q_u16(src_ptr + k_pos + 8);
        const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc01))), scale0));
        const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc01))), scale0));
        const int32x4_t q2 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc23))), scale0));
        const int32x4_t q3 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc23))), scale0));
        const int16x8_t h0 =
            vqaddq_s16(vqmovn_high_s32(vqmovn_s32(q0), q1), vnzp_s16);
        const int16x8_t h1 =
            vqaddq_s16(vqmovn_high_s32(vqmovn_s32(q2), q3), vnzp_s16);
        vst1q_s8((int8_t*)dst_ptr + k_pos, vqmovn_high_s16(vqmovn_s16(h0), h1));
      }
      for (; k_pos + 8 <= k; k_pos += 8) {
        const uint16x8_t vsrc01 = vld1q_u16(src_ptr + k_pos);
        const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc01))), scale0));
        const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc01))), scale0));
        const int16x8_t h0 =
            vqaddq_s16(vqmovn_high_s32(vqmovn_s32(q0), q1), vnzp_s16);
        vst1_s8((int8_t*)dst_ptr + k_pos, vqmovn_s16(h0));
      }
      for (; k_pos < k; ++k_pos) {
        const float src0 =
            xnn_float16_to_float(xnn_float16_from_bits(src_ptr[k_pos]));
        int32_t v0_s32 = (int32_t)roundf(src0 * scale0) + nudged_zero_point0;
        v0_s32 = math_max_s32(v0_s32, INT8_MIN);
        v0_s32 = math_min_s32(v0_s32, INT8_MAX);
        ((int8_t*)dst_ptr)[k_pos] = (int8_t)v0_s32;
      }
      if (k_pos < k_internal) {
        const float src_last =
            xnn_float16_to_float(xnn_float16_from_bits(src_ptr[k - 1]));
        int32_t vlast_s32 =
            (int32_t)roundf(src_last * scale0) + nudged_zero_point0;
        vlast_s32 = math_max_s32(vlast_s32, INT8_MIN);
        vlast_s32 = math_min_s32(vlast_s32, INT8_MAX);
        const int8_t q_last = (int8_t)vlast_s32;
        for (; k_pos < k_internal; ++k_pos) {
          ((int8_t*)dst_ptr)[k_pos] = q_last;
        }
      }
    } else {
      const size_t num_blocks_k = k / k_block_len;
      const size_t num_blocks_k_internal = k_internal / k_block_len;
      size_t block_idx = 0;

      if (k_block_len == 8) {
        for (; block_idx < num_blocks_k; ++block_idx) {
          const uint16x8_t vsrc01 = vld1q_u16(src_ptr + block_idx * 8);
          const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(
              vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc01))),
              scale0));
          const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(
              vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc01))),
              scale0));
          const int16x8_t h0 =
              vqaddq_s16(vqmovn_high_s32(vqmovn_s32(q0), q1), vnzp_s16);
          vst1_s8((int8_t*)dst_ptr, vqmovn_s16(h0));
          dst_ptr += mr * 8;
        }
      } else if (k_block_len == 4) {
        for (; block_idx + 2 <= num_blocks_k; block_idx += 2) {
          const uint16x8_t vsrc01 = vld1q_u16(src_ptr + block_idx * 4);
          const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(
              vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vsrc01))),
              scale0));
          const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(
              vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(vsrc01))),
              scale0));
          const int16x8_t h0 =
              vqaddq_s16(vqmovn_high_s32(vqmovn_s32(q0), q1), vnzp_s16);
          const uint32x2_t b01 = vreinterpret_u32_s8(vqmovn_s16(h0));
          const uint32_t b0 = vget_lane_u32(b01, 0);
          const uint32_t b1 = vget_lane_u32(b01, 1);
          memcpy(dst_ptr, &b0, sizeof(uint32_t));
          memcpy(dst_ptr + mr * 4, &b1, sizeof(uint32_t));
          dst_ptr += 2 * mr * 4;
        }
        for (; block_idx < num_blocks_k; ++block_idx) {
          const uint16x4_t vsrc0 = vld1_u16(src_ptr + block_idx * 4);
          const int32x4_t q0 = vcvtaq_s32_f32(
              vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(vsrc0)), scale0));
          const int16x4_t h0 =
              vqadd_s16(vqmovn_s32(q0), vget_low_s16(vnzp_s16));
          const uint32_t b0 = vget_lane_u32(
              vreinterpret_u32_s8(vqmovn_s16(vcombine_s16(h0, h0))), 0);
          memcpy(dst_ptr, &b0, sizeof(uint32_t));
          dst_ptr += mr * 4;
        }
      } else {
        for (; block_idx < num_blocks_k; ++block_idx) {
          const size_t k_pos = block_idx * k_block_len;
          for (size_t k_block_idx = 0; k_block_idx < k_block_len;
               ++k_block_idx) {
            const float src0 = xnn_float16_to_float(
                xnn_float16_from_bits(src_ptr[k_pos + k_block_idx]));
            int32_t v0_s32 =
                (int32_t)roundf(src0 * scale0) + nudged_zero_point0;
            v0_s32 = math_max_s32(v0_s32, INT8_MIN);
            v0_s32 = math_min_s32(v0_s32, INT8_MAX);
            *((int8_t*)dst_ptr) = (int8_t)v0_s32;
            dst_ptr += sizeof(int8_t);
          }
          dst_ptr += (mr - 1) * k_block_len;
        }
      }

      for (; block_idx < num_blocks_k_internal; ++block_idx) {
        const size_t k_pos = block_idx * k_block_len;
        for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
          const size_t k_idx_start =
              (k_pos + k_block_idx < k) ? (k_pos + k_block_idx) : (k - 1);
          const float src0 =
              xnn_float16_to_float(xnn_float16_from_bits(src_ptr[k_idx_start]));
          int32_t v0_s32 = (int32_t)roundf(src0 * scale0) + nudged_zero_point0;
          v0_s32 = math_max_s32(v0_s32, INT8_MIN);
          v0_s32 = math_min_s32(v0_s32, INT8_MAX);
          *((int8_t*)dst_ptr) = (int8_t)v0_s32;
          dst_ptr += sizeof(int8_t);
        }
        dst_ptr += (mr - 1) * k_block_len;
      }
    }

    dst_ptr = (uint8_t*)lhs_packed + mr * k_internal;
    dst_ptr += dst_x * sizeof(int32_t);
    *((int32_t*)dst_ptr) = -nudged_zero_point0;
    dst_ptr += (mr - dst_x) * sizeof(int32_t) + dst_x * sizeof(float);
    *((float*)dst_ptr) = recip_scale0;

    if (dst_x == mr - 1) {
      lhs_packed = (void*)((uint8_t*)lhs_packed + mr * dst_stride);
    }
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
