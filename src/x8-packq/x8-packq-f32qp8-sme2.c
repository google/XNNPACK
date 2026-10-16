// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_SME2_ACLE
#include <arm_neon.h>
#include <arm_sme.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/packq.h"

static void xnn_x8_packq_f32qp8_calc_params__neon(
    size_t mr, size_t slot, size_t valid_rows, size_t k, const float* lhs_blk,
    float* scales, int32_t* zps, float* recip_scales,
    void* lhs_packed_zp)
{
  const float qmin = (float)INT8_MIN;
  const float qmax = (float)INT8_MAX;

  for (size_t row_idx = 0; row_idx < mr; row_idx++) {
    if (row_idx < valid_rows) {
      const float* src_ptr = lhs_blk + (k * row_idx);
      float32x4_t vmax0 = vdupq_n_f32(0.0f);
      float32x4_t vmin0 = vdupq_n_f32(0.0f);
      float32x4_t vmax1 = vdupq_n_f32(0.0f);
      float32x4_t vmin1 = vdupq_n_f32(0.0f);

      size_t k_idx = 0;
      for (; k_idx + 8 <= k; k_idx += 8) {
        float32x4_t src0 = vld1q_f32(src_ptr + k_idx);
        float32x4_t src1 = vld1q_f32(src_ptr + k_idx + 4);
        vmax0 = vmaxq_f32(vmax0, src0);
        vmin0 = vminq_f32(vmin0, src0);
        vmax1 = vmaxq_f32(vmax1, src1);
        vmin1 = vminq_f32(vmin1, src1);
      }
      for (; k_idx + 4 <= k; k_idx += 4) {
        float32x4_t src0 = vld1q_f32(src_ptr + k_idx);
        vmax0 = vmaxq_f32(vmax0, src0);
        vmin0 = vminq_f32(vmin0, src0);
      }

      float32x4_t vmax = vmaxq_f32(vmax0, vmax1);
      float32x4_t vmin = vminq_f32(vmin0, vmin1);

      float max_val = vmaxvq_f32(vmax);
      float min_val = vminvq_f32(vmin);

      for (; k_idx < k; k_idx++) {
        float val = src_ptr[k_idx];
        max_val = math_max_f32(max_val, val);
        min_val = math_min_f32(min_val, val);
      }

      const float rmin = math_min_f32(0.0F, min_val);
      const float rmax = math_max_f32(0.0F, max_val);
      const float scale = rmin == rmax ? 1.F : (qmax - qmin) / (rmax - rmin);
      const float recip_scale = scale ? 1.0F / scale : 0.0F;
      const float descaled_min = rmin * scale;
      const float descaled_max = rmax * scale;
      const float zero_point_from_min_error = qmin + descaled_min;
      const float zero_point_from_max_error = qmax + descaled_max;
      float zero_point = zero_point_from_min_error + zero_point_from_max_error > 0 ? qmin - descaled_min : qmax - descaled_max;
      zero_point = math_max_f32(zero_point, qmin);
      zero_point = math_min_f32(zero_point, qmax);
      const int32_t nudged_zero_point = (int32_t)rintf(zero_point);

      scales[row_idx] = scale;
      zps[row_idx] = nudged_zero_point;
      recip_scales[row_idx] = recip_scale;
    } else {
      scales[row_idx] = 1.0f;
      zps[row_idx] = 0;
      recip_scales[row_idx] = 1.0f;
    }
  }

  int32_t* zp_ptr = (int32_t*)lhs_packed_zp + slot;
  float* recip_scale_ptr = (float*)((int32_t*)lhs_packed_zp + mr) + slot;
  const size_t num_slots = slot ? valid_rows : mr;

  for (size_t i = 0; i < num_slots; ++i) {
    zp_ptr[i] = -zps[i];
    recip_scale_ptr[i] = recip_scales[i];
  }
}

static void xnn_x8_packq_f32qp8_quantize_block__neon(
    size_t mr, size_t kblk, size_t valid_rows, size_t k, const float* lhs_blk,
    const float* scales, const int32_t* zps,
    int8_t* temp_q, size_t k_idx_block)
{
  int32x4_t vmin_int8 = vdupq_n_s32(INT8_MIN);
  int32x4_t vmax_int8 = vdupq_n_s32(INT8_MAX);

  for (size_t row_idx = 0; row_idx < mr; row_idx++) {
    if (row_idx < valid_rows) {
      const float* src_ptr = lhs_blk + (k * row_idx);
      const float scale = scales[row_idx];
      const int32_t nudged_zero_point = zps[row_idx];

      int8_t* dst_row = temp_q + row_idx * kblk;
      const float* slice_src = src_ptr + k_idx_block;

      float32x4_t vscale = vdupq_n_f32(scale);
      int32x4_t vzero_point = vdupq_n_s32(nudged_zero_point);

      size_t col = 0;
      size_t rem_k = (k > k_idx_block) ? (k - k_idx_block) : 0;
      size_t valid_in_block = (rem_k > kblk) ? kblk : rem_k;

      for (; col + 8 <= valid_in_block; col += 8) {
        float32x4_t src0 = vld1q_f32(slice_src + col);
        float32x4_t scaled0 = vmulq_f32(src0, vscale);
        int32x4_t src_s32_0 = vcvtaq_s32_f32(scaled0);
        src_s32_0 = vaddq_s32(src_s32_0, vzero_point);
        src_s32_0 = vmaxq_s32(src_s32_0, vmin_int8);
        src_s32_0 = vminq_s32(src_s32_0, vmax_int8);

        float32x4_t src1 = vld1q_f32(slice_src + col + 4);
        float32x4_t scaled1 = vmulq_f32(src1, vscale);
        int32x4_t src_s32_1 = vcvtaq_s32_f32(scaled1);
        src_s32_1 = vaddq_s32(src_s32_1, vzero_point);
        src_s32_1 = vmaxq_s32(src_s32_1, vmin_int8);
        src_s32_1 = vminq_s32(src_s32_1, vmax_int8);

        int16x4_t src_s16_0 = vmovn_s32(src_s32_0);
        int16x4_t src_s16_1 = vmovn_s32(src_s32_1);
        int8x8_t src_s8 = vmovn_s16(vcombine_s16(src_s16_0, src_s16_1));
        vst1_s8(dst_row + col, src_s8);
      }

      for (; col + 4 <= valid_in_block; col += 4) {
        float32x4_t src = vld1q_f32(slice_src + col);
        float32x4_t scaled = vmulq_f32(src, vscale);
        int32x4_t src_s32 = vcvtaq_s32_f32(scaled);
        src_s32 = vaddq_s32(src_s32, vzero_point);
        src_s32 = vmaxq_s32(src_s32, vmin_int8);
        src_s32 = vminq_s32(src_s32, vmax_int8);

        int16x4_t src_s16 = vmovn_s32(src_s32);
        int8x8_t src_s8 = vmovn_s16(vcombine_s16(src_s16, vdup_n_s16(0)));
        vst1_lane_u32((uint32_t*)(dst_row + col), vreinterpret_u32_s8(src_s8), 0);
      }

      int8_t last_q = 0;
      for (; col < valid_in_block; col++) {
        float val = slice_src[col];
        int32_t q = (int32_t)roundf(val * scale) + nudged_zero_point;
        q = math_max_s32(q, INT8_MIN);
        q = math_min_s32(q, INT8_MAX);
        dst_row[col] = (int8_t)q;
        last_q = (int8_t)q;
      }

      if (k > 0 && col > 0) {
        last_q = dst_row[col - 1];
      } else if (k > 0 && k_idx_block >= k) {
        float val = src_ptr[k - 1];
        int32_t q = (int32_t)roundf(val * scale) + nudged_zero_point;
        q = math_max_s32(q, INT8_MIN);
        q = math_min_s32(q, INT8_MAX);
        last_q = (int8_t)q;
      }

      for (; col < kblk; col++) {
        dst_row[col] = last_q;
      }
    } else {
      memset(temp_q + row_idx * kblk, 0, kblk);
    }
  }
}

static void xnn_x8_packq_f32qp8_m1__neon(
    size_t k, const float* lhs, void* lhs_packed, void* lhs_packed_zp)
{
  const float qmin = (float)INT8_MIN;
  const float qmax = (float)INT8_MAX;

  float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
  float32x4_t vmin = vdupq_n_f32(FLT_MAX);

  size_t k_idx = 0;
  for (; k_idx + 4 <= k; k_idx += 4) {
    float32x4_t src = vld1q_f32(lhs + k_idx);
    vmax = vmaxq_f32(vmax, src);
    vmin = vminq_f32(vmin, src);
  }

  float max_val = vmaxvq_f32(vmax);
  float min_val = vminvq_f32(vmin);

  for (; k_idx < k; k_idx++) {
    float val = lhs[k_idx];
    max_val = math_max_f32(max_val, val);
    min_val = math_min_f32(min_val, val);
  }

  const float rmin = math_min_f32(0.0F, min_val);
  const float rmax = math_max_f32(0.0F, max_val);
  const float s = rmin == rmax ? 1.F : (qmax - qmin) / (rmax - rmin);
  const float rs = s ? 1.0F / s : 0.0F;
  const float descaled_min = rmin * s;
  const float descaled_max = rmax * s;
  const float zero_point_from_min_error = qmin + descaled_min;
  const float zero_point_from_max_error = qmax + descaled_max;
  float zero_point = zero_point_from_min_error + zero_point_from_max_error > 0 ? qmin - descaled_min : qmax - descaled_max;
  zero_point = math_max_f32(zero_point, qmin);
  zero_point = math_min_f32(zero_point, qmax);
  const int32_t nudged_zero_point = (int32_t)rintf(zero_point);

  *(int32_t*)lhs_packed_zp = -nudged_zero_point;
  *((float*)lhs_packed_zp + 1) = rs;

  int8_t* lhs_packed_ptr = (int8_t*)lhs_packed;
  const float* src_ptr = lhs;

  float32x4_t vscale = vdupq_n_f32(s);
  int32x4_t vzero_point = vdupq_n_s32(nudged_zero_point);
  int32x4_t vmin_int8 = vdupq_n_s32(INT8_MIN);
  int32x4_t vmax_int8 = vdupq_n_s32(INT8_MAX);

  k_idx = 0;
  for (; k_idx + 4 <= k; k_idx += 4) {
    float32x4_t src = vld1q_f32(src_ptr + k_idx);
    float32x4_t scaled = vmulq_f32(src, vscale);
    int32x4_t src_s32 = vcvtaq_s32_f32(scaled);
    src_s32 = vaddq_s32(src_s32, vzero_point);
    src_s32 = vmaxq_s32(src_s32, vmin_int8);
    src_s32 = vminq_s32(src_s32, vmax_int8);

    int16x4_t src_s16 = vmovn_s32(src_s32);
    int8x8_t src_s8 = vmovn_s16(vcombine_s16(src_s16, vdup_n_s16(0)));

    vst1_lane_u32((uint32_t*)lhs_packed_ptr, vreinterpret_u32_s8(src_s8), 0);
    lhs_packed_ptr += 4;
  }

  for (; k_idx < k; k_idx++) {
    float val = src_ptr[k_idx];
    float scaled = val * s;
    float rounded = roundf(scaled);
    float biased = rounded + nudged_zero_point;
    int32_t src_s32 = (int32_t)biased;
    src_s32 = math_max_s32(src_s32, INT8_MIN);
    src_s32 = math_min_s32(src_s32, INT8_MAX);
    *lhs_packed_ptr++ = (int8_t)src_s32;
  }

  const size_t k_internal = ((k + 31) / 32) * 32;
  if (k_idx < k_internal) {
    float val = src_ptr[k - 1];
    float scaled = val * s;
    float rounded = roundf(scaled);
    float biased = rounded + nudged_zero_point;
    int32_t src_s32 = (int32_t)biased;
    src_s32 = math_max_s32(src_s32, INT8_MIN);
    src_s32 = math_min_s32(src_s32, INT8_MAX);
    int8_t last_quant = (int8_t)src_s32;
    for (; k_idx < k_internal; k_idx++) {
      *lhs_packed_ptr++ = last_quant;
    }
  }
}

__arm_locally_streaming
static void xnn_x8_packq_f32qp8_stage2__sme2(
    size_t k, size_t mr, void* lhs_packed, const int8_t* temp_q,
    size_t num_k_blocks, size_t mr_blk_idx, size_t k_idx_start) __arm_inout("za")
{
  const size_t vl = svcntw();
  const size_t kblk = 4 * vl;
  size_t k_rup_32 = ((k+31)/32)*32;

  int8_t* lhs_packed_blk = (int8_t*)lhs_packed + mr_blk_idx * (k_rup_32 + (4+4));

  svcount_t pcnt_8_all = svptrue_c8();
  svbool_t pg_8_all = svptrue_b8();

  for (size_t blk = 0; blk < num_k_blocks; blk++) {
  const size_t k_idx = k_idx_start + blk * kblk;
  const int8_t* blk_q = temp_q + blk * mr * kblk;
  int8_t* lhs_packed_inc = lhs_packed_blk + k_idx * mr;

  for (size_t row_idx = 0; row_idx < mr; row_idx += 2) {
    svint8_t r0 = svld1_s8(pg_8_all, blk_q + row_idx * kblk);
    svint8_t r1 = svld1_s8(pg_8_all, blk_q + (row_idx + 1) * kblk);

    if (row_idx < vl) {
      svwrite_hor_za32_u32_vg2(0, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
    } else {
      svwrite_hor_za32_u32_vg2(1, row_idx, svreinterpret_u32(svcreate2(r0, r1)));
    }
  }

  size_t k_groups = (k_rup_32 - k_idx) / 4;
  k_groups = (k_groups < vl) ? k_groups : vl;
  for (size_t j = 0; j < k_groups; j += 4) {
    svuint32x4_t vout_x4_0 = svread_ver_za32_u32_vg4(0, j); //tile, slice
    svuint32x4_t vout_x4_1 = svread_ver_za32_u32_vg4(1, j);
    svuint32x4_t vout_x4_comb0 = svcreate4(svget4(vout_x4_0,0), svget4(vout_x4_1,0), svget4(vout_x4_0,1), svget4(vout_x4_1, 1));
    svuint32x4_t vout_x4_comb1 = svcreate4(svget4(vout_x4_0,2), svget4(vout_x4_1,2), svget4(vout_x4_0,3), svget4(vout_x4_1, 3));

    svst1_s8_x4(pcnt_8_all, lhs_packed_inc, svreinterpret_s8(vout_x4_comb0)); lhs_packed_inc += 16 * vl;
    svst1_s8_x4(pcnt_8_all, lhs_packed_inc, svreinterpret_s8(vout_x4_comb1)); lhs_packed_inc += 16 * vl;
  }
  }
}

static void xnn_x8_packq_f32qp8_m_small__neon(
    size_t mr, size_t slot, size_t valid_rows, size_t k, const float* lhs_blk,
    int8_t* lhs_packed_blk, int8_t* lhs_packed_zp)
{
  float scales[64];
  int32_t zps[64];
  float recip_scales[64];

  // Calculate quantization parameters using calc_params__neon for correct -zp and scale encoding
  xnn_x8_packq_f32qp8_calc_params__neon(
      mr, slot, valid_rows, k, lhs_blk, scales, zps, recip_scales, lhs_packed_zp);

  int32x4_t vmin_int8 = vdupq_n_s32(INT8_MIN);
  int32x4_t vmax_int8 = vdupq_n_s32(INT8_MAX);

  float32x4_t vscales[64];
  int32x4_t vzps[64];
  for (size_t r = 0; r < valid_rows; r++) {
    vscales[r] = vdupq_n_f32(scales[r]);
    vzps[r] = vdupq_n_s32(zps[r]);
  }

  for (size_t k_idx = 0; k_idx < k; k_idx += 32) {
    int8_t* blk_ptr = lhs_packed_blk + (k_idx / 32) * (mr * 32);
    size_t k_len = (k - k_idx > 32) ? 32 : (k - k_idx);

    for (size_t s = 0; s < (k_len + 3) / 4; s++) {
      size_t col_start = k_idx + s * 4;
      int8_t* sub_ptr = blk_ptr + s * (mr * 4);

      for (size_t r = 0; r < valid_rows; r++) {
        const float* src = lhs_blk + r * k + col_start;
        float scale = scales[r];
        int32_t zp = zps[r];
        int8_t* out_r = sub_ptr + (slot + r) * 4;

        if (col_start + 4 <= k) {
          float32x4_t vsrc = vld1q_f32(src);
          float32x4_t vscaled = vmulq_f32(vsrc, vscales[r]);
          int32x4_t vsrc_s32 = vcvtaq_s32_f32(vscaled);
          vsrc_s32 = vaddq_s32(vsrc_s32, vzps[r]);
          vsrc_s32 = vmaxq_s32(vsrc_s32, vmin_int8);
          vsrc_s32 = vminq_s32(vsrc_s32, vmax_int8);

          int16x4_t vsrc_s16 = vmovn_s32(vsrc_s32);
          int8x8_t vsrc_s8 = vmovn_s16(vcombine_s16(vsrc_s16, vdup_n_s16(0)));
          vst1_lane_u32((uint32_t*)out_r, vreinterpret_u32_s8(vsrc_s8), 0);
        } else {
          size_t c = 0;
          size_t valid_c = k - col_start;
          int8_t last_q = 0;
          for (; c < valid_c; c++) {
            int32_t q = (int32_t)roundf(src[c] * scale) + zp;
            q = math_max_s32(q, INT8_MIN);
            q = math_min_s32(q, INT8_MAX);
            out_r[c] = (int8_t)q;
            last_q = (int8_t)q;
          }
          for (; c < 4; c++) {
            out_r[c] = last_q;
          }
        }
      }
    }
  }
}

#define XNN_PACKQ_SME2_TEMP_BYTES 32768

__arm_new("za")
static void xnn_x8_packq_f32qp8_m_large__sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed)
{
  size_t k_rup_32 = ((k+31)/32)*32;
  const size_t kblk = 2 * mr;
  const size_t k_rup_kblk = ((k + kblk - 1) / kblk) * kblk;
  const size_t kc = kblk * (XNN_PACKQ_SME2_TEMP_BYTES / (mr * kblk));

  XNN_ALIGN(64) __attribute__((uninitialized))
  int8_t temp_q[XNN_PACKQ_SME2_TEMP_BYTES];

  for (size_t mr_blk_idx = 0; mr_blk_idx < ((m + mr - 1) / mr) * mr; mr_blk_idx += mr) {
    size_t valid_rows = (m - mr_blk_idx < mr) ? (m - mr_blk_idx) : mr;
    const float* lhs_blk = lhs + mr_blk_idx * k;
    int8_t* lhs_packed_blk = (int8_t*)lhs_packed + mr_blk_idx * (k_rup_32 + (4+4));
    int8_t* lhs_packed_zp  = (int8_t*)lhs_packed_blk + mr * k_rup_32;

    float scales[64];
    int32_t zps[64];
    float recip_scales[64];

    // Calculate quantization parameters once per mr-row block
    xnn_x8_packq_f32qp8_calc_params__neon(
        mr, 0, valid_rows, k, lhs_blk, scales, zps, recip_scales, lhs_packed_zp);

    for (size_t k_chunk = 0; k_chunk < k_rup_kblk; k_chunk += kc) {
      const size_t chunk_len = (k_rup_kblk - k_chunk < kc) ? (k_rup_kblk - k_chunk) : kc;
      const size_t num_k_blocks = chunk_len / kblk;

      for (size_t blk = 0; blk < num_k_blocks; blk++) {
        xnn_x8_packq_f32qp8_quantize_block__neon(
            mr, kblk, valid_rows, k, lhs_blk, scales, zps, temp_q + blk * mr * kblk, k_chunk + blk * kblk);
      }

      xnn_x8_packq_f32qp8_stage2__sme2(
          k, mr, lhs_packed, temp_q, num_k_blocks,
          mr_blk_idx, k_chunk);
    }
  }
}
#undef XNN_PACKQ_SME2_TEMP_BYTES

void xnn_x8_packq_f32qp8_ukernel__sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs,
    size_t lhs_stride, void* lhs_packed)
{
  if (kr != 4 || sr != 1 ||
      lhs_stride != k * sizeof(float) || !((m == 1 && mr == 1) || mr == 2 * svcntsw())) {
#if XNN_ENABLE_KLEIDIAI
    xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2(
        m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
#else
    xnn_x8_packq_f32qp8_ukernel__scalar_u1(
        m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
#endif
    return;
  }

  const size_t slot = m_idx_start % mr;
  if (slot != 0) {
    const size_t k_rup_32 = ((k+31)/32)*32;
    const size_t head = (m < mr - slot) ? m : mr - slot;
    int8_t* lhs_packed_blk = (int8_t*)lhs_packed;
    int8_t* lhs_packed_zp  = (int8_t*)lhs_packed_blk + mr * k_rup_32;
    xnn_x8_packq_f32qp8_m_small__neon(mr, slot, head, k, lhs, lhs_packed_blk, lhs_packed_zp);
    m -= head;
    if (m == 0) {
      return;
    }
    lhs += head * k;
    lhs_packed = (int8_t*)lhs_packed + mr * (k_rup_32 + (4+4));
  }

  if (m == 1 && mr == 1) {
    int8_t* lhs_packed_ptr = (int8_t*)lhs_packed;
    int8_t* lhs_packed_zp  = (int8_t*)lhs_packed + ((k+31)/32)*32;

    // Single-row NEON path
    xnn_x8_packq_f32qp8_m1__neon(k, lhs, lhs_packed_ptr, lhs_packed_zp);

  } else if (m <= mr / 2) {
    size_t k_rup_32 = ((k+31)/32)*32;
    int8_t* lhs_packed_blk = (int8_t*)lhs_packed;
    int8_t* lhs_packed_zp  = (int8_t*)lhs_packed_blk + mr * k_rup_32;

    // Direct NEON path for M <= mr/2
    xnn_x8_packq_f32qp8_m_small__neon(mr, 0, m, k, lhs, lhs_packed_blk, lhs_packed_zp);

  } else {
    // Two-stage NEON + SME2 transpose path for M > mr/2
    xnn_x8_packq_f32qp8_m_large__sme2(
        m, k, mr, kr, sr, 0, lhs, lhs_stride, lhs_packed);
  }
}
#endif  // XNN_ENABLE_ARM_SME2_ACLE
