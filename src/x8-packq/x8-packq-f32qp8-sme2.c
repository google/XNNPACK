// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
#include <arm_neon.h>
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/packq.h"

// Maximum possible mr (2 * svcntsw()) for architectural max SVL of 2048 bits.
#define XNN_SME_PACKQ_MAX_MR 128

static XNN_INLINE int8x8_t quantize_f32x8_neon(float32x4_t v0, float32x4_t v1,
                                               float32x4_t vscale,
                                               int16x8_t vzp) {
  const int32x4_t vi0 = vcvtaq_s32_f32(vmulq_f32(v0, vscale));
  const int32x4_t vi1 = vcvtaq_s32_f32(vmulq_f32(v1, vscale));
  const int16x8_t vs =
      vqaddq_s16(vcombine_s16(vqmovn_s32(vi0), vqmovn_s32(vi1)), vzp);
  return vqmovn_s16(vs);
}

static XNN_INLINE int8_t* store_packq_s8x8_neon(int8_t* dst_ptr, int8x8_t vq,
                                                size_t mr, size_t k_block_len,
                                                size_t k_idx) {
  if (mr == 1) {
    vst1_s8(dst_ptr, vq);
    return dst_ptr + 8;
  }
  if (k_block_len == 4) {
    const size_t step = mr * 4;
    vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq), 0);
    dst_ptr += step;
    vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq), 1);
    return dst_ptr + step;
  }
  if (k_block_len == 8) {
    vst1_s8(dst_ptr, vq);
    return dst_ptr + mr * 8;
  }
  if (k_block_len == 2) {
    const size_t step = mr * 2;
    vst1_lane_s16((int16_t*)dst_ptr, vreinterpret_s16_s8(vq), 0);
    dst_ptr += step;
    vst1_lane_s16((int16_t*)dst_ptr, vreinterpret_s16_s8(vq), 1);
    dst_ptr += step;
    vst1_lane_s16((int16_t*)dst_ptr, vreinterpret_s16_s8(vq), 2);
    dst_ptr += step;
    vst1_lane_s16((int16_t*)dst_ptr, vreinterpret_s16_s8(vq), 3);
    return dst_ptr + step;
  }
  if (k_block_len == 1) {
    vst1_lane_s8(dst_ptr, vq, 0);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 1);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 2);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 3);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 4);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 5);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 6);
    dst_ptr += mr;
    vst1_lane_s8(dst_ptr, vq, 7);
    return dst_ptr + mr;
  }
  vst1_s8(dst_ptr, vq);
  if (((k_idx + 8) & (k_block_len - 1)) == 0) {
    return dst_ptr + 8 + (mr - 1) * k_block_len;
  }
  return dst_ptr + 8;
}

static void packq_f32qp8_neon(size_t m, size_t k, size_t mr, size_t kr,
                              size_t sr, size_t m_idx_start,
                              const float* XNN_RESTRICT lhs, size_t lhs_stride,
                              void* XNN_RESTRICT lhs_packed) {
  assert((kr % sr) == 0);
  const float qmin = (float)INT8_MIN;
  const float qmax = (float)INT8_MAX;
  const size_t dst_stride = lhs_packed_stride(k, mr, kr, sr);
  const size_t k_internal = k_roundedup(k, kr, sr);
  const size_t k_block_len = kr / sr;

  for (size_t row_idx = 0; row_idx < m; ++row_idx) {
    const float* src_row =
        (const float*)((uintptr_t)lhs + row_idx * lhs_stride);
    float32x4_t vmax0 = vdupq_n_f32(0.0f);
    float32x4_t vmin0 = vdupq_n_f32(0.0f);
    float32x4_t vmax1 = vdupq_n_f32(0.0f);
    float32x4_t vmin1 = vdupq_n_f32(0.0f);

    size_t k_idx = 0;
    for (; k_idx + 16 <= k; k_idx += 16) {
      const float32x4_t v0 = vld1q_f32(src_row + k_idx);
      const float32x4_t v1 = vld1q_f32(src_row + k_idx + 4);
      const float32x4_t v2 = vld1q_f32(src_row + k_idx + 8);
      const float32x4_t v3 = vld1q_f32(src_row + k_idx + 12);
      vmax0 = vmaxq_f32(vmax0, vmaxq_f32(v0, v1));
      vmin0 = vminq_f32(vmin0, vminq_f32(v0, v1));
      vmax1 = vmaxq_f32(vmax1, vmaxq_f32(v2, v3));
      vmin1 = vminq_f32(vmin1, vminq_f32(v2, v3));
    }
    for (; k_idx + 4 <= k; k_idx += 4) {
      const float32x4_t v0 = vld1q_f32(src_row + k_idx);
      vmax0 = vmaxq_f32(vmax0, v0);
      vmin0 = vminq_f32(vmin0, v0);
    }
    float max0 = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    float min0 = vminvq_f32(vminq_f32(vmin0, vmin1));
    for (; k_idx < k; ++k_idx) {
      const float val = src_row[k_idx];
      max0 = math_max_f32(val, max0);
      min0 = math_min_f32(val, min0);
    }

    const float scale0 = min0 == max0 ? 1.0f : (qmax - qmin) / (max0 - min0);
    const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;
    const float descaled_min0 = min0 * scale0;
    const float descaled_max0 = max0 * scale0;
    const float zero_point_from_min_error0 = qmin + descaled_min0;
    const float zero_point_from_max_error0 = qmax + descaled_max0;
    float zero_point0 =
        zero_point_from_min_error0 + zero_point_from_max_error0 > 0.0f
            ? qmin - descaled_min0
            : qmax - descaled_max0;
    zero_point0 = math_max_f32(zero_point0, qmin);
    zero_point0 = math_min_f32(zero_point0, qmax);
    const int32_t nudged_zero_point0 =
        vgetq_lane_s32(vcvtnq_s32_f32(vdupq_n_f32(zero_point0)), 0);

    const float32x4_t vscale = vdupq_n_f32(scale0);
    const int16x8_t vzp = vdupq_n_s16((int16_t)nudged_zero_point0);
    const size_t dst_x = (row_idx + m_idx_start) % mr;
    int8_t* dst_ptr = (int8_t*)lhs_packed + dst_x * k_block_len;

    k_idx = 0;
    if (mr == 1) {
      for (; k_idx + 16 <= k; k_idx += 16) {
        const int8x8_t vq0 = quantize_f32x8_neon(
            vld1q_f32(src_row + k_idx), vld1q_f32(src_row + k_idx + 4), vscale,
            vzp);
        const int8x8_t vq1 = quantize_f32x8_neon(
            vld1q_f32(src_row + k_idx + 8), vld1q_f32(src_row + k_idx + 12),
            vscale, vzp);
        vst1q_s8(dst_ptr, vcombine_s8(vq0, vq1));
        dst_ptr += 16;
      }
    } else if (k_block_len == 4) {
      const size_t step = mr * 4;
      for (; k_idx + 16 <= k; k_idx += 16) {
        const int8x8_t vq0 = quantize_f32x8_neon(
            vld1q_f32(src_row + k_idx), vld1q_f32(src_row + k_idx + 4), vscale,
            vzp);
        const int8x8_t vq1 = quantize_f32x8_neon(
            vld1q_f32(src_row + k_idx + 8), vld1q_f32(src_row + k_idx + 12),
            vscale, vzp);
        vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq0), 0);
        dst_ptr += step;
        vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq0), 1);
        dst_ptr += step;
        vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq1), 0);
        dst_ptr += step;
        vst1_lane_s32((int32_t*)dst_ptr, vreinterpret_s32_s8(vq1), 1);
        dst_ptr += step;
      }
    }
    for (; k_idx + 8 <= k; k_idx += 8) {
      const int8x8_t vq = quantize_f32x8_neon(
          vld1q_f32(src_row + k_idx), vld1q_f32(src_row + k_idx + 4), vscale,
          vzp);
      dst_ptr = store_packq_s8x8_neon(dst_ptr, vq, mr, k_block_len, k_idx);
    }
    if (k_idx < k_internal) {
      const float last_val = src_row[k - 1];
      if (k_idx < k) {
        float buf[8];
        for (size_t j = 0; j < 8; ++j) {
          buf[j] = (k_idx + j < k) ? src_row[k_idx + j] : last_val;
        }
        const int8x8_t vq = quantize_f32x8_neon(vld1q_f32(buf),
                                                vld1q_f32(buf + 4), vscale, vzp);
        dst_ptr = store_packq_s8x8_neon(dst_ptr, vq, mr, k_block_len, k_idx);
        k_idx += 8;
      }
      if (k_idx < k_internal) {
        const float32x4_t vlast = vdupq_n_f32(last_val);
        const int8x8_t vpad = quantize_f32x8_neon(vlast, vlast, vscale, vzp);
        for (; k_idx < k_internal; k_idx += 8) {
          dst_ptr =
              store_packq_s8x8_neon(dst_ptr, vpad, mr, k_block_len, k_idx);
        }
      }
    }

    int32_t* zp_ptr =
        (int32_t*)((int8_t*)lhs_packed + mr * k_internal) + dst_x;
    *zp_ptr = -nudged_zero_point0;
    *((float*)(zp_ptr + mr)) = recip_scale0;

    if (((row_idx + 1 + m_idx_start) % mr) == 0) {
      lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
    }
  }
}

__arm_locally_streaming __arm_new("za") static bool packq_f32qp8_mr_sme2(
    size_t m, size_t k, size_t mr, const float* lhs, size_t lhs_stride,
    void* lhs_packed) {
  const size_t svl_w = svcntsw();
  const size_t svl_b = svcntb();
  if (mr != svl_w && mr != 2 * svl_w) {
    return false;
  }
  assert(mr <= XNN_SME_PACKQ_MAX_MR);
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t dst_stride = mr * (k_internal + sizeof(int32_t) + sizeof(float));
  const svbool_t pg_32_all = svptrue_b32();

  float min_vals[XNN_SME_PACKQ_MAX_MR];
  float max_vals[XNN_SME_PACKQ_MAX_MR];
  float last_vals[XNN_SME_PACKQ_MAX_MR];
  float scales[XNN_SME_PACKQ_MAX_MR];
  int32_t zps[XNN_SME_PACKQ_MAX_MR];
  int32_t last_quants[XNN_SME_PACKQ_MAX_MR];

  for (size_t mr_blk_idx = 0; mr_blk_idx < m; mr_blk_idx += mr) {
    const size_t rem_rows = m - mr_blk_idx;
    const size_t valid_rows = rem_rows < mr ? rem_rows : mr;
    const float* lhs_blk =
        (const float*)((uintptr_t)lhs + mr_blk_idx * lhs_stride);
    int8_t* lhs_packed_blk =
        (int8_t*)lhs_packed + (mr_blk_idx / mr) * dst_stride;
    int32_t* zp_ptr = (int32_t*)(lhs_packed_blk + mr * k_internal);
    float* recip_scale_ptr = (float*)(zp_ptr + mr);

    for (size_t r = 0; r < valid_rows; ++r) {
      const float* src_row =
          (const float*)((uintptr_t)lhs_blk + r * lhs_stride);
      svfloat32_t vmax0 = svdup_n_f32(0.0f);
      svfloat32_t vmin0 = svdup_n_f32(0.0f);
      svfloat32_t vmax1 = svdup_n_f32(0.0f);
      svfloat32_t vmin1 = svdup_n_f32(0.0f);

      size_t k_idx = 0;
      if (r + 1 < valid_rows) {
        const float* next_row = (const float*)((uintptr_t)src_row + lhs_stride);
        for (; k_idx + 2 * svl_w <= k; k_idx += 2 * svl_w) {
          svprfb(pg_32_all, next_row + k_idx, SV_PLDL2KEEP);
          const svfloat32_t v0 = svld1_f32(pg_32_all, src_row + k_idx);
          const svfloat32_t v1 = svld1_f32(pg_32_all, src_row + k_idx + svl_w);
          vmax0 = svmax_f32_m(pg_32_all, vmax0, v0);
          vmin0 = svmin_f32_m(pg_32_all, vmin0, v0);
          vmax1 = svmax_f32_m(pg_32_all, vmax1, v1);
          vmin1 = svmin_f32_m(pg_32_all, vmin1, v1);
        }
      } else {
        for (; k_idx + 2 * svl_w <= k; k_idx += 2 * svl_w) {
          const svfloat32_t v0 = svld1_f32(pg_32_all, src_row + k_idx);
          const svfloat32_t v1 = svld1_f32(pg_32_all, src_row + k_idx + svl_w);
          vmax0 = svmax_f32_m(pg_32_all, vmax0, v0);
          vmin0 = svmin_f32_m(pg_32_all, vmin0, v0);
          vmax1 = svmax_f32_m(pg_32_all, vmax1, v1);
          vmin1 = svmin_f32_m(pg_32_all, vmin1, v1);
        }
      }
      if (k_idx < k) {
        const svbool_t pg0 = svwhilelt_b32_u64(k_idx, k);
        const svfloat32_t v0 = svld1_f32(pg0, src_row + k_idx);
        vmax0 = svmax_f32_m(pg0, vmax0, v0);
        vmin0 = svmin_f32_m(pg0, vmin0, v0);
        k_idx += svl_w;
        if (k_idx < k) {
          const svbool_t pg1 = svwhilelt_b32_u64(k_idx, k);
          const svfloat32_t v1 = svld1_f32(pg1, src_row + k_idx);
          vmax1 = svmax_f32_m(pg1, vmax1, v1);
          vmin1 = svmin_f32_m(pg1, vmin1, v1);
        }
      }
      vmax0 = svmax_f32_m(pg_32_all, vmax0, vmax1);
      vmin0 = svmin_f32_m(pg_32_all, vmin0, vmin1);
      max_vals[r] = svmaxv_f32(pg_32_all, vmax0);
      min_vals[r] = svminv_f32(pg_32_all, vmin0);
      last_vals[r] = src_row[k - 1];
    }
    for (size_t h = 0; h < mr; h += svl_w) {
      if (h >= valid_rows) {
        svst1_s32(pg_32_all, zp_ptr + h, svdup_n_s32(0));
        svst1_f32(pg_32_all, recip_scale_ptr + h, svdup_n_f32(0.0f));
        continue;
      }
      const svbool_t pg_valid = svwhilelt_b32_u64(h, valid_rows);
      const svfloat32_t vmin = svld1_f32(pg_valid, min_vals + h);
      const svfloat32_t vmax = svld1_f32(pg_valid, max_vals + h);
      const svbool_t pg_eq = svcmpeq_f32(pg_valid, vmin, vmax);
      const svfloat32_t vscale =
          svsel_f32(pg_eq, svdup_n_f32(1.0f),
                    svdiv_f32_x(pg_valid, svdup_n_f32(255.0f),
                                svsub_f32_x(pg_valid, vmax, vmin)));
      const svbool_t pg_nz = svcmpne_n_f32(pg_valid, vscale, 0.0f);
      const svfloat32_t vrecip_scale = svsel_f32(
          pg_nz, svdivr_n_f32_x(pg_valid, vscale, 1.0f), svdup_n_f32(0.0f));

      const svfloat32_t vdescaled_min = svmul_f32_x(pg_valid, vmin, vscale);
      const svfloat32_t vdescaled_max = svmul_f32_x(pg_valid, vmax, vscale);
      const svfloat32_t verr = svadd_f32_x(
          pg_valid, svadd_n_f32_x(pg_valid, vdescaled_min, -128.0f),
          svadd_n_f32_x(pg_valid, vdescaled_max, 127.0f));
      const svbool_t pg_gt = svcmpgt_n_f32(pg_valid, verr, 0.0f);
      svfloat32_t vzp =
          svsel_f32(pg_gt, svsubr_n_f32_x(pg_valid, vdescaled_min, -128.0f),
                    svsubr_n_f32_x(pg_valid, vdescaled_max, 127.0f));
      vzp = svmin_n_f32_x(pg_valid, svmax_n_f32_x(pg_valid, vzp, -128.0f),
                          127.0f);
      const svint32_t vnudged_zp =
          svcvt_s32_f32_z(pg_valid, svrintn_f32_z(pg_valid, vzp));

      svst1_s32(pg_32_all, zp_ptr + h, svneg_s32_z(pg_valid, vnudged_zp));
      svst1_f32(pg_32_all, recip_scale_ptr + h, vrecip_scale);
      svst1_f32(pg_valid, scales + h, vscale);
      svst1_s32(pg_valid, zps + h, vnudged_zp);

      const svfloat32_t vlast = svld1_f32(pg_valid, last_vals + h);
      svint32_t vlast_q = svcvt_s32_f32_x(
          pg_valid,
          svrinta_f32_x(pg_valid, svmul_f32_x(pg_valid, vlast, vscale)));
      vlast_q = svadd_s32_x(pg_valid, vlast_q, vnudged_zp);
      vlast_q = svmax_n_s32_x(pg_valid, vlast_q, INT8_MIN);
      vlast_q = svmin_n_s32_x(pg_valid, vlast_q, INT8_MAX);
      svst1_s32(pg_valid, last_quants + h, vlast_q);
    }

    if (valid_rows < mr) {
      svzero_za();
    }

    const size_t rows0 = valid_rows < svl_w ? valid_rows : svl_w;
    const size_t k_tile = 4 * svl_w;
    int8_t* dst_ptr = lhs_packed_blk;
    for (size_t k_chunk = 0; k_chunk < k_internal; k_chunk += k_tile) {
      if (XNN_LIKELY(k_chunk + k_tile <= k)) {
        const size_t prfm_k_offset = (k_chunk + 2 * k_tile <= k) ? k_tile : 0;
        for (size_t r = 0; r < rows0; ++r) {
          const float* src_row =
              (const float*)((uintptr_t)lhs_blk + r * lhs_stride) + k_chunk;
          const float* prfm_row = src_row + prfm_k_offset;
          svprfb(pg_32_all, prfm_row, SV_PLDL1KEEP);
          svprfb(pg_32_all, prfm_row + 2 * svl_w, SV_PLDL1KEEP);
          const svfloat32_t vscale = svdup_n_f32(scales[r]);
          const svint32_t vzp = svdup_n_s32(zps[r]);

          const svfloat32_t vf0 = svld1_f32(pg_32_all, src_row);
          const svfloat32_t vf1 = svld1_f32(pg_32_all, src_row + svl_w);
          const svfloat32_t vf2 = svld1_f32(pg_32_all, src_row + 2 * svl_w);
          const svfloat32_t vf3 = svld1_f32(pg_32_all, src_row + 3 * svl_w);

          const svint32_t vi0 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf0, vscale))),
              vzp);
          const svint32_t vi1 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf1, vscale))),
              vzp);
          const svint32_t vi2 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf2, vscale))),
              vzp);
          const svint32_t vi3 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf3, vscale))),
              vzp);

          const svint8_t vq =
              svqcvt_s8_s32_x4(svcreate4_s32(vi0, vi1, vi2, vi3));
          svwrite_hor_za32_s32_m(0, r, pg_32_all, svreinterpret_s32_s8(vq));
        }
        for (size_t r = svl_w; r < valid_rows; ++r) {
          const float* src_row =
              (const float*)((uintptr_t)lhs_blk + r * lhs_stride) + k_chunk;
          const float* prfm_row = src_row + prfm_k_offset;
          svprfb(pg_32_all, prfm_row, SV_PLDL1KEEP);
          svprfb(pg_32_all, prfm_row + 2 * svl_w, SV_PLDL1KEEP);
          const svfloat32_t vscale = svdup_n_f32(scales[r]);
          const svint32_t vzp = svdup_n_s32(zps[r]);

          const svfloat32_t vf0 = svld1_f32(pg_32_all, src_row);
          const svfloat32_t vf1 = svld1_f32(pg_32_all, src_row + svl_w);
          const svfloat32_t vf2 = svld1_f32(pg_32_all, src_row + 2 * svl_w);
          const svfloat32_t vf3 = svld1_f32(pg_32_all, src_row + 3 * svl_w);

          const svint32_t vi0 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf0, vscale))),
              vzp);
          const svint32_t vi1 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf1, vscale))),
              vzp);
          const svint32_t vi2 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf2, vscale))),
              vzp);
          const svint32_t vi3 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf3, vscale))),
              vzp);

          const svint8_t vq =
              svqcvt_s8_s32_x4(svcreate4_s32(vi0, vi1, vi2, vi3));
          svwrite_hor_za32_s32_m(1, r - svl_w, pg_32_all,
                                 svreinterpret_s32_s8(vq));
        }
      } else {
        const svbool_t pg_f0 = svwhilelt_b32_u64(k_chunk, k);
        const svbool_t pg_f1 = svwhilelt_b32_u64(k_chunk + svl_w, k);
        const svbool_t pg_f2 = svwhilelt_b32_u64(k_chunk + 2 * svl_w, k);
        const svbool_t pg_f3 = svwhilelt_b32_u64(k_chunk + 3 * svl_w, k);
        const svbool_t pg_k = svwhilelt_b8_u64(k_chunk, k);

        for (size_t r = 0; r < valid_rows; ++r) {
          const float* src_row =
              (const float*)((uintptr_t)lhs_blk + r * lhs_stride) + k_chunk;
          const svfloat32_t vscale = svdup_n_f32(scales[r]);
          const svint32_t vzp = svdup_n_s32(zps[r]);

          const svfloat32_t vf0 = svld1_f32(pg_f0, src_row);
          const svfloat32_t vf1 = svld1_f32(pg_f1, src_row + svl_w);
          const svfloat32_t vf2 = svld1_f32(pg_f2, src_row + 2 * svl_w);
          const svfloat32_t vf3 = svld1_f32(pg_f3, src_row + 3 * svl_w);

          const svint32_t vi0 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf0, vscale))),
              vzp);
          const svint32_t vi1 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf1, vscale))),
              vzp);
          const svint32_t vi2 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf2, vscale))),
              vzp);
          const svint32_t vi3 = svadd_s32_x(
              pg_32_all,
              svcvt_s32_f32_x(
                  pg_32_all,
                  svrinta_f32_x(pg_32_all,
                                svmul_f32_x(pg_32_all, vf3, vscale))),
              vzp);

          svint8_t vq = svqcvt_s8_s32_x4(svcreate4_s32(vi0, vi1, vi2, vi3));
          vq = svsel_s8(pg_k, vq, svdup_n_s8((int8_t)last_quants[r]));

          if (r < svl_w) {
            svwrite_hor_za32_s32_m(0, r, pg_32_all, svreinterpret_s32_s8(vq));
          } else {
            svwrite_hor_za32_s32_m(1, r - svl_w, pg_32_all,
                                   svreinterpret_s32_s8(vq));
          }
        }
      }

      const size_t rem_s = (k_internal - k_chunk) / 4;
      const size_t num_s = svl_w < rem_s ? svl_w : rem_s;
      const svint32_t vzero = svdup_n_s32(0);
      if (mr == svl_w) {
        for (size_t s = 0; s < num_s; ++s) {
          const svint32_t vcol0 =
              svread_ver_za32_s32_m(vzero, pg_32_all, 0, s);
          svst1_s32(pg_32_all, (int32_t*)dst_ptr, vcol0);
          dst_ptr += svl_b;
        }
      } else if (valid_rows <= svl_w) {
        for (size_t s = 0; s < num_s; ++s) {
          const svint32_t vcol0 =
              svread_ver_za32_s32_m(vzero, pg_32_all, 0, s);
          svst1_s32(pg_32_all, (int32_t*)dst_ptr, vcol0);
          svst1_s32(pg_32_all, (int32_t*)(dst_ptr + svl_b), vzero);
          dst_ptr += 2 * svl_b;
        }
      } else {
        for (size_t s = 0; s < num_s; ++s) {
          const svint32_t vcol0 =
              svread_ver_za32_s32_m(vzero, pg_32_all, 0, s);
          const svint32_t vcol1 =
              svread_ver_za32_s32_m(vzero, pg_32_all, 1, s);
          svst1_s32(pg_32_all, (int32_t*)dst_ptr, vcol0);
          svst1_s32(pg_32_all, (int32_t*)(dst_ptr + svl_b), vcol1);
          dst_ptr += 2 * svl_b;
        }
      }
    }
  }
  return true;
}

void xnn_x8_packq_f32qp8_ukernel__sme2(size_t m, size_t k, size_t mr, size_t kr,
                                       size_t sr, size_t m_idx_start,
                                       const float* XNN_RESTRICT lhs,
                                       size_t lhs_stride,
                                       void* XNN_RESTRICT lhs_packed) {
  if (m == 0) {
    return;
  }
  if (kr == 4 * sr && m_idx_start == 0 && m > 1 && mr >= 16 &&
      (mr & (mr - 1)) == 0) {
    if (packq_f32qp8_mr_sme2(m, k, mr, lhs, lhs_stride, lhs_packed)) {
      return;
    }
  }
  packq_f32qp8_neon(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ARCH_ARM64
