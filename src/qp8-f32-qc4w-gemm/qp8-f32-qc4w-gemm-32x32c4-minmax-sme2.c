// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

__arm_locally_streaming __arm_new(
    "za") void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x32c4__sme2(size_t mr,
                                                                  size_t nc,
                                                                  size_t kc,
                                                                  const void* a,
                                                                  const void* w,
                                                                  float* c,
                                                                  size_t
                                                                      cm_stride,
                                                                  size_t
                                                                      cn_stride,
                                                                  const void*
                                                                      params) {
  (void)cn_stride;
  const size_t vl = svcntw();
  const size_t mr_packed = 2 * vl;
  const size_t nr = 2 * vl;
  const size_t k_internal = (kc + 31) & ~(size_t)31;
  const size_t lhs_block_stride = mr_packed * (k_internal * sizeof(int8_t) +
                                               sizeof(int32_t) + sizeof(float));
  const size_t rhs_block_stride =
      nr * ((k_internal / 2) + sizeof(int32_t) + sizeof(float) + sizeof(float));
  const struct xnn_f32_qc4w_minmax_params* minmax_params =
      (const struct xnn_f32_qc4w_minmax_params*)params;
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vmin = svdup_n_f32(minmax_params->scalar.min);
  const svfloat32_t vmax = svdup_n_f32(minmax_params->scalar.max);

  for (size_t m_idx = 0; m_idx < mr; m_idx += mr_packed) {
    const size_t m_rem = mr - m_idx < mr_packed ? mr - m_idx : mr_packed;
    const uint8_t* lhs_block =
        (const uint8_t*)a + (m_idx / mr_packed) * lhs_block_stride;
    const int32_t* lhs_neg_zps =
        (const int32_t*)(lhs_block + mr_packed * k_internal);
    const float* lhs_scales = (const float*)(lhs_neg_zps + mr_packed);

    for (size_t n_idx = 0; n_idx < nc; n_idx += nr) {
      if (m_rem <= vl && n_idx + nr < nc) {
        const uint8_t* rhs_block0 =
            (const uint8_t*)w + (n_idx / nr) * rhs_block_stride;
        const uint8_t* rhs_block1 = rhs_block0 + rhs_block_stride;
        svzero_za();

        const int8_t* a_ptr = (const int8_t*)lhs_block;
        const int8_t* w0_ptr = (const int8_t*)rhs_block0;
        const int8_t* w1_ptr = (const int8_t*)rhs_block1;
        for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
#pragma clang loop unroll(full)
          for (int u = 0; u < 4; ++u) {
            const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 2 * u);
            const svint8_t vraw0 = svld1_vnum_s8(pg8, w0_ptr, u);
            const svint8_t vraw1 = svld1_vnum_s8(pg8, w1_ptr, u);
            const svint8_t vlo0 = svlsl_n_s8_x(pg8, vraw0, 4);
            const svint8_t vhi0 = svand_n_s8_x(pg8, vraw0, (int8_t)0xF0);
            const svint8_t vb0 = svzip1_s8(vlo0, vhi0);
            const svint8_t vb1 = svzip2_s8(vlo0, vhi0);
            const svint8_t vlo1 = svlsl_n_s8_x(pg8, vraw1, 4);
            const svint8_t vhi1 = svand_n_s8_x(pg8, vraw1, (int8_t)0xF0);
            const svint8_t vb2 = svzip1_s8(vlo1, vhi1);
            const svint8_t vb3 = svzip2_s8(vlo1, vhi1);

            svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
            svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
            svmopa_za32_s8_m(2, pg8, pg8, va0, vb2);
            svmopa_za32_s8_m(3, pg8, pg8, va0, vb3);
          }
          a_ptr += 16 * mr_packed;
          w0_ptr += 8 * nr;
          w1_ptr += 8 * nr;
        }

        const int32_t* rhs_sums0 =
            (const int32_t*)(rhs_block0 + nr * (k_internal / 2));
        const float* rhs_scales0 = (const float*)(rhs_sums0 + nr);
        const float* rhs_biases0 = rhs_scales0 + nr;
        const int32_t* rhs_sums1 =
            (const int32_t*)(rhs_block1 + nr * (k_internal / 2));
        const float* rhs_scales1 = (const float*)(rhs_sums1 + nr);
        const float* rhs_biases1 = rhs_scales1 + nr;

        const svint32_t vrsum0 = svld1_vnum_s32(pg32, rhs_sums0, 0);
        const svint32_t vrsum1 = svld1_vnum_s32(pg32, rhs_sums0, 1);
        const svint32_t vrsum2 = svld1_vnum_s32(pg32, rhs_sums1, 0);
        const svint32_t vrsum3 = svld1_vnum_s32(pg32, rhs_sums1, 1);
        const svfloat32_t vrscale0 = svld1_vnum_f32(pg32, rhs_scales0, 0);
        const svfloat32_t vrscale1 = svld1_vnum_f32(pg32, rhs_scales0, 1);
        const svfloat32_t vrscale2 = svld1_vnum_f32(pg32, rhs_scales1, 0);
        const svfloat32_t vrscale3 = svld1_vnum_f32(pg32, rhs_scales1, 1);
        const svfloat32_t vrbias0 = svld1_vnum_f32(pg32, rhs_biases0, 0);
        const svfloat32_t vrbias1 = svld1_vnum_f32(pg32, rhs_biases0, 1);
        const svfloat32_t vrbias2 = svld1_vnum_f32(pg32, rhs_biases1, 0);
        const svfloat32_t vrbias3 = svld1_vnum_f32(pg32, rhs_biases1, 1);

        const size_t n_rem = nc - n_idx;
        const svbool_t p_out2 = svwhilelt_b32_u64(2 * vl, n_rem);
        const svbool_t p_out3 = svwhilelt_b32_u64(3 * vl, n_rem);

        for (size_t row = 0; row < m_rem; ++row) {
          const svint32_t vlhs_zp = svdup_n_s32(lhs_neg_zps[row]);
          const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scales[row]);

          svint32_t vacc0 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), 4);
          svint32_t vacc1 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), 4);
          svint32_t vacc2 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row), 4);
          svint32_t vacc3 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row), 4);

          vacc0 = svmla_s32_m(pg32, vacc0, vrsum0, vlhs_zp);
          vacc1 = svmla_s32_m(pg32, vacc1, vrsum1, vlhs_zp);
          vacc2 = svmla_s32_m(pg32, vacc2, vrsum2, vlhs_zp);
          vacc3 = svmla_s32_m(pg32, vacc3, vrsum3, vlhs_zp);

          svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
          svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);
          svfloat32_t vf2 = svcvt_f32_s32_x(pg32, vacc2);
          svfloat32_t vf3 = svcvt_f32_s32_x(pg32, vacc3);

          vf0 = svmla_f32_m(pg32, vrbias0, vf0,
                            svmul_f32_x(pg32, vlhs_scale, vrscale0));
          vf1 = svmla_f32_m(pg32, vrbias1, vf1,
                            svmul_f32_x(pg32, vlhs_scale, vrscale1));
          vf2 = svmla_f32_m(pg32, vrbias2, vf2,
                            svmul_f32_x(pg32, vlhs_scale, vrscale2));
          vf3 = svmla_f32_m(pg32, vrbias3, vf3,
                            svmul_f32_x(pg32, vlhs_scale, vrscale3));

          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
          vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
          vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

          float* dst_ptr =
              (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
          svst1_vnum_f32(pg32, dst_ptr, 0, vf0);
          svst1_vnum_f32(pg32, dst_ptr, 1, vf1);
          svst1_vnum_f32(p_out2, dst_ptr, 2, vf2);
          svst1_vnum_f32(p_out3, dst_ptr, 3, vf3);
        }
        n_idx += nr;
        continue;
      }

      const uint8_t* rhs_block =
          (const uint8_t*)w + (n_idx / nr) * rhs_block_stride;
      svzero_za();

      const int8_t* a_ptr = (const int8_t*)lhs_block;
      const int8_t* w_ptr = (const int8_t*)rhs_block;
      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 2 * u + 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 2 * u + 1);
          const svint8_t vraw = svld1_vnum_s8(pg8, w_ptr, u);
          const svint8_t vlo = svlsl_n_s8_x(pg8, vraw, 4);
          const svint8_t vhi = svand_n_s8_x(pg8, vraw, (int8_t)0xF0);
          const svint8_t vb0 = svzip1_s8(vlo, vhi);
          const svint8_t vb1 = svzip2_s8(vlo, vhi);

          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
          svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
          svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
        }
        a_ptr += 16 * mr_packed;
        w_ptr += 8 * nr;
      }

      const int32_t* rhs_sums =
          (const int32_t*)(rhs_block + nr * (k_internal / 2));
      const float* rhs_scales = (const float*)(rhs_sums + nr);
      const float* rhs_biases = rhs_scales + nr;

      const svint32_t vrsum0 = svld1_vnum_s32(pg32, rhs_sums, 0);
      const svint32_t vrsum1 = svld1_vnum_s32(pg32, rhs_sums, 1);
      const svfloat32_t vrscale0 = svld1_vnum_f32(pg32, rhs_scales, 0);
      const svfloat32_t vrscale1 = svld1_vnum_f32(pg32, rhs_scales, 1);
      const svfloat32_t vrbias0 = svld1_vnum_f32(pg32, rhs_biases, 0);
      const svfloat32_t vrbias1 = svld1_vnum_f32(pg32, rhs_biases, 1);

      const size_t n_rem = nc - n_idx;
      const svbool_t p_out0 = svwhilelt_b32_u64(0 * vl, n_rem);
      const svbool_t p_out1 = svwhilelt_b32_u64(1 * vl, n_rem);

      const size_t m0_limit = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < m0_limit; ++row) {
        const svint32_t vlhs_zp = svdup_n_s32(lhs_neg_zps[row]);
        const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scales[row]);

        svint32_t vacc0 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), 4);
        svint32_t vacc1 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), 4);

        vacc0 = svmla_s32_m(pg32, vacc0, vrsum0, vlhs_zp);
        vacc1 = svmla_s32_m(pg32, vacc1, vrsum1, vlhs_zp);

        svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
        svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);

        vf0 = svmla_f32_m(pg32, vrbias0, vf0,
                          svmul_f32_x(pg32, vlhs_scale, vrscale0));
        vf1 = svmla_f32_m(pg32, vrbias1, vf1,
                          svmul_f32_x(pg32, vlhs_scale, vrscale1));

        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

        float* dst_ptr =
            (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
      }

      for (size_t row = vl; row < m_rem; ++row) {
        const svint32_t vlhs_zp = svdup_n_s32(lhs_neg_zps[row]);
        const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scales[row]);

        svint32_t vacc0 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row - vl), 4);
        svint32_t vacc1 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row - vl), 4);

        vacc0 = svmla_s32_m(pg32, vacc0, vrsum0, vlhs_zp);
        vacc1 = svmla_s32_m(pg32, vacc1, vrsum1, vlhs_zp);

        svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
        svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);

        vf0 = svmla_f32_m(pg32, vrbias0, vf0,
                          svmul_f32_x(pg32, vlhs_scale, vrscale0));
        vf1 = svmla_f32_m(pg32, vrbias1, vf1,
                          svmul_f32_x(pg32, vlhs_scale, vrscale1));

        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);

        float* dst_ptr =
            (float*)((uintptr_t)c + (m_idx + row) * cm_stride) + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
      }
    }
  }
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x32c4__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_32x32c4__sme2_get_nr() {
  return 2 * svcntsw();
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
