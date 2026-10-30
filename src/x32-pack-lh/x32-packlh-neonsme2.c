// Copyright 2025 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

static struct kai_matmul_pack_lhs_uker_api xnn_kleidiai_pf32_lhs_pack_api(
    void) {
  return kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
}

#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

__arm_locally_streaming __arm_new("za") static void xnn_x32_packlh_sme_1vl(
    size_t m, size_t k, const uint8_t* lhs_ptr, size_t lhs_stride,
    uint32_t* dst_base) {
  const size_t vl = svcntw();
  const svbool_t pg32 = svptrue_b32();

  for (size_t i_m = 0; i_m < m; i_m += vl) {
    const size_t valid_rows = m - i_m < vl ? m - i_m : vl;
    uint32_t* dst_blk = dst_base + i_m * k;
    size_t k_idx = 0;

    if (XNN_LIKELY(valid_rows == vl)) {
      for (; k_idx + 4 * vl <= k; k_idx += 4 * vl) {
        const uint8_t* rp =
            lhs_ptr + i_m * lhs_stride + k_idx * sizeof(uint32_t);
        for (size_t r = 0; r < vl; ++r) {
          const uint32_t* row = (const uint32_t*)rp;
          svld1_hor_za32(0, r, pg32, row + 0 * vl);
          svld1_hor_za32(1, r, pg32, row + 1 * vl);
          svld1_hor_za32(2, r, pg32, row + 2 * vl);
          svld1_hor_za32(3, r, pg32, row + 3 * vl);
          rp += lhs_stride;
        }
        uint32_t* out_p = dst_blk + k_idx * vl;
        for (size_t c = 0; c < vl; ++c) {
          svst1_ver_za32(0, c, pg32, out_p + (0 * vl + c) * vl);
          svst1_ver_za32(1, c, pg32, out_p + (1 * vl + c) * vl);
          svst1_ver_za32(2, c, pg32, out_p + (2 * vl + c) * vl);
          svst1_ver_za32(3, c, pg32, out_p + (3 * vl + c) * vl);
        }
      }
    } else {
      svzero_za();
    }

    for (; k_idx < k; k_idx += vl) {
      const size_t k_rem = k - k_idx < vl ? k - k_idx : vl;
      const svbool_t pg = svwhilelt_b32_u64(0, k_rem);
      const uint8_t* rp = lhs_ptr + i_m * lhs_stride + k_idx * sizeof(uint32_t);
      for (size_t r = 0; r < valid_rows; ++r) {
        svld1_hor_za32(0, r, pg, (const uint32_t*)rp);
        rp += lhs_stride;
      }
      uint32_t* out_ptr = dst_blk + k_idx * vl;
      for (size_t c = 0; c < k_rem; ++c) {
        svst1_ver_za32(0, c, pg32, out_ptr + c * vl);
      }
    }
  }
}
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's
// `kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme`, but with a name that is
// recognized by our tooling.
void xnn_x32_pack_lh_ukernel__neonsme2(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr, size_t m_idx_start,
                                       const float* XNN_RESTRICT lhs,
                                       size_t lhs_stride,
                                       void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(m_idx_start == 0);
  assert(kr == 1);
  assert(sr == 1);
  (void)m_idx_start;
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(float) * k);
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf32_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape =
        {m, k};
    struct kai_matmul_pack_lhs_uker_args args = {
        .shape = {m, k},
        .operand.lhs_packed.ptr = lhs_packed,
        .operand.lhs_packed.stride =
            api.get_lhs_packed_stride(&config, &lhs_packed_shape),
        .operand.lhs.ptr = lhs,
        .operand.lhs.stride.m = lhs_stride,
    };
    api.run(&config, &args);
  }
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
  assert(m_idx_start == 0);
  assert(kr == 1);
  assert(sr == 1);
  (void)m_idx_start;
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    float* out = (float*)lhs_packed;
    for (size_t row = 0; row < m; ++row) {
      memcpy(out + row * k, (const uint8_t*)lhs + row * lhs_stride,
             k * sizeof(float));
    }
    return;
  }
  xnn_x32_packlh_sme_1vl(m, k, (const uint8_t*)lhs, lhs_stride,
                         (uint32_t*)lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x32_pack_lh_size__neonsme2(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(float) * k;
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf32_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {
        .m = m,
        .k = k,
    };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride =
        api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_size(&config, &shape, &stride);
  }
#elif XNN_ENABLE_ARM_SME_ACLE
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(float) * k;
  }
  const size_t m_step = svcntsw();
  return round_up(m, m_step) * k * sizeof(float);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x32_pack_lh_offset__neonsme2(size_t m, size_t k, size_t mr_packed,
                                        size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(float) * k;
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf32_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {
        .m = mr_packed,
        .k = k,
    };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args index = {
        .m = m,
        .k = 0,
    };
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride =
        api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_offset(&config, &index, &stride);
  }
#elif XNN_ENABLE_ARM_SME_ACLE
  (void)mr_packed;
  (void)kr;
  (void)sr;
  return m * k * sizeof(float);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
