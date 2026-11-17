// Copyright 2024 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

static struct kai_matmul_pack_lhs_uker_api
xnn_kleidiai_pf16_lhs_pack_api(void) {
  return kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme();
}
#endif  // XNN_ENABLE_KLEIDIAI

// Wraps KleidiAI's `kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme` with a name
// recognized by our tooling.
void xnn_x16_pack_lh_ukernel__neonsme2(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr, size_t m_idx_start,
                                       const xnn_float16* XNN_RESTRICT lhs,
                                       size_t lhs_stride,
                                       void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(m_idx_start == 0);
  assert(kr == 2);
  assert(sr == 1);
  (void)m_idx_start;
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(xnn_float16) * k);
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf16_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args packed_shape = {
        m, k};
    struct kai_matmul_pack_lhs_uker_args args = {
        .shape = {m, k},
        .operand.lhs_packed.ptr = lhs_packed,
        .operand.lhs_packed.stride =
            api.get_lhs_packed_stride(&config, &packed_shape),
        .operand.lhs.ptr = lhs,
        .operand.lhs.stride.m = lhs_stride,
    };
    api.run(&config, &args);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_size__neonsme2(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 2);
  assert(sr == 1);
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf16_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {m, k};
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride =
        api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_size(&config, &shape, &stride);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_offset__neonsme2(size_t m, size_t k, size_t mr_packed,
                                        size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 2);
  assert(sr == 1);
  (void)kr;
  (void)sr;
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    const struct kai_matmul_pack_lhs_uker_config config = {0};
    const struct kai_matmul_pack_lhs_uker_api api =
        xnn_kleidiai_pf16_lhs_pack_api();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {
        mr_packed, k};
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args index = {m, 0};
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride =
        api.get_lhs_packed_stride(&config, &shape);
    return api.get_lhs_packed_offset(&config, &index, &stride);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
