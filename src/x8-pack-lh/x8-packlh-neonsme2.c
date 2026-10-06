// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI

void xnn_x8_pack_lh_ukernel__neonsme2(
    size_t m, size_t k, size_t mr_packed, size_t kr, size_t sr,
    size_t m_idx_start, const int8_t* XNN_RESTRICT lhs, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_pack_lhs_uker_config config = {0};
  const struct kai_matmul_pack_lhs_uker_api api =
      kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
  if (mr_packed == 1) {
    for (size_t i = 0; i < m; ++i) {
      memcpy((int8_t*)lhs_packed + i * k, (const int8_t*)lhs + i * lhs_stride,
             sizeof(int8_t) * k);
    }
  } else {
    assert(mr_packed % api.get_step(&config).m == 0);
    assert(kr == 4);
    assert(sr == 1);
    // Callers offset both pointers before passing a tile-local start of zero.
    assert(m_idx_start == 0);
    (void)kr;
    (void)sr;
    (void)m_idx_start;
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args shape = {m, k};
    struct kai_matmul_pack_lhs_uker_args args = {0};
    args.shape.m = m;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride.m = lhs_stride;
    args.operand.lhs_packed.ptr = lhs_packed;
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &shape);
    api.run(&config, &args);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__neonsme2(size_t m, size_t k, size_t mr_packed,
                                     size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  (void)kr;
  (void)sr;
  const struct kai_matmul_pack_lhs_uker_config config = {0};
  const struct kai_matmul_pack_lhs_uker_api api =
      kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    assert(mr_packed % api.get_step(&config).m == 0);
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

size_t xnn_x8_pack_lh_offset__neonsme2(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  (void)kr;
  (void)sr;
  const struct kai_matmul_pack_lhs_uker_config config = {0};
  const struct kai_matmul_pack_lhs_uker_api api =
      kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
      assert(mr_packed % api.get_step(&config).m == 0);
      assert(m % api.get_step(&config).m == 0);
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

// The QC4 SME2 kernel retains the older p2VL packed layout.
void xnn_x8_pack_lh_ukernel__neonsme2_p2(
    size_t m, size_t k, size_t mr_packed, size_t kr, size_t sr,
    size_t m_idx_start, const int8_t* XNN_RESTRICT lhs, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    for (size_t i = 0; i < m; ++i) {
      memcpy((int8_t*)lhs_packed + i * k, (const int8_t*)lhs + i * lhs_stride,
             sizeof(int8_t) * k);
    }
  } else {
    kai_run_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed, kr, sr, m_idx_start, lhs,
                                     lhs_stride, lhs_packed);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__neonsme2_p2(size_t m, size_t k,
                                        size_t mr_packed, size_t kr,
                                        size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed,
                                                            kr, sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__neonsme2_p2(size_t m, size_t k,
                                          size_t mr_packed, size_t kr,
                                          size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(
        m, k, mr_packed, kr, sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
