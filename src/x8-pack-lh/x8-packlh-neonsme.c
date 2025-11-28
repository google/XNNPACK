// Copyright 2025 Google LLC
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
#include "src/xnnpack/config.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI


// This function just wraps KleidiAI's `kai_run_lhs_pack_x8p2vlx4_x8_sme`, but
// with a name that is recognized by our tooling.
void xnn_x8_pack_lh_ukernel__neonsme(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr, size_t m_idx_start,
                                      const int8_t* XNN_RESTRICT lhs,
                                      size_t lhs_stride,
                                      void* XNN_RESTRICT lhs_packed) {
 
#if XNN_ENABLE_KLEIDIAI
  const struct xnn_gemm_config* gemm_config = xnn_init_pqs8_qc8w_gemm_config();
  mr_packed = gemm_config->mr_packed;
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(int8_t) * k);
  } else {
    kai_run_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed, kr, sr, m_idx_start, lhs,
                                     lhs_stride, lhs_packed);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__neonsme(size_t m, size_t k, size_t mr_packed,
                                     size_t kr, size_t sr) {

#if XNN_ENABLE_KLEIDIAI
  const struct xnn_gemm_config* gemm_config = xnn_init_pqs8_qc8w_gemm_config();
  mr_packed = gemm_config->mr_packed;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed, kr,
                                                            sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__neonsme(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr) {

#if XNN_ENABLE_KLEIDIAI
  const struct xnn_gemm_config* gemm_config = xnn_init_pqs8_qc8w_gemm_config();
  mr_packed = gemm_config->mr_packed;
  if (mr_packed == 1) {
    return m * sizeof(int8_t) * k;
  } else {
    return kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(m, k, mr_packed,
                                                              kr, sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
