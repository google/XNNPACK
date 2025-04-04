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
  // Keep this line indented to avoid it being pulled out of the #ifdef when the
  // sources are amalgamated.
  #include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
  #include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI


// This function just wraps KleidiAI's `kai_run_lhs_pack_x16p2vlx2_x16_sme`, but
// with a name that is recognized by our tooling.
void xnn_x16_pack_lh_ukernel__neonsme2(size_t m, size_t k, size_t mr_packed,
                                       size_t kr, size_t sr, size_t m_idx_start,
                                       const xnn_float16* XNN_RESTRICT lhs,
                                       size_t lhs_stride,
                                       void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    memcpy(lhs_packed, lhs, sizeof(xnn_float16) * k);
  } else {
    kai_run_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed, kr, sr, m_idx_start,
                                       lhs, lhs_stride, lhs_packed);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_size__neonsme2(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    return kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed,
                                                              kr, sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x16_pack_lh_offset__neonsme2(size_t m, size_t k, size_t mr_packed,
                                        size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  if (mr_packed == 1) {
    return m * sizeof(xnn_float16) * k;
  } else {
    return kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme(m, k, mr_packed,
                                                                kr, sr);
  }
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
