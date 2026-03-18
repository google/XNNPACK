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

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI

// This function just wraps KleidiAI's `kai_run_lhs_pack_x8p2vlx4_x8_sme`, but
// with a name that is recognized by our tooling.
void xnn_x8_pack_lh_ukernel__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                            size_t mr_packed, size_t kr,
                                            size_t sr, const void** restrict a,
                                            size_t a_offset, const void* zero,
                                            void* lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(m, ks, kc, a,
                                            a_offset, zero, lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_size__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                           size_t mr_packed, size_t kr,
                                           size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
      m, ks, kc);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x8_pack_lh_offset__igemm_neonsme2(size_t m, size_t kc, size_t ks,
                                             size_t mr_packed, size_t kr,
                                             size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  assert(kr == 4);
  return kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(
      m, ks, kc);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
