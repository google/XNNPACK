// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/pack-lh.h"

#if XNN_ENABLE_KLEIDIAI
  // Keep this line indented to avoid it being pulled out of the #ifdef when the
  // sources are amalgamated.
  #include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI


// This function just wraps KleidiAI's `kai_run_lhs_pack_f32p2vlx1_f32_sme`, but
// with a name that is recognized by our tooling.

void xnn_x32_pack_lh_ukernel__neonsme2_u2(size_t m, size_t k, size_t mr,
                                          size_t kr, size_t sr,
                                          size_t m_idx_start,
                                          const float* XNN_RESTRICT lhs,
                                          size_t lhs_stride,
                                          void* XNN_RESTRICT lhs_packed) {
  printf("PACL LHS\n");fflush(stdout);
#if XNN_ENABLE_KLEIDIAI
  printf("m %zu k %zu mr %zu\n", m, k, mr);fflush(stdout);
  kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr, m_idx_start, lhs,
                                     lhs_stride, lhs_packed);
  //printf("LHS DATA size %zu\n", kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m,k,mr,kr,sr));
  //fflush(stdout);
  //for (int i = 0; i < m; ++i) {
  //  for (int j = 0; j < k; ++j) {
  //    printf("%f ", lhs[j + i * k]);
  //  }
  //  printf("\n");
  //}
  //float* lhs_packed_f = (float*)lhs_packed;
  //printf("PACKED LHS DATA\n");
  //for (int i = 0; i < mr; ++i) {
  //  for (int j = 0; j < k; ++j) {
  //    printf("%f ", lhs_packed_f[j+i*k]);
  //  }
  //  printf("\n");
  //}
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_x32_pack_lh_size__neonsme2_u2(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
