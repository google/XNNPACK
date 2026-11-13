// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x64c4__hmx_get_mr(void) {
  return 32;
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x64c4__hmx_get_nr(void) {
  return 64;
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x64c4__hmx(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* w, int8_t* c, size_t cm_stride, const void* params) {
  assert(mr != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(packed_lhs != NULL);
  assert(w != NULL);
  assert(c != NULL);
  assert(params != NULL);

  const size_t k = ks * round_up_po2(kc, 4);
  xnn_pqs8_qc8w_gemm_minmax_32x64c4_hmx(
      mr, nc, k, packed_lhs, w, c, cm_stride, sizeof(int8_t),
      (const union xnn_qs8_qc8w_conv_minmax_params*) params);
}
