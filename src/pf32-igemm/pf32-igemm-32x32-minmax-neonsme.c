// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#endif // XNN_ENABLE_KLEIDIAI

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_mr(void)
{
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

size_t xnn_pf32_igemm_minmax_ukernel_32x32__neonsme_get_nr(void)
{
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
}

void xnn_pf32_igemm_minmax_ukernel_32x32__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    const struct xnn_f32_minmax_params* minmax_params) {

  const size_t kai_kr = 1;
  const size_t k = ks * round_up(kc/sizeof(float), kai_kr);

#if XNN_ENABLE_KLEIDIAI
  kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
      mr, nc, k , lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
