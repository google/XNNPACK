// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f16_neon.h"
#endif  // XNN_ENABLE_KLEIDIAI

// Wraps KleidiAI's FP16 `qai8dxp` LHS quantize-and-pack microkernel.
void xnn_x8_packq_f16qp8_ukernel__aarch64_neon_u2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const void* XNN_RESTRICT lhs, size_t lhs_stride,
    void* XNN_RESTRICT lhs_packed) {
#if XNN_ENABLE_KLEIDIAI
  kai_run_lhs_quant_pack_qai8dxp_f16_neon(m, k, mr, kr, sr, m_idx_start, lhs,
                                          lhs_stride, lhs_packed);
#else
  assert("Not compiled with XNN_ENABLE_KLEIDIAI" && 0);
#endif  // XNN_ENABLE_KLEIDIAI
}
