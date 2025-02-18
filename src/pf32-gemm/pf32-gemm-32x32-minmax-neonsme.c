// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include "xnnpack/microparams.h"
#include "pf32-gemm/gen/pf32-gemm-minmax-utils.h"

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_mr() {
    return xnn_pf32_gemm_minmax_mr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme_get_nr() {
    return xnn_pf32_gemm_minmax_nr * xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme();
}

// Wraps the `xnn_pf32_gemm_minmax__asm_aarch64_neonsme`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    union xnn_f32_minmax_params
        minmax_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]) {
    
    xnn_pf32_gemm_minmax__asm_aarch64_neonsme(lhs_packed, rhs_packed, dst, (k/sizeof(float)), &minmax_params->scalar.max,
                    &minmax_params->scalar.min, m, n, NULL, 0, dst_stride_row);    
                
}
