// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
    size_t batch,                                    \
    size_t samples,                                  \
    int16_t* data,                                   \
    const int16_t* twiddle,                          \
    size_t stride);

DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x1)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x2)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples1_ukernel__neon)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples1_ukernel__scalar)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples4_ukernel__neon)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_samples4_ukernel__scalar)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__neon_x1)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__neon_x4)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x1)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x2)
DECLARE_CS16_BFLY4_UKERNEL_FUNCTION(xnn_cs16_bfly4_ukernel__scalar_x4)

#define DECLARE_CS16_FFTR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
    size_t samples,                                 \
    int16_t* data,                                  \
    const int16_t* twiddle);

DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__asm_aarch32_neon_x1)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__asm_aarch32_neon_x4)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__neon_x4)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x1)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x2)
DECLARE_CS16_FFTR_UKERNEL_FUNCTION(xnn_cs16_fftr_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif
