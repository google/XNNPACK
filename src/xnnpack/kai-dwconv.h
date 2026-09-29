// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_KAI_DWCONV_H_
#define XNNPACK_SRC_XNNPACK_KAI_DWCONV_H_

#include <stddef.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL size_t
xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_channel_tile(void);
XNN_INTERNAL size_t
xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_output_height_tile(void);
XNN_INTERNAL void xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_KAI_DWCONV_H_
