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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_KAI_DWCONV_H_
