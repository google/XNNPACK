// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/kai-dwconv.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_channel_tile(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_sme_vector_length_u32();
#else
  return 1;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2_get_output_height_tile(
    void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
#else
  return 1;
#endif  // XNN_ENABLE_KLEIDIAI
}
