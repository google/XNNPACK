// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"

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

void xnn_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    size_t channels, size_t output_width, const float** input,
    const float* weights, float* output, intptr_t input_stride,
    size_t output_increment, size_t input_offset, size_t input_pixel_stride,
    const float* zero, const struct xnn_f32_minmax_params* params) {
  // This symbol identifies the KAI configuration in XNNPACK's DWConv plumbing.
  // The planar KAI microkernel is invoked by xnn_compute_kai_f32_dwconv.
  (void)channels;
  (void)output_width;
  (void)input;
  (void)weights;
  (void)output;
  (void)input_stride;
  (void)output_increment;
  (void)input_offset;
  (void)input_pixel_stride;
  (void)zero;
  (void)params;
  assert("KleidiAI F32 DWConv must use the direct planar compute path." && 0);
  XNN_UNREACHABLE;
}
