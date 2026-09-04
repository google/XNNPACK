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

void xnn_kai_f32_dwconv_minmax_ukernel_9pvc__neonsme2(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max) {
  kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
      input, packed_weights, output, input_height_stride, input_pixel_stride,
      output_height_stride, output_pixel_stride, valid_input_rows,
      valid_output_rows, input_padding_left, input_padding_top, padding_value,
      output_min, output_max);
}
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
