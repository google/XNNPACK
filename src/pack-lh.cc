// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/pack-lh.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/quantization.h"

// Wraps a templated function that generates `xnn_qd8_quantization_params` from
// parameters of the templated type.
template <typename T>
struct InitQuantizationParams {
  typedef struct xnn_qd8_quantization_params (*fn)(T min, T max, T* scale);
};

template <typename InputT, typename OutputT, typename qs8_cvt_params_t,
          typename InitQuantizationParams<InputT>::fn init_quantization_params>
static void pack_lh_fx_qd(size_t m, size_t k, size_t mr_packed, size_t kr,
                          size_t sr, size_t m_idx_start, const InputT* lhs,
                          size_t lhs_stride, void* lhs_packed,
                          xnn_vunary_ukernel_fn convert_ukernel,
                          xnn_reduce_ukernel_fn minmax_ukernel) {
  assert(m_idx_start == 0);

  struct xnn_f32_default_params minmax_params;
  qs8_cvt_params_t convert_params;

  const size_t k_scaled = k * sizeof(InputT);
  const uintptr_t packed_row_stride = round_up(k, kr * sr) * sizeof(OutputT);

  while (m) {
    // Pointers to the input and output data for this set of `mr` rows.
    struct xnn_qd8_quantization_params* quantization_params =
        (struct xnn_qd8_quantization_params*)lhs_packed;
    OutputT* packed_weights =
        (OutputT*)((uintptr_t)lhs_packed +
                   mr_packed * sizeof(struct xnn_qd8_quantization_params));

    // For each row in this block of `mr` rows...
    for (size_t row_id = 0; row_id < min(mr_packed, m); row_id++) {
      // Compute the quantization params for this row.
      InputT minmax[2] = {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()};
      InputT scale;
      minmax_ukernel(k_scaled, lhs, minmax, &minmax_params);
      quantization_params[row_id] =
          init_quantization_params(minmax[0], minmax[1], &scale);

      // Quantize the row.
      convert_params.scalar.scale = scale;
      convert_params.scalar.output_zero_point =
          quantization_params[row_id].zero_point;
      convert_ukernel(k_scaled, lhs, packed_weights,
                      (union xnn_unary_uparams*)&convert_params);

      // Advance the input and output pointers.
      lhs = (const InputT*)((uintptr_t)lhs + lhs_stride);
      packed_weights =
          (OutputT*)((uintptr_t)packed_weights + packed_row_stride);
    }

    // Copy any extra quantization params if needed.
    for (size_t row_id = m; row_id < mr_packed; row_id++) {
      quantization_params[row_id] = quantization_params[m - 1];
    }

    // Advance the pointers and counters.
    lhs_packed = (void*)((uintptr_t)lhs_packed +
                         xnn_pack_lh_fx_qd8_packed_size(
                             /*m=*/mr_packed, k, mr_packed, kr, sr));
    m -= min(mr_packed, m);
  }
}

extern "C" {

size_t xnn_pack_lh_fx_qd8_packed_size(size_t m, size_t k, size_t mr_packed,
                                      size_t kr, size_t sr) {
  // Each packed row starts with the `mr` quantization params, followed by the
  // `mr` rows of quantized data.
  m = round_up(m, mr_packed);
  k = round_up(k, kr * sr);
  return m * sizeof(struct xnn_qd8_quantization_params) +
         m * k * sizeof(int8_t);
}

size_t xnn_pack_lh_fx_qd8_packed_offset(size_t m, size_t k, size_t mr_packed,
                                        size_t kr, size_t sr) {
  // Each packed row starts with the `mr` quantization params, followed by the
  // `mr` rows of quantized data.
  m = round_up(m, mr_packed);
  k = round_up(k, kr * sr);
  return m * sizeof(struct xnn_qd8_quantization_params) +
         m * k * sizeof(int8_t);
}

void xnn_pack_lh_f32_qdint8(size_t m, size_t k, size_t mr_packed, size_t kr,
                            size_t sr, size_t m_idx_start, const void* lhs,
                            size_t lhs_stride, void* lhs_packed) {
  static const xnn_vunary_ukernel_fn convert_ukernel =
      xnn_init_f32_to_qs8_cvt_config()->ukernel;
  static const xnn_reduce_ukernel_fn minmax_ukernel =
      xnn_init_f32_rminmax_config()->ukernel;
  pack_lh_fx_qd</*InputT=*/float, /*OutputT=*/int8_t,
                /*qs8_cvt_params_t=*/struct xnn_f32_qs8_cvt_params,
                xnn_f32_qd8_asymmetric_quantization_params>(
      m, k, mr_packed, kr, sr, m_idx_start, (const float*)lhs, lhs_stride,
      lhs_packed, convert_ukernel, minmax_ukernel);
}

void xnn_pack_lh_f32_qduint8(size_t m, size_t k, size_t mr_packed, size_t kr,
                             size_t sr, size_t m_idx_start, const void* lhs,
                             size_t lhs_stride, void* lhs_packed) {
  static const xnn_vunary_ukernel_fn convert_ukernel =
      xnn_init_f32_to_qu8_cvt_config()->ukernel;
  static const xnn_reduce_ukernel_fn minmax_ukernel =
      xnn_init_f32_rminmax_config()->ukernel;
  pack_lh_fx_qd</*InputT=*/float, /*OutputT=*/uint8_t,
                /*qs8_cvt_params_t=*/struct xnn_f32_qs8_cvt_params,
                xnn_f32_qdu8_asymmetric_quantization_params>(
      m, k, mr_packed, kr, sr, m_idx_start, (const float*)lhs, lhs_stride,
      lhs_packed, convert_ukernel, minmax_ukernel);
}

void xnn_pack_lh_f16_qdint8(size_t m, size_t k, size_t mr_packed, size_t kr,
                            size_t sr, size_t m_idx_start, const void* lhs,
                            size_t lhs_stride, void* lhs_packed) {
  static const xnn_vunary_ukernel_fn convert_ukernel =
      xnn_init_f16_to_qs8_cvt_config()->ukernel;
  static const xnn_reduce_ukernel_fn minmax_ukernel =
      xnn_init_f16_rminmax_config()->ukernel;
  pack_lh_fx_qd</*InputT=*/xnn_float16, /*OutputT=*/int8_t,
                /*qs8_cvt_params_t=*/struct xnn_f16_qs8_cvt_params,
                xnn_f16_qd8_asymmetric_quantization_params>(
      m, k, mr_packed, kr, sr, m_idx_start, (const xnn_float16*)lhs, lhs_stride,
      lhs_packed, convert_ukernel, minmax_ukernel);
}

void xnn_pack_lh_f16_qduint8(size_t m, size_t k, size_t mr_packed, size_t kr,
                             size_t sr, size_t m_idx_start, const void* lhs,
                             size_t lhs_stride, void* lhs_packed) {
  static const xnn_vunary_ukernel_fn convert_ukernel =
      xnn_init_f16_to_qu8_cvt_config()->ukernel;
  static const xnn_reduce_ukernel_fn minmax_ukernel =
      xnn_init_f16_rminmax_config()->ukernel;
  pack_lh_fx_qd</*InputT=*/xnn_float16, /*OutputT=*/uint8_t,
                /*qs8_cvt_params_t=*/struct xnn_f16_qs8_cvt_params,
                xnn_f16_qdu8_asymmetric_quantization_params>(
      m, k, mr_packed, kr, sr, m_idx_start, (const xnn_float16*)lhs, lhs_stride,
      lhs_packed, convert_ukernel, minmax_ukernel);
}

}  // extern "C"
