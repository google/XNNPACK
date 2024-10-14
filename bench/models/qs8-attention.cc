// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t QS8Attention(size_t b, size_t t, size_t h, size_t n, size_t s) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/4, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));

  // External inputs and outputs
  uint32_t value_proj = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> value_proj_dims = {{b, s, n, t}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, value_proj_dims.size(), value_proj_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &value_proj);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor value_proj" << std::endl;
    return nullptr;
  }

  uint32_t query_proj = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> query_proj_dims = {{b, t, n, h}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, query_proj_dims.size(), query_proj_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &query_proj);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query_proj" << std::endl;
    return nullptr;
  }

  uint32_t key_proj = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> key_proj_dims = {{b, s, n, h}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, key_proj_dims.size(), key_proj_dims.data(),
      /*data=*/nullptr, 2, XNN_VALUE_FLAG_EXTERNAL_INPUT, &key_proj);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key_proj" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> output_dims = {{b, t, n, t}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, output_dims.size(), output_dims.data(),
      /*data=*/nullptr, 3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)>
      scale_data;
  uint32_t scale = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> scale_dims = {{1}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, scale_dims.size(), scale_dims.data(),
      /*data=*/scale_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &scale);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor scale" << std::endl;
    return nullptr;
  }
  std::generate(scale_data.begin(), scale_data.end(), std::ref(f32rng));

  uint32_t query_after_scale = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> query_after_scale_dims = {{b, t, n, h}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, query_after_scale_dims.size(),
      query_after_scale_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &query_after_scale);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query_after_scale" << std::endl;
    return nullptr;
  }

  xnn_binary_params binary_inf_params = {
      -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
  };
  status = xnn_define_binary(subgraph, xnn_binary_multiply, &binary_inf_params,
                             query_proj, scale, query_after_scale, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  static const std::array<size_t, 4> transpose_0213_data = {0, 2, 1, 3};

  uint32_t query_permuted = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> query_permuted_dims = {{b, n, t, h}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, query_permuted_dims.size(),
      query_permuted_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &query_permuted);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query_permuted" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/transpose_0213_data.size(),
                                       /*perm=*/transpose_0213_data.data(),
                                       query_after_scale, query_permuted, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  uint32_t key_permuted = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> key_permuted_dims = {{b, n, s, h}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, key_permuted_dims.size(), key_permuted_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &key_permuted);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key_permuted" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/transpose_0213_data.size(),
                                       /*perm=*/transpose_0213_data.data(),
                                       key_proj, key_permuted, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  // BatchMatMul only accepts fp16, fp32, or a very restrictive set of
  // quantized inputs (qd8 x qc8 -> fp16). Furthermore, you can convert
  // from fp16 -> qd8 but not fp16 -> qs8. For now, we convert
  // everything to fp32 to avoid this. TODO: figure out how to improve this.
  uint32_t query_permuted_fp32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> query_permuted_fp32_dims = {{b, n, t, h}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   query_permuted_fp32_dims.size(),
                                   query_permuted_fp32_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &query_permuted_fp32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor query_permuted_fp32" << std::endl;
    return nullptr;
  }

  uint32_t key_permuted_fp32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> key_permuted_fp32_dims = {{b, n, s, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, key_permuted_fp32_dims.size(),
      key_permuted_fp32_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &key_permuted_fp32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor key_permuted_fp32" << std::endl;
    return nullptr;
  }

  uint32_t bmm_result_fp32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> bmm_result_fp32_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, bmm_result_fp32_dims.size(),
      bmm_result_fp32_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &bmm_result_fp32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor bmm_result_fp32" << std::endl;
    return nullptr;
  }

  status = xnn_define_convert(subgraph, query_permuted, query_permuted_fp32, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node query_permuted_fp32" << std::endl;
    return nullptr;
  }

  status = xnn_define_convert(subgraph, key_permuted, key_permuted_fp32, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node key_permuted_fp32" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, query_permuted_fp32,
                                            key_permuted_fp32, bmm_result_fp32,
                                            XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node bmm_result_fp32" << std::endl;
    return nullptr;
  }

  uint32_t bmm_result = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> bmm_result_dims = {{b, n, t, s}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, bmm_result_dims.size(), bmm_result_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &bmm_result);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor bmm_result" << std::endl;
    return nullptr;
  }

  status = xnn_define_convert(subgraph, bmm_result_fp32, bmm_result, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node bmm_result" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1, float)> w15_data;
  uint32_t w15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w15_dims = {{1}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, w15_dims.size(), w15_dims.data(),
      /*data=*/w15_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w15" << std::endl;
    return nullptr;
  }
  std::generate(w15_data.begin(), w15_data.end(), std::ref(f32rng));

  uint32_t bmm_scaled = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> bmm_scaled_dims = {{b, n, t, s}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, bmm_scaled_dims.size(), bmm_scaled_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &bmm_scaled);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor bmm_scaled" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph, xnn_binary_multiply, &binary_inf_params,
                             bmm_result, w15, bmm_scaled,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  uint32_t logits = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> logits_dims = {{b, n, t, s}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, logits_dims.size(), logits_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &logits);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor logits" << std::endl;
    return nullptr;
  }

  // Tanh: fp16, fp32, qint8
  status = xnn_define_tanh(subgraph, bmm_scaled, logits, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  // Softmax only accepts fp16 or fp32, so convert logits to fp16 for that
  uint32_t logits_fp16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> logits_fp16_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, logits_fp16_dims.size(),
      logits_fp16_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &logits_fp16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor logits_fp16_dims" << std::endl;
    return nullptr;
  }
  status = xnn_define_convert(subgraph, logits, logits_fp16, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node logits_fp16_dims" << std::endl;
    return nullptr;
  }

  uint32_t probs = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> probs_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, probs_dims.size(), probs_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &probs);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor probs" << std::endl;
    return nullptr;
  }

  status = xnn_define_softmax(subgraph, logits_fp16, probs, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  uint32_t value_permuted = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> value_permuted_dims = {{b, n, t, s}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, value_permuted_dims.size(),
      value_permuted_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &value_permuted);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor value_permuted" << std::endl;
    return nullptr;
  }

  static const std::array<size_t, 4> transpose_0231_data = {0, 2, 3, 1};
  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/transpose_0231_data.size(),
                                       /*perm=*/transpose_0231_data.data(),
                                       value_proj, value_permuted, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  uint32_t value_permutedfp32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> value_permutedfp32_dims = {{b, n, s, h}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, value_permutedfp32_dims.size(),
      value_permutedfp32_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &value_permutedfp32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor value_permutedfp32" << std::endl;
    return nullptr;
  }

  uint32_t outcome_before_permutefp32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> outcome_before_permutefp32_dims = {{b, n, t, s}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   outcome_before_permutefp32_dims.size(),
                                   outcome_before_permutefp32_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &outcome_before_permutefp32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor outcome_before_permutefp32"
              << std::endl;
    return nullptr;
  }

  status = xnn_define_convert(subgraph, value_permuted, value_permutedfp32, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node value_permutedfp32" << std::endl;
    return nullptr;
  }

  status = xnn_define_batch_matrix_multiply(subgraph, probs, value_permutedfp32,
                                            outcome_before_permutefp32,
                                            XNN_FLAG_TRANSPOSE_B);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node outcome_before_permutefp32"
              << std::endl;
    return nullptr;
  }

  uint32_t outcome_before_permute = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> outcome_before_permute_dims = {{b, n, t, t}};
  status = xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8,
      /*zero_point=*/0,
      /*scale=*/0.0078125f, outcome_before_permute_dims.size(),
      outcome_before_permute_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0,
      &outcome_before_permute);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor outcome_before_permute" << std::endl;
    return nullptr;
  }

  status = xnn_define_convert(subgraph, outcome_before_permutefp32,
                              outcome_before_permute, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node outcome_before_permute" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_transpose(subgraph,
                                       /*num_dims=*/transpose_0213_data.size(),
                                       /*perm=*/transpose_0213_data.data(),
                                       outcome_before_permute, output, 0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
