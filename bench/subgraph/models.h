// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_BENCH_MODELS_MODELS_H_
#define THIRD_PARTY_XNNPACK_BENCH_MODELS_MODELS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "include/xnnpack.h"

namespace models {

struct QD8AttentionWeights {
  std::vector<int8_t> query_data;
  std::vector<float> query_scale;
  std::vector<int8_t> key_data;
  std::vector<float> key_scale;
  std::vector<int8_t> value_data;
  std::vector<float> value_scale;
  std::vector<int8_t> post_proj_data;
  std::vector<float> post_proj_scale;
};

xnn_subgraph_t FP32Attention(size_t b, size_t t, size_t h, size_t n, size_t s);
xnn_subgraph_t FP32MobileNetV1();
xnn_subgraph_t FP32MobileNetV2();
xnn_subgraph_t FP32MobileNetV3Large();
xnn_subgraph_t FP32MobileNetV3Small();
xnn_subgraph_t QD8Attention(size_t batch_size, size_t seq_len,
                            size_t embedding_dim, size_t num_heads,
                            size_t head_dim, QD8AttentionWeights& weights);
xnn_subgraph_t QS8MobileNetV2();

// This is a sequence of {add, multiply} x `depth` ops, on `batch_size` x
// `num_elements` values.
xnn_subgraph_t FP32Elementwise(size_t batch_size, size_t num_elements,
                               size_t depth);

// Compute the layer norm of [m x n x k] tensors, where the mean and variance
// are computed over the dimensions in `norm_mask`. This computation is
// equivalent to: input_mean = mean(input, norm_mask) (input - input_mean) /
// sqrt(mean(squared_difference(input, input_mean), norm_mask) + epsilon) *
// weight + bias
//
// Where `mean(x, norm_mask)` means computing the mean of the dimensions in the
// `norm_mask`.
xnn_subgraph_t FP32LayerNorm(size_t m, size_t n, size_t k, uint32_t norm_mask);

// Similar to the above, but computes the L2 norm.
xnn_subgraph_t FP32L2Norm(size_t m, size_t n, size_t k, uint32_t norm_mask);

// Similar to the above, but computes the softmax instead of the layer norm.
// If `use_softmax` is `false`, a "decomposed" subgraph which computes the
// softmax using unary and binary elementwise ops and reduction ops, is used,
// whereas if `use_softmax` is `true`, a single `softmax` subgraph node is
// created.
xnn_subgraph_t FP32Softmax(size_t m, size_t n, size_t k, uint32_t norm_mask,
                           bool use_softmax);

struct FP32DepthwiseSeparableWeights {
  std::vector<float> w0;
  std::vector<float> w1;
  std::vector<float> w2;
  std::vector<float> w3;
};

// Depthwise convolution with kernel size kw x kw, followed by a 1x1 conv with
// ci -> co channels. This is a common pattern in imaging models.
xnn_subgraph_t FP32DepthwiseSeparable(size_t w, size_t h, size_t kw, size_t ci,
                                      size_t co,
                                      FP32DepthwiseSeparableWeights& weights);

// Creates a single Gemma3-like Transformer block.
xnn_subgraph_t QD8TransformerBlock(size_t batch_size, size_t sequence_length,
                                   size_t embedding_dim, size_t num_heads,
                                   size_t head_dim, size_t hidden_dim);
xnn_subgraph_t FP32TransformerBlock(size_t batch_size, size_t sequence_length,
                                    size_t embedding_dim, size_t num_heads,
                                    size_t head_dim, size_t hidden_dim);

}  // namespace models

#endif  // THIRD_PARTY_XNNPACK_BENCH_MODELS_MODELS_H_
