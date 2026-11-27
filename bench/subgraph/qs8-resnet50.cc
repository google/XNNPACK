// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "bench/subgraph/models.h"
#include "include/xnnpack.h"
#include "test/replicable_random_device.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

namespace {

struct StageConfig {
  size_t num_blocks;
  size_t mid_channels;
  size_t out_channels;
  uint32_t first_stride;
};

constexpr std::array<StageConfig, 4> kResNet50Stages = {{
    {3, 64, 256, 1},
    {4, 128, 512, 2},
    {6, 256, 1024, 2},
    {3, 512, 2048, 2},
}};

}  // namespace

xnn_subgraph_t QS8ResNet50(size_t batch_size) {
  const float inf = std::numeric_limits<float>::infinity();
  constexpr float kActivationScale = 0.02f;

  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status =
      xnn_create_subgraph(/*num_external_values=*/2, /*flags=*/0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  static std::deque<std::vector<int8_t>> weight_buffers;
  static std::deque<std::vector<int32_t>> bias_buffers;
  static std::deque<std::vector<float>> scale_buffers;
  weight_buffers.clear();
  bias_buffers.clear();
  scale_buffers.clear();

  xnnpack::ReplicableRandomDevice rng;
  auto qc8rng = std::bind(
      std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max()),
      std::ref(rng));
  auto qc32rng =
      std::bind(std::uniform_int_distribution<int>(-1000, 1000), std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 0.2f),
                            std::ref(rng));

  auto define_qint8_tensor = [&](const size_t* dims, size_t num_dims,
                                 int32_t zero_point, float scale,
                                 uint32_t* id_out) -> bool {
    *id_out = XNN_INVALID_VALUE_ID;
    status = xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, zero_point, scale, num_dims, dims,
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, id_out);
    return status == xnn_status_success;
  };

  auto define_qc8_filter_and_bias =
      [&](const size_t* filter_dims, size_t num_filter_dims, float input_scale,
          uint32_t* filter_id, uint32_t* bias_id) -> bool {
    const size_t out_c = filter_dims[0];
    size_t num_elements = 1;
    for (size_t i = 0; i < num_filter_dims; ++i) {
      num_elements *= filter_dims[i];
    }

    weight_buffers.emplace_back(XNN_PAD_EXTRA_BYTES(num_elements, int8_t));
    std::generate(weight_buffers.back().begin(), weight_buffers.back().end(),
                  std::ref(qc8rng));

    scale_buffers.emplace_back(out_c);
    std::vector<float>& filter_scales = scale_buffers.back();
    std::generate(filter_scales.begin(), filter_scales.end(),
                  std::ref(scalerng));

    *filter_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_channelwise_quantized_tensor_value(
        subgraph, xnn_datatype_qcint8, filter_scales.data(), num_filter_dims,
        /*channel_dim=*/0, filter_dims, weight_buffers.back().data(),
        XNN_INVALID_VALUE_ID, /*flags=*/0, filter_id);
    if (status != xnn_status_success) {
      return false;
    }

    bias_buffers.emplace_back(XNN_PAD_EXTRA_BYTES(out_c, int32_t));
    std::generate(bias_buffers.back().begin(), bias_buffers.back().end(),
                  std::ref(qc32rng));

    scale_buffers.emplace_back(out_c);
    std::vector<float>& bias_scales = scale_buffers.back();
    for (size_t c = 0; c < out_c; ++c) {
      bias_scales[c] = input_scale * filter_scales[c];
    }

    const std::array<size_t, 1> bias_dims = {{out_c}};
    *bias_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_channelwise_quantized_tensor_value(
        subgraph, xnn_datatype_qcint32, bias_scales.data(), bias_dims.size(),
        /*channel_dim=*/0, bias_dims.data(), bias_buffers.back().data(),
        XNN_INVALID_VALUE_ID, /*flags=*/0, bias_id);
    return status == xnn_status_success;
  };

  auto define_conv2d = [&](uint32_t input_id, size_t in_h, size_t in_w,
                           size_t in_c, size_t out_c, uint32_t kernel_size,
                           uint32_t stride, uint32_t padding, bool fuse_relu,
                           uint32_t* output_id) -> bool {
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    uint32_t filter_id = XNN_INVALID_VALUE_ID;
    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    const std::array<size_t, 4> filter_dims = {
        {out_c, kernel_size, kernel_size, in_c}};
    if (!define_qc8_filter_and_bias(filter_dims.data(), filter_dims.size(),
                                    kActivationScale, &filter_id, &bias_id)) {
      return false;
    }

    const int32_t out_zero_point = fuse_relu ? -128 : 0;
    const std::array<size_t, 4> out_dims = {{batch_size, out_h, out_w, out_c}};
    if (!define_qint8_tensor(out_dims.data(), out_dims.size(), out_zero_point,
                             kActivationScale, output_id)) {
      return false;
    }

    const float output_min = fuse_relu ? 0.0f : -inf;
    status = xnn_define_convolution_2d(
        subgraph,
        /*input_padding_top=*/padding, /*input_padding_right=*/padding,
        /*input_padding_bottom=*/padding, /*input_padding_left=*/padding,
        /*kernel_height=*/kernel_size, /*kernel_width=*/kernel_size,
        /*subsampling_height=*/stride, /*subsampling_width=*/stride,
        /*dilation_height=*/1, /*dilation_width=*/1,
        /*groups=*/1,
        /*group_input_channels=*/in_c,
        /*group_output_channels=*/out_c, output_min, /*output_max=*/inf,
        input_id, filter_id, bias_id, *output_id, /*flags=*/0);
    return status == xnn_status_success;
  };

  uint32_t v_in = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 4> in_dims = {{batch_size, 224, 224, 3}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, in_dims.size(),
                                   in_dims.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &v_in);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t v_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> out_dims = {{batch_size, 1000}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, out_dims.size(),
                                   out_dims.data(),
                                   /*data=*/nullptr, /*external_id=*/1,
                                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v_out);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t v_in_q = XNN_INVALID_VALUE_ID;
  if (!define_qint8_tensor(in_dims.data(), in_dims.size(), /*zero_point=*/0,
                           kActivationScale, &v_in_q)) {
    return nullptr;
  }
  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            v_in, v_in_q, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  // Stem: 7x7 conv (stride 2, pad 3, ReLU) -> 3x3 maxpool (stride 2, pad 1).
  uint32_t conv1_out = XNN_INVALID_VALUE_ID;
  if (!define_conv2d(v_in_q, /*in_h=*/224, /*in_w=*/224, /*in_c=*/3,
                     /*out_c=*/64, /*kernel_size=*/7, /*stride=*/2,
                     /*padding=*/3, /*fuse_relu=*/true, &conv1_out)) {
    return nullptr;
  }

  uint32_t pool1_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 4> pool1_dims = {{batch_size, 56, 56, 64}};
  if (!define_qint8_tensor(pool1_dims.data(), pool1_dims.size(),
                           /*zero_point=*/-128, kActivationScale, &pool1_out)) {
    return nullptr;
  }
  status = xnn_define_max_pooling_2d(
      subgraph,
      /*input_padding_top=*/1, /*input_padding_right=*/1,
      /*input_padding_bottom=*/1, /*input_padding_left=*/1,
      /*pooling_height=*/3, /*pooling_width=*/3,
      /*stride_height=*/2, /*stride_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*output_min=*/-inf, /*output_max=*/inf, conv1_out, pool1_out,
      /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t cur_id = pool1_out;
  size_t cur_h = 56;
  size_t cur_w = 56;
  size_t cur_c = 64;

  // Residual bottleneck stages.
  for (const StageConfig& stage : kResNet50Stages) {
    for (size_t block = 0; block < stage.num_blocks; ++block) {
      const uint32_t stride = (block == 0) ? stage.first_stride : 1;
      const size_t next_h = cur_h / stride;
      const size_t next_w = cur_w / stride;

      uint32_t b_conv1 = XNN_INVALID_VALUE_ID;
      if (!define_conv2d(cur_id, cur_h, cur_w, cur_c, stage.mid_channels,
                         /*kernel_size=*/1, /*stride=*/1, /*padding=*/0,
                         /*fuse_relu=*/true, &b_conv1)) {
        return nullptr;
      }

      uint32_t b_conv2 = XNN_INVALID_VALUE_ID;
      if (!define_conv2d(b_conv1, cur_h, cur_w, stage.mid_channels,
                         stage.mid_channels, /*kernel_size=*/3, stride,
                         /*padding=*/1, /*fuse_relu=*/true, &b_conv2)) {
        return nullptr;
      }

      uint32_t b_conv3 = XNN_INVALID_VALUE_ID;
      if (!define_conv2d(b_conv2, next_h, next_w, stage.mid_channels,
                         stage.out_channels, /*kernel_size=*/1, /*stride=*/1,
                         /*padding=*/0, /*fuse_relu=*/false, &b_conv3)) {
        return nullptr;
      }

      uint32_t shortcut_id = cur_id;
      if (block == 0) {
        if (!define_conv2d(cur_id, cur_h, cur_w, cur_c, stage.out_channels,
                           /*kernel_size=*/1, stride, /*padding=*/0,
                           /*fuse_relu=*/false, &shortcut_id)) {
          return nullptr;
        }
      }

      // Post-ReLU output tensor with zero_point = -128 naturally clamps at
      // 0.0f.
      uint32_t block_out = XNN_INVALID_VALUE_ID;
      const std::array<size_t, 4> block_out_dims = {
          {batch_size, next_h, next_w, stage.out_channels}};
      if (!define_qint8_tensor(block_out_dims.data(), block_out_dims.size(),
                               /*zero_point=*/-128, kActivationScale,
                               &block_out)) {
        return nullptr;
      }

      const xnn_binary_params add_params = {-inf, inf};
      status = xnn_define_binary(subgraph, xnn_binary_add, &add_params, b_conv3,
                                 shortcut_id, block_out, /*flags=*/0);
      if (status != xnn_status_success) {
        return nullptr;
      }

      cur_id = block_out;
      cur_h = next_h;
      cur_w = next_w;
      cur_c = stage.out_channels;
    }
  }

  // Global average pooling over spatial dimensions (1, 2).
  uint32_t avgpool_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> avgpool_dims = {{batch_size, cur_c}};
  if (!define_qint8_tensor(avgpool_dims.data(), avgpool_dims.size(),
                           /*zero_point=*/-128, kActivationScale,
                           &avgpool_out)) {
    return nullptr;
  }
  const std::array<int64_t, 2> reduction_axes = {{1, 2}};
  status = xnn_define_static_reduce_v2(
      subgraph, xnn_reduce_mean, reduction_axes.size(), reduction_axes.data(),
      cur_id, avgpool_out, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  // Classification head: fully connected 2048 -> 1000.
  uint32_t fc_filter = XNN_INVALID_VALUE_ID;
  uint32_t fc_bias = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> fc_filter_dims = {{1000, cur_c}};
  if (!define_qc8_filter_and_bias(fc_filter_dims.data(), fc_filter_dims.size(),
                                  kActivationScale, &fc_filter, &fc_bias)) {
    return nullptr;
  }

  uint32_t fc_out = XNN_INVALID_VALUE_ID;
  if (!define_qint8_tensor(out_dims.data(), out_dims.size(), /*zero_point=*/0,
                           kActivationScale, &fc_out)) {
    return nullptr;
  }

  status = xnn_define_fully_connected(subgraph, /*output_min=*/-inf,
                                      /*output_max=*/inf, avgpool_out,
                                      fc_filter, fc_bias, fc_out, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            fc_out, v_out, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  return subgraph;
}

xnn_subgraph_t QD8ResNet50(size_t batch_size) {
  const float inf = std::numeric_limits<float>::infinity();

  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status =
      xnn_create_subgraph(/*num_external_values=*/2, /*flags=*/0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  static std::deque<std::vector<int8_t>> weight_buffers;
  static std::deque<std::vector<float>> float_buffers;
  weight_buffers.clear();
  float_buffers.clear();

  xnnpack::ReplicableRandomDevice rng;
  auto qc8rng = std::bind(
      std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max()),
      std::ref(rng));
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-0.05f, +0.05f),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 0.1f),
                            std::ref(rng));

  auto define_fp32_tensor = [&](const size_t* dims, size_t num_dims,
                                uint32_t* id_out) -> bool {
    *id_out = XNN_INVALID_VALUE_ID;
    status =
        xnn_define_tensor_value(subgraph, xnn_datatype_fp32, num_dims, dims,
                                /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                /*flags=*/0, id_out);
    return status == xnn_status_success;
  };

  auto define_qc8_filter_and_fp32_bias =
      [&](const size_t* filter_dims, size_t num_filter_dims,
          uint32_t* filter_id, uint32_t* bias_id) -> bool {
    const size_t out_c = filter_dims[0];
    size_t num_elements = 1;
    for (size_t i = 0; i < num_filter_dims; ++i) {
      num_elements *= filter_dims[i];
    }

    weight_buffers.emplace_back(XNN_PAD_EXTRA_BYTES(num_elements, int8_t));
    std::generate(weight_buffers.back().begin(), weight_buffers.back().end(),
                  std::ref(qc8rng));

    float_buffers.emplace_back(out_c);
    std::vector<float>& filter_scales = float_buffers.back();
    std::generate(filter_scales.begin(), filter_scales.end(),
                  std::ref(scalerng));

    *filter_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_channelwise_quantized_tensor_value(
        subgraph, xnn_datatype_qcint8, filter_scales.data(), num_filter_dims,
        /*channel_dim=*/0, filter_dims, weight_buffers.back().data(),
        XNN_INVALID_VALUE_ID, /*flags=*/0, filter_id);
    if (status != xnn_status_success) {
      return false;
    }

    float_buffers.emplace_back(XNN_PAD_EXTRA_BYTES(out_c, float));
    std::generate(float_buffers.back().begin(), float_buffers.back().end(),
                  std::ref(f32rng));

    const std::array<size_t, 1> bias_dims = {{out_c}};
    *bias_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph, xnn_datatype_fp32, bias_dims.size(), bias_dims.data(),
        /*data=*/float_buffers.back().data(), XNN_INVALID_VALUE_ID,
        /*flags=*/0, bias_id);
    return status == xnn_status_success;
  };

  auto define_qd8_conv2d =
      [&](uint32_t input_id, size_t in_h, size_t in_w, size_t in_c,
          size_t out_c, uint32_t kernel_size, uint32_t stride, uint32_t padding,
          float output_min, float output_max, uint32_t* output_id) -> bool {
    const size_t out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    const size_t out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    const std::array<size_t, 4> in_dims = {{batch_size, in_h, in_w, in_c}};
    uint32_t dq_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_dynamically_quantized_tensor_value(
        subgraph, xnn_datatype_qdint8, in_dims.size(),
        /*num_nonbatch_dims=*/3, in_dims.data(), XNN_INVALID_VALUE_ID,
        /*flags=*/0, &dq_id);
    if (status != xnn_status_success) {
      return false;
    }

    status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                              input_id, dq_id, /*flags=*/0);
    if (status != xnn_status_success) {
      return false;
    }

    uint32_t filter_id = XNN_INVALID_VALUE_ID;
    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    const std::array<size_t, 4> filter_dims = {
        {out_c, kernel_size, kernel_size, in_c}};
    if (!define_qc8_filter_and_fp32_bias(filter_dims.data(), filter_dims.size(),
                                         &filter_id, &bias_id)) {
      return false;
    }

    const std::array<size_t, 4> out_dims = {{batch_size, out_h, out_w, out_c}};
    if (!define_fp32_tensor(out_dims.data(), out_dims.size(), output_id)) {
      return false;
    }

    status = xnn_define_convolution_2d(
        subgraph,
        /*input_padding_top=*/padding, /*input_padding_right=*/padding,
        /*input_padding_bottom=*/padding, /*input_padding_left=*/padding,
        /*kernel_height=*/kernel_size, /*kernel_width=*/kernel_size,
        /*subsampling_height=*/stride, /*subsampling_width=*/stride,
        /*dilation_height=*/1, /*dilation_width=*/1,
        /*groups=*/1,
        /*group_input_channels=*/in_c,
        /*group_output_channels=*/out_c, output_min, output_max, dq_id,
        filter_id, bias_id, *output_id, /*flags=*/0);
    return status == xnn_status_success;
  };

  uint32_t v_in = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 4> in_dims = {{batch_size, 224, 224, 3}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, in_dims.size(),
                                   in_dims.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &v_in);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t v_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> out_dims = {{batch_size, 1000}};
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, out_dims.size(),
                                   out_dims.data(),
                                   /*data=*/nullptr, /*external_id=*/1,
                                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v_out);
  if (status != xnn_status_success) {
    return nullptr;
  }

  // Stem: 7x7 conv (stride 2, pad 3, ReLU) -> 3x3 maxpool (stride 2, pad 1).
  uint32_t conv1_out = XNN_INVALID_VALUE_ID;
  if (!define_qd8_conv2d(v_in, /*in_h=*/224, /*in_w=*/224, /*in_c=*/3,
                         /*out_c=*/64, /*kernel_size=*/7, /*stride=*/2,
                         /*padding=*/3, /*output_min=*/0.0f,
                         /*output_max=*/inf, &conv1_out)) {
    return nullptr;
  }

  uint32_t pool1_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 4> pool1_dims = {{batch_size, 56, 56, 64}};
  if (!define_fp32_tensor(pool1_dims.data(), pool1_dims.size(), &pool1_out)) {
    return nullptr;
  }
  status = xnn_define_max_pooling_2d(
      subgraph,
      /*input_padding_top=*/1, /*input_padding_right=*/1,
      /*input_padding_bottom=*/1, /*input_padding_left=*/1,
      /*pooling_height=*/3, /*pooling_width=*/3,
      /*stride_height=*/2, /*stride_width=*/2,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*output_min=*/-inf, /*output_max=*/inf, conv1_out, pool1_out,
      /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t cur_id = pool1_out;
  size_t cur_h = 56;
  size_t cur_w = 56;
  size_t cur_c = 64;

  // Residual bottleneck stages.
  for (const StageConfig& stage : kResNet50Stages) {
    for (size_t block = 0; block < stage.num_blocks; ++block) {
      const uint32_t stride = (block == 0) ? stage.first_stride : 1;
      const size_t next_h = cur_h / stride;
      const size_t next_w = cur_w / stride;

      uint32_t b_conv1 = XNN_INVALID_VALUE_ID;
      if (!define_qd8_conv2d(cur_id, cur_h, cur_w, cur_c, stage.mid_channels,
                             /*kernel_size=*/1, /*stride=*/1, /*padding=*/0,
                             /*output_min=*/0.0f, /*output_max=*/inf,
                             &b_conv1)) {
        return nullptr;
      }

      uint32_t b_conv2 = XNN_INVALID_VALUE_ID;
      if (!define_qd8_conv2d(b_conv1, cur_h, cur_w, stage.mid_channels,
                             stage.mid_channels, /*kernel_size=*/3, stride,
                             /*padding=*/1, /*output_min=*/0.0f,
                             /*output_max=*/inf, &b_conv2)) {
        return nullptr;
      }

      uint32_t b_conv3 = XNN_INVALID_VALUE_ID;
      if (!define_qd8_conv2d(b_conv2, next_h, next_w, stage.mid_channels,
                             stage.out_channels, /*kernel_size=*/1,
                             /*stride=*/1, /*padding=*/0, /*output_min=*/-inf,
                             /*output_max=*/inf, &b_conv3)) {
        return nullptr;
      }

      uint32_t shortcut_id = cur_id;
      if (block == 0) {
        if (!define_qd8_conv2d(cur_id, cur_h, cur_w, cur_c, stage.out_channels,
                               /*kernel_size=*/1, stride, /*padding=*/0,
                               /*output_min=*/-inf, /*output_max=*/inf,
                               &shortcut_id)) {
          return nullptr;
        }
      }

      uint32_t block_out = XNN_INVALID_VALUE_ID;
      const std::array<size_t, 4> block_out_dims = {
          {batch_size, next_h, next_w, stage.out_channels}};
      if (!define_fp32_tensor(block_out_dims.data(), block_out_dims.size(),
                              &block_out)) {
        return nullptr;
      }

      const xnn_binary_params relu_params = {0.0f, inf};
      status = xnn_define_binary(subgraph, xnn_binary_add, &relu_params,
                                 b_conv3, shortcut_id, block_out, /*flags=*/0);
      if (status != xnn_status_success) {
        return nullptr;
      }

      cur_id = block_out;
      cur_h = next_h;
      cur_w = next_w;
      cur_c = stage.out_channels;
    }
  }

  // Global average pooling over spatial dimensions (1, 2).
  uint32_t avgpool_out = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> avgpool_dims = {{batch_size, cur_c}};
  if (!define_fp32_tensor(avgpool_dims.data(), avgpool_dims.size(),
                          &avgpool_out)) {
    return nullptr;
  }
  const std::array<int64_t, 2> reduction_axes = {{1, 2}};
  status = xnn_define_static_reduce_v2(
      subgraph, xnn_reduce_mean, reduction_axes.size(), reduction_axes.data(),
      cur_id, avgpool_out, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  // Classification head: dynamically quantize avgpool_out -> fully connected.
  uint32_t fc_dq_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, avgpool_dims.size(),
      /*num_nonbatch_dims=*/1, avgpool_dims.data(), XNN_INVALID_VALUE_ID,
      /*flags=*/0, &fc_dq_id);
  if (status != xnn_status_success) {
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            avgpool_out, fc_dq_id, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  uint32_t fc_filter = XNN_INVALID_VALUE_ID;
  uint32_t fc_bias = XNN_INVALID_VALUE_ID;
  const std::array<size_t, 2> fc_filter_dims = {{1000, cur_c}};
  if (!define_qc8_filter_and_fp32_bias(
          fc_filter_dims.data(), fc_filter_dims.size(), &fc_filter, &fc_bias)) {
    return nullptr;
  }

  status = xnn_define_fully_connected(subgraph, /*output_min=*/-inf,
                                      /*output_max=*/inf, fc_dq_id, fc_filter,
                                      fc_bias, v_out, /*flags=*/0);
  if (status != xnn_status_success) {
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
