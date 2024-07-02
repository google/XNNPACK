// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "convolution-test-helpers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "xnnpack/microparams.h"

namespace xnnpack {

void compute_convolution_qd8_f32_qc8w_reference_results(
    size_t batch_size, size_t output_height, size_t output_width,
    size_t input_height, size_t input_width, size_t input_padding_top,
    size_t input_padding_right, size_t input_padding_bottom,
    size_t input_padding_left, size_t kernel_height, size_t kernel_width,
    size_t subsampling_height, size_t subsampling_width, size_t dilation_height,
    size_t dilation_width, size_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    const std::vector<int8_t>& input, const std::vector<int8_t>& filter,
    const std::vector<float>& filter_scale,
    const std::vector<xnn_qd8_quantization_params>& quantization_params,
    std::vector<float>& output, bool has_bias, const std::vector<float>& bias) {
  std::fill(output.begin(), output.end(), 0.0f);

  for (size_t i = 0; i < batch_size; i++) {
    int32_t zero_point = quantization_params[i].zero_point;
    float inv_scale = quantization_params[i].inv_scale;
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        // Compute reference results.
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling_height + ky * dilation_height -
                            input_padding_top;
          for (size_t kx = 0; kx < kernel_width; kx++) {
            const size_t ix = ox * subsampling_width + kx * dilation_width -
                              input_padding_left;
            for (size_t g = 0; g < groups; g++) {
              for (size_t oc = 0; oc < group_output_channels; oc++) {
                for (size_t ic = 0; ic < group_input_channels; ic++) {
                  if (iy < input_height && ix < input_width) {
                    output[(((i * output_height + oy) * output_width + ox) *
                                groups +
                            g) *
                               group_output_channels +
                           oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width +
                                        ix) *
                                           input_channel_stride +
                                       g * group_input_channels + ic]) -
                         zero_point) *
                        int32_t(filter[(((g * group_output_channels + oc) *
                                             kernel_height +
                                         ky) *
                                            kernel_width +
                                        kx) *
                                           group_input_channels +
                                       ic]);
                  }
                }
              }
            }
          }
        }
        // Initialize Bias
        for (size_t g = 0; g < groups; g++) {
          for (size_t oc = 0; oc < group_output_channels; oc++) {
            size_t n_index = g * group_output_channels + oc;
            output[(((i * output_height + oy) * output_width + ox) * groups +
                    g) *
                       group_output_channels +
                   oc] *= (inv_scale * filter_scale[n_index]);
            if (has_bias) {
              output[(((i * output_height + oy) * output_width + ox) * groups +
                      g) *
                         group_output_channels +
                     oc] += bias[g * group_output_channels + oc];
            }
          }
        }
      }
    }
  }
}

void compute_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_channel_stride,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  if (!has_bias) {
    std::fill(accumulators.begin(), accumulators.end(), 0);
  }

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        // Initialize Bias
        if (has_bias) {
          for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < group_output_channels; oc++) {
              accumulators[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] =
                  bias[g * group_output_channels + oc];
            }
          }
        }
        // Compute reference results.
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling_height + ky * dilation_height - input_padding_top;
          if (iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t ix = ox * subsampling_width + kx * dilation_width - input_padding_left;
              if (ix < input_width) {
                for (size_t g = 0; g < groups; g++) {
                  for (size_t oc = 0; oc < group_output_channels; oc++) {
                    for (size_t ic = 0; ic < group_input_channels; ic++) {
                      accumulators[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * input_channel_stride +
                                    g * group_input_channels + ic]) -
                         int32_t(input_zero_point)) *
                        int32_t(filter[(((g * group_output_channels + oc) * kernel_height + ky) * kernel_width + kx) * group_input_channels + ic]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void compute_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  compute_convolution_qs8_reference_results(
      batch_size,
      output_height,
      output_width,
      input_height,
      input_width,
      input_padding_top,
      input_padding_right,
      input_padding_bottom,
      input_padding_left,
      kernel_height,
      kernel_width,
      subsampling_height,
      subsampling_width,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      groups * group_input_channels,
      input_zero_point,
      input,
      filter,
      accumulators,
      has_bias,
      bias);
}

void compute_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_channel_stride,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  if (!has_bias) {
    std::fill(accumulators.begin(), accumulators.end(), 0);
  }

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        // Initialize Bias
        if (has_bias) {
          for (size_t g = 0; g < groups; g++) {
            for (size_t oc = 0; oc < group_output_channels; oc++) {
              accumulators[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] =
                  bias[g * group_output_channels + oc];
            }
          }
        }
        // Compute reference results.
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling_height + ky * dilation_height - input_padding_top;
          if (iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t ix = ox * subsampling_width + kx * dilation_width - input_padding_left;
              if (ix < input_width) {
                for (size_t g = 0; g < groups; g++) {
                  for (size_t oc = 0; oc < group_output_channels; oc++) {
                    for (size_t ic = 0; ic < group_input_channels; ic++) {
                      accumulators[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * input_channel_stride + g * group_input_channels + ic]) -
                         int32_t(input_zero_point)) *
                        (int32_t(filter[(((g * group_output_channels + oc) * kernel_height + ky) * kernel_width + kx) * group_input_channels + ic]) - int32_t(kernel_zero_point));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void compute_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  compute_convolution_qu8_reference_results(
      batch_size,
      output_height,
      output_width,
      input_height,
      input_width,
      input_padding_top,
      input_padding_right,
      input_padding_bottom,
      input_padding_left,
      kernel_height,
      kernel_width,
      subsampling_height,
      subsampling_width,
      dilation_height,
      dilation_width,
      groups,
      group_input_channels,
      group_output_channels,
      groups * group_input_channels,
      input_zero_point,
      kernel_zero_point,
      input,
      filter,
      accumulators,
      has_bias,
      bias);
}

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    size_t input_channel_stride,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  if (!has_bias) {
    std::fill(accumulators.begin(), accumulators.end(), 0);
  }

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        // Initialize Bias
        if (has_bias) {
          for (size_t g = 0; g < input_channels; g++) {
            for (size_t oc = 0; oc < depth_multiplier; oc++) {
              accumulators[(((i * output_height + oy) * output_width + ox) * input_channels + g) * depth_multiplier + oc] =
                  bias[g * depth_multiplier + oc];
            }
          }
        }
        // Compute reference results.
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling_height + ky * dilation_height - input_padding_top;
          if (iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t ix = ox * subsampling_width + kx * dilation_width - input_padding_left;
              if (ix < input_width) {
                for (size_t g = 0; g < input_channels; g++) {
                  for (size_t oc = 0; oc < depth_multiplier; oc++) {
                    accumulators[(((i * output_height + oy) * output_width + ox) * input_channels + g) * depth_multiplier + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * input_channel_stride + g]) - int32_t(input_zero_point)) *
                        int32_t(filter[((ky * kernel_width + kx) * input_channels + g) * depth_multiplier + oc]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void compute_depthwise_convolution_qs8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    int8_t input_zero_point,
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  compute_depthwise_convolution_qs8_reference_results(
    batch_size,
    output_height,
    output_width,
    input_height,
    input_width,
    input_padding_top,
    input_padding_right,
    input_padding_bottom,
    input_padding_left,
    kernel_height,
    kernel_width,
    subsampling_height,
    subsampling_width,
    dilation_height,
    dilation_width,
    input_channels,
    depth_multiplier,
    input_channels,
    input_zero_point,
    input,
    filter,
    accumulators,
    has_bias,
    bias);
}

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    size_t input_channel_stride,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  if (!has_bias) {
    std::fill(accumulators.begin(), accumulators.end(), 0);
  }

  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        // Initialize Bias
        if (has_bias) {
          for (size_t g = 0; g < input_channels; g++) {
            for (size_t oc = 0; oc < depth_multiplier; oc++) {
              accumulators[(((i * output_height + oy) * output_width + ox) * input_channels + g) * depth_multiplier + oc] =
                  bias[g * depth_multiplier + oc];
            }
          }
        }
        // Compute reference results.
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling_height + ky * dilation_height - input_padding_top;
          if (iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t ix = ox * subsampling_width + kx * dilation_width - input_padding_left;
              if (ix < input_width) {
                for (size_t g = 0; g < input_channels; g++) {
                  for (size_t oc = 0; oc < depth_multiplier; oc++) {
                    accumulators[(((i * output_height + oy) * output_width + ox) * input_channels + g) * depth_multiplier + oc] +=
                        (int32_t(input[((i * input_height + iy) * input_width + ix) * input_channel_stride + g]) - int32_t(input_zero_point)) *
                        (int32_t(filter[((ky * kernel_width + kx) * input_channels + g) * depth_multiplier + oc]) - int32_t(kernel_zero_point));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void compute_depthwise_convolution_qu8_reference_results(
    size_t batch_size,
    size_t output_height,
    size_t output_width,
    size_t input_height,
    size_t input_width,
    size_t input_padding_top,
    size_t input_padding_right,
    size_t input_padding_bottom,
    size_t input_padding_left,
    size_t kernel_height,
    size_t kernel_width,
    size_t subsampling_height,
    size_t subsampling_width,
    size_t dilation_height,
    size_t dilation_width,
    size_t input_channels,
    size_t depth_multiplier,
    uint8_t input_zero_point,
    uint8_t kernel_zero_point,
    const std::vector<uint8_t>& input,
    const std::vector<uint8_t>& filter,
    std::vector<int32_t>& accumulators,
    bool has_bias,
    const std::vector<int32_t>& bias)
{
  compute_depthwise_convolution_qu8_reference_results(
    batch_size,
    output_height,
    output_width,
    input_height,
    input_width,
    input_padding_top,
    input_padding_right,
    input_padding_bottom,
    input_padding_left,
    kernel_height,
    kernel_width,
    subsampling_height,
    subsampling_width,
    dilation_height,
    dilation_width,
    input_channels,
    depth_multiplier,
    input_channels,
    input_zero_point,
    kernel_zero_point,
    input,
    filter,
    accumulators,
    has_bias,
    bias);
}
}  // namespace xnnpack
