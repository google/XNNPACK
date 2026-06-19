// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <vector>
#include "bench/utils.h"
#include "src/xnnpack/indirection.h"

static void BM_indirection_init_conv2d(benchmark::State& state) {
  const size_t output_tile_size = 4;
  const size_t output_height = state.range(0);
  const size_t output_width = state.range(1);
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;

  const size_t input_height = output_height;
  const size_t input_width = output_width;
  const size_t input_pixel_stride = 1;
  const size_t stride_height = 1;
  const size_t stride_width = 1;
  const size_t dilation_height = 1;
  const size_t dilation_width = 1;
  const size_t input_padding_top = 1;
  const size_t input_padding_left = 1;

  std::vector<const void*> indirection_buffer(output_height * output_width * kernel_height * kernel_width);
  std::vector<char> input(input_height * input_width);
  std::vector<char> zero_buffer(1);

  for (auto _ : state) {
    xnn_indirection_init_conv2d(
      output_tile_size, 0, output_height * output_width,
      indirection_buffer.data(), input.data(), zero_buffer.data(),
      input_pixel_stride, input_height, input_width,
      output_height, output_width, kernel_height, kernel_width,
      stride_height, stride_width, dilation_height, dilation_width,
      input_padding_top, input_padding_left);
  }
}
BENCHMARK(BM_indirection_init_conv2d)->Args({224, 224})->Args({64, 64});

static void BM_indirection_init_deconv2d(benchmark::State& state) {
  const size_t output_tile_size = 4;
  const size_t output_height = state.range(0);
  const size_t output_width = state.range(1);
  const size_t kernel_height = 3;
  const size_t kernel_width = 3;

  const size_t input_height = output_height;
  const size_t input_width = output_width;
  const size_t input_pixel_stride = 1;
  const size_t stride_height = 1;
  const size_t stride_width = 1;
  const size_t dilation_height = 1;
  const size_t dilation_width = 1;
  const size_t padding_top = 1;
  const size_t padding_left = 1;

  std::vector<const void*> indirection_buffer(output_height * output_width * kernel_height * kernel_width);
  std::vector<char> input(input_height * input_width);
  std::vector<char> zero_buffer(1);

  for (auto _ : state) {
    xnn_indirection_init_deconv2d(
      output_tile_size,
      indirection_buffer.data(), input.data(), input_pixel_stride, zero_buffer.data(),
      input_height, input_width, output_height, output_width,
      kernel_height, kernel_width, stride_height, stride_width,
      dilation_height, dilation_width, padding_top, padding_left);
  }
}
BENCHMARK(BM_indirection_init_deconv2d)->Args({224, 224})->Args({64, 64});

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
