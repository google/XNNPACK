// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include <benchmark/benchmark.h>

#define XNN_INVALID_NODE_ID UINT32_MAX
void xnnpack_fully_connected_qd8_f32_qc4w(benchmark::State& state,
                                          const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_channels = state.range(1);
  const size_t output_channels = state.range(2);
  const size_t num_threads = state.range(3);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
  std::uniform_real_distribution<float> f32idist(0.5f, 2.0f);
  std::uniform_int_distribution<int32_t> w8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  xnnpack::Buffer<float> input(batch_size * input_channels,
                               xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> kernel(output_channels * input_channels / 1);
  xnnpack::Buffer<float> bias(output_channels);
  xnnpack::Buffer<float> output(batch_size * output_channels);
  xnnpack::Buffer<float> kernel_scale(output_channels);

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::generate(kernel.begin(), kernel.end(), [&]() { return w8dist(rng); });
  std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                [&]() { return f32idist(rng); });
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();

  std::vector<size_t> input_dims{batch_size, input_channels};
  std::vector<size_t> kernel_dims{output_channels, input_channels};
  std::vector<size_t> bias_dims{output_channels};
  std::vector<size_t> output_dims{batch_size, output_channels};
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*external_value_ids=*/4,
                               /*flags=*/0, &subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(),
      nullptr, /*external_id=*/0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define tensor0");
    return;
  }

  uint32_t dq_quantized_id = XNN_INVALID_NODE_ID;
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, input_dims.size(),
      /*num_nonbatch_dims=*/1, input_dims.data(), XNN_INVALID_VALUE_ID,
      /*flags=*/0, &dq_quantized_id);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define tensor1");
    return;
  }
  const uint8_t kernel_zero_point = 8;
  uint32_t kernel_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_channelwise_quantized_tensor_value_v2(
      subgraph, xnn_datatype_qcint4, kernel_zero_point, kernel_scale.data(),
      kernel_dims.size(),
      /*channel_dim=*/0, kernel_dims.data(), kernel.data(),
      /*external_id=*/1, /*flags=*/0, &kernel_id);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define tensor2");
    return;
  }

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  status =
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, bias_dims.size(),
                              bias_dims.data(), bias.data(),
                              /*external_id=*/2, /*flags=*/0, &bias_id);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define tensor3");
    return;
  }
  uint32_t output_id = XNN_INVALID_NODE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(),
      nullptr,
      /*external_id=*/3, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define tensor4");
    return;
  }

  xnn_runtime_t runtime = nullptr;
  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            input_id, dq_quantized_id, /*flags=*/0);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define convert");
    return;
  }
  status = xnn_define_fully_connected(subgraph, output_min, output_max,
                                      dq_quantized_id, kernel_id, bias_id,
                                      output_id, /*flags=*/0);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to define fc");
    return;
  }
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool{
      nullptr, &pthreadpool_destroy};
  threadpool.reset(pthreadpool_create(num_threads));
  status =
      xnn_create_runtime_v3(subgraph, nullptr, threadpool.get(), 0, &runtime);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, output.data()}};
  status = xnn_setup_runtime(runtime, external.size(), external.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  for (auto _ : state) {
    state.PauseTiming();
    state.ResumeTiming();

    status = xnn_status_success, xnn_invoke_runtime(runtime);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Fully Connected operator");
      return;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * batch_size *
                             input_channels * output_channels,
                         benchmark::Counter::kIsRate);
}

void xnnpack_fully_connected_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_channels = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<float> input(batch_size * input_channels,
                               xnnpack::XnnExtraBytes);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  xnnpack::Buffer<float> kernel(input_channels * output_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  xnnpack::Buffer<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (kernel.size() + bias.size() + output_elements));
  xnnpack::Buffer<float> output(output_elements * num_buffers);

  xnnpack::Buffer<xnn_operator_t> ops(num_buffers);
  for (xnn_operator_t& op : ops) {
    status = xnn_create_fully_connected_nc_f32(
        input_channels, output_channels, input_channels, output_channels,
        kernel.data(), bias.data(), -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        /*flags=*/0, nullptr, nullptr, &op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Fully Connected operator");
      return;
    }
  }

  for (size_t i = 0; i < ops.size(); i++) {
    status = xnn_reshape_fully_connected_nc_f32(ops[i], batch_size,
                                                /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Fully Connected operator");
      return;
    }
  }

  for (size_t i = 0; i < ops.size(); i++) {
    status = xnn_setup_fully_connected_nc_f32(
        ops[i], input.data(), output.data() + i * output_elements);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Fully Connected operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(ops[buffer_index], /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Fully Connected operator");
      return;
    }
  }

  for (xnn_operator_t& op : ops) {
    status = xnn_delete_operator(op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Fully Connected operator");
      return;
    }
    op = nullptr;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * batch_size *
                             input_channels * output_channels,
                         benchmark::Counter::kIsRate);
}

void xnnpack_dynamic_fully_connected_f32(benchmark::State& state,
                                         const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_channels = state.range(1);
  const size_t output_channels = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<float> input(batch_size * input_channels,
                               xnnpack::XnnExtraBytes);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  xnnpack::Buffer<float> kernel(input_channels * output_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  xnnpack::Buffer<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (kernel.size() + bias.size() + output_elements));
  xnnpack::Buffer<float> output(output_elements * num_buffers);

  xnnpack::Buffer<xnn_operator_t> ops(num_buffers);
  for (xnn_operator_t& op : ops) {
    status = xnn_create_dynamic_fully_connected_nc_f32(
        -std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        /*flags=*/0, &op);
    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to create FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  std::vector<std::unique_ptr<xnnpack::Buffer<char>>> workspaces;

  for (size_t i = 0; i < ops.size(); i++) {
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    status = xnn_reshape_dynamic_fully_connected_nc_f32(
        ops[i], batch_size, input_channels, output_channels, input_channels,
        output_channels, &workspace_size, &workspace_alignment,
        /*threadpool=*/nullptr);

    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to reshape FP32 Dynamic Fully Connected operator");
      return;
    }

    auto workspace = std::make_unique<xnnpack::Buffer<char>>(workspace_size);
    char* workspace_ptr = workspace->data();

    workspaces.push_back(std::move(workspace));

    status = xnn_setup_dynamic_fully_connected_nc_f32(
        ops[i], workspace_ptr, input.data(), kernel.data(), bias.data(),
        output.data() + i * output_elements);

    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to setup FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(ops[buffer_index], /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to run FP32 Dynamic Fully Connected operator");
      return;
    }
  }

  for (xnn_operator_t& op : ops) {
    status = xnn_delete_operator(op);
    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to delete FP32 Dynamic Fully Connected operator");
      return;
    }
    op = nullptr;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] =
      benchmark::Counter(uint64_t(state.iterations()) * 2 * batch_size *
                             input_channels * output_channels,
                         benchmark::Counter::kIsRate);
}

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
