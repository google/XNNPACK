// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vcvt.h>


namespace {
static void f16_f32_vcvt(
  benchmark::State& state,
  xnn_f16_f32_vcvt_ukernel_fn cvt,
  xnn_init_f16_f32_cvt_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f16rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  xnn_f16_f32_cvt_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint16_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void f32_f16_vcvt(
  benchmark::State& state,
  xnn_f32_f16_vcvt_ukernel_fn cvt,
  xnn_init_f32_f16_cvt_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT16_C(0x7E00));

  xnn_f32_f16_cvt_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint16_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void f32_qs8_vcvt(
  benchmark::State& state,
  xnn_f32_qs8_vcvt_ukernel_fn cvt,
  xnn_init_f32_qs8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), INT8_C(0xA5));

  xnn_f32_qs8_cvt_params params;
  init_params(&params,
    25.0f /* scale */,
    1 /* output zero point */,
    std::numeric_limits<int8_t>::min() + 1 /* output min */,
    std::numeric_limits<int8_t>::max() - 1 /* output max */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(int8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void f32_qu8_vcvt(
  benchmark::State& state,
  xnn_f32_qu8_vcvt_ukernel_fn cvt,
  xnn_init_f32_qu8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT8_C(0xA5));

  xnn_f32_qu8_cvt_params params;
  init_params(&params,
    25.0f /* scale */,
    127 /* output zero point */,
    std::numeric_limits<uint8_t>::min() + 1 /* output min */,
    std::numeric_limits<uint8_t>::max() - 1 /* output max */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qs8_vcvt(
  benchmark::State& state,
  xnn_qs8_vcvt_ukernel_fn cvt,
  xnn_init_qs8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(i8rng));
  std::fill(y.begin(), y.end(), INT8_C(0xAA));

  xnn_qs8_cvt_params params;
  init_params(&params, 1.25f /* scale */, -1 /* input zero point */, 1 /* output zero point */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(int8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(int8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qu8_vcvt(
  benchmark::State& state,
  xnn_qu8_vcvt_ukernel_fn cvt,
  xnn_init_qu8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(u8rng));
  std::fill(y.begin(), y.end(), UINT8_C(0xAA));

  xnn_qu8_cvt_params params;
  init_params(&params, 1.25f /* scale */, 127 /* input zero point */, 129 /* output zero point */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint8_t) + sizeof(uint8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qs16_qs8_vcvt(
  benchmark::State& state,
  xnn_qs16_qs8_vcvt_ukernel_fn cvt,
  xnn_init_qs16_qs8_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if ((isa_check != nullptr) && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i16rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()),
    std::ref(rng));

  std::vector<int16_t, AlignedAllocator<int16_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(int16_t));
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(i16rng));
  std::fill(y.begin(), y.end(), INT8_C(0xAA));

  xnn_qs16_qs8_cvt_params params;
  init_params(&params, 1.25f /* scale */, 1 /* output zero point */);

  for (auto _ : state) {
    cvt(num_elements * sizeof(int16_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(int8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qs8_f16_vcvt(
  benchmark::State& state,
  xnn_qs8_f16_vcvt_ukernel_fn cvt,
  xnn_init_qs8_f16_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(i8rng));
  std::fill(y.begin(), y.end(), UINT16_C(0x7E00));

  xnn_qs8_f16_cvt_params params;
  init_params(&params,
    fp16_ieee_from_fp32_value(0.25f) /* scale */,
    1 /* output zero point */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(int8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(uint16_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qs8_f32_vcvt(
  benchmark::State& state,
  xnn_qs8_f32_vcvt_ukernel_fn cvt,
  xnn_init_qs8_f32_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(i8rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  xnn_qs8_f32_cvt_params params;
  init_params(&params,
    0.25f /* scale */,
    1 /* output zero point */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(int8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void qu8_f32_vcvt(
  benchmark::State& state,
  xnn_qu8_f32_vcvt_ukernel_fn cvt,
  xnn_init_qu8_f32_cvt_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(u8rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  xnn_qu8_f32_cvt_params params;
  init_params(&params,
    0.25f /* scale */,
    127 /* output zero point */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

}  // namespace
