#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/raddexpminusmax.h>
#include <xnnpack/raddextexp.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>
#include <xnnpack/vscale.h>
#include <xnnpack/vscaleexpminusmax.h>
#include <xnnpack/vscaleextexp.h>

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_INTEL_DNNL
#include <dnnl.h>
#endif  // BENCHMARK_INTEL_DNNL


#ifdef BENCHMARK_INTEL_DNNL
static void DNNLSoftArgMax(
  benchmark::State& state)
{
  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float> x(n);
  std::vector<float> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  dnnl_engine_t engine;
  if (dnnl_engine_create(&engine, dnnl_cpu, 0) != dnnl_success) {
    state.SkipWithError("failed to create CPU engine");
    return;
  }

  dnnl_dim_t input_output_shape[1] = { static_cast<int>(n) };

  dnnl_memory_desc_t memory_descriptor = { 0 };
  if (dnnl_memory_desc_init_by_tag(
    &memory_descriptor, 1, input_output_shape, dnnl_f32, dnnl_x) != dnnl_success)
  {
    state.SkipWithError("failed to create input memory descriptor");
    return;
  }

  dnnl_memory_t input_memory = nullptr;
  if (dnnl_memory_create(
    &input_memory, &memory_descriptor, engine, x.data()) != dnnl_success)
  {
    state.SkipWithError("failed to create input memory");
    return;
  }

  dnnl_memory_t output_memory = nullptr;
  if (dnnl_memory_create(
    &output_memory, &memory_descriptor, engine, y.data()) != dnnl_success)
  {
    state.SkipWithError("failed to create output memory");
    return;
  }

  dnnl_softmax_desc_t softmax_forward_descriptor = {};
  if (dnnl_softmax_forward_desc_init(
    &softmax_forward_descriptor, dnnl_forward_inference,
    &memory_descriptor, 0) != dnnl_success)
  {
    state.SkipWithError("failed to create SoftMax forward descriptor");
    return;
  }

  dnnl_primitive_desc_t softmax_primitive_descriptor = nullptr;
  if (dnnl_primitive_desc_create(
    &softmax_primitive_descriptor, &softmax_forward_descriptor,
    nullptr /* primitive attributes */, engine, nullptr /* hint */) != dnnl_success)
  {
    state.SkipWithError("failed to create SoftMax primitive descriptor");
    return;
  }

  dnnl_primitive_t softmax_primitive = nullptr;
  if (dnnl_primitive_create(
    &softmax_primitive, softmax_primitive_descriptor) != dnnl_success)
  {
    state.SkipWithError("failed to create SoftMax primitive");
    return;
  }

  dnnl_exec_arg_t softmax_args[2] = {
    {DNNL_ARG_SRC, input_memory},
    {DNNL_ARG_DST, output_memory},
  };

  dnnl_stream_t stream = nullptr;
  if (dnnl_stream_create(&stream, engine, dnnl_stream_default_flags) != dnnl_success) {
    state.SkipWithError("failed to create stream");
    return;
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    if (dnnl_primitive_execute(
      softmax_primitive, stream, 2, softmax_args) != dnnl_success)
    {
      state.SkipWithError("failed to execute SoftMax");
      return;
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  if (dnnl_stream_destroy(stream) != dnnl_success) {
    state.SkipWithError("failed to destroy stream");
    return;
  }

  if (dnnl_primitive_desc_destroy(softmax_primitive_descriptor) != dnnl_success) {
    state.SkipWithError("failed to destroy SoftMax primitive descriptor");
    return;
  }

  if (dnnl_primitive_destroy(softmax_primitive) != dnnl_success) {
    state.SkipWithError("failed to destroy SoftMax primitive");
    return;
  }

  if (dnnl_memory_destroy(input_memory) != dnnl_success) {
    state.SkipWithError("failed to destroy input memory");
    return;
  }

  if (dnnl_memory_destroy(output_memory) != dnnl_success) {
    state.SkipWithError("failed to destroy output memory");
    return;
  }

  if (dnnl_engine_destroy(engine) != dnnl_success) {
    state.SkipWithError("failed to destroy engine");
    return;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * 2 * sizeof(float) * n, benchmark::Counter::kIsRate);
}
#endif  // BENCHMARK_INTEL_DNNL

static void ThreePassSoftMaxWithRecomputing(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddexpminusmax_ukernel_function raddexpminusmax,
  xnn_f32_vscaleexpminusmax_ukernel_function vscaleexpminusmax,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float> x(n);
  std::vector<float> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float x_max = nanf("");
    rmax(n * sizeof(float), x.data(), &x_max);
    float y_sum = nanf("");
    raddexpminusmax(n * sizeof(float), x.data(), &y_sum, x_max);
    vscaleexpminusmax(n * sizeof(float), x.data(), y.data() + packed_n * buffer_index, x_max, 1.0f / y_sum);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * 2 * sizeof(float) * n, benchmark::Counter::kIsRate);
}

static void ThreePassSoftMaxWithReloading(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax,
  xnn_f32_vscale_ukernel_function vscale,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float> x(n);
  std::vector<float> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float x_max = nanf("");
    rmax(n * sizeof(float), x.data(), &x_max);
    float y_sum = nanf("");
    raddstoreexpminusmax(n * sizeof(float), x.data(), y.data() + packed_n * buffer_index, &y_sum, x_max);
    vscale(n * sizeof(float), y.data() + packed_n * buffer_index, y.data() + packed_n * buffer_index, 1.0f / y_sum);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * 2 * sizeof(float) * n, benchmark::Counter::kIsRate);
}

static void TwoPassSoftMax(
  benchmark::State& state,
  xnn_f32_raddextexp_ukernel_function raddextexp,
  xnn_f32_vscaleextexp_ukernel_function vscaleextexp,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float> x(n);
  std::vector<float> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float scale[2];
    raddextexp(n * sizeof(float), x.data(), scale);
    vscaleextexp(n * sizeof(float), x.data(), y.data() + packed_n * buffer_index, 1.0f / scale[0], -scale[1]);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * 2 * sizeof(float) * n, benchmark::Counter::kIsRate);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  for (int32_t n = 1000; n <= 100000000; n *= 10) {
    b->Arg(n);
    b->Arg(3 * n);
  }
}

#ifdef BENCHMARK_INTEL_DNNL
  BENCHMARK(DNNLSoftArgMax)->Apply(CharacteristicArguments)->UseManualTime();
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_p5,
    xnn_f32_raddextexp_ukernel__avx2_p5_x96,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_p5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_p5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2,
    xnn_f32_vscale_ukernel__avx_unroll32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  BENCHMARK_CAPTURE(TwoPassSoftMax, avx512f_p5_scalef,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x144_acc3,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx512f_p5_scalef,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx512f_p5_scalef,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscale_ukernel__avx512f_unroll64,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
