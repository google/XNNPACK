#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_INTEL_DNNL
#include <dnnl.h>
#endif  // BENCHMARK_INTEL_DNNL
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/raddexpminusmax.h"
#include "xnnpack/raddextexp.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/reduce.h"
#include "xnnpack/vscaleexpminusmax.h"
#include "xnnpack/vscaleextexp.h"


#ifdef BENCHMARK_INTEL_DNNL

// Macros to help transition from oneDNN v2 to v3.
#if DNNL_VERSION_MAJOR == 2
#define DNNL_MEMORY_DESC_INIT dnnl_memory_desc_init_by_tag
#define DNNL_MEMORY_CREATE(mem, mem_desc, engine, handle) \
  dnnl_memory_create(&mem, &mem_desc, engine, handle)
#else   // DNNL_VERSION_MAJOR == 3
#define DNNL_MEMORY_DESC_INIT dnnl_memory_desc_create_with_tag
#define DNNL_MEMORY_CREATE(mem, mem_desc, engine, handle) \
  dnnl_memory_create(&mem, mem_desc, engine, handle)
#endif  // DNNL_VERSION_MAJOR == 2

static void DNNLSoftArgMax(
  benchmark::State& state)
{
  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  std::vector<float> x(elements);
  std::vector<float> y(packed_elements * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  dnnl_engine_t engine;
  if (dnnl_engine_create(&engine, dnnl_cpu, 0) != dnnl_success) {
    state.SkipWithError("failed to create CPU engine");
    return;
  }

  dnnl_dim_t input_output_shape[1] = { static_cast<int>(elements) };

  dnnl_memory_desc_t memory_descriptor = { 0 };
  if (DNNL_MEMORY_DESC_INIT(
    &memory_descriptor, 1, input_output_shape, dnnl_f32, dnnl_x) != dnnl_success)
  {
    state.SkipWithError("failed to create input memory descriptor");
    return;
  }

  dnnl_memory_t input_memory = nullptr;
  if (DNNL_MEMORY_CREATE(
    input_memory, memory_descriptor, engine, x.data()) != dnnl_success)
  {
    state.SkipWithError("failed to create input memory");
    return;
  }

  dnnl_memory_t output_memory = nullptr;
  if (DNNL_MEMORY_CREATE(
    output_memory, memory_descriptor, engine, y.data()) != dnnl_success)
  {
    state.SkipWithError("failed to create output memory");
    return;
  }

#if DNNL_VERSION_MAJOR == 2
  dnnl_softmax_desc_t softmax_forward_descriptor = {};
  if (dnnl_softmax_forward_desc_init(
    &softmax_forward_descriptor, dnnl_forward_inference,
    &memory_descriptor, 0) != dnnl_success)
  {
    state.SkipWithError("failed to create SoftMax forward descriptor");
    return;
  }
#endif  // DNNL_VERSION_MAJOR == 2

  dnnl_primitive_desc_t softmax_primitive_descriptor = nullptr;
#if DNNL_VERSION_MAJOR == 2
  if (dnnl_primitive_desc_create(
    &softmax_primitive_descriptor, &softmax_forward_descriptor,
    nullptr /* primitive attributes */, engine, nullptr /* hint */) != dnnl_success)
#else   // DNNL_VERSION_MAJOR == 3
  if (dnnl_softmax_forward_primitive_desc_create(
    &softmax_primitive_descriptor, engine, dnnl_forward_inference,
    dnnl_softmax_accurate, /*src_desc=*/ memory_descriptor,
    /*dst_dsc=*/ memory_descriptor, /*softmax_axis=*/ 0,
    /*attr=*/ nullptr) != dnnl_success)
#endif  // DNNL_VERSION_MAJOR == 2
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

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}
#endif  // BENCHMARK_INTEL_DNNL

static void ThreePassSoftMaxWithRecomputing(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_fn rmax,
  xnn_init_f32_default_params_fn init_rmax_params,
  xnn_f32_raddexpminusmax_ukernel_fn raddexpminusmax,
  xnn_f32_vscaleexpminusmax_ukernel_fn vscaleexpminusmax,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  std::vector<float> x(elements);
  std::vector<float> y(packed_elements * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  xnn_f32_default_params rmax_params;
  if (init_rmax_params) {
    init_rmax_params(&rmax_params);
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float x_max = nanf("");
    rmax(elements * sizeof(float), x.data(), &x_max, &rmax_params);
    float y_sum = nanf("");
    raddexpminusmax(elements * sizeof(float), x.data(), &y_sum, x_max);
    vscaleexpminusmax(elements * sizeof(float), x.data(), y.data() + packed_elements * buffer_index, x_max, 1.0f / y_sum);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void ThreePassSoftMaxWithReloading(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_fn rmax,
  xnn_init_f32_default_params_fn init_rmax_params,
  xnn_f32_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
  xnn_init_f32_expminus_params_fn init_expminus_params,
  xnn_f32_vbinary_minmax_ukernel_fn vmulc,
  xnn_init_f32_minmax_params_fn init_minmax_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  std::vector<float> x(elements);
  std::vector<float> y(packed_elements * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  xnn_f32_default_params rmax_params;
  xnn_f32_expminus_params expminus_params;
  xnn_f32_minmax_params minmax_params;
  if (init_rmax_params) {
    init_rmax_params(&rmax_params);
  }
  init_expminus_params(&expminus_params);
  init_minmax_params(&minmax_params, -INFINITY, INFINITY);

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::PrefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float x_max = nanf("");
    rmax(elements * sizeof(float), x.data(), &x_max, &rmax_params);
    float y_sum = nanf("");
    raddstoreexpminusmax(elements * sizeof(float), x.data(), &x_max, y.data() + packed_elements * buffer_index, &y_sum, &expminus_params);
    const float inv_y_sum = 1.0f / y_sum;
    vmulc(elements * sizeof(float), y.data() + packed_elements * buffer_index, &inv_y_sum, y.data() + packed_elements * buffer_index, &minmax_params);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void TwoPassSoftMax(
  benchmark::State& state,
  xnn_f32_raddextexp_ukernel_fn raddextexp,
  xnn_f32_vscaleextexp_ukernel_fn vscaleextexp,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  std::vector<float> x(elements);
  std::vector<float> y(packed_elements * num_buffers);

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
    raddextexp(elements * sizeof(float), x.data(), scale);
    vscaleextexp(elements * sizeof(float), x.data(), y.data() + packed_elements * buffer_index, 1.0f / scale[0], -scale[1]);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  // Size        Iterations  Parameters used by Stable Diffusion
  b->Arg( 128);  // 1
  b->Arg( 154);  // 421
  b->Arg( 512);  // 20
  b->Arg(2048);  // 80
  b->Arg(8192);  // 320
  for (int32_t n = 10000; n <= 100000000; n *= 10) {
    b->Arg(n);
  }
}

#ifdef BENCHMARK_INTEL_DNNL
  BENCHMARK(DNNLSoftArgMax)->Apply(CharacteristicArguments)->UseManualTime();
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_p5,
    xnn_f32_raddextexp_ukernel__avx2_p5_u96,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_u40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_p5,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    (xnn_init_f32_default_params_fn) nullptr,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u96,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_p5,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    (xnn_init_f32_default_params_fn) nullptr,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc2,
    nullptr,
    xnn_f32_vmulc_minmax_ukernel__avx_u16,
    xnn_init_f32_minmax_avx_params,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  BENCHMARK_CAPTURE(TwoPassSoftMax, avx512f_p5_scalef,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx512f_p5_scalef,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    (xnn_init_f32_default_params_fn) nullptr,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc4,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx512f_p5_scalef,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    (xnn_init_f32_default_params_fn) nullptr,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc2,
    nullptr,
    xnn_f32_vmulc_minmax_ukernel__avx512f_u32,
    xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, rvv_p6_rmax_m8_exp_m4_vmulc_m8,
    xnn_f32_rmax_ukernel__rvv_u8v,
    (xnn_init_f32_default_params_fn) nullptr,
    xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v,
    nullptr,
    xnn_f32_vmulc_minmax_ukernel__rvv_u8v,
    xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckRVV)->Apply(CharacteristicArguments)->UseManualTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
