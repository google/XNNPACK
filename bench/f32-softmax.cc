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

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  // Parameters auto-tuned for a mix
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_blend,
    xnn_f32_raddextexp_ukernel__avx2_p5_x96,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_blend,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_blend,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2,
    xnn_f32_vscale_ukernel__avx_unroll32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  // Parameters auto-tuned for Broadwell
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_broadwell,
    xnn_f32_raddextexp_ukernel__avx2_p5_x96,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_broadwell,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_broadwell,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64,
    xnn_f32_vscale_ukernel__avx_unroll32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  // Parameters auto-tuned for Zen 2
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_zen2,
    xnn_f32_raddextexp_ukernel__avx2_p5_x72,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_zen2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_zen2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64,
    xnn_f32_vscale_ukernel__avx_unroll32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  // Parameters auto-tuned for Skylake
  BENCHMARK_CAPTURE(TwoPassSoftMax, avx2_skylake,
    xnn_f32_raddextexp_ukernel__avx2_p5_x64,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx2_skylake,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx2_skylake,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscale_ukernel__avx_unroll32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseManualTime();

  BENCHMARK_CAPTURE(TwoPassSoftMax, avx512f_skylake,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x144_acc3,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithRecomputing, avx512f_skylake,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
  BENCHMARK_CAPTURE(ThreePassSoftMaxWithReloading, avx512f_skylake,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscale_ukernel__avx512f_unroll64,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseManualTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
