// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <random>
#include <vector>

#include <cpuinfo.h>

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_GEMMLOWP
#include "gemmlowp/public/gemmlowp.h"
#endif  // BENCHMARK_GEMMLOWP
#ifdef BENCHMARK_RUY
#include "ruy/ruy.h"
#endif  // BENCHMARK_RUY
#include "bench/gemm.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static void GEMMBenchmark(benchmark::State& state,
  xnn_q8_gemm_ukernel_function q8gemm,
  size_t mr, size_t nr, size_t kr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::vector<uint8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<int32_t> b(nc);
  std::generate(b.begin(), b.end(), std::ref(s32rng));

  const size_t w_elements = kc_stride * nc_stride + nc_stride * sizeof(int32_t) / sizeof(uint8_t);
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint8_t) * (w_elements + c_elements));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_q8_gemm_goi_w(1 /* groups */, nc, kc, nr, kr, 127, 127, k.data(), b.data(), w.data());
  std::vector<uint8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), 0xA5);

  union xnn_q8_gemm_params quantizationParams =
    xnn_init_q8_gemm_params(127, 127, 0.75f, 127, 1, 254);

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      for (uint32_t n = 0; n < nc; n += nr) {
        const uint32_t nb = min(nc - n, nr);
        q8gemm(
          mb, nb, kc * sizeof(uint8_t),
          a.data() + m * kc, kc * sizeof(uint8_t),
          w.data() + (w_elements * buffer_index + n * (kc_stride + sizeof(int32_t))) / sizeof(uint8_t),
          c.data() + (mc * buffer_index + m) * nc + n, nc * sizeof(uint8_t), nr * sizeof(uint8_t),
          &quantizationParams);
      }
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_GEMMLOWP
struct GemmlowpOutputPipeline {
  typedef gemmlowp::VectorMap<const int32_t, gemmlowp::VectorShape::Col> ColVectorMap;
  typedef std::tuple<
      gemmlowp::OutputStageBiasAddition<ColVectorMap>,
      gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint,
      gemmlowp::OutputStageClamp,
      gemmlowp::OutputStageSaturatingCastToUint8>
      Pipeline;

  static Pipeline Make(
      const int32_t* bias_data,
      int output_rows,
      int32_t output_offset,
      int32_t output_multiplier,
      int output_shift,
      int32_t output_activation_min,
      int32_t output_activation_max)
  {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
    quantize_down_stage.result_shift = output_shift;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage, clamp_stage, saturating_cast_stage);
  }
};

static void GemmlowpBenchmark(benchmark::State& state, uint32_t threads)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  std::vector<uint8_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(u8rng));

  const size_t kElements = nc * kc;
  const size_t bElements = nc;
  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      kElements * sizeof(uint8_t) + bElements * sizeof(int32_t) + c_elements * sizeof(uint8_t));

  std::vector<uint8_t> k(kElements * num_buffers);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<int32_t> b(bElements * num_buffers);
  std::generate(b.begin(), b.end(), std::ref(s32rng));
  std::vector<uint8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), 0xA5);

  gemmlowp::MultiThreadGemmContext threadingContext;
  threadingContext.set_max_num_threads(threads);

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor> AM(a.data(), mc, kc, kc);
    gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor> BM(k.data() + buffer_index * kElements, kc, nc, kc);
    gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::RowMajor> CM(c.data() + buffer_index * c_elements, mc, nc, nc);
    const auto& outputPipeline = GemmlowpOutputPipeline::Make(b.data() + buffer_index * bElements, nc, 127, 127, 127, 0, 255);
    gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
        &threadingContext, AM, BM, &CM, 127, 127, outputPipeline);
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void gemmlowp_st(benchmark::State& state, const char* net)
{
  GemmlowpBenchmark(state, 1);
}
#endif  // BENCHMARK_GEMMLOWP


#ifdef BENCHMARK_RUY
static void RuyBenchmark(benchmark::State& state, size_t threads)
{
  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      nc * (sizeof(uint8_t) * (mc + kc) + sizeof(int32_t)));

  std::vector<uint8_t> a(mc * kc);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  std::vector<uint8_t> k(num_buffers * nc * kc);
  std::generate(k.begin(), k.end(), std::ref(u8rng));
  std::vector<int32_t> b(num_buffers * nc);
  std::generate(b.begin(), b.end(), std::ref(s32rng));
  std::vector<uint8_t> c(num_buffers * nc * mc);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Note: context must be static to avoid the cost of re-creating it for each benchmark.
  static ruy::Context context;
  context.set_max_num_threads(threads);

  ruy::Matrix<uint8_t> ruy_a;
  ruy::MakeSimpleLayout(nc, kc, ruy::Order::kRowMajor, ruy_a.mutable_layout());
  ruy_a.set_zero_point(127);
  ruy::Matrix<uint8_t> ruy_b;
  ruy::MakeSimpleLayout(kc, mc, ruy::Order::kColMajor, ruy_b.mutable_layout());
  ruy_b.set_data(a.data());
  ruy_b.set_zero_point(127);
  ruy::Matrix<uint8_t> ruy_c;
  ruy::MakeSimpleLayout(nc, mc, ruy::Order::kColMajor, ruy_c.mutable_layout());
  ruy_c.set_zero_point(127);

  ruy::MulParams<int32_t, uint8_t> mul_params;
  mul_params.set_multiplier_fixedpoint(0x40000000);

  // ruy::Context uses deferred initialization, which affects percieved GEMM performance. Initialization happens during
  // the first GEMM calls, and per Benoit Jacob it takes up to ~250 milliseconds for performance to stabilize.
  // Thus, on the first benchmark, we compute GEMM for 500 milliseconds (to be safe) without recording performance, and
  // keep the ruy::Context object initialized (by being static) between subsequent benchmarks.
  static std::once_flag warmup;
  std::call_once(warmup, [&](){
    auto start = std::chrono::steady_clock::now();
    do {
      ruy_a.set_data(k.data());
      ruy_c.set_data(c.data());
      mul_params.set_bias(b.data());

      ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.5);
  });

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - K is not in cache (for any cache level)
    // - B is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    ruy_a.set_data(k.data() + buffer_index * nc * kc);
    ruy_c.set_data(c.data() + buffer_index * mc * nc);
    mul_params.set_bias(b.data() + buffer_index * nc);

    ruy::Mul(ruy_a, ruy_b, mul_params, &context, &ruy_c);
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void ruy_st(benchmark::State& state, const char* net)
{
  RuyBenchmark(state, 1);
}
#endif  // BENCHMARK_RUY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void q8gemm_4x8__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_q8_gemm_minmax_ukernel_4x8__neon, 4, 8, 1);
  }

  static void q8gemm_8x8__neon(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_q8_gemm_minmax_ukernel_8x8__neon, 8, 8, 1);
  }

  BENCHMARK_GEMM(q8gemm_4x8__neon)
  BENCHMARK_GEMM(q8gemm_8x8__neon)
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void q8gemm_4x4c2__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_q8_gemm_minmax_ukernel_4x4c2__sse2, 4, 4, 2);
  }

  static void q8gemm_2x4c8__sse2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state, xnn_q8_gemm_minmax_ukernel_2x4c8__sse2, 2, 4, 8);
  }

  BENCHMARK_GEMM(q8gemm_4x4c2__sse2)
  BENCHMARK_GEMM(q8gemm_2x4c8__sse2)
#endif

#ifdef BENCHMARK_RUY
BENCHMARK_GEMM(ruy_st)
#endif  // BENCHMARK_RUY
#ifdef BENCHMARK_GEMMLOWP
BENCHMARK_GEMM(gemmlowp_st)
#endif  // BENCHMARK_GEMMLOWP

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
