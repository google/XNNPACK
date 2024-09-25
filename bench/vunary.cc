// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/vunary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vhswish.h"
#include "xnnpack/vlrelu.h"
#include <benchmark/benchmark.h>

template <typename T>
struct UniformDistribution {
  std::uniform_real_distribution<T> dist{-10.0f, 10.0f};

  template <class Generator>
  T operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<xnn_float16> {
  std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

  template <class Generator>
  xnn_float16 operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<int8_t> {
  std::uniform_int_distribution<int> dist{
      std::numeric_limits<int8_t>::lowest(),
      std::numeric_limits<int8_t>::max()};

  template <class Generator>
  int8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<uint8_t> {
  std::uniform_int_distribution<int> dist{
      std::numeric_limits<uint8_t>::lowest(),
      std::numeric_limits<uint8_t>::max()};

  template <class Generator>
  uint8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <typename T, typename InitFn, typename... Args>
T make_params(InitFn init_fn, Args... args) {
  T result;
  init_fn(&result, args...);
  return result;
}

template <typename TIn, typename Params>
struct Config {
  Params params;
};

template <>
struct Config<xnn_float16, xnn_f16_minmax_params> {
  xnn_f16_minmax_params params = {{-1.0f, 1.0f}};
};

template <>
struct Config<float, xnn_f32_minmax_params> {
  xnn_f32_minmax_params params = {{-1.0f, 1.0f}};
};

template <>
struct Config<xnn_float16, xnn_f16_elu_params> {
  xnn_f16_elu_params params = {{1.0f, 1.0f, 1.0f}};
};

template <>
struct Config<float, xnn_f32_elu_params> {
  xnn_f32_elu_params params = {{1.0f, 1.0f, 1.0f}};
};

template <>
struct Config<xnn_float16, xnn_f16_lrelu_params> {
  xnn_f16_lrelu_params params = {{0.01f}};
};

template <>
struct Config<float, xnn_f32_lrelu_params> {
  xnn_f32_lrelu_params params = {{0.01f}};
};

template <>
struct Config<int8_t, xnn_s8_minmax_params> {
  xnn_s8_minmax_params params = {{-100, 100}};
};

template <>
struct Config<uint8_t, xnn_u8_minmax_params> {
  xnn_u8_minmax_params params = {{0, 200}};
};

template <>
struct Config<int8_t, xnn_qs8_lrelu_params> {
  xnn_qs8_lrelu_params params = make_params<xnn_qs8_lrelu_params>(
      xnn_init_qs8_lrelu_scalar_params, 0.1f, 1.0f, 1, 1);
};

template <>
struct Config<uint8_t, xnn_qu8_lrelu_params> {
  xnn_qu8_lrelu_params params = make_params<xnn_qu8_lrelu_params>(
      xnn_init_qu8_lrelu_scalar_params, 0.1f, 1.0f, 1, 1);
};

template <>
struct Config<int8_t, xnn_qs8_hswish_params> {
  xnn_qs8_hswish_params params = make_params<xnn_qs8_hswish_params>(
      xnn_init_qs8_hswish_scalar_params, 0, 0, 1.0f, 1.0f);
};

template <>
struct Config<uint8_t, xnn_qu8_hswish_params> {
  xnn_qu8_hswish_params params = make_params<xnn_qu8_hswish_params>(
      xnn_init_qu8_hswish_scalar_params, 0, 0, 1.0f, 1.0f);
};

// Microkernel function, templated on the `params` type.
template <typename TIn, typename TOut, typename UKernelParams>
using UKernelFn = void (*)(size_t, const TIn*, TOut*,
                           const UKernelParams* params);

template <typename TIn, typename TOut, typename UKernelParams>
void vunary(benchmark::State& state, uint64_t arch_flags,
            UKernelFn<TIn, TOut, UKernelParams> ukernel) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  Config<TOut, UKernelParams> config;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  UniformDistribution<TIn> dist;

  std::vector<TIn, AlignedAllocator<TIn, 64>> x(num_elements);
  std::vector<TOut, AlignedAllocator<TOut, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  std::fill(y.begin(), y.end(), 0);

  for (auto _ : state) {
    ukernel(num_elements * sizeof(TOut), x.data(), y.data(), &config.params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      num_elements * (sizeof(TIn) + sizeof(TOut));
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  BENCHMARK_CAPTURE(vunary, ukernel, arch_flags, ukernel)                     \
      ->Apply(                                                                \
          benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)   \
      ->UseRealTime();
#include "src/f16-vabs/f16-vabs.h"
#include "src/f16-vclamp/f16-vclamp.h"
#include "src/f16-velu/f16-velu.h"
#include "src/f16-vhswish/f16-vhswish.h"
#include "src/f16-vlrelu/f16-vlrelu.h"
#include "src/f16-vneg/f16-vneg.h"
#include "src/f16-vrnd/f16-vrndd.h"
#include "src/f16-vrnd/f16-vrndne.h"
#include "src/f16-vrnd/f16-vrndu.h"
#include "src/f16-vrnd/f16-vrndz.h"
#include "src/f16-vrsqrt/f16-vrsqrt.h"
#include "src/f16-vsigmoid/f16-vsigmoid.h"
#include "src/f16-vsqr/f16-vsqr.h"
#include "src/f16-vsqrt/f16-vsqrt.h"
#include "src/f16-vtanh/f16-vtanh.h"
#include "src/f32-vabs/f32-vabs.h"
#include "src/f32-vclamp/f32-vclamp.h"
#include "src/f32-velu/f32-velu.h"
#include "src/f32-vgelu/f32-vgelu.h"
#include "src/f32-vhswish/f32-vhswish.h"
#include "src/f32-vlog/f32-vlog.h"
#include "src/f32-vlrelu/f32-vlrelu.h"
#include "src/f32-vneg/f32-vneg.h"
#include "src/f32-vrelu/f32-vrelu.h"
#include "src/f32-vrnd/f32-vrndd.h"
#include "src/f32-vrnd/f32-vrndne.h"
#include "src/f32-vrnd/f32-vrndu.h"
#include "src/f32-vrnd/f32-vrndz.h"
#include "src/f32-vrsqrt/f32-vrsqrt.h"
#include "src/f32-vsigmoid/f32-vsigmoid.h"
#include "src/f32-vsqr/f32-vsqr.h"
#include "src/f32-vsqrt/f32-vsqrt.h"
#include "src/f32-vtanh/f32-vtanh.h"
#include "src/qs8-vhswish/qs8-vhswish.h"
#include "src/qs8-vlrelu/qs8-vlrelu.h"
#include "src/qu8-vhswish/qu8-vhswish.h"
#include "src/qu8-vlrelu/qu8-vlrelu.h"
#include "src/s8-vclamp/s8-vclamp.h"
#include "src/u8-vclamp/u8-vclamp.h"
#undef XNN_UKERNEL_WITH_PARAMS

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
