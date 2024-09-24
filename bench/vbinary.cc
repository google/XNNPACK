// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/vbinary.h"

#include <algorithm>
#include <cmath>
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
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
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

template <typename Params>
struct ParamsWrapper {
  Params params;
};

xnn_quantization_params quantization = {0, 1.0f};

template <>
struct ParamsWrapper<xnn_qs8_add_minmax_params> {
  xnn_qs8_add_minmax_params params = make_params<xnn_qs8_add_minmax_params>(
      xnn_init_qs8_add_minmax_scalar_params, &quantization, &quantization, &quantization);
};

template <>
struct ParamsWrapper<xnn_qu8_add_minmax_params> {
  xnn_qu8_add_minmax_params params = make_params<xnn_qu8_add_minmax_params>(
      xnn_init_qu8_add_minmax_scalar_params, &quantization, &quantization, &quantization);
};

template <>
struct ParamsWrapper<xnn_qs8_mul_minmax_params> {
  xnn_qs8_mul_minmax_params params = make_params<xnn_qs8_mul_minmax_params>(
      xnn_init_qs8_mul_minmax_scalar_params, &quantization, &quantization, &quantization);
};

template <>
struct ParamsWrapper<xnn_qu8_mul_minmax_params> {
  xnn_qu8_mul_minmax_params params = make_params<xnn_qu8_mul_minmax_params>(
      xnn_init_qu8_mul_minmax_scalar_params, &quantization, &quantization, &quantization);
};

// Microkernel function, templated on the `params` type.
template <typename T, typename UKernelParams>
using UKernelFn = void (*)(size_t, const T*, const T*, T*,
                           const UKernelParams* params);

template <typename T, typename UKernelParams>
static void vbinary(benchmark::State& state, uint64_t arch_flags,
                    UKernelFn<T, UKernelParams> ukernel) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  UniformDistribution<T> dist;

  std::vector<T, AlignedAllocator<T, 64>> a(num_elements);
  std::vector<T, AlignedAllocator<T, 64>> b(num_elements);
  std::vector<T, AlignedAllocator<T, 64>> output(num_elements);
  std::generate(a.begin(), a.end(), [&]() { return dist(rng); });
  std::generate(b.begin(), b.end(), [&]() { return dist(rng); });

  ParamsWrapper<UKernelParams> params;
  for (auto _ : state) {
    ukernel(num_elements * sizeof(T), a.data(), b.data(), output.data(),
            &params.params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * num_elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * num_elements * sizeof(T);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  BENCHMARK_CAPTURE(vbinary, ukernel, arch_flags, ukernel)                    \
      ->Apply(                                                                \
          benchmark::utils::BinaryElementwiseParameters<datatype, datatype>)  \
      ->UseRealTime();
#include "src/f16-vbinary/f16-vadd.h"
#include "src/f16-vbinary/f16-vaddc.h"
#include "src/f16-vbinary/f16-vdiv.h"
#include "src/f16-vbinary/f16-vdivc.h"
#include "src/f16-vbinary/f16-vmax.h"
#include "src/f16-vbinary/f16-vmaxc.h"
#include "src/f16-vbinary/f16-vmin.h"
#include "src/f16-vbinary/f16-vminc.h"
#include "src/f16-vbinary/f16-vmul.h"
#include "src/f16-vbinary/f16-vmulc.h"
#include "src/f16-vbinary/f16-vprelu.h"
#include "src/f16-vbinary/f16-vpreluc.h"
#include "src/f16-vbinary/f16-vrdivc.h"
#include "src/f16-vbinary/f16-vrpreluc.h"
#include "src/f16-vbinary/f16-vrsubc.h"
#include "src/f16-vbinary/f16-vsqrdiff.h"
#include "src/f16-vbinary/f16-vsqrdiffc.h"
#include "src/f16-vbinary/f16-vsub.h"
#include "src/f16-vbinary/f16-vsubc.h"
#include "src/f32-vbinary/f32-vadd.h"
#include "src/f32-vbinary/f32-vaddc.h"
#include "src/f32-vbinary/f32-vcopysign.h"
#include "src/f32-vbinary/f32-vcopysignc.h"
#include "src/f32-vbinary/f32-vdiv.h"
#include "src/f32-vbinary/f32-vdivc.h"
#include "src/f32-vbinary/f32-vmax.h"
#include "src/f32-vbinary/f32-vmaxc.h"
#include "src/f32-vbinary/f32-vmin.h"
#include "src/f32-vbinary/f32-vminc.h"
#include "src/f32-vbinary/f32-vmul.h"
#include "src/f32-vbinary/f32-vmulc.h"
#include "src/f32-vbinary/f32-vprelu.h"
#include "src/f32-vbinary/f32-vpreluc.h"
#include "src/f32-vbinary/f32-vrcopysignc.h"
#include "src/f32-vbinary/f32-vrdivc.h"
#include "src/f32-vbinary/f32-vrpreluc.h"
#include "src/f32-vbinary/f32-vrsubc.h"
#include "src/f32-vbinary/f32-vsqrdiff.h"
#include "src/f32-vbinary/f32-vsqrdiffc.h"
#include "src/f32-vbinary/f32-vsub.h"
#include "src/f32-vbinary/f32-vsubc.h"
#include "src/qs8-vadd/qs8-vadd-minmax.h"
#include "src/qs8-vaddc/qs8-vaddc-minmax.h"
#include "src/qs8-vmul/qs8-vmul-minmax-fp32.h"
#include "src/qs8-vmul/qs8-vmul-minmax-rndnu.h"
#include "src/qs8-vmulc/qs8-vmulc-minmax-fp32.h"
#include "src/qs8-vmulc/qs8-vmulc-minmax-rndnu.h"
#include "src/qu8-vadd/qu8-vadd-minmax.h"
#include "src/qu8-vaddc/qu8-vaddc-minmax.h"
#include "src/qu8-vmul/qu8-vmul-minmax-fp32.h"
#include "src/qu8-vmul/qu8-vmul-minmax-rndnu.h"
#include "src/qu8-vmulc/qu8-vmulc-minmax-fp32.h"
#include "src/qu8-vmulc/qu8-vmulc-minmax-rndnu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
