// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/vbinary.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
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
    return static_cast<xnn_float16>(dist(g));
  }
};

template <>
struct UniformDistribution<int8_t> {
  std::uniform_int_distribution<int> dist{std::numeric_limits<int8_t>::lowest(),
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
      xnn_init_qs8_add_minmax_scalar_params, &quantization, &quantization,
      &quantization);
};

template <>
struct ParamsWrapper<xnn_qu8_add_minmax_params> {
  xnn_qu8_add_minmax_params params = make_params<xnn_qu8_add_minmax_params>(
      xnn_init_qu8_add_minmax_scalar_params, &quantization, &quantization,
      &quantization);
};

template <>
struct ParamsWrapper<xnn_qs8_mul_minmax_params> {
  xnn_qs8_mul_minmax_params params = make_params<xnn_qs8_mul_minmax_params>(
      xnn_init_qs8_mul_minmax_scalar_params, &quantization, &quantization,
      &quantization);
};

template <>
struct ParamsWrapper<xnn_qu8_mul_minmax_params> {
  xnn_qu8_mul_minmax_params params = make_params<xnn_qu8_mul_minmax_params>(
      xnn_init_qu8_mul_minmax_scalar_params, &quantization, &quantization,
      &quantization);
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

  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> a(num_elements);
  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> b(num_elements);
  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> output(num_elements);
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
  state.counters["num_elements"] = benchmark::Counter(
      uint64_t(state.iterations()) * num_elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * num_elements * sizeof(T);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype,  \
                    params_type, init_params)                                \
  BENCHMARK_CAPTURE(vbinary, ukernel, arch_flags, ukernel)                   \
      ->Apply(                                                               \
          benchmark::utils::BinaryElementwiseParameters<datatype, datatype>) \
      ->UseRealTime();
#include "src/f16-vbinary/f16-vadd.inc"
#include "src/f16-vbinary/f16-vaddc.inc"
#include "src/f16-vbinary/f16-vdiv.inc"
#include "src/f16-vbinary/f16-vdivc.inc"
#include "src/f16-vbinary/f16-vmax.inc"
#include "src/f16-vbinary/f16-vmaxc.inc"
#include "src/f16-vbinary/f16-vmin.inc"
#include "src/f16-vbinary/f16-vminc.inc"
#include "src/f16-vbinary/f16-vmul.inc"
#include "src/f16-vbinary/f16-vmulc.inc"
#include "src/f16-vbinary/f16-vprelu.inc"
#include "src/f16-vbinary/f16-vpreluc.inc"
#include "src/f16-vbinary/f16-vrdivc.inc"
#include "src/f16-vbinary/f16-vrpreluc.inc"
#include "src/f16-vbinary/f16-vrsubc.inc"
#include "src/f16-vbinary/f16-vsqrdiff.inc"
#include "src/f16-vbinary/f16-vsqrdiffc.inc"
#include "src/f16-vbinary/f16-vsub.inc"
#include "src/f16-vbinary/f16-vsubc.inc"
#include "src/f32-vbinary/f32-vadd.inc"
#include "src/f32-vbinary/f32-vaddc.inc"
#include "src/f32-vbinary/f32-vcopysign.inc"
#include "src/f32-vbinary/f32-vcopysignc.inc"
#include "src/f32-vbinary/f32-vdiv.inc"
#include "src/f32-vbinary/f32-vdivc.inc"
#include "src/f32-vbinary/f32-vmax.inc"
#include "src/f32-vbinary/f32-vmaxc.inc"
#include "src/f32-vbinary/f32-vmin.inc"
#include "src/f32-vbinary/f32-vminc.inc"
#include "src/f32-vbinary/f32-vmul.inc"
#include "src/f32-vbinary/f32-vmulc.inc"
#include "src/f32-vbinary/f32-vprelu.inc"
#include "src/f32-vbinary/f32-vpreluc.inc"
#include "src/f32-vbinary/f32-vrcopysignc.inc"
#include "src/f32-vbinary/f32-vrdivc.inc"
#include "src/f32-vbinary/f32-vrpreluc.inc"
#include "src/f32-vbinary/f32-vrsubc.inc"
#include "src/f32-vbinary/f32-vsqrdiff.inc"
#include "src/f32-vbinary/f32-vsqrdiffc.inc"
#include "src/f32-vbinary/f32-vsub.inc"
#include "src/f32-vbinary/f32-vsubc.inc"
#include "src/qs8-vadd/qs8-vadd-minmax.inc"
#include "src/qs8-vaddc/qs8-vaddc-minmax.inc"
#include "src/qs8-vmul/qs8-vmul-minmax-fp32.inc"
#include "src/qs8-vmul/qs8-vmul-minmax-rndnu.inc"
#include "src/qs8-vmulc/qs8-vmulc-minmax-fp32.inc"
#include "src/qs8-vmulc/qs8-vmulc-minmax-rndnu.inc"
#include "src/qu8-vadd/qu8-vadd-minmax.inc"
#include "src/qu8-vaddc/qu8-vaddc-minmax.inc"
#include "src/qu8-vmul/qu8-vmul-minmax-fp32.inc"
#include "src/qu8-vmul/qu8-vmul-minmax-rndnu.inc"
#include "src/qu8-vmulc/qu8-vmulc-minmax-fp32.inc"
#include "src/qu8-vmulc/qu8-vmulc-minmax-rndnu.inc"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
