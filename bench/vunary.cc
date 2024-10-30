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

#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vunary.h"
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
  std::uniform_int_distribution<int32_t> dist{
      std::numeric_limits<int8_t>::lowest(),
      std::numeric_limits<int8_t>::max()};

  template <class Generator>
  int8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<uint8_t> {
  std::uniform_int_distribution<int32_t> dist{
      std::numeric_limits<uint8_t>::lowest(),
      std::numeric_limits<uint8_t>::max()};

  template <class Generator>
  uint8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<int16_t> {
  std::uniform_int_distribution<int32_t> dist{
      std::numeric_limits<int16_t>::lowest(),
      std::numeric_limits<int16_t>::max()};

  template <class Generator>
  int16_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<int32_t> {
  std::uniform_int_distribution<int32_t> dist{
      std::numeric_limits<int32_t>::lowest(),
      std::numeric_limits<int32_t>::max()};

  template <class Generator>
  int32_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<uint32_t> {
  std::uniform_int_distribution<uint32_t> dist{
      std::numeric_limits<uint32_t>::lowest(),
      std::numeric_limits<uint32_t>::max()};

  template <class Generator>
  uint32_t operator()(Generator& g) {
    return dist(g);
  }
};

template <typename T>
xnn_quantization_params InputQuantization(T) {
  return {0, 1.0f};
}
template <typename T>
xnn_quantization_params OutputQuantization(T) {
  return {0, 1.0f};
}
xnn_quantization_params InputQuantization(int8_t) { return {1, 0.5f}; }
xnn_quantization_params OutputQuantization(int8_t) { return {-1, 0.7f}; }
xnn_quantization_params InputQuantization(uint8_t) { return {129, 0.5f}; }
xnn_quantization_params OutputQuantization(uint8_t) { return {127, 0.7f}; }

// Microkernel function, templated on the `params` type.
template <typename TIn, typename TOut, typename UKernelParams>
using UKernelFn = void (*)(size_t, const TIn*, TOut*,
                           const UKernelParams* params);

template <typename TIn, typename TOut, typename UKernelParams>
void vunary(benchmark::State& state, uint64_t arch_flags,
            UKernelFn<TIn, TOut, UKernelParams> ukernel,
            xnn_init_unary_uparams_fn init_params,
            const xnn_unary_params* params = nullptr,
            const xnn_quantization_params& input_quantization =
                InputQuantization(TIn()),
            const xnn_quantization_params& output_quantization =
                OutputQuantization(TOut())) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  xnn_unary_uparams uparams;
  if (init_params) {
    init_params(&uparams, params, &input_quantization, &output_quantization);
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  UniformDistribution<TIn> dist;

  xnnpack::Buffer<TIn, XNN_ALLOCATION_ALIGNMENT> x(
      num_elements + XNN_EXTRA_BYTES / sizeof(TIn));
  xnnpack::Buffer<TOut, XNN_ALLOCATION_ALIGNMENT> y(num_elements);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });

  for (auto _ : state) {
    ukernel(num_elements * sizeof(TIn), x.data(), y.data(),
            (UKernelParams*)&uparams);
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
  BENCHMARK_CAPTURE(vunary, ukernel, arch_flags, ukernel, init_params)        \
      ->Apply(                                                                \
          benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)   \
      ->UseRealTime();
#include "f16-vabs/f16-vabs.h"
#include "f16-vhswish/f16-vhswish.h"
#include "f16-vneg/f16-vneg.h"
#include "f16-vrnd/f16-vrndd.h"
#include "f16-vrnd/f16-vrndne.h"
#include "f16-vrnd/f16-vrndu.h"
#include "f16-vrnd/f16-vrndz.h"
#include "f16-vrsqrt/f16-vrsqrt.h"
#include "f16-vsigmoid/f16-vsigmoid.h"
#include "f16-vsqr/f16-vsqr.h"
#include "f16-vsqrt/f16-vsqrt.h"
#include "f16-vtanh/f16-vtanh.h"
#include "f32-vabs/f32-vabs.h"
#include "f32-vgelu/f32-vgelu.h"
#include "f32-vhswish/f32-vhswish.h"
#include "f32-vlog/f32-vlog.h"
#include "f32-vneg/f32-vneg.h"
#include "f32-vrelu/f32-vrelu.h"
#include "f32-vrnd/f32-vrndd.h"
#include "f32-vrnd/f32-vrndne.h"
#include "f32-vrnd/f32-vrndu.h"
#include "f32-vrnd/f32-vrndz.h"
#include "f32-vrsqrt/f32-vrsqrt.h"
#include "f32-vsigmoid/f32-vsigmoid.h"
#include "f32-vsqr/f32-vsqr.h"
#include "f32-vsqrt/f32-vsqrt.h"
#include "f32-vtanh/f32-vtanh.h"
#undef XNN_UKERNEL_WITH_PARAMS

template <typename TIn, typename TOut, typename UKernelParams>
void velu(benchmark::State& state, uint64_t arch_flags,
          UKernelFn<TIn, TOut, UKernelParams> ukernel,
          xnn_init_unary_uparams_fn init_params) {
  xnn_unary_params params;
  params.elu.alpha = 1.0f;
  vunary(state, arch_flags, ukernel, init_params, &params);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  BENCHMARK_CAPTURE(velu, ukernel, arch_flags, ukernel, init_params)          \
      ->Apply(                                                                \
          benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)   \
      ->UseRealTime();
#include "f16-velu/f16-velu.h"
#include "f32-velu/f32-velu.h"
#undef XNN_UKERNEL_WITH_PARAMS

template <typename TIn, typename TOut, typename UKernelParams>
void vclamp(benchmark::State& state, uint64_t arch_flags,
            UKernelFn<TIn, TOut, UKernelParams> ukernel,
            xnn_init_unary_uparams_fn init_params) {
  xnn_unary_params params;
  params.clamp.min = -1.0f;
  params.clamp.max = 1.0f;
  // These kernels cannot handle changing quantization parameters.
  xnn_quantization_params input_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
  vunary(state, arch_flags, ukernel, init_params, &params, input_quantization,
         output_quantization);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  BENCHMARK_CAPTURE(vclamp, ukernel, arch_flags, ukernel, init_params)        \
      ->Apply(                                                                \
          benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)   \
      ->UseRealTime();
#include "f16-vclamp/f16-vclamp.h"
#include "f32-vclamp/f32-vclamp.h"
#include "s8-vclamp/s8-vclamp.h"
#include "u8-vclamp/u8-vclamp.h"
#undef XNN_UKERNEL_WITH_PARAMS

template <typename TIn, typename TOut, typename UKernelParams>
void vlrelu(benchmark::State& state, uint64_t arch_flags,
            UKernelFn<TIn, TOut, UKernelParams> ukernel,
            xnn_init_unary_uparams_fn init_params) {
  xnn_unary_params params;
  params.leaky_relu.negative_slope = 0.5f;
  vunary(state, arch_flags, ukernel, init_params, &params);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  BENCHMARK_CAPTURE(vlrelu, ukernel, arch_flags, ukernel, init_params)        \
      ->Apply(                                                                \
          benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)   \
      ->UseRealTime();
#include "f16-vlrelu/f16-vlrelu.h"
#include "f32-vlrelu/f32-vlrelu.h"
#include "qs8-vlrelu/qs8-vlrelu.h"
#include "qu8-vlrelu/qu8-vlrelu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile,        \
                                    vector_tile, datatype_in, datatype_out, \
                                    params_type, init_params)               \
  BENCHMARK_CAPTURE(vunary, ukernel, arch_flags, ukernel, init_params)      \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype_in,     \
                                                           datatype_out>)   \
      ->UseRealTime();
#include "f16-f32-vcvt/f16-f32-vcvt.h"
#include "f16-qs8-vcvt/f16-qs8-vcvt.h"
#include "f32-f16-vcvt/f32-f16-vcvt.h"
#include "f32-qs8-vcvt/f32-qs8-vcvt.h"
#include "f32-qu8-vcvt/f32-qu8-vcvt.h"
#include "qs16-qs8-vcvt/qs16-qs8-vcvt.h"
#include "qs8-f16-vcvt/qs8-f16-vcvt.h"
#include "qs8-f32-vcvt/qs8-f32-vcvt.h"
#include "qs8-vcvt/qs8-vcvt.h"
#include "qu8-f32-vcvt/qu8-f32-vcvt.h"
#include "qu8-vcvt/qu8-vcvt.h"
#include "s32-f32-vcvt/s32-f32-vcvt.h"
#include "u32-f32-vcvt/u32-f32-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
