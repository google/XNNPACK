// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "models.h"
#include "bench/utils.h"
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

struct ModelRuntime {
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> model;
  pthreadpool_t threadpool = nullptr;
  xnn_runtime_t runtime = nullptr;
  std::vector<xnn_external_value> external_values;

  explicit ModelRuntime(int num_threads) : model(nullptr, xnn_delete_subgraph) {
    xnn_delete_runtime(runtime);
    threadpool = pthreadpool_create(num_threads);
  }

  ~ModelRuntime() {
    if (runtime) {
      xnn_delete_runtime(runtime);
    }
    if (threadpool) {
      pthreadpool_destroy(threadpool);
    }
    for (xnn_external_value& i : external_values) {
      xnn_release_simd_memory(i.data);
    }
  }

  bool CreateModel(std::function<xnn_subgraph_t()> model_factory) {
    model.reset(model_factory());
    if (!model) {
      return false;
    }
    for (uint32_t i = 0; i < model->num_values; ++i) {
      if ((model->values[i].flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                                     XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
        continue;
      }
      // Make a buffer for this external value.
      size_t size = xnn_tensor_get_size(&model->values[i]) + XNN_EXTRA_BYTES;
      external_values.push_back(
          xnn_external_value{i, xnn_allocate_zero_simd_memory(size)});
    }
    return model != nullptr;
  }

  bool CreateRuntime(uint32_t flags) {
    assert(!runtime);
    return xnn_status_success == xnn_create_runtime_v4(model.get(), nullptr,
                                                       nullptr, threadpool,
                                                       flags, &runtime);
  }
  bool ReshapeRuntime() {
    return xnn_status_success == xnn_reshape_runtime(runtime);
  }

  bool SetupRuntime() {
    return xnn_status_success == xnn_setup_runtime_v2(runtime,
                                                      external_values.size(),
                                                      external_values.data());
  }

  bool Invoke() { return xnn_status_success == xnn_invoke_runtime(runtime); }
};

static void BenchmarkInvoke(benchmark::State& state,
                            std::function<xnn_subgraph_t()> model_factory,
                            uint32_t flags = 0) {
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  ModelRuntime model_runtime(state.range(0));
  if (!model_runtime.CreateModel(model_factory)) {
    state.SkipWithError("failed to create model");
    return;
  }

  // TODO(dsharlet): We should have benchmarks of these steps too.
  if (!model_runtime.CreateRuntime(flags)) {
    state.SkipWithError("failed to create runtime");
    return;
  }

  if (!model_runtime.ReshapeRuntime()) {
    state.SkipWithError("failed to reshape runtime");
    return;
  }

  if (!model_runtime.SetupRuntime()) {
    state.SkipWithError("failed to setup runtime");
    return;
  }

  for (auto _ : state) {
    if (!model_runtime.Invoke()) {
      state.SkipWithError("failed to invoke runtime");
      return;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void FP32MobileNetV1(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV1);
}

static void FP32MobileNetV2(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV2);
}

static void FP32MobileNetV3Large(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV3Large);
}

static void FP32MobileNetV3Small(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV3Small);
}

static void FP16MobileNetV1(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV1,
                  XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV2(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV2,
                  XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Large(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV3Large,
                  XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void FP16MobileNetV3Small(benchmark::State& state) {
  BenchmarkInvoke(state, models::FP32MobileNetV3Small,
                  XNN_FLAG_FORCE_FP16_INFERENCE);
}

static void QS8MobileNetV2(benchmark::State& state) {
  BenchmarkInvoke(state, models::QS8MobileNetV2);
}

BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP16MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QS8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
