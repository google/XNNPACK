// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "models.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

int FLAGS_num_threads = 1;

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

  ModelRuntime model_runtime(FLAGS_num_threads);
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

static void FP32Attention(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32Attention(state.range(0), state.range(1), state.range(2),
                                 state.range(3), state.range(4));
  });
}

static void FP16Attention(benchmark::State& state) {
  BenchmarkInvoke(
      state,
      [&state]() {
        return models::FP32Attention(state.range(0), state.range(1),
                                     state.range(2), state.range(3),
                                     state.range(4));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
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

static void QD8Attention(benchmark::State& state) {
  models::QD8AttentionWeights weights;
  BenchmarkInvoke(
      state,
      [&state, &weights]() {
        return models::QD8Attention(state.range(0), state.range(1),
                                    state.range(2), state.range(3),
                                    state.range(4), weights);
      },
      0);
}

static void QS8MobileNetV2(benchmark::State& state) {
  BenchmarkInvoke(state, models::QS8MobileNetV2);
}

static void AttentionArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"B", "T", "H", "N", "S"});
  b->Args({1, 16, 25, 24, 4});
  b->Args({1, 1536, 128, 12, 18});
  b->Args({1, 1024, 256, 4, 46});
  b->Args({1, 1792, 256, 8, 36});
  b->Args({1, 1536, 256, 6, 22});
  b->Args({1, 2048, 256, 8, 18});
  b->Args({1, 3072, 256, 16, 28});
  b->Args({1, 2304, 256, 8, 26});
  b->Args({1, 2048, 64, 32, 24});
}

BENCHMARK(FP32Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(FP16Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(FP32MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Large)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Small)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP16MobileNetV1)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Large)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Small)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QD8Attention)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(AttentionArguments);

BENCHMARK(QS8MobileNetV2)->Unit(benchmark::kMicrosecond)->UseRealTime();

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "--num_threads=", 14) == 0) {
      FLAGS_num_threads = atoi(argv[i] + 14);
      if (FLAGS_num_threads <= 0) {
        std::cerr << "Invalid --num_threads: " << FLAGS_num_threads << "\n";
        return 1;
      }
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else {
      ++i;
    }
  }
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}

