// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "bench/subgraph/models.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

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
                            uint32_t extra_flags = 0) {
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
  if (!model_runtime.CreateRuntime(FLAGS_xnn_runtime_flags | extra_flags)) {
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

  int num_iters = FLAGS_benchmark_min_iters;
  while (state.KeepRunningBatch(num_iters)) {
    for (int iter = 0; iter < num_iters; iter++) {
      benchmark::utils::WipePthreadpoolL2Caches(state,
                                                model_runtime.threadpool);
      if (!model_runtime.Invoke()) {
        state.SkipWithError("failed to invoke runtime");
        return;
      }
    }
    num_iters = 1;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void FP32Attention(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                 state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP16Attention(benchmark::State& state) {
  BenchmarkInvoke(
      state,
      [&state]() {
        return models::FP32Attention(FLAGS_batch_size, state.range(0),
                                     state.range(1), state.range(2),
                                     state.range(3));
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
  BenchmarkInvoke(state, [&state, &weights]() {
    return models::QD8Attention(FLAGS_batch_size, state.range(0),
                                state.range(1), state.range(2), state.range(3),
                                weights);
  });
}

static void QS8MobileNetV2(benchmark::State& state) {
  BenchmarkInvoke(state, models::QS8MobileNetV2);
}

static void FP32Elementwise(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32Elementwise(/*batch_size=*/state.range(0),
                                   /*num_elements=*/state.range(1),
                                   /*depth=*/state.range(2));
  });
}

static void FP32LayerNorm(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32LayerNorm(state.range(0), state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void FP32SoftmaxDecomp(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32Softmax(state.range(0), state.range(1), state.range(2),
                               state.range(3), /*use_softmax=*/false);
  });
}

static void FP32SoftmaxFused(benchmark::State& state) {
  BenchmarkInvoke(state, [&state]() {
    return models::FP32Softmax(state.range(0), state.range(1), state.range(2),
                               state.range(3), /*use_softmax=*/true);
  });
}

static void FP32DepthwiseSeparable(benchmark::State& state) {
  models::FP32DepthwiseSeparableWeights weights;
  BenchmarkInvoke(state, [&state, &weights]() {
    return models::FP32DepthwiseSeparable(state.range(0), state.range(1),
                                          state.range(2), state.range(3),
                                          state.range(4), weights);
  });
}

static void AttentionArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"T", "H", "N", "S"});
  b->Args({16, 25, 24, 4});
  b->Args({1536, 128, 12, 18});
  b->Args({1024, 256, 4, 46});
  b->Args({1792, 256, 8, 36});
  b->Args({1536, 256, 6, 22});
  b->Args({2048, 256, 8, 18});
  b->Args({3072, 256, 16, 28});
  b->Args({2304, 256, 8, 26});
  b->Args({2048, 64, 32, 24});
}

static void LayerNormArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

static void SoftmaxArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

static void DepthwiseSeparableArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"W", "H", "KW", "CI", "CO"});

  // Mobilenet v2-ish
  b->Args({112, 112, 3, 32, 16});
  b->Args({56, 56, 3, 96, 24});
  b->Args({28, 28, 3, 144, 32});
  b->Args({14, 14, 3, 192, 64});
  b->Args({14, 14, 3, 384, 96});
  b->Args({14, 14, 3, 576, 160});
  b->Args({7, 7, 3, 960, 320});

  // Bigger
  b->Args({512, 512, 3, 128, 128});
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

BENCHMARK(FP32Elementwise)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->ArgNames({"B", "N", "D"})
    ->Args({1024, 1024, 6})
    ->Args({1024, 1024, 10})
    ->Args({1024, 1024, 18})
    ->Args({1024, 1024, 34});

BENCHMARK(FP32LayerNorm)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(LayerNormArguments);

BENCHMARK(FP32SoftmaxDecomp)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(SoftmaxArguments);

BENCHMARK(FP32SoftmaxFused)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(SoftmaxArguments);

BENCHMARK(FP32DepthwiseSeparable)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(DepthwiseSeparableArguments);

XNN_BENCHMARK_MAIN();
