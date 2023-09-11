// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/models.h>


static void End2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory)
{
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_threads = state.range(0);
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  auto execution_plan = model_factory(threadpool.get());
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), threadpool.get());
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void FP32MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV1);
}

static void FP32MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV2);
}

static void FP32MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3Large);
}

static void FP32MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3Small);
}

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
static void FP32MobileNetV3SmallFused(benchmark::State& state) {
  End2EndBenchmark(state, models::FP32MobileNetV3SmallFused);
}
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

static void FP32Sparse80MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV1(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV2(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV3Large(0.8f, threadpool);
  });
}

static void FP32Sparse80MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP32SparseMobileNetV3Small(0.8f, threadpool);
  });
}

static void FP16MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV1);
}

static void FP16MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV2);
}

static void FP16MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV3Large);
}

static void FP16MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::FP16MobileNetV3Small);
}

static void FP16Sparse80MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP16SparseMobileNetV1(0.8f, threadpool);
  });
}

static void FP16Sparse80MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP16SparseMobileNetV2(0.8f, threadpool);
  });
}

static void FP16Sparse80MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP16SparseMobileNetV3Large(0.8f, threadpool);
  });
}

static void FP16Sparse80MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, [](pthreadpool_t threadpool) {
    return models::FP16SparseMobileNetV3Small(0.8f, threadpool);
  });
}

static void QC8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QC8MobileNetV1);
}

static void QC8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QC8MobileNetV2);
}

static void QS8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QS8MobileNetV1);
}

static void QS8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QS8MobileNetV2);
}

static void QU8MobileNetV1(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV1);
}

static void QU8MobileNetV2(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV2);
}

static void QU8MobileNetV3Large(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV3Large);
}

static void QU8MobileNetV3Small(benchmark::State& state) {
  End2EndBenchmark(state, models::QU8MobileNetV3Small);
}

BENCHMARK(FP32MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP32Sparse80MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse80MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse80MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP32Sparse80MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP16MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(FP16Sparse80MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16Sparse80MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16Sparse80MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(FP16Sparse80MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QC8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(QC8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QS8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(QS8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

BENCHMARK(QU8MobileNetV1)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(QU8MobileNetV2)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(QU8MobileNetV3Large)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
BENCHMARK(QU8MobileNetV3Small)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();

#if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
BENCHMARK(FP32MobileNetV3SmallFused)->Apply(benchmark::utils::MultiThreadingParameters)->Unit(benchmark::kMicrosecond)->UseRealTime();
#endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
