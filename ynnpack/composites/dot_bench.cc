// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

#define BENCH_ASSERT_SUCCESS(call)                       \
  do {                                                   \
    ynn_status status = (call);                          \
    if (status != ynn_status_success) {                  \
      state.SkipWithError(#call " failed with status " + \
                          std::to_string(status));       \
      return;                                            \
    }                                                    \
  } while (0)

subgraph_ptr create_subgraph(size_t num_external_values, uint32_t flags = 0) {
  ynn_subgraph_t subgraph = nullptr;
  ynn_create_subgraph(num_external_values, flags, &subgraph);
  return subgraph_ptr(subgraph, &ynn_delete_subgraph);
}

runtime_ptr create_runtime(ynn_subgraph_t subgraph,
                           ynn_threadpool_t threadpool = nullptr,
                           uint32_t flags = 0) {
  ynn_runtime_t runtime = nullptr;
  ynn_create_runtime(subgraph, threadpool, flags, &runtime);
  return runtime_ptr(runtime, &ynn_delete_runtime);
}

threadpool_ptr create_threadpool(ynn_scheduler_t scheduler,
                                 void* scheduler_context, uint32_t flags = 0) {
  ynn_threadpool_t threadpool = nullptr;
  ynn_create_threadpool(scheduler, scheduler_context, flags, &threadpool);
  return threadpool_ptr(threadpool, &ynn_delete_threadpool);
}

std::unique_ptr<float[]> ones(size_t n) {
  auto result = std::make_unique<float[]>(n);
  for (size_t i = 0; i < n; ++i) {
    result[i] = 1.0f;
  }
  return result;
}

void bench_standard(benchmark::State& state, ynn_threadpool_t threadpool, int m,
                    int n, int k) {
  subgraph_ptr subgraph = create_subgraph(3);

  size_t a_shape[2] = {0, 0};
  size_t b_shape[2] = {0, 0};
  size_t output_shape[2] = {0, 0};
  if (m > 0) a_shape[0] = output_shape[0] = m;
  if (n > 0) b_shape[1] = output_shape[1] = n;
  if (k > 0) a_shape[1] = b_shape[0] = k;

  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, &a_shape[0], nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id));
  BENCH_ASSERT_SUCCESS(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, &b_shape[0], nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id));
  BENCH_ASSERT_SUCCESS(ynn_define_tensor(
      subgraph.get(), ynn_type_fp32, 2, &output_shape[0], nullptr,
      /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  BENCH_ASSERT_SUCCESS(ynn_define_dot(subgraph.get(), /*num_k_dims=*/1, a_id,
                                      b_id, YNN_INVALID_VALUE_ID, &output_id,
                                      0));

  BENCH_ASSERT_SUCCESS(
      ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0));

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);
  if (!runtime) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  m = std::abs(m);
  n = std::abs(n);
  k = std::abs(k);

  a_shape[0] = m;
  a_shape[1] = k;
  b_shape[0] = k;
  b_shape[1] = n;
  BENCH_ASSERT_SUCCESS(ynn_set_external_value_shape(runtime.get(), a_id,
                                                    /*rank=*/2, &a_shape[0]));
  BENCH_ASSERT_SUCCESS(ynn_set_external_value_shape(runtime.get(), b_id,
                                                    /*rank=*/2, &b_shape[0]));
  BENCH_ASSERT_SUCCESS(ynn_reshape_runtime(runtime.get()));

  auto a = ones(m * k);
  auto b = ones(k * n);
  auto output = std::make_unique<float[]>(m * n);

  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), a_id, a.get()));
  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), b_id, b.get()));
  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), output_id, output.get()));

  for (auto _ : state) {
    BENCH_ASSERT_SUCCESS(ynn_invoke_runtime(runtime.get()));
  }

  for (size_t i = 0; i < m * n; ++i) {
    if (output[i] != k) {
      state.SkipWithError("Incorrect result");
      break;
    }
  }

  const size_t ops = static_cast<size_t>(m) * n * k * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

template <size_t NumA, size_t NumB, int MaxSumIndex = -1>
void bench_dot_sum(benchmark::State& state, ynn_threadpool_t threadpool, int m,
                   int n, int k) {
  subgraph_ptr subgraph = create_subgraph(3);

  size_t a_shape[2] = {0, 0};
  size_t b_shape[2] = {0, 0};
  size_t output_shape[2] = {0, 0};
  if (m > 0) a_shape[0] = output_shape[0] = m;
  if (n > 0) b_shape[1] = output_shape[1] = n;
  if (k > 0) a_shape[1] = b_shape[0] = k;

  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, &a_shape[0], nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id));
  BENCH_ASSERT_SUCCESS(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, &b_shape[0], nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id));
  BENCH_ASSERT_SUCCESS(ynn_define_tensor(
      subgraph.get(), ynn_type_fp32, 2, &output_shape[0], nullptr,
      /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  // Define splits
  std::vector<uint32_t> a_values(NumA, YNN_INVALID_VALUE_ID);
  for (size_t i = 0; i < NumA; ++i) {
    BENCH_ASSERT_SUCCESS(ynn_define_tensor(subgraph.get(), ynn_type_bf16, 2,
                                           &a_shape[0], nullptr, 0,
                                           &a_values[i]));
  }
  BENCH_ASSERT_SUCCESS(define_convert_f32_to_bf16_sum(
      subgraph.get(), a_id, NumA, a_values.data(), 0));

  std::vector<uint32_t> b_values(NumB, YNN_INVALID_VALUE_ID);
  for (size_t i = 0; i < NumB; ++i) {
    BENCH_ASSERT_SUCCESS(ynn_define_tensor(subgraph.get(), ynn_type_bf16, 2,
                                           &b_shape[0], nullptr, 0,
                                           &b_values[i]));
  }
  BENCH_ASSERT_SUCCESS(define_convert_f32_to_bf16_sum(
      subgraph.get(), b_id, NumB, b_values.data(), 0));

  // Call define_dot_sum
  BENCH_ASSERT_SUCCESS(define_dot_sum(
      subgraph.get(), /*num_k_dims=*/1, NumA, a_values.data(), NumB,
      b_values.data(), YNN_INVALID_VALUE_ID, output_id, 0, MaxSumIndex));

  BENCH_ASSERT_SUCCESS(
      ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0));

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);
  if (!runtime) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  m = std::abs(m);
  n = std::abs(n);
  k = std::abs(k);

  a_shape[0] = m;
  a_shape[1] = k;
  b_shape[0] = k;
  b_shape[1] = n;
  BENCH_ASSERT_SUCCESS(ynn_set_external_value_shape(runtime.get(), a_id,
                                                    /*rank=*/2, &a_shape[0]));
  BENCH_ASSERT_SUCCESS(ynn_set_external_value_shape(runtime.get(), b_id,
                                                    /*rank=*/2, &b_shape[0]));
  BENCH_ASSERT_SUCCESS(ynn_reshape_runtime(runtime.get()));

  auto a = ones(m * k);
  auto b = ones(k * n);
  auto output = std::make_unique<float[]>(m * n);

  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), a_id, a.get()));
  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), b_id, b.get()));
  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), output_id, output.get()));

  for (auto _ : state) {
    BENCH_ASSERT_SUCCESS(ynn_invoke_runtime(runtime.get()));
  }

  for (size_t i = 0; i < m * n; ++i) {
    if (std::isnan(output[i]) || std::abs(output[i] - k) > k * 0.1f) {
      state.SkipWithError("Incorrect result: expected " + std::to_string(k) +
                          ", got " + std::to_string(output[i]));
      break;
    }
  }

  const size_t ops = static_cast<size_t>(m) * n * k * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

template <size_t BlockSize, ynn_type TypeOfB = ynn_type_int4,
          ynn_type TypeOfBScale = ynn_type_bf16>
void bench_blockwise(benchmark::State& state, ynn_threadpool_t threadpool,
                     int m, int n, int k) {
  if (std::abs(k) % BlockSize != 0) {
    state.SkipWithError("k must be a multiple of the block size");
    return;
  }
  const size_t num_blocks = std::abs(k) / BlockSize;

  subgraph_ptr subgraph = create_subgraph(2);

  size_t a_shape[2] = {0, 0};
  if (m > 0) a_shape[0] = m;
  if (k > 0) a_shape[1] = k;

  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, &a_shape[0], nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id));
  size_t output_shape[2] = {0, 0};
  if (m > 0) output_shape[0] = m;
  if (n > 0) output_shape[1] = n;
  BENCH_ASSERT_SUCCESS(ynn_define_tensor(
      subgraph.get(), ynn_type_fp32, 2, &output_shape[0], nullptr,
      /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  // B and its blockwise scales are static.
  const size_t b_elements = static_cast<size_t>(std::abs(k)) * std::abs(n);
  std::vector<uint8_t> b_data(b_elements / ynn::type_element_count(TypeOfB));
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<uint8_t>((i * 0x9E5F) >> 3);
  }
  const size_t b_dims[2] = {static_cast<size_t>(std::abs(k)),
                            static_cast<size_t>(std::abs(n))};
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(ynn_define_tensor(subgraph.get(), TypeOfB, 2, &b_dims[0],
                                         b_data.data(),
                                         YNN_VALUE_FLAG_COPY_DATA, &b_id));

  auto b_scale_data = ones(std::abs(n) * num_blocks);
  const size_t b_scale_dims[2] = {static_cast<size_t>(std::abs(n)), num_blocks};
  uint32_t b_scale_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(ynn_define_tensor(
      subgraph.get(), TypeOfBScale, 2, &b_scale_dims[0], b_scale_data.get(),
      YNN_VALUE_FLAG_COPY_DATA_FP32, &b_scale_id));

  // Dynamically quantize A, like the LLM int8/int4 weight-only paths do.
  int32_t reduce_axis = -1;
  uint32_t min_max_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(ynn_define_reduce(
      subgraph.get(), ynn_reduce_min_max, 1, &reduce_axis, a_id,
      YNN_INVALID_VALUE_ID, &min_max_id, YNN_NODE_FLAG_KEEP_DIMS));
  uint32_t a_zp_id = YNN_INVALID_VALUE_ID;
  uint32_t a_scale_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(ynn_define_dynamic_quantization(
      subgraph.get(), min_max_id, ynn_type_int8, &a_zp_id, &a_scale_id, 0));
  uint32_t quantized_a_id = YNN_INVALID_VALUE_ID;
  BENCH_ASSERT_SUCCESS(ynn_define_quantize(subgraph.get(), a_id, ynn_type_int8,
                                           a_zp_id, a_scale_id, &quantized_a_id,
                                           0));

  BENCH_ASSERT_SUCCESS(
      define_blockwise_dot(subgraph.get(), quantized_a_id, a_zp_id, a_scale_id,
                           b_id, YNN_INVALID_VALUE_ID, b_scale_id, BlockSize,
                           YNN_INVALID_VALUE_ID, ynn_type_fp32, output_id, 0));

  BENCH_ASSERT_SUCCESS(
      ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0));

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);
  if (!runtime) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  m = std::abs(m);
  n = std::abs(n);
  k = std::abs(k);

  a_shape[0] = m;
  a_shape[1] = k;
  BENCH_ASSERT_SUCCESS(ynn_set_external_value_shape(runtime.get(), a_id,
                                                    /*rank=*/2, &a_shape[0]));

  auto a = std::make_unique<float[]>(static_cast<size_t>(m) * k);
  for (size_t i = 0; i < static_cast<size_t>(m) * k; ++i) {
    a[i] = static_cast<float>(i % 63) * 0.125f - 4.0f;
  }
  auto output = std::make_unique<float[]>(static_cast<size_t>(m) * n);

  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), a_id, a.get()));
  BENCH_ASSERT_SUCCESS(
      ynn_set_external_value_data(runtime.get(), output_id, output.get()));
  BENCH_ASSERT_SUCCESS(ynn_reshape_runtime(runtime.get()));

  for (auto _ : state) {
    BENCH_ASSERT_SUCCESS(ynn_invoke_runtime(runtime.get()));
  }

  for (size_t i = 0; i < static_cast<size_t>(m) * n; ++i) {
    if (std::isnan(output[i]) || std::isinf(output[i])) {
      state.SkipWithError("Non-finite result");
      break;
    }
  }

  const size_t ops = static_cast<size_t>(m) * n * k * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

}  // namespace
}  // namespace ynn

int parse(const char* str, int) { return std::stoi(str); }
std::string parse(const char* str, std::string) { return std::string(str); }

template <typename T>
void parse_list(const char* str, std::vector<T>& result) {
  std::stringstream ss(str);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    result.push_back(parse(segment.c_str(), T{}));
  }
}

void usage(const char* name) {
  std::cout << "Usage: " << name << " [options]\n";
  std::cout << R"(
Options:
  --benchmarks=b1,b2,... (standard, bench_dot_sum, blockwise, all)
  --thread_count=N
  --shape=m,n,k
  -m=m1,m2,...
  -n=n1,n2,...
  -k=k1,k2,...

Notes:
  Multiple --benchmarks, --shape, -m, -n, -k options are allowed. These options
  form lists. The Cartesian product of the -m, -n, and -k lists are added to the
  --shape list.
)";
}

int main(int argc, char** argv) {
  constexpr unsigned max_threads = 32;
  int thread_count = std::min(max_threads, std::thread::hardware_concurrency());
  std::vector<int> ms;
  std::vector<int> ns;
  std::vector<int> ks;
  std::vector<std::array<int, 3>> shapes;
  std::vector<std::string> benchmark_types;
  benchmark::Initialize(&argc, argv);

  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "-m=", 3) == 0) {
      parse_list(argv[i] + 3, ms);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-n=", 3) == 0) {
      parse_list(argv[i] + 3, ns);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-k=", 3) == 0) {
      parse_list(argv[i] + 3, ks);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--shape=", 8) == 0) {
      std::vector<int> shape;
      parse_list(argv[i] + 8, shape);
      if (shape.size() != 3) {
        usage(argv[0]);
        return -1;
      }
      shapes.push_back({shape[0], shape[1], shape[2]});
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--benchmarks=", 13) == 0) {
      parse_list(argv[i] + 13, benchmark_types);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--thread_count=", 15) == 0) {
      thread_count = std::stoi(argv[i] + 15);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else {
      usage(argv[0]);
      return -1;
    }
  }

  if (thread_count < 1) {
    usage(argv[0]);
    return -1;
  }

  bool enable_standard = false;
  bool enable_dot_sum = false;
  bool enable_blockwise = false;

  if (benchmark_types.empty()) {
    enable_standard = true;
    enable_dot_sum = true;
    enable_blockwise = true;
  } else {
    for (const auto& btype : benchmark_types) {
      if (btype == "all") {
        enable_standard = true;
        enable_dot_sum = true;
        enable_blockwise = true;
      } else if (btype == "standard") {
        enable_standard = true;
      } else if (btype == "dot_sum") {
        enable_dot_sum = true;
      } else if (btype == "blockwise") {
        enable_blockwise = true;
      } else {
        usage(argv[0]);
        return -1;
      }
    }
  }

  if ((ms.empty() || ns.empty() || ks.empty()) && shapes.empty()) {
    if (!ms.empty() || !ns.empty() || !ks.empty()) {
      usage(argv[0]);
      return -1;
    }
    shapes.push_back({256, 256, 256});
  }

  // `thread_count` (from --thread_count) is the total number of threads that
  // should run the work. The runtime's invoking thread participates as a
  // worker, so the scheduler only needs `thread_count - 1` background threads.
  ynn::TestScheduler scheduler(thread_count - 1);
  ynn::threadpool_ptr threadpool =
      ynn::create_threadpool(scheduler.scheduler(), &scheduler);

  auto register_bench = [&](const std::string& name, auto bench_fn) {
    auto* b = benchmark::RegisterBenchmark(
        name, [=, &threadpool](benchmark::State& state) {
          const int m = state.range(0);
          const int n = state.range(1);
          const int k = state.range(2);
          bench_fn(state, threadpool.get(), m, n, k);
        });
    b->ArgNames({"m", "n", "k"});
    b->UseRealTime();
    b->MeasureProcessCPUTime();
    for (const auto& shape : shapes) {
      b->Args({shape[0], shape[1], shape[2]});
    }
    for (int m : ms) {
      for (int n : ns) {
        for (int k : ks) {
          b->Args({m, n, k});
        }
      }
    }
  };

  if (enable_standard) {
    register_bench("fp32", ynn::bench_standard);
  }
  if (enable_dot_sum) {
    register_bench("bf16x1", ynn::bench_dot_sum<1, 1>);
    register_bench("bf16x3", ynn::bench_dot_sum<2, 2, 2>);
    register_bench("bf16x6", ynn::bench_dot_sum<3, 3, 3>);
    register_bench("bf16x9", ynn::bench_dot_sum<3, 3, 6>);
  }
  if (enable_blockwise) {
    register_bench("blockwise_int8_bs32",
                   ynn::bench_blockwise<32, ynn_type_int8>);
    register_bench("blockwise_int4_bs32", ynn::bench_blockwise<32>);
    register_bench("blockwise_int4_bs256", ynn::bench_blockwise<256>);
  }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
