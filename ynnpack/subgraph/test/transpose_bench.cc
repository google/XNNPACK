// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
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

#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;
using threadpool_ptr =
    std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>;

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

// Allocate storage for n elements of T (T may be a packed sub-byte type).
template <typename T>
std::unique_ptr<T[]> allocate(size_t n) {
  const size_t elem_count = type_element_count(type_of<T>());
  return std::make_unique<T[]>((n + elem_count - 1) / elem_count);
}

template <typename T>
void bench(benchmark::State& state, ynn_threadpool_t threadpool, int dim0,
           int dim1, int dim2, const std::array<int, 3>& perm) {
  subgraph_ptr subgraph = create_subgraph(2);

  // If we have static shapes, set them now. A dynamic dimension of the input
  // makes the corresponding (permuted) dimension of the output dynamic.
  size_t input_shape[3] = {0, 0, 0};
  if (dim0 > 0) input_shape[0] = dim0;
  if (dim1 > 0) input_shape[1] = dim1;
  if (dim2 > 0) input_shape[2] = dim2;
  size_t output_shape[3];
  int32_t permutation[3];
  for (int d = 0; d < 3; ++d) {
    output_shape[d] = input_shape[perm[d]];
    permutation[d] = perm[d];
  }

  uint32_t input_id = 0;
  uint32_t output_id = 1;
  if (ynn_define_tensor(subgraph.get(), type_of<T>(), 3, &input_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &input_id) != ynn_status_success) {
    state.SkipWithError("Failed to define input tensor");
    return;
  }
  if (ynn_define_tensor(subgraph.get(), type_of<T>(), 3, &output_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                        &output_id) != ynn_status_success) {
    state.SkipWithError("Failed to define output tensor");
    return;
  }

  if (ynn_define_static_transpose(subgraph.get(), 3, &permutation[0], input_id,
                                  &output_id,
                                  /*flags=*/0) != ynn_status_success) {
    state.SkipWithError("Failed to define transpose node");
    return;
  }

  if (ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0) !=
      ynn_status_success) {
    state.SkipWithError("Failed to optimize subgraph");
    return;
  }

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);
  if (!runtime) {
    state.SkipWithError("Failed to create ynnpack runtime");
    return;
  }

  // A negative shape indicates a dynamic shape of the same magnitude.
  const size_t dims[3] = {static_cast<size_t>(std::abs(dim0)),
                          static_cast<size_t>(std::abs(dim1)),
                          static_cast<size_t>(std::abs(dim2))};
  const size_t size = dims[0] * dims[1] * dims[2];

  ynn_set_external_value_shape(runtime.get(), input_id, 3, &dims[0]);
  if (ynn_reshape_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  auto input = allocate<T>(size);
  auto output = allocate<T>(size);
  // Materialize the input pages so the benchmark doesn't read the zero page.
  memset(input.get(), 1, type_size_bytes(type_of<T>(), size));

  ynn_set_external_value_data(runtime.get(), input_id, input.get());
  ynn_set_external_value_data(runtime.get(), output_id, output.get());

  // Warm up (and materialize the output pages) outside of the timed loop.
  if (ynn_invoke_runtime(runtime.get()) != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  const size_t total_bytes = 2 * type_size_bytes(type_of<T>(), size);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * total_bytes,
                                               benchmark::Counter::kIsRate);
}

template <typename Fn>
bool switch_bits(int bits, Fn&& fn) {
  switch (bits) {
    case 2:
      fn(int2x4());
      return true;
    case 4:
      fn(int4x2());
      return true;
    case 8:
      fn(int8_t());
      return true;
    case 16:
      fn(half());
      return true;
    case 32:
      fn(float());
      return true;
    case 64:
      fn(double());
      return true;
    default:
      return false;
  }
}

void bench(benchmark::State& state, ynn_threadpool_t threadpool, int bits,
           int dim0, int dim1, int dim2, const std::array<int, 3>& perm) {
  bool ok = switch_bits(bits, [&](auto type_val) {
    bench<decltype(type_val)>(state, threadpool, dim0, dim1, dim2, perm);
  });
  if (!ok) {
    state.SkipWithError("Unsupported bit width");
  }
}

}  // namespace ynn

int parse(const char* str, int) { return std::stoi(str); }

std::string parse(const char* str, std::string) { return std::string(str); }

// Verifies that a permutation string has exactly 3 digits and is a valid
// permutation over axes 0, 1, 2.
bool valid_perm(const std::string& str, std::array<int, 3>& perm) {
  if (str.size() != 3) return false;
  for (int i = 0; i < 3; ++i) {
    if (str[i] < '0' || str[i] > '2') return false;
    perm[i] = str[i] - '0';
  }
  std::array<int, 3> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  return sorted == std::array<int, 3>{0, 1, 2};
}

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
  --thread_count=N
  --bits=b1,b2,...  (element size in bits, e.g. 2, 4, 8, 16, 32, 64)
  --shape=d0,d1,d2
  -perm=p1,p2,...   (3-digit permutation strings over the axes, e.g. 021, 210;
                     012 is the identity and measures the aliased copy path)

Notes:
  Multiple --bits, --shape, and -perm options are allowed. These options form
  lists. The registered benchmarks are the Cartesian product of the shapes,
  permutations, and bit widths.

  If a shape value is positive, it is a static shape. If it is negative, it is
  a dynamic shape of the same magnitude.

  The output type of the transpose is the same as the input type.
)";
}

int main(int argc, char** argv) {
  constexpr unsigned max_threads = 32;
  int thread_count = std::min(max_threads, std::thread::hardware_concurrency());
  std::vector<std::array<int, 3>> shapes;
  std::vector<std::string> perms;
  std::vector<int> bits;
  benchmark::Initialize(&argc, argv);

  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "-perm=", 6) == 0) {
      parse_list(argv[i] + 6, perms);
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
    } else if (strncmp(argv[i], "--bits=", 7) == 0) {
      parse_list(argv[i] + 7, bits);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--thread_count=", 15) == 0) {
      thread_count = std::stoi(argv[i] + 15);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--benchmark_", 12) == 0 ||
               strncmp(argv[i], "-benchmark_", 11) == 0) {
      i++;
    } else {
      usage(argv[0]);
      return -1;
    }
  }

  if (bits.empty()) {
    bits = {2, 4, 8, 16, 32, 64};
  }

  if (thread_count < 1) {
    usage(argv[0]);
    return -1;
  }

  if (shapes.empty()) {
    shapes.push_back({256, 256, 256});
  }

  if (perms.empty()) {
    // All non-identity permutations.
    perms = {"021", "102", "120", "201", "210"};
  }
  for (const std::string& p : perms) {
    std::array<int, 3> perm;
    if (!valid_perm(p, perm)) {
      usage(argv[0]);
      return -1;
    }
  }

  ynn::TestScheduler scheduler(thread_count - 1);
  ynn::threadpool_ptr threadpool =
      ynn::create_threadpool(scheduler.scheduler(), &scheduler);

  for (int b : bits) {
    for (const std::string& p : perms) {
      std::array<int, 3> perm;
      valid_perm(p, perm);
      std::stringstream name;
      name << "transpose_" << b << "bit/perm:" << p;
      auto* transpose_bench = benchmark::RegisterBenchmark(
          name.str(), [=, &threadpool](benchmark::State& state) {
            const int dim0 = state.range(0);
            const int dim1 = state.range(1);
            const int dim2 = state.range(2);
            ynn::bench(state, threadpool.get(), b, dim0, dim1, dim2, perm);
          });
      transpose_bench->ArgNames({"dim0", "dim1", "dim2"});
      transpose_bench->UseRealTime();
      transpose_bench->MeasureProcessCPUTime();
      for (const auto& shape : shapes) {
        transpose_bench->Args({shape[0], shape[1], shape[2]});
      }
    }
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
