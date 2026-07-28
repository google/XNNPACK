// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
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

#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include <benchmark/benchmark.h>

namespace ynn {

// TODO(dsharlet): We should have a minimal C++ wrapper API in ynnpack.h...
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

// Allocate an array of n values of `type` with the value 1.
template <typename T>
std::unique_ptr<T[]> ones(size_t n) {
  auto result = std::make_unique<T[]>(n / type_element_count(type_of<T>()));
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(result.get(), i, 1);
  }
  return result;
}

template <typename A, typename B, typename Output>
void bench(benchmark::State& state, ynn_threadpool_t threadpool,
           bool transpose_b, bool static_b, int m, int n, int k) {
  subgraph_ptr subgraph = create_subgraph(3);

  // If we have static shapes, set them now.
  size_t a_shape[2] = {0, 0};
  size_t b_shape[2] = {0, 0};
  size_t output_shape[2] = {0, 0};
  if (m > 0) a_shape[0] = output_shape[0] = m;
  if (n > 0) b_shape[1] = output_shape[1] = n;
  if (k > 0) a_shape[1] = b_shape[0] = k;
  if (transpose_b) {
    std::swap(b_shape[0], b_shape[1]);
  }

  auto b = ones<B>(std::abs(k) * std::abs(n));

  uint32_t a_id = 0;
  uint32_t b_id = 1;
  uint32_t output_id = 2;
  ynn_define_tensor(subgraph.get(), type_of<A>(), 2, &a_shape[0], nullptr,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id);
  if (static_b) {
    ynn_define_tensor(subgraph.get(), type_of<B>(), 2, &b_shape[0], b.get(),
                      /*flags=*/0, &b_id);
  } else {
    ynn_define_tensor(subgraph.get(), type_of<B>(), 2, &b_shape[0], nullptr,
                      /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id);
  }
  ynn_define_tensor(subgraph.get(), type_of<Output>(), 2, nullptr, nullptr,
                    /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);

  uint32_t b_dot_id = b_id;
  if (transpose_b) {
    b_dot_id = YNN_INVALID_VALUE_ID;
    ynn_define_tensor(subgraph.get(), type_of<B>(), 2, nullptr, nullptr,
                      /*flags=*/0, &b_dot_id);
    const int32_t perm[2] = {1, 0};
    ynn_define_static_transpose(subgraph.get(), 2, perm, b_id, &b_dot_id, 0);
  }

  ynn_define_dot(subgraph.get(), /*num_k_dims=*/1, a_id, b_dot_id,
                 YNN_INVALID_VALUE_ID, &output_id, 0);

  ynn_optimize_subgraph(subgraph.get(), threadpool, /*flags=*/0);

  runtime_ptr runtime = create_runtime(subgraph.get(), threadpool);

  // A negative shape indicates a dynamic shape of the same magnitude, from here
  // on we want the dynamic value.
  m = std::abs(m);
  n = std::abs(n);
  k = std::abs(k);

  // If the shapes were not static, we need to set them now.
  a_shape[0] = m;
  a_shape[1] = k;
  b_shape[0] = k;
  b_shape[1] = n;
  if (transpose_b) {
    std::swap(b_shape[0], b_shape[1]);
  }
  ynn_set_external_value_shape(runtime.get(), a_id, /*rank=*/2, &a_shape[0]);
  if (!static_b) {
    ynn_set_external_value_shape(runtime.get(), b_id, /*rank=*/2, &b_shape[0]);
  }
  ynn_reshape_runtime(runtime.get());

  // Create buffers for the inputs and output. We use inputs of ones so we can
  // quickly check the result is plausibly correct.
  auto a = ones<A>(m * k);
  auto output = std::make_unique<Output[]>(m * n);

  ynn_set_external_value_data(runtime.get(), a_id, a.get());
  if (!static_b) {
    ynn_set_external_value_data(runtime.get(), b_id, b.get());
  }
  ynn_set_external_value_data(runtime.get(), output_id, output.get());

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  // Check that the result is expected. We filled the inputs with 1, so the
  // result should be k.
  const Output expected = static_cast<Output>(k);
  if (std::any_of(output.get(), output.get() + m * n,
                  [expected](Output x) { return x != expected; })) {
    state.SkipWithError("Incorrect result");
  }

  const size_t ops = static_cast<size_t>(m) * n * k * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

void bench(benchmark::State& state, ynn_threadpool_t threadpool,
           multi_type type, bool transpose_b, bool static_b, int m, int n,
           int k) {
  SwitchThreeTypes(type, [&](auto a, auto b, auto output) {
    bench<decltype(a), decltype(b), decltype(output)>(
        state, threadpool, transpose_b, static_b, m, n, k);
  });
}

}  // namespace ynn

int parse(const char* str, int) { return std::stoi(str); }

bool parse(const char* str, bool) {
  return strcmp(str, "true") == 0 || strcmp(str, "1") == 0 ||
         strcmp(str, "t") == 0;
}

ynn::multi_type parse(const char* str, ynn::multi_type) {
  if (!strcmp(str, "int8_int2")) return ynn::multi_type::int8_int2_int32;
  if (!strcmp(str, "int8_int4")) return ynn::multi_type::int8_int4_int32;
  if (!strcmp(str, "int8")) return ynn::multi_type::int8_int8_int32;
  if (!strcmp(str, "fp8_e4m3")) return ynn::multi_type::fp8_e4m3_fp8_e4m3_fp32;
  if (!strcmp(str, "fp8_e5m2")) return ynn::multi_type::fp8_e5m2_fp8_e5m2_fp32;
  if (!strcmp(str, "bf16")) return ynn::multi_type::bf16_bf16_fp32;
  if (!strcmp(str, "fp16")) return ynn::multi_type::fp16_fp16_fp32;
  if (!strcmp(str, "fp32")) return ynn::multi_type::fp32;

  return static_cast<ynn::multi_type>(-1);
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
  --type=t1,t2,...     (int8, int8xint4, int8xint2, bf16, fp16, fp32)
  --transpose_b=b1,... (0, 1, false, true)
  --static_b=b1,...    (0, 1, false, true)
  --shape=m,n,k
  -m=m1,m2,...
  -n=n1,n2,...
  -k=k1,k2,...

Notes:
  Multiple --type, --transpose_b, --shape, -m, -n, -k options are allowed. These
  options form lists. The Cartesian product of the -m, -n, and -k lists are added
  to the --shape list. The full set of benchmarks is then the Cartesian product
  of the --shape, --type, --transpose_b, and --static_b lists.

  If a shape value is positive, it is a static shape. If it is negative, it is a
  dynamic shape of the same magnitude. If static_b is true, the B shape must be
  // static.

  --type indicates the type of the inputs and outputs.
  --transpose_b transposes the B matrix input.
)";
}

int main(int argc, char** argv) {
  // Our thread pool isn't very good with a large number of threads.
  constexpr unsigned max_threads = 32;
  int thread_count = std::min(max_threads, std::thread::hardware_concurrency());
  std::vector<int> ms;
  std::vector<int> ns;
  std::vector<int> ks;
  std::vector<std::array<int, 3>> shapes;
  std::vector<ynn::multi_type> types;
  std::vector<bool> transpose_bs;
  std::vector<bool> static_bs;
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
    } else if (strncmp(argv[i], "--type=", 7) == 0) {
      parse_list(argv[i] + 7, types);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--transpose_b=", 14) == 0) {
      const char* eq = strchr(argv[i], '=');
      parse_list(eq + 1, transpose_bs);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strcmp(argv[i], "--transpose_b") == 0) {
      transpose_bs.push_back(true);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "--static_b=", 11) == 0) {
      const char* eq = strchr(argv[i], '=');
      parse_list(eq + 1, static_bs);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strcmp(argv[i], "--static_b") == 0) {
      static_bs.push_back(true);
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

  if (types.empty()) {
    types = {ynn::multi_type::int8_int8_int32, ynn::multi_type::int8_int4_int32,
             ynn::multi_type::int8_int2_int32, ynn::multi_type::bf16_bf16_fp32,
             ynn::multi_type::fp16_fp16_fp32,  ynn::multi_type::fp32};
  }

  if (transpose_bs.empty()) {
    transpose_bs = {false};
  }

  if (static_bs.empty()) {
    static_bs = {false};
  }

  if (thread_count < 1) {
    usage(argv[0]);
    return -1;
  }

  if ((ms.empty() || ns.empty() || ks.empty()) && shapes.empty()) {
    if (!ms.empty() || !ns.empty() || !ks.empty()) {
      usage(argv[0]);
      return -1;
    }
    // If there is no shape, use a default.
    shapes.push_back({256, 256, 256});
  }

  // `thread_count` (from --thread_count) is the total number of threads that
  // should run the work. The runtime's invoking thread participates as a
  // worker, so the scheduler only needs `thread_count - 1` background threads.
  ynn::TestScheduler scheduler(thread_count - 1);
  ynn::threadpool_ptr threadpool =
      ynn::create_threadpool(scheduler.scheduler(), &scheduler);

  for (ynn::multi_type type : types) {
    if (static_cast<int>(type) < 0) {
      usage(argv[0]);
      return -1;
    }

    for (bool transpose_b : transpose_bs) {
      for (bool static_b : static_bs) {
        std::stringstream name("dot_");
        name << ynn::to_string(type);
        if (transpose_b) {
          name << "_transpose_b";
        }
        auto* dot_bench = benchmark::RegisterBenchmark(
            name.str(), [=, &threadpool](benchmark::State& state) {
              const int m = state.range(0);
              const int n = state.range(1);
              const int k = state.range(2);
              ynn::bench(state, threadpool.get(), type, transpose_b, static_b,
                         m, n, k);
            });
        dot_bench->ArgNames({"m", "n", "k"});
        dot_bench->UseRealTime();
        dot_bench->MeasureProcessCPUTime();
        for (const auto& shape : shapes) {
          dot_bench->Args({shape[0], shape[1], shape[2]});
        }
        for (int m : ms) {
          for (int n : ns) {
            for (int k : ks) {
              dot_bench->Args({m, n, k});
            }
          }
        }
      }
    }
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
