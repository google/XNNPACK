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

#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/util.h"
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

// Allocate an array of n values of `type` with the value 1.
template <typename T>
std::unique_ptr<T[]> ones(size_t n) {
  auto result = std::make_unique<T[]>(n / type_element_count(type_of<T>()));
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(result.get(), i, 1);
  }
  return result;
}

template <typename Input, typename Output>
void bench(benchmark::State& state, ynn_threadpool_t threadpool, int dim0,
           int dim1, int dim2, int kdims) {
  subgraph_ptr subgraph = create_subgraph(2);

  // If we have static shapes, set them now.
  size_t input_shape[3] = {0, 0, 0};
  if (dim0 > 0) input_shape[0] = dim0;
  if (dim1 > 0) input_shape[1] = dim1;
  if (dim2 > 0) input_shape[2] = dim2;

  std::vector<int32_t> reduce_axes;
  size_t output_shape[3] = {input_shape[0], input_shape[1], input_shape[2]};
  for (int i = 0; i < 3; ++i) {
    if (((1 << i) & kdims) != 0) {
      reduce_axes.push_back(i);
      output_shape[i] = 1;
    }
  }

  uint32_t input_id = 0;
  uint32_t output_id = 1;
  if (ynn_define_tensor(subgraph.get(), type_of<Input>(), 3, &input_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_INPUT,
                        &input_id) != ynn_status_success) {
    state.SkipWithError("Failed to define input tensor");
    return;
  }
  if (ynn_define_tensor(subgraph.get(), type_of<Output>(), 3, &output_shape[0],
                        nullptr,
                        /*flags=*/YNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                        &output_id) != ynn_status_success) {
    state.SkipWithError("Failed to define output tensor");
    return;
  }

  if (ynn_define_reduce(
          subgraph.get(), ynn_reduce_sum, reduce_axes.size(),
          reduce_axes.data(), input_id, YNN_INVALID_VALUE_ID, &output_id,
          /*flags=*/YNN_NODE_FLAG_KEEP_DIMS) != ynn_status_success) {
    state.SkipWithError("Failed to define reduce node");
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
  dim0 = std::abs(dim0);
  dim1 = std::abs(dim1);
  dim2 = std::abs(dim2);

  size_t runtime_input_shape[3] = {static_cast<size_t>(dim0),
                                   static_cast<size_t>(dim1),
                                   static_cast<size_t>(dim2)};
  size_t runtime_output_shape[3] = {
      runtime_input_shape[0], runtime_input_shape[1], runtime_input_shape[2]};
  for (int axis : reduce_axes) {
    runtime_output_shape[axis] = 1;
  }

  ynn_set_external_value_shape(runtime.get(), input_id, 3,
                               &runtime_input_shape[0]);
  ynn_reshape_runtime(runtime.get());

  const size_t input_size = static_cast<size_t>(dim0) * dim1 * dim2;
  const size_t output_size = runtime_output_shape[0] * runtime_output_shape[1] *
                             runtime_output_shape[2];

  auto input = ones<Input>(input_size);
  auto output = std::make_unique<Output[]>(output_size);

  ynn_set_external_value_data(runtime.get(), input_id, input.get());
  ynn_set_external_value_data(runtime.get(), output_id, output.get());

  for (auto _ : state) {
    ynn_invoke_runtime(runtime.get());
  }

  float expected_val = 1.0f;
  const int dims[3] = {dim0, dim1, dim2};
  for (int axis : reduce_axes) {
    expected_val *= dims[axis];
  }
  const Output expected = static_cast<Output>(expected_val);

  if (std::any_of(output.get(), output.get() + output_size,
                  [expected](Output x) { return x != expected; })) {
    state.SkipWithError("Incorrect result");
  }

  const size_t input_bytes = (input_size * type_size_bytes(type_of<Input>())) /
                             type_element_count(type_of<Input>());
  const size_t output_bytes =
      (output_size * type_size_bytes(type_of<Output>())) /
      type_element_count(type_of<Output>());
  const size_t total_bytes = input_bytes + output_bytes;

  state.counters["Bytes"] = benchmark::Counter(state.iterations() * total_bytes,
                                               benchmark::Counter::kIsRate);
  state.counters["OP"] = benchmark::Counter(state.iterations() * input_size,
                                            benchmark::Counter::kIsRate);
}

void bench(benchmark::State& state, ynn_threadpool_t threadpool,
           multi_type type, int dim0, int dim1, int dim2, int kdims) {
  SwitchTwoTypes(type, [&](auto input_val, auto output_val) {
    using Input = decltype(input_val);
    using Output = decltype(output_val);
    bench<Input, Output>(state, threadpool, dim0, dim1, dim2, kdims);
  });
}

multi_type multi_type_for_input_type(ynn_type input_type) {
  switch (input_type) {
    case ynn_type_int2:
      return multi_type::int2_int32;
    case ynn_type_uint2:
      return multi_type::uint2_int32;
    case ynn_type_int4:
      return multi_type::int4_int32;
    case ynn_type_uint4:
      return multi_type::uint4_int32;
    case ynn_type_int8:
      return multi_type::int8_int32;
    case ynn_type_uint8:
      return multi_type::uint8_int32;
    case ynn_type_int32:
      return multi_type::int32;
    case ynn_type_fp8_e5m2:
      return multi_type::fp8_e5m2_fp32;
    case ynn_type_fp8_e4m3:
      return multi_type::fp8_e4m3_fp32;
    case ynn_type_fp16:
      return multi_type::fp16_fp32;
    case ynn_type_bf16:
      return multi_type::bf16_fp32;
    case ynn_type_fp32:
      return multi_type::fp32;
    case ynn_type_fp64:
      return multi_type::fp64;
    default:
      std::cerr << "Unsupported input type: " << ynn::to_string(input_type);
      std::abort();
  }
}

ynn_type output_type_for_input_type(ynn_type input_type) {
  if (input_type == ynn_type_fp64) return ynn_type_fp64;
  if (ynn::type_is_floating_point(input_type)) return ynn_type_fp32;
  return ynn_type_int32;
}

}  // namespace ynn

int parse(const char* str, int) { return std::stoi(str); }

ynn_type parse(const char* str, ynn_type) {
  if (strcmp(str, "int2") == 0) return ynn_type_int2;
  if (strcmp(str, "uint2") == 0) return ynn_type_uint2;
  if (strcmp(str, "int4") == 0) return ynn_type_int4;
  if (strcmp(str, "uint4") == 0) return ynn_type_uint4;
  if (strcmp(str, "int8") == 0) return ynn_type_int8;
  if (strcmp(str, "uint8") == 0) return ynn_type_uint8;
  if (strcmp(str, "int32") == 0) return ynn_type_int32;
  if (strcmp(str, "fp8_e5m2") == 0) return ynn_type_fp8_e5m2;
  if (strcmp(str, "fp8_e4m3") == 0) return ynn_type_fp8_e4m3;
  if (strcmp(str, "fp16") == 0) return ynn_type_fp16;
  if (strcmp(str, "bf16") == 0) return ynn_type_bf16;
  if (strcmp(str, "fp32") == 0) return ynn_type_fp32;
  if (strcmp(str, "fp64") == 0) return ynn_type_fp64;
  return ynn_type_invalid;
}

int parse_kdims(const char* str) {
  int mask = 0;
  for (const char* p = str; *p != '\0'; ++p) {
    if (*p >= '0' && *p <= '9') {
      mask |= (1 << (*p - '0'));
    }
  }
  return mask;
}

template <typename T>
void parse_list(const char* str, std::vector<T>& result) {
  std::stringstream ss(str);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    result.push_back(parse(segment.c_str(), T{}));
  }
}

void parse_list_kdims(const char* str, std::vector<int>& result) {
  std::stringstream ss(str);
  std::string segment;
  while (std::getline(ss, segment, ',')) {
    result.push_back(parse_kdims(segment.c_str()));
  }
}

void usage(const char* name) {
  std::cout << "Usage: " << name << " [options]\n";
  std::cout << R"(
Options:
  --thread_count=N
  --type=t1,t2,...  (int2, uint2, int4, uint4, int8, uint8, fp8_e5m2, fp8_e4m3, fp16, bf16, fp32, fp64)
  --shape=d0,d1,d2
  -d0=d0_1,d0_2,...
  -d1=d1_1,d1_2,...
  -d2=d2_1,d2_2,...
  -kdims=r1,r2,...  (e.g., 0, 1, 01, 012)

Notes:
  Multiple --type, --shape, -d0, -d1, -d2, and -kdims options are allowed.
  These options form lists. The Cartesian product of the -d0, -d1, -d2, and
  -kdims lists are registered. If --shape is given, the Cartesian product
  of --shape and -kdims is registered. The full set of benchmarks is then the
  Cartesian product of the shape/dimension configurations and the --type options.

  If a shape value is positive, it is a static shape. If it is negative, it is a
  dynamic shape of the same magnitude.

  --type specifies the input type. Output type is int32 for integral input types,
  fp32 for floating-point input types (except fp64 which outputs fp64).

  -kdims specifies reduction dimensions as digit strings (e.g., 012 reduces dimensions 0, 1, and 2).
)";
}

int main(int argc, char** argv) {
  constexpr unsigned max_threads = 32;
  int thread_count = std::min(max_threads, std::thread::hardware_concurrency());
  std::vector<int> dim0s;
  std::vector<int> dim1s;
  std::vector<int> dim2s;
  std::vector<int> kdims_list;
  std::vector<std::array<int, 3>> shapes;
  std::vector<ynn_type> types;
  benchmark::Initialize(&argc, argv);

  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "-d0=", 4) == 0) {
      parse_list(argv[i] + 4, dim0s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-d1=", 4) == 0) {
      parse_list(argv[i] + 4, dim1s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-d2=", 4) == 0) {
      parse_list(argv[i] + 4, dim2s);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else if (strncmp(argv[i], "-kdims=", 7) == 0) {
      parse_list_kdims(argv[i] + 7, kdims_list);
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
    types = {
        ynn_type_int2,  ynn_type_int4,     ynn_type_int8,
        ynn_type_uint8, ynn_type_fp8_e5m2, ynn_type_fp8_e4m3,
        ynn_type_fp16,  ynn_type_bf16,     ynn_type_fp32,
    };
  }

  if (thread_count < 1) {
    usage(argv[0]);
    return -1;
  }

  if (!dim0s.empty() || !dim1s.empty() || !dim2s.empty()) {
    if (dim0s.empty()) dim0s = {1};
    if (dim1s.empty()) dim1s = {1};
    if (dim2s.empty()) dim2s = {1};
  } else if (shapes.empty()) {
    // If there is no shape specified, use a default shape.
    shapes.push_back({256, 256, 256});
  }

  if (kdims_list.empty()) {
    kdims_list = {
        parse_kdims("0"),   parse_kdims("1"),  parse_kdims("2"),
        parse_kdims("01"),  parse_kdims("02"), parse_kdims("12"),
        parse_kdims("012"),
    };
  }

  ynn::TestScheduler scheduler(thread_count - 1);
  ynn::threadpool_ptr threadpool =
      ynn::create_threadpool(scheduler.scheduler(), &scheduler);

  for (ynn_type input_type : types) {
    ynn::multi_type type = ynn::multi_type_for_input_type(input_type);
    if (type == static_cast<ynn::multi_type>(-1)) {
      usage(argv[0]);
      return -1;
    }

    std::stringstream name("reduce_");
    name << ynn::to_string(input_type) << "_"
         << ynn::to_string(ynn::output_type_for_input_type(input_type));
    auto* reduce_bench = benchmark::RegisterBenchmark(
        name.str(), [=, &threadpool](benchmark::State& state) {
          const int dim0 = state.range(0);
          const int dim1 = state.range(1);
          const int dim2 = state.range(2);
          const int kdims = state.range(3);
          ynn::bench(state, threadpool.get(), type, dim0, dim1, dim2, kdims);
        });
    reduce_bench->ArgNames({"dim0", "dim1", "dim2", "kdims"});
    reduce_bench->UseRealTime();
    reduce_bench->MeasureProcessCPUTime();
    for (const auto& shape : shapes) {
      for (int r : kdims_list) {
        reduce_bench->Args({shape[0], shape[1], shape[2], r});
      }
    }
    for (int d0 : dim0s) {
      for (int d1 : dim1s) {
        for (int d2 : dim2s) {
          for (int r : kdims_list) {
            reduce_bench->Args({d0, d1, d2, r});
          }
        }
      }
    }
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
