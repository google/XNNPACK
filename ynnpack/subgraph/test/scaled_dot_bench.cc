// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Benchmarks `ynn_define_scaled_dot` on the expert projection of a Mixture of
// Experts (MoE) layer with dense (not gathered) expert weights:
//
//   output[e, t, :] = scale[e, t] * dot(a[e or 0, t, :], w[e, :, :])
//
// where `scale[e, t]` is the routing weight of token `t` for expert `e`, or 0
// if the router did not pick expert `e` for token `t`. Only `top_k` of the
// `num_experts` scales of each token are non-zero.
//
// Layouts:
// - kDense: `multiply(dot(a, w), scale)`. Computes every (expert, token) row,
//   i.e. what you get without `scaled_dot`. This is the baseline.
// - kScattered: `scaled_dot(a, w, scale)` with `a` broadcast over experts, so
//   the routed rows of an expert are scattered over the tokens.
// - kSorted: `scaled_dot(a, w, scale)` with the routed tokens of each expert
//   compacted to the first rows of its slice of `a` (`a` is [E, T, D]). This
//   models sorting tokens by expert before the dot. The gather that produces
//   this `a` is not timed.
// - kNoneActive: `scaled_dot` with all scales zero, to measure the fixed
//   overhead (mask scan, zero fill, the scale multiply).
// - kGather: gathers the weights of the routed experts of each token and dots
//   each token with its `top_k` experts, like the MoE lowering of the TFLite
//   delegate. Only run for small token counts.
//
// The `useful_FLOP` counter only counts the routed rows, so it is directly
// comparable between layouts.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

enum Layout {
  kDense = 0,
  kScattered = 1,
  kSorted = 2,
  kNoneActive = 3,
  kGather = 4,
};

// Number of distinct routings the timed loop cycles through, so that the same
// few experts do not stay resident in cache across iterations.
constexpr int kRoutingRingSize = 16;

// Returns `num_tokens * top_k` expert ids, `top_k` distinct uniformly random
// experts per token.
std::vector<int> Route(int num_tokens, int num_experts, int top_k,
                       std::mt19937& rng) {
  std::vector<int> experts(num_experts);
  std::vector<int> result(num_tokens * top_k);
  for (int t = 0; t < num_tokens; ++t) {
    std::iota(experts.begin(), experts.end(), 0);
    for (int k = 0; k < top_k; ++k) {
      std::uniform_int_distribution<int> pick(k, num_experts - 1);
      std::swap(experts[k], experts[pick(rng)]);
      result[t * top_k + k] = experts[k];
    }
  }
  return result;
}

// Returns the [E, T] scales for `routing`, either with the routed tokens of
// each expert at their token index (`sorted` = false), or compacted to the
// first rows of the expert (`sorted` = true).
std::vector<float> MakeScale(const std::vector<int>& routing, int num_tokens,
                             int num_experts, int top_k, bool sorted) {
  std::vector<float> scale(static_cast<size_t>(num_experts) * num_tokens, 0.0f);
  std::vector<int> count(num_experts, 0);
  const float weight = 1.0f / top_k;
  for (int t = 0; t < num_tokens; ++t) {
    for (int k = 0; k < top_k; ++k) {
      const int e = routing[t * top_k + k];
      const int row = sorted ? count[e] : t;
      ++count[e];
      scale[static_cast<size_t>(e) * num_tokens + row] = weight;
    }
  }
  return scale;
}

template <typename T>
void BM_MoeGatherDot(benchmark::State& state, size_t d_in, size_t d_out,
                     size_t num_experts, size_t top_k) {
  const size_t M = state.range(0);
  const int thread_count = state.range(1);
  const size_t E = num_experts;
  const size_t K = top_k;

  const uint32_t a_id = 0;
  const uint32_t index_id = 1;
  const uint32_t weight_id = 2;
  const uint32_t out_id = 3;
  SubgraphBuilder subgraph(4);
  subgraph.AddInput(type_of<T>(), {M, d_in}, a_id)
      .AddInput(ynn_type_int32, {M, K}, index_id)
      .AddInput(ynn_type_fp32, {M, K}, weight_id)
      .AddOutput(ynn_type_fp32, {M, K, 1, d_out}, out_id);

  Tensor<T> w({E, d_in, d_out});
  w.fill(static_cast<T>(1));
  uint32_t w_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), w.extents(), w_id, w.data());

  uint32_t a_4d = YNN_INVALID_VALUE_ID;
  uint32_t index_4d = YNN_INVALID_VALUE_ID;
  uint32_t weight_4d = YNN_INVALID_VALUE_ID;
  uint32_t gathered = YNN_INVALID_VALUE_ID;
  uint32_t dot_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), 4, a_4d)
      .AddTensor(ynn_type_int32, 4, index_4d)
      .AddTensor(ynn_type_fp32, 4, weight_4d)
      .AddTensor(type_of<T>(), 4, gathered)
      .AddTensor(
          type_of<T>() == ynn_type_fp32 ? ynn_type_fp32 : ynn_type_int32, 4,
          dot_id);
  subgraph.AddExpandDims({1, 2}, a_id, a_4d)
      .AddExpandDims({2, 3}, index_id, index_4d)
      .AddExpandDims({2, 3}, weight_id, weight_4d)
      .AddGather({0}, /*output_rank=*/4, w_id, index_4d, gathered)
      .AddDot(/*num_k_dims=*/1, a_4d, gathered, YNN_INVALID_VALUE_ID, dot_id)
      .AddBinary(ynn_binary_multiply, dot_id, weight_4d, out_id);

  TestScheduler scheduler(thread_count - 1);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  Tensor<T> a({M, d_in});
  a.fill(static_cast<T>(1));
  Tensor<int32_t> index({M, K});
  Tensor<float> weight({M, K});
  weight.fill(1.0f / K);
  Tensor<float> out({M, K, 1, d_out});

  std::mt19937 rng(0x5eed1234u);
  std::vector<std::vector<int>> routings(kRoutingRingSize);
  for (int i = 0; i < kRoutingRingSize; ++i) {
    routings[i] = Route(M, E, K, rng);
  }

  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(index.data(), index_id)
      .SetupExternalTensor(weight.data(), weight_id)
      .SetupExternalTensor(out.data(), out_id)
      .ReshapeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  int ring_index = 0;
  for (auto _ : state) {
    std::copy(routings[ring_index].begin(), routings[ring_index].end(),
              index.data());
    ring_index = ring_index + 1 == kRoutingRingSize ? 0 : ring_index + 1;
    runtime.InvokeRuntime();
  }
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }

  const double useful_flops_per_iter = 2.0 * d_in * d_out * M * K;
  state.counters["useful_FLOP"] =
      benchmark::Counter(state.iterations() * useful_flops_per_iter,
                         benchmark::Counter::kIsRate);
  state.counters["active_rows"] = static_cast<double>(M * K);
}

template <typename T>
void BM_MoeScaledDot(benchmark::State& state, size_t d_in, size_t d_out,
                     size_t num_experts, size_t top_k) {
  const size_t num_tokens = state.range(0);
  const int thread_count = state.range(1);
  const Layout layout = static_cast<Layout>(state.range(2));
  if (layout == kGather) {
    BM_MoeGatherDot<T>(state, d_in, d_out, num_experts, top_k);
    return;
  }

  const size_t E = num_experts;
  const size_t M = num_tokens;
  const std::vector<size_t> a_shape =
      layout == kSorted ? std::vector<size_t>{E, M, d_in}
                        : std::vector<size_t>{M, d_in};

  const uint32_t a_id = 0;
  const uint32_t scale_id = 1;
  const uint32_t out_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<T>(), a_shape, a_id)
      .AddInput(ynn_type_fp32, {E, M, 1}, scale_id)
      .AddOutput(ynn_type_fp32, {E, M, d_out}, out_id);

  Tensor<T> w({E, d_in, d_out});
  w.fill(static_cast<T>(1));
  uint32_t w_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), w.extents(), w_id, w.data());

  if (layout == kDense) {
    uint32_t dot_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<T>() == ynn_type_fp32 ? ynn_type_fp32
                                                     : ynn_type_int32,
                       3, dot_id);
    subgraph.AddDot(/*num_k_dims=*/1, a_id, w_id, YNN_INVALID_VALUE_ID, dot_id)
        .AddBinary(ynn_binary_multiply, dot_id, scale_id, out_id);
  } else {
    subgraph.AddScaledDot(/*num_k_dims=*/1, a_id, w_id, YNN_INVALID_VALUE_ID,
                          scale_id, out_id);
  }

  TestScheduler scheduler(thread_count - 1);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  Tensor<T> a(a_shape);
  a.fill(static_cast<T>(1));
  Tensor<float> scale({E, M, 1});
  Tensor<float> out({E, M, d_out});

  std::mt19937 rng(0x5eed1234u);
  std::vector<std::vector<float>> scales(kRoutingRingSize);
  size_t active_rows = 0;
  for (int i = 0; i < kRoutingRingSize; ++i) {
    std::vector<int> routing = Route(M, E, top_k, rng);
    scales[i] = MakeScale(routing, M, E, top_k, layout == kSorted);
    if (layout == kNoneActive) {
      std::fill(scales[i].begin(), scales[i].end(), 0.0f);
    }
    active_rows += std::count_if(scales[i].begin(), scales[i].end(),
                                 [](float s) { return s != 0.0f; });
  }

  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(scale.data(), scale_id)
      .SetupExternalTensor(out.data(), out_id)
      .ReshapeRuntime();
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  int index = 0;
  for (auto _ : state) {
    std::copy(scales[index].begin(), scales[index].end(), scale.data());
    index = index + 1 == kRoutingRingSize ? 0 : index + 1;
    runtime.InvokeRuntime();
  }
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }

  const double useful_flops_per_iter = 2.0 * d_in * d_out *
                                       static_cast<double>(active_rows) /
                                       kRoutingRingSize;
  state.counters["useful_FLOP"] =
      benchmark::Counter(state.iterations() * useful_flops_per_iter,
                         benchmark::Counter::kIsRate);
  state.counters["active_rows"] =
      static_cast<double>(active_rows) / kRoutingRingSize;
}

void Arguments(benchmark::Benchmark* b) {
  b->ArgNames({"tokens", "threads", "layout"});
  b->UseRealTime();
  for (int tokens : {1, 16, 128, 512}) {
    for (int threads : {1, 4}) {
      for (int layout : {kDense, kScattered, kSorted, kNoneActive}) {
        b->Args({tokens, threads, layout});
      }
      // Gathered weights have shape [tokens, top_k, d_in, d_out], which is
      // multiple GBs for large token counts.
      if (tokens <= 16) {
        b->Args({tokens, threads, kGather});
      }
    }
  }
}

void BM_Gemma700M_F32(benchmark::State& state) {
  BM_MoeScaledDot<float>(state, /*d_in=*/512, /*d_out=*/448,
                         /*num_experts=*/32, /*top_k=*/4);
}
void BM_Gemma700M_Int8(benchmark::State& state) {
  BM_MoeScaledDot<int8_t>(state, /*d_in=*/512, /*d_out=*/448,
                          /*num_experts=*/32, /*top_k=*/4);
}

void BM_Qwen_Int8(benchmark::State& state) {
  BM_MoeScaledDot<int8_t>(state, /*d_in=*/2048, /*d_out=*/1408,
                          /*num_experts=*/60, /*top_k=*/4);
}

void BM_Gemma26B_Int8(benchmark::State& state) {
  BM_MoeScaledDot<int8_t>(state, /*d_in=*/2816, /*d_out=*/704,
                          /*num_experts=*/128, /*top_k=*/8);
}

BENCHMARK(BM_Gemma700M_F32)
    ->Apply(Arguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_Gemma700M_Int8)
    ->Apply(Arguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_Qwen_Int8)
    ->Apply(Arguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_Gemma26B_Int8)
    ->Apply(Arguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);

// End to end expert projection, including combining the experts' outputs per
// token:
//
//   output[t, :] = sum_k routing_weight[t, k] *
//                  dot(a[t, :], w[expert_index[t, k], :, :])
//
// Pipelines:
// - kPipelineScattered: `scaled_dot` of the tokens with all experts, [E, T, N],
//   with the routing weights as the scale, then a sum over the experts.
// - kPipelineSorted: the tokens are bucketed by expert on the host (a counting
//   sort, timed), and gathered to [E, C, D], where C is the largest number of
//   tokens routed to one expert, rounded up to `kCapacityAlignment`. Then
//   `scaled_dot` with the routing weights of the slots as the scale (zero for
//   the padding slots), [E, C, N], a gather of each (token, k)'s slot, [T, K,
//   N], and a sum over k. C depends on the routing, so the runtime is reshaped
//   on every invocation.
// - kPipelineSortedFixedCapacity: like kPipelineSorted, but C is the largest
//   capacity of all the routings, so the runtime is not reshaped in the timed
//   loop. The difference to kPipelineSorted is the cost of reshaping.
// - kPipelineSortedUnaligned: like kPipelineSorted, but C is not rounded up,
//   so the runtime is reshaped more often, with fewer padding slots.
enum Pipeline {
  kPipelineScattered = 0,
  kPipelineSorted = 1,
  kPipelineSortedFixedCapacity = 2,
  kPipelineSortedUnaligned = 3,
};

constexpr int kCapacityAlignment = 16;

struct Dispatch {
  int capacity = 0;
  std::vector<int32_t> slot_token;   // [E, capacity]
  std::vector<float> slot_weight;    // [E, capacity]
  std::vector<int32_t> token_slot;   // [T, K], flat index into [E, capacity]
};

void MakeDispatch(const std::vector<int>& routing, int num_tokens,
                  int num_experts, int top_k, int min_capacity, int alignment,
                  Dispatch& dispatch) {
  std::vector<int> count(num_experts, 0);
  for (int e : routing) ++count[e];
  const int max_count = *std::max_element(count.begin(), count.end());
  const int capacity = std::max(
      min_capacity, (max_count + alignment - 1) / alignment * alignment);
  dispatch.capacity = capacity;
  dispatch.slot_token.assign(static_cast<size_t>(num_experts) * capacity, 0);
  dispatch.slot_weight.assign(static_cast<size_t>(num_experts) * capacity,
                              0.0f);
  dispatch.token_slot.resize(static_cast<size_t>(num_tokens) * top_k);
  std::fill(count.begin(), count.end(), 0);
  const float weight = 1.0f / top_k;
  for (int t = 0; t < num_tokens; ++t) {
    for (int k = 0; k < top_k; ++k) {
      const int e = routing[t * top_k + k];
      const int slot = e * capacity + count[e]++;
      dispatch.slot_token[slot] = t;
      dispatch.slot_weight[slot] = weight;
      dispatch.token_slot[t * top_k + k] = slot;
    }
  }
}

template <typename T>
void BM_MoePipeline(benchmark::State& state, size_t d_in, size_t d_out,
                    size_t num_experts, size_t top_k) {
  const size_t M = state.range(0);
  const int thread_count = state.range(1);
  const Pipeline pipeline = static_cast<Pipeline>(state.range(2));
  const size_t E = num_experts;
  const size_t K = top_k;
  const bool sorted = pipeline != kPipelineScattered;
  const int alignment =
      pipeline == kPipelineSortedUnaligned ? 1 : kCapacityAlignment;

  std::mt19937 rng(0x5eed1234u);
  std::vector<std::vector<int>> routings(kRoutingRingSize);
  for (int i = 0; i < kRoutingRingSize; ++i) {
    routings[i] = Route(M, E, K, rng);
  }
  int max_capacity = 0;
  Dispatch dispatch;
  for (const std::vector<int>& routing : routings) {
    MakeDispatch(routing, M, E, K, /*min_capacity=*/0, alignment, dispatch);
    max_capacity = std::max(max_capacity, dispatch.capacity);
  }
  const int min_capacity =
      pipeline == kPipelineSortedFixedCapacity ? max_capacity : 0;

  const uint32_t a_id = 0;
  const uint32_t scale_id = 1;
  const uint32_t slot_token_id = 2;
  const uint32_t token_slot_id = 3;
  const uint32_t out_id = 4;
  SubgraphBuilder subgraph(5);
  subgraph.AddInput(type_of<T>(), {M, d_in}, a_id)
      .AddOutput(ynn_type_fp32, {M, d_out}, out_id);

  Tensor<T> w({E, d_in, d_out});
  w.fill(static_cast<T>(1));
  uint32_t w_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), w.extents(), w_id, w.data());

  if (!sorted) {
    uint32_t y_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(ynn_type_fp32, {E, M, 1}, scale_id)
        .AddTensor(ynn_type_fp32, 3, y_id)
        .AddScaledDot(/*num_k_dims=*/1, a_id, w_id, YNN_INVALID_VALUE_ID,
                      scale_id, y_id)
        .AddReduce(ynn_reduce_sum, {0}, y_id, YNN_INVALID_VALUE_ID, out_id);
  } else {
    uint32_t slot_token_3d = YNN_INVALID_VALUE_ID;
    uint32_t scale_3d = YNN_INVALID_VALUE_ID;
    uint32_t a_sorted = YNN_INVALID_VALUE_ID;
    uint32_t y_id = YNN_INVALID_VALUE_ID;
    uint32_t y_flat = YNN_INVALID_VALUE_ID;
    uint32_t token_slot_3d = YNN_INVALID_VALUE_ID;
    uint32_t y_tokens = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(ynn_type_fp32, 2, scale_id)
        .AddInput(ynn_type_int32, 2, slot_token_id)
        .AddInput(ynn_type_int32, {M, K}, token_slot_id)
        .AddTensor(ynn_type_int32, 3, slot_token_3d)
        .AddTensor(ynn_type_fp32, 3, scale_3d)
        .AddTensor(type_of<T>(), 3, a_sorted)
        .AddTensor(ynn_type_fp32, 3, y_id)
        .AddTensor(ynn_type_fp32, 2, y_flat)
        .AddTensor(ynn_type_int32, 3, token_slot_3d)
        .AddTensor(ynn_type_fp32, 3, y_tokens)
        .AddExpandDims({2}, slot_token_id, slot_token_3d)
        .AddExpandDims({2}, scale_id, scale_3d)
        .AddGather({0}, /*output_rank=*/3, a_id, slot_token_3d, a_sorted)
        .AddScaledDot(/*num_k_dims=*/1, a_sorted, w_id, YNN_INVALID_VALUE_ID,
                      scale_3d, y_id)
        .AddFuseDim(0, 2, y_id, y_flat)
        .AddExpandDims({2}, token_slot_id, token_slot_3d)
        .AddGather({0}, /*output_rank=*/3, y_flat, token_slot_3d, y_tokens)
        .AddReduce(ynn_reduce_sum, {1}, y_tokens, YNN_INVALID_VALUE_ID,
                   out_id);
  }

  TestScheduler scheduler(thread_count - 1);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to create runtime");
    return;
  }

  Tensor<T> a({M, d_in});
  a.fill(static_cast<T>(1));
  Tensor<float> out({M, d_out});
  std::vector<float> scale(E * std::max<size_t>(M, max_capacity));
  std::vector<int32_t> slot_token(E * max_capacity);
  std::vector<int32_t> token_slot(M * K);
  std::vector<std::vector<float>> scattered_scales;
  if (!sorted) {
    scattered_scales.resize(routings.size());
    for (size_t i = 0; i < routings.size(); ++i) {
      scattered_scales[i] = MakeScale(routings[i], M, E, K, /*sorted=*/false);
    }
  }

  runtime.SetupExternalTensor(a.data(), a_id)
      .SetupExternalTensor(out.data(), out_id);
  int reshaped_capacity = -1;
  auto setup = [&](int ring_index) {
    if (!sorted) {
      std::copy(scattered_scales[ring_index].begin(),
                scattered_scales[ring_index].end(), scale.data());
      if (reshaped_capacity < 0) {
        runtime.SetupExternalTensor(scale.data(), scale_id).ReshapeRuntime();
        reshaped_capacity = 0;
      }
      return;
    }
    MakeDispatch(routings[ring_index], M, E, K, min_capacity, alignment,
                 dispatch);
    std::copy(dispatch.slot_weight.begin(), dispatch.slot_weight.end(),
              scale.data());
    std::copy(dispatch.slot_token.begin(), dispatch.slot_token.end(),
              slot_token.data());
    std::copy(dispatch.token_slot.begin(), dispatch.token_slot.end(),
              token_slot.data());
    if (dispatch.capacity != reshaped_capacity) {
      const size_t c = dispatch.capacity;
      runtime.ReshapeExternalTensor({E, c}, scale.data(), scale_id)
          .ReshapeExternalTensor({E, c}, slot_token.data(), slot_token_id)
          .SetupExternalTensor(token_slot.data(), token_slot_id)
          .ReshapeRuntime();
      reshaped_capacity = dispatch.capacity;
    }
  };
  setup(0);
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to reshape runtime");
    return;
  }

  int ring_index = 0;
  double reshapes = 0;
  for (auto _ : state) {
    const int capacity_before = reshaped_capacity;
    setup(ring_index);
    reshapes += capacity_before != reshaped_capacity;
    ring_index = ring_index + 1 == kRoutingRingSize ? 0 : ring_index + 1;
    runtime.InvokeRuntime();
  }
  if (runtime.Status() != ynn_status_success) {
    state.SkipWithError("Failed to invoke runtime");
    return;
  }

  const double useful_flops_per_iter = 2.0 * d_in * d_out * M * K;
  state.counters["useful_FLOP"] =
      benchmark::Counter(state.iterations() * useful_flops_per_iter,
                         benchmark::Counter::kIsRate);
  state.counters["max_capacity"] = max_capacity;
  state.counters["reshapes_per_iter"] = reshapes / state.iterations();
}

void PipelineArguments(benchmark::Benchmark* b) {
  b->ArgNames({"tokens", "threads", "pipeline"});
  b->UseRealTime();
  for (int tokens : {1, 16, 128, 512}) {
    for (int threads : {1, 4}) {
      for (int pipeline : {kPipelineScattered, kPipelineSorted,
                           kPipelineSortedFixedCapacity,
                           kPipelineSortedUnaligned}) {
        b->Args({tokens, threads, pipeline});
      }
    }
  }
}

void BM_Qwen_Int8_Pipeline(benchmark::State& state) {
  BM_MoePipeline<int8_t>(state, /*d_in=*/2048, /*d_out=*/1408,
                         /*num_experts=*/60, /*top_k=*/4);
}

void BM_Gemma26B_Int8_Pipeline(benchmark::State& state) {
  BM_MoePipeline<int8_t>(state, /*d_in=*/2816, /*d_out=*/704,
                         /*num_experts=*/128, /*top_k=*/8);
}

BENCHMARK(BM_Qwen_Int8_Pipeline)
    ->Apply(PipelineArguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);
BENCHMARK(BM_Gemma26B_Int8_Pipeline)
    ->Apply(PipelineArguments)
    ->Unit(benchmark::TimeUnit::kMicrosecond);

}  // namespace
}  // namespace ynn
