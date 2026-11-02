// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
#define XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/ref_count.h"
#include "ynnpack/base/span.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"
#include "slinky/runtime/stmt.h"

#define YNN_FLAG_ENABLE_SLINKY_TRACE (1 << 31)

namespace ynn {

// The estimated duration of a task below which it is not worth running in
// parallel with other tasks: waking up threads and synchronizing with them
// costs ~5-20us per parallel loop on x86 and ARM servers.
constexpr float min_task_seconds = 16e-6f;

// Conservative estimates of the bandwidths (in bytes per second) at which one
// core streams the operands of a task from L3 or DRAM, and writes its outputs
// (to cache, which is faster). The kernel cost models are fit to L1-resident
// benchmarks, and don't account for these, which dominate small tasks: adding
// them makes the estimated duration of tasks track their measured duration
// within ~2x on x86 and ARM servers, including AMX kernels, whose cost models
// alone underestimate it by ~10x.
constexpr float task_bytes_per_second = 8e9f;
constexpr float task_output_bytes_per_second = 16e9f;

}  // namespace ynn

struct ynn_runtime_value : public ynn_value {
  explicit ynn_runtime_value(const ynn_value& value) : ynn_value(value) {}
  ynn_runtime_value() = default;

  slinky::buffer_expr_ptr buffer;

  void make_buffer(ynn_runtime& runtime);
  void make_buffer(ynn_runtime& runtime, slinky::expr elem_size);
};

struct ynn_runtime {
  ynn::ref_count<const ynn_subgraph> subgraph;
  std::vector<ynn_runtime_value> values;
  uint32_t flags;

  slinky::eval_config eval_config;
  slinky::thread_pool* threadpool() { return eval_config.thread_pool; };

  // Symbols we've named in Slinky. We copy this from the subgraph, so we
  // inherit symbols from the subgraph, but we can add more names here without
  // corrupting the subgraph.
  ynn::slinky_globals globals;

  // Keep the slinky funcs and scheduling info alive until we build the pipeline
  // with them.
  std::vector<slinky::func> funcs;
  std::vector<std::unique_ptr<ynn::scheduling_info>> scheduling_info_storage;

  // This implements the logic to evaluate the symbolic shapes.
  slinky::stmt reshape_impl;

  // The evaluateable slinky pipeline, and the context to run it.
  slinky::pipeline pipeline;
  slinky::eval_context eval_context;

  struct cost_term {
    std::vector<slinky::expr> extents;
    float seconds_per_element = 0.0f;
  };
  std::vector<cost_term> cost_terms;
  float estimated_duration = std::numeric_limits<float>::infinity();

  const ynn_runtime_value& value(uint32_t id) const {
    assert(id < values.size());
    return values[id];
  }
  ynn_runtime_value& value(uint32_t id) {
    assert(id < values.size());
    return values[id];
  }

  std::unique_ptr<ynn::scheduling_info> make_schedule(
      ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
      ynn::span<const slinky::expr> splits,
      ynn::span<const int> loop_order = {});

  std::unique_ptr<ynn::scheduling_info> make_schedule(
      ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
      const slinky::expr& element_cost,
      ynn::span<const slinky::expr> given_splits = {},
      ynn::span<const int> loop_order = {});

  slinky::buffer_expr_ptr null_buffer();

  void schedule();

  ynn_runtime(ynn::ref_count<const ynn_subgraph> subgraph,
              slinky::thread_pool* threadpool, uint32_t flags);

  ynn_status build();
  ynn_status reshape();
  ynn_status setup();
  ynn_status invoke();

  slinky::buffer_expr_ptr null_buffer_ = nullptr;
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
