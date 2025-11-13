// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
#define XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/base/span.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/pipeline.h"
#include "slinky/runtime/stmt.h"

struct ynn_runtime_value : public ynn_value {
  explicit ynn_runtime_value(const ynn_value& value) : ynn_value(value) {}
  ynn_runtime_value() = default;

  slinky::buffer_expr_ptr buffer;

  void make_buffer(ynn_runtime& runtime);
  void make_buffer(ynn_runtime& runtime, slinky::expr elem_size);
};

struct ynn_runtime {
  const ynn_subgraph& subgraph;
  std::vector<ynn_runtime_value> values;
  uint32_t flags;

  slinky::eval_config eval_config;
  slinky::thread_pool* threadpool() { return eval_config.thread_pool; };

  // Symbols we've named in Slinky. We copy this from the subgraph, so we
  // inherit symbols from the subgraph, but we can add more names here without
  // corrupting the subgraph.
  slinky::node_context symbols;

  // Keep the slinky funcs and scheduling info alive until we build the pipeline
  // with them.
  std::vector<slinky::func> funcs;
  std::vector<std::unique_ptr<ynn::scheduling_info>> scheduling_info_storage;

  // This implements the logic to evaluate the symbolic shapes.
  slinky::stmt reshape_impl;

  // This is a list of global variables and their (symbolic) value that will be
  // lifted out of the pipeline.
  std::vector<std::pair<slinky::var, slinky::expr>> globals;

  // The evaluateable slinky pipeline, and the context to run it.
  slinky::pipeline pipeline;
  slinky::eval_context eval_context;

  const ynn_runtime_value& value(uint32_t id) const {
    assert(id < values.size());
    return values[id];
  }
  ynn_runtime_value& value(uint32_t id) {
    assert(id < values.size());
    return values[id];
  }

  // Make a global variable for the given expression. Deduplicates identical
  // expressions to the same variable.
  slinky::var make_global_variable(slinky::expr value,
                                   const char* prefix = "r");
  std::unique_ptr<ynn::scheduling_info> make_schedule(
      const std::vector<slinky::var>& dims, slinky::buffer_expr_ptr output,
      uint32_t output_value, slinky::span<const slinky::expr> given_splits = {},
      const slinky::expr& element_cost = 1,
      const std::vector<slinky::index_t>& loop_order = {});

  slinky::buffer_expr_ptr null_buffer();

  void schedule();

  ynn_runtime(const ynn_subgraph& subgraph, slinky::thread_pool* threadpool,
              uint32_t flags);

  ynn_status build();
  ynn_status reshape();
  ynn_status setup();
  ynn_status invoke();

  slinky::buffer_expr_ptr null_buffer_ = nullptr;
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
