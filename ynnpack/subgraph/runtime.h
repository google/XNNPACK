// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
#define XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_

#include <cassert>
#include <cstdint>
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

struct ynn_runtime_value : public ynn_value {
  explicit ynn_runtime_value(const ynn_value& value) : ynn_value(value) {}
  ynn_runtime_value() = default;

  slinky::buffer_expr_ptr buffer;

  // The strides required of `data`, computed by `ynn_runtime::reshape`.
  // `slinky::dim::auto_stride` indicates the stride of a dimension is not
  // required to have a particular value.
  std::vector<slinky::index_t> required_strides;

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

  struct scheduled_func {
    slinky::func func;
    ynn::scheduling_info info;

    scheduled_func(slinky::func func,  // NOLINT(google-explicit-constructor)
                   ynn::scheduling_info info = {})
        : func(std::move(func)), info(std::move(info)) {}
  };

  // Keep the slinky funcs and scheduling info alive until we build the pipeline
  // with them.
  std::vector<scheduled_func> funcs;

  // This implements the logic to evaluate the symbolic shapes.
  slinky::stmt reshape_impl;

  struct reshape_check {
    const ynn_node* node;
    const ynn_node::check* check;
  };
  std::vector<reshape_check> reshape_checks;
  std::vector<uint32_t> external_value_ids;
  std::vector<uint32_t> external_output_ids;

  // True if `reshape` has succeeded since the shape of an external value last
  // changed.
  bool is_reshaped = false;

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

  ynn::scheduling_info make_schedule(
      ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
      ynn::span<const slinky::expr> splits,
      ynn::span<const int> loop_order = {});

  ynn::scheduling_info make_schedule(
      ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
      const slinky::expr& element_cost,
      ynn::span<const slinky::expr> given_splits = {},
      ynn::span<const int> loop_order = {},
      ynn::span<const slinky::expr> alignments = {});

  slinky::buffer_expr_ptr null_buffer();

  slinky::loop_nest schedule();

  ynn_runtime(ynn::ref_count<const ynn_subgraph> subgraph,
              slinky::thread_pool* threadpool, uint32_t flags);

  ynn_status build();
  ynn_status reshape();
  ynn_status invoke();

  slinky::buffer_expr_ptr null_buffer_ = nullptr;
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_RUNTIME_H_
