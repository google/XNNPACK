// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/runtime.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifdef YNN_ENABLE_PERFETTO
#include "ynnpack/subgraph/perfetto.h"
#endif
#include "ynnpack/base/build_config.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/tensor.h"
#include "slinky/base/arithmetic.h"
#include "slinky/base/span.h"
#include "slinky/base/thread_pool.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

void ynn_runtime_value::make_buffer(ynn_runtime& runtime,
                                    slinky::expr elem_size) {
  if (!buffer) {
    if (!symbol.defined()) {
      symbol = runtime.symbols.insert_unique(name());
    }
  }
  buffer = ynn::make_buffer_expr(symbol, rank(), std::move(elem_size));
  assert(buffer->sym() == symbol);
}

void ynn_runtime_value::make_buffer(ynn_runtime& runtime) {
  make_buffer(runtime, ynn::type_size_bytes(type));
}

slinky::var ynn_runtime::make_global_variable(slinky::expr value,
                                              const char* prefix) {
  value = slinky::simplify(value);
  assert(value.defined());

  auto i = std::find_if(globals.begin(), globals.end(),
                        [&](const auto& j) { return match(j.second, value); });
  if (i == globals.end()) {
    slinky::var r = symbols.insert_unique(prefix);
    globals.push_back(std::make_pair(r, value));
    return r;
  } else {
    return i->first;
  }
}

std::unique_ptr<ynn::scheduling_info> ynn_runtime::make_schedule(
    const std::vector<slinky::var>& dims, const slinky::buffer_expr_ptr output,
    uint32_t output_value, slinky::span<const slinky::expr> given_splits,
    const slinky::expr& element_cost) {
  auto sched = std::make_unique<ynn::scheduling_info>();

  int max_threads = threadpool() ? threadpool()->thread_count() : 1;
  const std::vector<slinky::expr>& output_extents = value(output_value).extents;
  // Enough tasks to have good load balancing.
  slinky::index_t target_task_count = max_threads > 1 ? max_threads * 2 : 1;

  // Area is selected such that tiles fit better into cache, this is a
  // constant for now, but we could add a more advanced logic based on
  // hardware info.
  assert(dims.size() == output->rank() || dims.size() + 1 == output->rank());
  // For min_max reductions dims.size() + 1 == output.rank().
  // Otherwise, dims.size() == output->rank().
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return sched;
  }
  assert(output->rank() == output_extents.size());

  slinky::expr tile_area = slinky::ceil_div(slinky::expr(32768 * 4),
                                            output->elem_size() * element_cost);
  std::vector<slinky::expr> splits(rank);
  slinky::expr tile_area_so_far = 1;
  for (int d = 0; d < rank; ++d) {
    if (!output_extents[d].defined()) continue;
    if (d < given_splits.size()) {
      splits[d] = given_splits[d];
    } else {
      slinky::expr s = slinky::simplify(slinky::max(
          1, slinky::min(tile_area / tile_area_so_far, output_extents[d])));
      s = make_global_variable(s, "s");
      splits[d] = s;
    }
    if (splits[d].defined()) {
      tile_area_so_far = slinky::simplify(tile_area_so_far * splits[d]);
    } else {
      tile_area_so_far = slinky::simplify(tile_area_so_far * output_extents[d]);
    }
  }

  std::vector<slinky::expr> workers(rank);
  slinky::expr threads_so_far = 1;

  for (int d = rank - 1; d >= 0; --d) {
    if (max_threads == 1) {
      workers[d] = slinky::loop::serial;
    } else if (output_extents[d].defined() && splits[d].defined()) {
      slinky::expr w =
          slinky::ceil_div(slinky::expr(target_task_count), threads_so_far);
      w = make_global_variable(w, "w");

      workers[d] = slinky::simplify(slinky::select::make(
          w > 1, slinky::loop::parallel, slinky::loop::serial));

      threads_so_far = slinky::simplify(threads_so_far *
                                        ceil_div(output_extents[d], splits[d]));
    }
  }

  for (int d = 0; d < rank; ++d) {
    if (output_extents[d].defined() && splits[d].defined()) {
      sched->loop_splits.push_back({dims[d], splits[d], workers[d], d});
    }
  }

  sched->base_buffer_id = output_value;

  // Schedule the output buffer to be stored at the same level as it's
  // computed at.
  ynn::scheduled_buffer sched_output_buffer = {output, 0};
  sched->scheduled_buffers.push_back(std::move(sched_output_buffer));

  return sched;
}

namespace {

template <typename T, typename Target>
bool find_n(const T* data, size_t size, const Target& x) {
  for (size_t i = 0; i < size; ++i) {
    if (data[i] == x) {
      return true;
    }
  }
  return false;
}

}  // namespace

// Logically this function has multiple separate blocks:
// 1) computing a list of possible compute_at locations for a given function.
//    This is a very concrete thing and doesn't require any heuristics.
// 2) using the set of locations from 1) decide if we want for this function
//    to be computed at root or at one of the existing loops based on the
//    available information such as scheduling_info attached to the function
//.   or forward bounds.
// 3) if we decide to share the loop location possibly update loop parameters
//    such as step based on the specific of the given function (this is pretty
//.   much a no-op right now and is solely defined by a "parent" function of
//    the loop, but we can use it in the future to figure out, for example, a
//    step size based on the *all* functions which were assigned to the loop).
// 4) potentially add new loop(s) into the loop nest based on a given
//    function.
// 5) based on compute locations computed in 1) - 4), set up the
//    func-s. This is done in a separate loop once all of the functions from
//    the pipeline were processed.
void ynn_runtime::schedule() {
  // Just a helper function to track information about loop levels.
  struct loop_level {
    slinky::loop_id loop_id;
    slinky::expr extent;
    slinky::expr step;
    bool step_is_required = false;
  };

  struct scheduling_data {
    // Loop nest should be a pair of function and loop_id. This is in fact a
    // tree, but for the sake of simplicity we store it as a set of pathes
    // (loop nests in this case) from the root of the tree (outermost
    // location) to a leaf node (the innermost loop of a given function). Loop
    // nests for each of the functions scheduled so far with the indices
    // pointing to the global loop nest. Loop nests can overlap with each
    // other if functions are scheduled within the same loop (only prefixes,
    // i.e. from the root of the loop nest to the most innermost common loop).
    std::vector<int> loop_nest;
    // Compute_at locations of a function -- this an index within a
    // loop nest of a given function, i.e from root (0) to the innermost loop
    // (loop_nest[f_index].back()).
    int compute_at = 0;
    // How many loops from the scheduling_info were matched to the existing
    // loop nest (this is currently done by comparing extents computed in
    // forward bounds).
    int splits_match = 0;
    // This is a product of matched extents which we can use to decide if we
    // should add remaining (not-matched) loops of the function.
    slinky::expr match_volume = 1;
  };

  // This a list of indices of consumers of a given buffer.
  std::map<slinky::var, std::vector<int>> consumers;
  // This is a tree representing a global loop nest of a whole pipeline so
  // far. For efficiency and convenience, it's stored as an array of nodes
  // with auxiliary structures using indices to refer to the loop levels.
  std::vector<loop_level> global_loop_nest;

  std::vector<scheduling_data> func_scheduling_data(funcs.size());
  for (int i = funcs.size() - 1; i >= 0; --i) {
    slinky::func& f = funcs[i];
    scheduling_data& sched_data = func_scheduling_data[i];
    std::vector<int>& loop_nest = sched_data.loop_nest;
    // First of all, we need to find where this function can be scheduled
    // based on its customers. The options are a range of loop levels starting
    // from outermost root to a common subnest of its consumer loop nests.
    // In order to find a common subnest, we iterate over loop nests of the
    // customers and find a common prefix. This also can be viewed as finding
    // a least common ancestor.
    bool loop_nest_initialized = false;
    for (const auto& output : f.outputs()) {
      // Find common subnest of the consumers.
      for (int consumer_index = 0;
           consumer_index < consumers[output.buffer->sym()].size();
           consumer_index++) {
        int consumer = consumers[output.buffer->sym()][consumer_index];
        const std::vector<int>& consumer_loop_nest =
            func_scheduling_data[consumer].loop_nest;
        if (!loop_nest_initialized) {
          loop_nest = consumer_loop_nest;
          loop_nest_initialized = true;
          continue;
        }
        if (consumer_loop_nest.size() < loop_nest.size()) {
          loop_nest.erase(loop_nest.begin() + consumer_loop_nest.size(),
                          loop_nest.end());
        }
        for (int j = 0;
             j < std::min(consumer_loop_nest.size(), loop_nest.size()); ++j) {
          if (loop_nest[j] != consumer_loop_nest[j]) {
            loop_nest.erase(loop_nest.begin() + j, loop_nest.end());
            break;
          }
        }
      }
    }

    int compute_at = -1;
    // The total number of elements shared between the producer and consumer
    // at the proposed compute_at level.
    sched_data.splits_match = 0;
    sched_data.match_volume = 1;
    ynn::scheduling_info* sched =
        static_cast<ynn::scheduling_info*>(f.user_data());

    // If they are not matching and the extents of f are smaller than parent
    // we are risking over-compute. If it doesn't have schedule info then we
    // just compute at the innermost location.
    if (sched && !sched->loop_splits.empty()) {
      std::vector<ynn::scheduling_split>& loop_splits = sched->loop_splits;
      // Make sure that extents of the dims belonging to subnest match.
      // Reverse to simplify indexing below.
      std::reverse(loop_splits.begin(), loop_splits.end());

      const ynn_runtime_value& v = value(sched->base_buffer_id);
      assert(v.is_valid());
      const std::vector<slinky::expr> extents = v.extents;
      compute_at = 0;
      for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
        if (compute_at >= loop_nest.size()) {
          break;
        }
        const ynn::scheduling_split& split = loop_splits[split_i];
        int loop_nest_id = loop_nest[compute_at];
        loop_level& global_loop = global_loop_nest[loop_nest_id];
        // Loops can't be shared if the existing loop step and this loop
        // step are not equal and both are required.
        if (split.step_is_required && global_loop.step_is_required &&
            !prove_true(split.step == global_loop.step)) {
          break;
        }
        if (prove_true(extents[split.axis] == global_loop.extent)) {
          // We can overwrite the current loop step if it's not required, but
          // this one is.
          if (split.step_is_required) {
            global_loop.step = split.step;
            global_loop.step_is_required = true;
          }
          // NOTE(vksnk): Another example of how can we use scheduling_info from
          // all functions assigned to a loop to compute a loop step.
          // global_loop_nest[loop_nest[compute_at]].step = slinky::simplify(
          //     slinky::min(global_loop_nest[loop_nest[compute_at]].step,
          //                 loop_splits[splits_match].step));
          sched_data.match_volume *= global_loop.extent;
          compute_at++;
          sched_data.splits_match = split_i;
        }
      }
      // Remove the inner part of the loop nest which we were not able to
      // match.
      loop_nest.erase(loop_nest.begin() + compute_at, loop_nest.end());
    }

    if (sched && sched->force_root) {
      compute_at = 0;
      sched_data.splits_match = 0;
      loop_nest.clear();
    }

    sched_data.compute_at = compute_at;

    // NOTE: potentially we could also track how much specific loop are
    // computing by keeping a sum of compute amounts for each of the functions
    // inside of this loop and only schedule loops which have more
    // computations than certain threshold.
    if (sched && !sched->loop_splits.empty() &&
        prove_true(sched_data.match_volume == 1)) {
      const std::vector<ynn::scheduling_split>& loop_splits =
          sched->loop_splits;
      // Update the global loop nest by adding loops of this function.
      const ynn_runtime_value& v = value(sched->base_buffer_id);
      assert(v.is_valid());
      const std::vector<slinky::expr> extents = v.extents;
      int splits_match = sched_data.splits_match;
      for (int j = splits_match; j < loop_splits.size(); j++) {
        const ynn::scheduling_split& dim = loop_splits[j];
        global_loop_nest.push_back(
            {{&f, dim.var}, extents[dim.axis], dim.step, dim.step_is_required});
        loop_nest.push_back(global_loop_nest.size() - 1);
      }
    }

    // Record which buffers this function is consuming.
    for (const auto& input : f.inputs()) {
      consumers[input.buffer->sym()].push_back(i);
    }
  }

  // Use previously computed information to actually schedule the functions.
  for (int i = funcs.size() - 1; i >= 0; --i) {
    slinky::func& f = funcs[i];
    const scheduling_data& sched_data = func_scheduling_data[i];
    ynn::scheduling_info* sched =
        static_cast<ynn::scheduling_info*>(f.user_data());
    int compute_at = sched_data.compute_at;
    const slinky::expr& match_volume = sched_data.match_volume;
    // Now we know a compute_at location of this function
    if (compute_at == 0) {
      f.compute_root();
    } else {
      const std::vector<int>& loop_nest = sched_data.loop_nest;
      if (compute_at > 0) {
        const slinky::loop_id& lid =
            global_loop_nest[loop_nest[compute_at - 1]].loop_id;
        f.compute_at(lid);
      }
      if (sched) {
        for (auto& b : sched->scheduled_buffers) {
          if (b.store_at_min_depth == 0) {
            b.buffer->store_at({&funcs[i], slinky::var()});
          } else if (b.store_at_min_depth < loop_nest.size()) {
            const slinky::loop_id& lid =
                global_loop_nest[loop_nest[b.store_at_min_depth - 1]].loop_id;
            b.buffer->store_at(lid);
          } else {
            b.buffer->store_root();
          }
        }
      }
    }

    if (sched && !sched->loop_splits.empty() && prove_true(match_volume == 1)) {
      std::vector<ynn::scheduling_split>& loop_splits = sched->loop_splits;
      const std::vector<int>& loop_nest = sched_data.loop_nest;
      // Reverse it back.
      std::reverse(loop_splits.begin(), loop_splits.end());

      const int splits_match = sched_data.splits_match;
      std::vector<slinky::func::loop_info> loops;
      loops.reserve(loop_splits.size() - splits_match);
      for (int j = 0; j < loop_splits.size() - splits_match; ++j) {
        const ynn::scheduling_split& dim = loop_splits[j];
        loops.push_back(
            {dim.var,
             global_loop_nest[loop_nest[loop_nest.size() - j - 1]].step,
             dim.workers});
      }

      f.loops(std::move(loops));
    }
  }
}

slinky::buffer_expr_ptr ynn_runtime::null_buffer() {
  if (!null_buffer_) {
    slinky::var null(symbols, "null");
    null_buffer_ = slinky::buffer_expr::make_constant(
        null, slinky::raw_buffer::make(0, 0));
  }
  return null_buffer_;
}

namespace {

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// This generates a call that populates all the shapes of the xnn_values using
// the forward bounds expressions.
auto make_reshape_impl(ynn_runtime* runtime) {
  // Gather only what we need for capturing in the callback lambda.
  return [runtime](const slinky::call_stmt*,
                   slinky::eval_context& ctx) -> slinky::index_t {
    int errors = 0;
    for (const ynn_node& node : runtime->subgraph.nodes) {
      if (!node.is_valid()) continue;
      for (const auto& check : node.checks) {
        if (!slinky::evaluate(check.condition, ctx)) {
          std::stringstream error;
          for (const auto& i : check.message) {
            std::visit(overloaded{
                           [&](const char* s) { error << s; },
                           [&](slinky::expr e) { error << evaluate(e, ctx); },
                           [&](ynn_node::input_idx i) {
                             error << "input " << i.idx << " (id "
                                   << node.inputs[i.idx] << ")";
                           },
                           [&](ynn_node::output_idx i) {
                             error << "output " << i.idx << " (id "
                                   << node.outputs[i.idx] << ")";
                           },
                       },
                       i);
          }
          YNN_LOG_ERROR() << "Error in node '" << node.name() << ": "
                          << error.str();
          ++errors;
        }
      }
    }
    if (errors) {
      return ynn_status_error;
    }

    for (auto& i : runtime->values) {
      if (!i.is_valid()) continue;
      if (i.is_external_output()) {
        assert(i.data);
        assert(i.data->rank == i.rank());
        for (size_t d = 0; d < i.rank(); ++d) {
          if (i.extents[d].defined()) {
            i.data->dim(d).set_min_extent(
                0, evaluate(slinky::max(0, i.extents[d]), ctx));
          } else {
            i.data->dim(d).set_min_extent(0, 1);
          }
        }
        ynn::init_buffer_strides(*i.data);
      }
    }
    return 0;
  };
}

slinky::expr type_elem_size(ynn_type type) {
  const int size = ynn::type_size_bytes(type);
  return size > 0 ? slinky::expr(size) : slinky::expr{};
}

#ifdef YNN_ENABLE_PERFETTO
// TODO(dsharlet): We need a better way to control tracing output.
const char* get_trace_filename() { return getenv("YNN_TRACE"); }
#endif

}  // namespace

extern "C" {

ynn_runtime::ynn_runtime(const ynn_subgraph& subgraph,
                         slinky::thread_pool* threadpool, uint32_t flags)
    : subgraph(subgraph),
      flags(flags),
      symbols(subgraph.symbols),
      globals(subgraph.globals) {
  // Implement our required alignment for heap allocations.
  eval_config.allocate = [](slinky::var sym, slinky::raw_buffer* buffer) {
    return buffer->allocate(YNN_ALLOCATION_ALIGNMENT);
  };
  eval_config.free = [](slinky::var sym, slinky::raw_buffer* buffer,
                        void* ptr) { std::free(ptr); };
  eval_config.thread_pool = threadpool;
  // Slinky's default check failure handler calls std::abort(), don't let that
  // happen here.
  eval_config.check_failed = [](const slinky::expr& e) {
    YNN_LOG_ERROR() << "Check failed";
  };
  eval_config.call_failed = [](const slinky::call_stmt* c) {
    YNN_LOG_ERROR() << c->attrs.name << " failed";
  };
  eval_config.base_alignment = YNN_ALLOCATION_ALIGNMENT;

#ifdef YNN_ENABLE_PERFETTO
  eval_config.trace_begin = [](const char* name) {
    ynn::perfetto_session::global()->begin(name);
    return reinterpret_cast<slinky::index_t>(name);
  };
  eval_config.trace_end = [](slinky::index_t token) {
    ynn::perfetto_session::global()->end();
  };
#endif
  eval_context.config = &eval_config;

  values.reserve(subgraph.values.size());
  for (const ynn_value& i : subgraph.values) {
    values.push_back(ynn_runtime_value(i));
    ynn_runtime_value& value = values.back();
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (!value.symbol.defined()) {
      value.symbol = symbols.insert_unique(value.name());
    }
    if (value.is_static()) {
      value.buffer =
          slinky::buffer_expr::make_constant(value.symbol, value.data);
    } else if (value.is_external_input()) {
      value.buffer = ynn::make_buffer_expr(value.symbol, value.rank(),
                                           type_elem_size(value.type));
      for (int d = 0; d < i.rank(); ++d) {
        value.buffer->dim(d).fold_factor = slinky::dim::unfolded;
      }

      if (!value.data) {
        value.data =
            slinky::raw_buffer::make(i.rank(), ynn::type_size_bytes(i.type));
      } else {
        assert(value.data->rank == value.rank());
      }
    }
  }
}

// This function takes the subgraph and turns it into a slinky pipeline.
ynn_status ynn_runtime::build() {
  std::vector<slinky::buffer_expr_ptr> inputs;
  std::vector<slinky::buffer_expr_ptr> outputs;
  funcs.clear();
  for (ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (value.is_static()) {
      assert(value.buffer);
      assert(value.buffer->constant());
    } else if (value.is_external_input()) {
      assert(value.buffer);
      inputs.push_back(value.buffer);
    }
  }

  for (const ynn_node& i : subgraph.nodes) {
    if (!i.is_valid()) continue;
    ynn_status status = i.create(i, *this);
    if (status != ynn_status_success) {
      return status;
    }

    for (uint32_t j : i.outputs) {
      ynn_runtime_value& value = values[j];
      if (!value.is_valid()) continue;
      assert(value.buffer->elem_size().defined());
      if (value.is_external_output() &&
          (!value.data || value.data->rank != value.rank())) {
        value.data = slinky::raw_buffer::make(
            value.rank(), *as_constant(value.buffer->elem_size()));
      }
    }
  }

  for (ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (value.is_external_output()) {
      assert(value.buffer);
      outputs.push_back(value.buffer);
    }
  }

  if ((flags & YNN_RUNTIME_FLAG_NO_SCHEDULE) == 0) {
    schedule();
  }

  slinky::build_options options;
#ifdef YNN_ENABLE_PERFETTO
  options.trace = get_trace_filename() != nullptr;
#endif
#ifdef NDEBUG
  options.no_checks = true;
#endif

  pipeline =
      slinky::build_pipeline(symbols, {}, inputs, outputs, globals, options);

  slinky::call_stmt::attributes attrs;
  attrs.name = "ynn_reshape_runtime";
  reshape_impl = slinky::let_stmt::make(
      globals, slinky::call_stmt::make(make_reshape_impl(this), {}, {}, {},
                                       std::move(attrs)));
  return ynn_status_success;
}

ynn_status ynn_runtime::reshape() {
  setup();
  return slinky::evaluate(reshape_impl, eval_context) ? ynn_status_error
                                                      : ynn_status_success;
}

ynn_status ynn_runtime::setup() {
  std::vector<const slinky::raw_buffer*> inputs;
  std::vector<const slinky::raw_buffer*> outputs;
  for (const ynn_runtime_value& value : values) {
    if (!value.is_valid()) {
      // This value was removed or never defined.
    } else if (value.is_external_input()) {
      assert(value.data);
      inputs.push_back(value.data.get());
    } else if (value.is_external_output()) {
      assert(value.data);
      outputs.push_back(value.data.get());
    }
  }

  pipeline.setup(inputs, outputs, eval_context);
  return ynn_status_success;
}

ynn_status ynn_create_runtime(ynn_subgraph_t subgraph,
                              ynn_threadpool_t threadpool, uint32_t flags,
                              ynn_runtime_t* runtime_out) {
  slinky::thread_pool* slinky_threadpool =
      reinterpret_cast<slinky::thread_pool*>(threadpool);
  auto runtime =
      std::make_unique<ynn_runtime>(*subgraph, slinky_threadpool, flags);
  ynn_status status = runtime->build();
  if (status != ynn_status_success) {
    return status;
  }

  status = runtime->setup();
  if (status != ynn_status_success) {
    return status;
  }

  *runtime_out = runtime.release();
  return ynn_status_success;
}

ynn_status ynn_update_runtime_with_threadpool(ynn_runtime_t runtime,
                                              ynn_threadpool_t threadpool) {
  runtime->eval_config.thread_pool =
      reinterpret_cast<slinky::thread_pool*>(threadpool);
  return ynn_status_success;
}

ynn_status ynn_runtime::invoke() {
  return pipeline.evaluate(eval_context) ? ynn_status_error
                                         : ynn_status_success;
}

ynn_status ynn_set_external_value_shape(ynn_runtime_t runtime,
                                        uint32_t external_id, size_t rank,
                                        const size_t* dims) {
  ynn_runtime_value& value = runtime->value(external_id);
  return value.set_external_shape(rank, dims);
}

ynn_status ynn_get_external_value_shape(ynn_runtime_t runtime,
                                        uint32_t external_id, size_t* rank,
                                        size_t* dims) {
  const ynn_runtime_value& value = runtime->value(external_id);
  return value.get_external_shape(rank, dims);
}

ynn_status ynn_reshape_runtime(ynn_runtime_t runtime) {
  return runtime->reshape();
}

ynn_status ynn_set_external_value_data(ynn_runtime_t runtime,
                                       uint32_t external_id, void* data) {
  ynn_value& value = runtime->values[external_id];
  value.data->base = data;
  return ynn_status_success;
}

ynn_status ynn_invoke_runtime(ynn_runtime_t runtime) {
  return runtime->invoke();
}

void ynn_delete_runtime(ynn_runtime_t runtime) { delete runtime; }

}  // extern "C"
