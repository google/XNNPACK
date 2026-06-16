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
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifdef YNN_ENABLE_PERFETTO
#include "ynnpack/subgraph/perfetto.h"
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
#include "xla/tsl/profiler/lib/traceme.h"
#endif
#include "ynnpack/base/base.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/ref_count.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/tensor.h"
#include "slinky/base/arithmetic.h"
#include "slinky/base/thread_pool.h"
#include "slinky/builder/node_mutator.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/builder/substitute.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/depends_on.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

void ynn_runtime_value::make_buffer(ynn_runtime& runtime,
                                    slinky::expr elem_size) {
  if (buffer) {
    assert(buffer->sym() == symbol);
    return;
  }
  if (!symbol.defined()) {
    symbol = runtime.globals.symbols.insert_unique(name());
  }
  buffer = ynn::make_buffer_expr(symbol, rank(), std::move(elem_size));
  for (size_t i = 0; i < rank(); ++i) {
    if (!extents[i].defined() || slinky::is_constant(extents[i], 1)) {
      buffer->dim(i) = slinky::dim::broadcast();
    }
  }
}

void ynn_runtime_value::make_buffer(ynn_runtime& runtime) {
  make_buffer(runtime, ynn::type_size_bytes(type));
}

std::unique_ptr<ynn::scheduling_info> ynn_runtime::make_schedule(
    ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
    const slinky::expr& element_cost,
    ynn::span<const slinky::expr> given_splits,
    ynn::span<const int> loop_order) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }

  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, element_cost, given_splits, loop_order);

  return make_schedule(dims, extents, splits, loop_order);
}

std::unique_ptr<ynn::scheduling_info> ynn_runtime::make_schedule(
    ynn::span<const slinky::var> dims, ynn::span<const slinky::expr> extents,
    ynn::span<const slinky::expr> splits, ynn::span<const int> loop_order) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }

  int max_threads = threadpool() ? threadpool()->thread_count() : 1;
  // Enough tasks to have good load balancing.
  slinky::index_t target_task_count = max_threads > 1 ? max_threads * 2 : 1;

  std::vector<slinky::expr> workers(rank);
  slinky::expr threads_so_far = 1;

  auto get_loop_dim = [&](int index_d) {
    return index_d < loop_order.size() ? loop_order[index_d] : index_d;
  };

  for (int index_d = rank - 1; index_d >= 0; --index_d) {
    int d = get_loop_dim(index_d);
    if (max_threads == 1 || globals.is_reduction_dim(dims[d])) {
      workers[d] = slinky::loop::serial;
    } else if (extents[d].defined() && splits[d].defined()) {
      slinky::expr w =
          slinky::ceil_div(slinky::expr(target_task_count), threads_so_far);
      w = globals.get(w, "w");

      workers[d] = slinky::simplify(slinky::select::make(
          w > 1, slinky::loop::parallel, slinky::loop::serial));

      threads_so_far =
          slinky::simplify(threads_so_far * ceil_div(extents[d], splits[d]));
    }
  }

  std::vector<ynn::scheduling_split> loop_splits;
  for (int index_d = 0; index_d < rank; ++index_d) {
    int d = get_loop_dim(index_d);
    if (extents[d].defined() && splits[d].defined()) {
      loop_splits.push_back({dims[d], splits[d], workers[d], extents[d]});
    }
  }

  auto scheduling_info = std::make_unique<ynn::scheduling_info>();
  scheduling_info->loop_splits = std::move(loop_splits);
  return scheduling_info;
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

// Finds which {`buffer`, `dim`} corresponds to the output dimension variable
// `v`.
std::pair<slinky::var, int> find_output_dim(const slinky::func* f,
                                            const slinky::var& v) {
  if (f) {
    for (const auto& out : f->outputs()) {
      for (int i = 0; i < out.dims.size(); ++i) {
        if (out.dims[i] == v) {
          return {out.sym(), i};
        }
      }
    }
  }
  return {slinky::var(), -1};
}

}  // namespace

// Logically this function has multiple separate blocks:
// 1) infer symbolic source regions for all buffers to ensure loops are only
//    fused if they share a common source region origin.
// 2) computing a list of possible compute_at locations for a given function.
//    This is a very concrete thing and doesn't require any heuristics.
// 3) using the set of locations from 2) decide if we want for this function
//    to be computed at root or at one of the existing loops based on the
//    available information such as scheduling_info attached to the function
//    or source regions inferred in 1).
// 4) if we decide to share the loop location possibly update loop parameters
//    such as step based on the specific of the given function (this is pretty
//    much a no-op right now and is solely defined by a "parent" function of
//    the loop, but we can use it in the future to figure out, for example, a
//    step size based on the *all* functions which were assigned to the loop).
// 5) potentially add new loop(s) into the loop nest based on a given
//    function.
// 6) based on compute locations computed in 2) - 5), set up the
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
  };

  // This a list of indices of consumers of a given buffer.
  std::map<slinky::var, std::vector<int>> consumers;
  // This is a tree representing a global loop nest of a whole pipeline so
  // far. For efficiency and convenience, it's stored as an array of nodes
  // with auxiliary structures using indices to refer to the loop levels.
  std::vector<loop_level> global_loop_nest;

  std::vector<scheduling_data> func_scheduling_data(funcs.size());

  // This pass infers symbolic source regions for all buffers, traversing the
  // pipeline in reverse topological order (consumers to producers). It ensures
  // that loops are only fused if their dimensions share a common source
  // region origin, which naturally prevents incorrect fusions of unrelated
  // dimensions that happen to have the same constant size. This is similar
  // to the backward bounds inference, but much more lightweight because we
  // only care if given extents are the same in terms of consumer extents.

  std::map<slinky::var, const slinky::func*> buffer_to_producer;
  for (const auto& f : funcs) {
    for (const auto& out : f.outputs()) {
      buffer_to_producer[out.sym()] = &f;
    }
  }

  // Maps {buffer_sym, dim_index} to its inferred source region unique
  // identifier.
  std::map<std::pair<slinky::var, int>, int> source_regions;

  int next_source_region_id = 0;

  // Lazily creates a new symbolic source region identifier if one doesn't
  // exist.
  auto get_source_region = [&](slinky::var buf, int dim) {
    auto key = std::make_pair(buf, dim);
    if (source_regions.find(key) == source_regions.end()) {
      source_regions[key] = next_source_region_id++;
    }
    return source_regions[key];
  };

  // Traverses operations backwards to propagate source region symbols.
  for (int i = funcs.size() - 1; i >= 0; --i) {
    const slinky::func& f = funcs[i];
    if (f.outputs().empty()) continue;

    // Collect all unique output variables for this function.
    std::set<slinky::var> out_vars;
    for (const auto& out : f.outputs()) {
      for (auto v : out.dims) {
        out_vars.insert(v);
      }
    }

    for (const auto& in : f.inputs()) {
      for (int d = 0; d < in.bounds.size(); ++d) {
        slinky::interval_expr bound = in.bounds[d];

        // If the producer provided a custom scheduler_bound, we use it instead
        // of the forward bounds. This allows tricks like fusing pack_b's
        // blocks_n (extent N/16) with dot's n (extent N) by forcing a virtual
        // 1-to-1 mapping.
        if (buffer_to_producer.count(in.sym()) > 0) {
          const slinky::func* f_prod = buffer_to_producer[in.sym()];
          slinky::var v_prod;
          for (const auto& out : f_prod->outputs()) {
            if (out.sym() == in.sym() && d < out.dims.size()) {
              v_prod = out.dims[d];
              break;
            }
          }

          if (v_prod.defined()) {
            if (f_prod->user_data()) {
              const auto* sched =
                  static_cast<const ynn::scheduling_info*>(f_prod->user_data());
              for (const auto& split : sched->loop_splits) {
                if (split.var == v_prod && split.scheduler_bound.has_value()) {
                  bound = *split.scheduler_bound;
                  break;
                }
              }
            }
          }
        }

        // Find which output variables this input dimension depends on.
        slinky::var correlated_var;
        for (auto v : out_vars) {
          if (slinky::depends_on(bound, v).any()) {
            if (correlated_var.defined()) {
              correlated_var = slinky::var();
              break;
            }
            correlated_var = v;
          }
        }

        int inferred_region = next_source_region_id++;

        if (correlated_var.defined()) {
          slinky::var v = correlated_var;

          // Collect source regions for this variable from all outputs that
          // contain it.
          std::vector<int> parent_source_regions;
          for (const auto& out : f.outputs()) {
            for (int od = 0; od < out.dims.size(); ++od) {
              if (out.dims[od] == v) {
                parent_source_regions.push_back(
                    get_source_region(out.sym(), od));
              }
            }
          }

          if (slinky::is_variable(bound.min, v) &&
              slinky::is_variable(bound.max, v) &&
              !parent_source_regions.empty()) {
            // Check if all parent extents are equivalent.
            bool all_equal = true;
            for (size_t k = 1; k < parent_source_regions.size(); ++k) {
              if (parent_source_regions[0] != parent_source_regions[k]) {
                all_equal = false;
                break;
              }
            }
            if (all_equal) {
              inferred_region = parent_source_regions[0];
            }
          }
        }

        auto key = std::make_pair(in.sym(), d);
        if (source_regions.count(key) > 0) {
          // If this buffer has multiple consumers with conflicting inferred
          // regions, merge them into a new unique ID (breaks fusion for this
          // dimension).
          if (source_regions[key] != inferred_region) {
            source_regions[key] = next_source_region_id++;
          }
        } else {
          source_regions[key] = inferred_region;
        }
      }
    }
  }

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

      compute_at = 0;
      for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
        if (compute_at >= loop_nest.size()) {
          break;
        }
        const ynn::scheduling_split& split = loop_splits[split_i];
        if (prove_true(split.extent == 1)) {
          // If the split is 1, it doesn't matter if we fuse or not.
          continue;
        }
        int loop_nest_id = loop_nest[compute_at];
        loop_level& global_loop = global_loop_nest[loop_nest_id];
        if (split.step_is_required && global_loop.step_is_required &&
            !prove_true(split.step == global_loop.step)) {
          // Loops can't be shared if the existing loop step and this loop
          // step are not equal and both are required.
          break;
        }
        if (!globals.is_pure_dim(split.var)) {
          // We don't want to fuse a reduction dimension because it is likely
          // being broadcasted here.
          break;
        }
        // Map the consumer's loop variable back to its output dimension
        // index.
        auto [consumer_buf, consumer_dim] =
            find_output_dim(global_loop.loop_id.func, global_loop.loop_id.var);

        // Map the producer's loop variable back to its output dimension index.
        auto [producer_buf, producer_dim] = find_output_dim(&f, split.var);

        // Instead of comparing forward extents (which causes false positives
        // for unrelated constant extents), we check if both loops share the
        // exact same inferred source region identifier.
        bool extents_match = false;
        if (producer_dim != -1 && consumer_dim != -1 &&
            producer_buf.defined() && consumer_buf.defined()) {
          int producer_source_region =
              get_source_region(producer_buf, producer_dim);
          int consumer_source_region =
              get_source_region(consumer_buf, consumer_dim);

          if (producer_source_region == consumer_source_region) {
            extents_match = true;
          }
        }

        if (!extents_match) {
          break;
        }
        // We can overwrite the current loop step if it's not required, but
        // this one is.
        if (split.step_is_required) {
          if (std::optional<slinky::var> v =
                  slinky::as_variable(global_loop.step)) {
            // This is a special variable which defines partial reduction
            // bounds, so we need to override to match the loop step.
            if (globals.symbols.name(*v).rfind("pr_split", 0) == 0) {
              globals.update_let(*v, split.step);
            }
          }
          global_loop.step = split.step;
          global_loop.step_is_required = true;
        }
        compute_at++;
        sched_data.splits_match = split_i + 1;
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
    if (sched && !sched->loop_splits.empty()) {
      const std::vector<ynn::scheduling_split>& loop_splits =
          sched->loop_splits;
      // Update the global loop nest by adding loops of this function.
      int splits_match = sched_data.splits_match;
      for (int j = splits_match; j < loop_splits.size(); j++) {
        const ynn::scheduling_split& dim = loop_splits[j];
        global_loop_nest.push_back(
            {{&f, dim.var}, dim.extent, dim.step, dim.step_is_required});
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
      if (!sched || sched->scheduled_buffers.empty()) {
        f.store_outputs_innermost();
      } else {
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

    if (sched && !sched->loop_splits.empty()) {
      std::vector<ynn::scheduling_split>& loop_splits = sched->loop_splits;
      const std::vector<int>& loop_nest = sched_data.loop_nest;
      // Reverse it back.
      std::reverse(loop_splits.begin(), loop_splits.end());

      const int splits_match = sched_data.splits_match;
      std::vector<slinky::func::loop_info> loops;
      loops.reserve(loop_splits.size() - splits_match);
      for (int j = 0; j < loop_splits.size() - splits_match; ++j) {
        const ynn::scheduling_split& dim = loop_splits[j];
        slinky::expr step =
            global_loop_nest[loop_nest[loop_nest.size() - j - 1]].step;
        loops.push_back({dim.var, step, dim.workers});
      }

      f.loops(std::move(loops));
    }
  }
}

slinky::buffer_expr_ptr ynn_runtime::null_buffer() {
  if (!null_buffer_) {
    slinky::var null(globals.symbols, "null");
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
    for (const ynn_node& node : runtime->subgraph->nodes) {
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
        std::vector<slinky::expr> phys_extents = i.physical_extents();
        for (size_t d = 0; d < i.rank(); ++d) {
          slinky::expr extent_d = i.physical_extent(d);
          if (extent_d.defined()) {
            i.data->mutable_dim(d).set_min_extent(0, evaluate(extent_d, ctx));
          } else {
            i.data->mutable_dim(d).set_min_extent(0, 1);
          }
        }
        ynn::init_buffer_strides(*i.data);
      }
    }
    return 0;
  };
}

#ifdef YNN_ENABLE_PERFETTO
// TODO(dsharlet): We need a better way to control tracing output.
const char* get_trace_filename() { return getenv("YNN_TRACE"); }
#endif

#ifdef YNN_ENABLE_TSL_PROFILER
bool ynn_traceme_enabled() {
  // We can't use `TraceMe::Active` here, because it returns false when called,
  // even if it would later return true when we actually want to trace. We
  // should also gate this behind an extra flag because our tracing might be a
  // lot higher frequency than other xprof tracing.
  const char* traceme = getenv("YNN_TRACEME");
  return traceme && strcmp(traceme, "0") != 0;
}
#endif

}  // namespace

extern "C" {

ynn_runtime::ynn_runtime(ynn::ref_count<const ynn_subgraph> subgraph,
                         slinky::thread_pool* threadpool, uint32_t flags)
    : subgraph(subgraph), flags(flags), globals(subgraph->globals) {
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
  if (ynn::perfetto_session::global()) {
    eval_config.trace_begin = [](const char* name) {
      ynn::perfetto_session::global()->begin(name);
      return reinterpret_cast<slinky::index_t>(name);
    };
    eval_config.trace_end = [](slinky::index_t token) {
      ynn::perfetto_session::global()->end();
    };
  }
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
  if (ynn_traceme_enabled()) {
    if (ynn::perfetto_session::global()) {
      YNN_LOG_WARNING()
          << "tsl::profiler tracing is overriding perfetto tracing.";
    }
    eval_config.trace_begin = [](const char* name) {
      return static_cast<slinky::index_t>(
          tsl::profiler::TraceMe::ActivityStart(name));
    };
    eval_config.trace_end = [](slinky::index_t token) {
      tsl::profiler::TraceMe::ActivityEnd(token);
    };
  }
#endif
  eval_context.config = &eval_config;

  values.reserve(subgraph->values.size());
  for (const ynn_value& i : subgraph->values) {
    values.push_back(ynn_runtime_value(i));
    ynn_runtime_value& value = values.back();
    if (!value.is_valid()) {
      // This value was removed or never defined.
      continue;
    }
    if (!value.symbol.defined()) {
      value.symbol = globals.symbols.insert_unique(value.name());
    }
    if (value.is_static()) {
      for (size_t d = 0; d < value.extents.size(); ++d) {
        if (!value.extents[d].defined() ||
            slinky::is_constant(value.extents[d], 1)) {
          value.data->mutable_dim(d) = slinky::dim::broadcast();
        }
      }

      value.buffer =
          slinky::buffer_expr::make_constant(value.symbol, value.data);
    } else if (value.is_external_input()) {
      value.make_buffer(*this);

      for (size_t d = 0; d < value.extents.size(); ++d) {
        slinky::expr extent_d = i.physical_extent(d);
        if (!extent_d.defined()) {
          value.buffer->dim(d).bounds = slinky::point(0);
        } else if (const auto v = as_constant(extent_d)) {
          value.buffer->dim(d).bounds = slinky::range(0, *v);
        }
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

  for (const ynn_node& i : subgraph->nodes) {
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

    if (value.data) {
      for (size_t d = 0; d < value.extents.size(); ++d) {
        if (!value.extents[d].defined() ||
            slinky::is_constant(value.extents[d], 1)) {
          value.data->mutable_dim(d) = slinky::dim::broadcast();
        }
      }
    }
  }

  if ((flags & YNN_RUNTIME_FLAG_NO_SCHEDULE) == 0) {
    schedule();
  }

  slinky::build_options options;
#ifdef YNN_ENABLE_PERFETTO
  options.trace = options.trace || get_trace_filename() != nullptr;
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
  options.trace = options.trace || ynn_traceme_enabled();
#endif
#ifdef NDEBUG
  options.no_checks = true;
#endif

  pipeline = slinky::build_pipeline(globals.symbols, {}, inputs, outputs,
                                    globals.lets, options);

  slinky::call_stmt::attributes attrs;
  attrs.name = "ynn_reshape_runtime";
  reshape_impl = slinky::let_stmt::make(
      globals.lets, slinky::call_stmt::make(make_reshape_impl(this), {}, {}, {},
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
  YNN_RETURN_IF_ERROR(ynn::validate_subgraph("create_runtime", subgraph));
  if (runtime_out == nullptr) {
    YNN_LOG_ERROR() << "runtime_out must be non-null";
    return ynn_status_invalid_parameter;
  }

  slinky::thread_pool* slinky_threadpool =
      reinterpret_cast<slinky::thread_pool*>(threadpool);
  auto runtime =
      std::make_unique<ynn_runtime>(subgraph, slinky_threadpool, flags);
  YNN_RETURN_IF_ERROR(runtime->build());
  YNN_RETURN_IF_ERROR(runtime->setup());

  *runtime_out = runtime.release();
  return ynn_status_success;
}

ynn_status ynn_update_runtime_with_threadpool(ynn_runtime_t runtime,
                                              ynn_threadpool_t threadpool) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
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
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  ynn_runtime_value& value = runtime->value(external_id);
  return value.set_external_shape(rank, dims);
}

ynn_status ynn_get_external_value_shape(ynn_runtime_t runtime,
                                        uint32_t external_id, size_t* rank,
                                        size_t* dims) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  const ynn_runtime_value& value = runtime->value(external_id);
  return value.get_external_shape(rank, dims);
}

ynn_status ynn_reshape_runtime(ynn_runtime_t runtime) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  return runtime->reshape();
}

ynn_status ynn_set_external_value_data(ynn_runtime_t runtime,
                                       uint32_t external_id, void* data) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!runtime->subgraph->is_valid_value(external_id)) {
    YNN_LOG_ERROR() << "invalid value ID: " << external_id;
    return ynn_status_invalid_parameter;
  }
  ynn_value& value = runtime->values[external_id];
  assert(value.data);
  value.data->base = data;
  return ynn_status_success;
}

ynn_status ynn_invoke_runtime(ynn_runtime_t runtime) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  return runtime->invoke();
}
namespace {

int32_t get_max_concurrency(const ynn_runtime& runtime) {
  // Traverse the pipeline body for any loops. If we find a parallel loop, we
  // return `max_int32`. Otherwise, we return 1.
  class visitor : public slinky::recursive_node_visitor {
   public:
    int32_t result = 1;
    void visit(const slinky::loop* op) override {
      if (!slinky::prove_true(op->max_workers == 1)) {
        result = std::numeric_limits<int32_t>::max();
      }
      slinky::recursive_node_visitor::visit(op);
    }
  } v;
  if (runtime.pipeline.body.defined()) {
    runtime.pipeline.body.accept(&v);
  }
  return v.result;
}

}  // namespace

ynn_status ynn_query_runtime(ynn_runtime_t runtime,
                             enum ynn_runtime_property property, void* result,
                             size_t* result_size) {
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  if (!result_size || !result) {
    YNN_LOG_ERROR() << "result and result_size must be non-null";
    return ynn_status_invalid_parameter;
  }

  switch (property) {
    case ynn_runtime_property_concurrency: {
      memset(result, 0, *result_size);
      if (*result_size < sizeof(int32_t)) {
        YNN_LOG_ERROR() << "result must be an int32_t.";
        return ynn_status_error;
      }
      *result_size = sizeof(int32_t);
      int32_t max_threads = get_max_concurrency(*runtime);
      memcpy(result, &max_threads, sizeof(int32_t));
      return ynn_status_success;
    }
  }
  YNN_LOG_ERROR() << "Unknown runtime property: " << property;
  return ynn_status_error;
}

void ynn_delete_runtime(ynn_runtime_t runtime) { delete runtime; }

}  // extern "C"
