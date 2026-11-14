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
#include "ynnpack/kernels/dot/cost_model/cost_model.h"
#include "ynnpack/kernels/dot/dot.h"
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
    ynn::span<const slinky::expr> given_splits, ynn::span<const int> loop_order,
    ynn::span<const slinky::expr> alignments) {
  const int rank = dims.size();
  if (rank <= 0) {
    // Nothing to schedule here.
    return {};
  }

  std::vector<slinky::expr> splits = make_split_factors(
      globals, extents, element_cost, given_splits, loop_order, alignments);

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

  auto get_loop_dim = [&](int index_d) {
    return index_d < loop_order.size() ? loop_order[index_d] : index_d;
  };

  std::vector<ynn::scheduling_split> loop_splits;
  for (int index_d = 0; index_d < rank; ++index_d) {
    int d = get_loop_dim(index_d);
    if (extents[d].defined() && splits[d].defined()) {
      loop_splits.push_back({dims[d], splits[d], extents[d]});
    }
  }

  auto scheduling_info = std::make_unique<ynn::scheduling_info>();
  scheduling_info->loop_splits = std::move(loop_splits);
  return scheduling_info;
}

namespace {

// Per-loop metadata indexed by `slinky::loop_ref::index()` while building
// `slinky::loop_nest`.
struct loop_metadata {
  slinky::expr extent;
  bool step_is_required = false;
  int funcs_count = 0;
  int source_region = -1;
  // A finer step for this loop, proposed by a function fused into it. Set by
  // reconcile_step(); compute_workers() decides whether to use it, because
  // that depends on the rest of the loop nest, which isn't built yet when
  // functions are matched.
  slinky::expr proposed_step;
};

int physical_output_dim(const slinky::func& f, int d) {
  // Skip dimensions sliced by `slinky::func::make_stack` when indexing the
  // buffer's physical dimensions.
  int od = d;
  if (!f.inputs().empty()) {
    const std::vector<slinky::expr>& slices = f.inputs()[0].output_slice;
    for (int slice_d = 0;
         slice_d <= od && slice_d < static_cast<int>(slices.size());
         ++slice_d) {
      if (slices[slice_d].defined()) {
        ++od;
      }
    }
  }
  return od;
}

// Least common multiple of two (possibly symbolic) loop steps, evaluated at
// runtime. Used to reconcile two producers that require different tiles for a
// shared loop: a multiple of both keeps the fused loop an integer number of
// each producer's tile (and hence a multiple of each kernel's m/n block).
// If the LCM overflows, it clamps at the max index_t value.
slinky::expr lcm_sat(ynn::slinky_globals& globals, slinky::expr a,
                     slinky::expr b) {
  if (slinky::prove_true(a == b, globals.fact_bounds, globals.fact_alignment)) {
    return a;
  }
  auto impl = [](const slinky::call* op,
                 slinky::eval_context& ctx) -> slinky::index_t {
    slinky::index_t a_val = slinky::evaluate(op->args[0], ctx);
    slinky::index_t b_val = slinky::evaluate(op->args[1], ctx);
    assert(a_val != 0 && b_val != 0);
    return slinky::mul_sat(a_val / slinky::gcd(a_val, b_val), b_val);
  };
  return globals.get(slinky::call::make(impl, {std::move(a), std::move(b)}),
                     "lcm_sat");
}

// This pass infers symbolic source regions for all buffers, traversing the
// pipeline in reverse topological order (consumers to producers). It ensures
// that loops are only fused if their dimensions share a common source
// region origin, which naturally prevents incorrect fusions of unrelated
// dimensions that happen to have the same constant size. This is similar
// to the backward bounds inference, but much more lightweight because we
// only care if given extents are the same in terms of consumer extents.
std::map<std::pair<slinky::var, int>, int> infer_source_regions(
    const std::vector<slinky::func>& funcs) {
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

    const auto* sched = static_cast<const ynn::scheduling_info*>(f.user_data());

    for (size_t in_idx = 0; in_idx < f.inputs().size(); ++in_idx) {
      const auto& in = f.inputs()[in_idx];
      const std::vector<slinky::interval_expr>* scheduler_bounds = nullptr;
      if (sched && in_idx < sched->input_scheduler_bounds.size()) {
        scheduler_bounds = &sched->input_scheduler_bounds[in_idx];
      }

      for (int d = 0; d < in.bounds.size(); ++d) {
        slinky::interval_expr bound = in.bounds[d];

        // If this function provided a custom scheduler bound for this input
        // dimension, we use it instead of the real bounds. This allows tricks
        // like fusing pack_b's blocks_n (extent N/16) with dot's n (extent N)
        // by declaring a virtual 1-to-1 mapping. The bounds are attached to
        // the consumer that understands the layout of the input, so they work
        // regardless of which function produced the input.
        if (scheduler_bounds && d < scheduler_bounds->size() &&
            (*scheduler_bounds)[d].min.defined()) {
          bound = (*scheduler_bounds)[d];
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
                parent_source_regions.push_back(get_source_region(
                    out.sym(), physical_output_dim(f, od)));
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

  return source_regions;
}

// If `e` is a global let variable, return the let's value (an existing,
// shared expression -- nothing is constructed); otherwise return `e`
// unchanged. One level is enough for the analyses below: loop steps are let
// variables whose immediate values expose the min/max caps the constant
// bounds need (variables remaining inside them are simply unknowns to the
// bounds evaluator), and loop extents are stored in raw form. This
// deliberately avoids substituting lets into expressions: full expansion
// can grow combinatorially when let values nest.
slinky::expr resolve_let_var(const ynn::slinky_globals& globals,
                             const slinky::expr& e) {
  if (auto v = slinky::as_variable(e)) {
    for (const auto& let : globals.lets) {
      if (let.first == *v) return let.second;
    }
  }
  return e;
}

// The condition under which `l` runs exactly one full-extent iteration, i.e.
// whoever created it made no real splitting decision for this dimension:
// make_split_factors() hands out a cache-sized tile area and returns the whole
// extent for any dimension that fits in what is left of it. This is the
// weakest claim a function can make on a loop step. Such a loop can be folded
// away entirely, it can take a finer step from a function fused into it, and
// it cannot be the loop that parallelizes the nest.
//
// The step is resolved one let level deep and the extent simplified before
// comparing: split factors are usually `min(...)` expressions the simplifier
// already reduced to the extent itself, but hidden behind a let variable the
// prover can't see through.
slinky::expr is_single_iteration(const ynn::slinky_globals& globals,
                                 const slinky::expr& extent,
                                 const slinky::expr& step) {
  if (!extent.defined() || !step.defined()) return slinky::expr();
  return slinky::simplify(extent) <= resolve_let_var(globals, step);
}

// Decide how many workers each loop of `loops` should use. This must run after
// the whole nest is built (and all the steps are final): after fusion, a
// function's loops can end up inside loops of other functions, so the number
// of tasks produced outside each loop is only known once the nest is complete.
//
// `loop_meta[i].funcs_count` is the number of functions whose body executes
// inside loop `i`. A loop whose ancestors provably always produce enough tasks
// can never run more than one worker, so its
// `select(w > 1, parallel, serial)` is folded to `serial`. If such a loop
// also contains a single function and its step is not required (no alignment
// constraint), the loop is pure overhead: one kernel call per tile with
// nothing to interleave or parallelize. Setting its step to the full extent
// makes it a provable single iteration, which slinky then folds away,
// leaving one kernel call over the whole range.
void compute_workers(ynn::slinky_globals& globals, int max_threads,
                     slinky::loop_nest& loops,
                     const std::vector<loop_metadata>& loop_meta) {
  // Enough tasks to have good load balancing.
  // TODO(b/570143574): Evaluate using tasks_per_thread = 4 unconditionally or
  // detecting heterogeneous cores instead of checking __ANDROID__.
#if defined(__ANDROID__)
  const slinky::index_t tasks_per_thread = 4;
#else
  const slinky::index_t tasks_per_thread = 2;
#endif
  const slinky::index_t target_task_count =
      max_threads > 1 ? max_threads * tasks_per_thread : 1;

  // A guaranteed lower bound of the number of iterations of loop `l`:
  // ceil_div(lower bound of extent, upper bound of step), or 1 when either
  // bound is unknown (a scheduled loop runs at least one iteration).
  auto min_iterations = [&](slinky::loop_ref l,
                            const loop_metadata& meta) -> slinky::index_t {
    std::optional<slinky::index_t> extent_lb =
        slinky::evaluate_constant_lower_bound(meta.extent);
    std::optional<slinky::index_t> step_ub =
        slinky::evaluate_constant_upper_bound(
            resolve_let_var(globals, loops[l].step));
    if (extent_lb && step_ub && *step_ub > 0) {
      return std::max<slinky::index_t>(slinky::ceil_div(*extent_lb, *step_ub),
                                       1);
    }
    return 1;
  };

  // For each loop, whether any of its descendants can be parallelized: a pure
  // or partial-reduction ("r") loop that provably runs more than one iteration.
  // Serial reduction ("k") loops never count, they run inside one task by
  // construction. Loops are appended after their parent, so a single backward
  // pass propagates each loop's answer up to its parent.
  std::vector<bool> parallel_in_subtree(loops.size(), false);
  for (int i = static_cast<int>(loops.size()) - 1; i >= 1; --i) {
    const slinky::loop_ref l = loops.loop_at(i);
    const slinky::loop_ref parent = loops[l].parent;
    if (parent.is_root()) {
      // This is a root loop.
      continue;
    }
    if (parallel_in_subtree[i]) {
      parallel_in_subtree[parent.index()] = true;
      continue;
    }
    const loop_metadata& meta = loop_meta[i];
    if (globals.is_reduction_dim(loops[l].dims[0].dim)) {
      // Serial reduction loops run inside one task.
      continue;
    }
    const slinky::expr once =
        is_single_iteration(globals, meta.extent, loops[l].step);
    if (!once.defined() || !slinky::prove_false(once, globals.fact_bounds,
                                                globals.fact_alignment)) {
      // This loop might only run one iteration.
      continue;
    }
    parallel_in_subtree[parent.index()] = true;
  }

  // The number of tasks the loops from the root down to (and including) each
  // loop can produce. Serial loops (reductions) run their iterations within
  // one task, so they don't contribute to this count. `tasks_lb` is a
  // guaranteed constant lower bound of the same quantity.
  std::vector<slinky::expr> tasks(loops.size(), 1);
  std::vector<slinky::index_t> tasks_lb(loops.size(), 1);
  for (size_t i = 1; i < loops.size(); ++i) {
    const slinky::loop_ref l = loops.loop_at(i);
    slinky::loop_nest::loop& node = loops[l];
    const slinky::loop_ref parent = node.parent;
    assert(static_cast<size_t>(parent.index()) < i);
    const loop_metadata& meta = loop_meta[i];
    const slinky::expr& tasks_above = tasks[parent.index()];
    const slinky::index_t tasks_above_lb = tasks_lb[parent.index()];
    // Take the finer step a fused function proposed for this loop, if this
    // loop is where the nest has to get its parallelism. It isn't if the
    // ancestors already produce enough tasks, or if the subtree below has a
    // level that can be parallelized instead - splitting here would then cost
    // parallelism rather than add it, because the extra outer tasks push the
    // inner levels past the task target and turn them serial. The inner
    // levels are also the better place to split: finer tasks over contiguous
    // memory.
    if (meta.proposed_step.defined() && max_threads > 1 &&
        !parallel_in_subtree[i] && tasks_above_lb < target_task_count) {
      // Behind a global so per-task closures reference a variable evaluated
      // once per invoke, not the whole select tree.
      node.step =
          globals.get(slinky::simplify(meta.proposed_step, globals.fact_bounds,
                                       globals.fact_alignment),
                      "s");
    }
    // A loop that provably runs exactly one iteration is identical for any
    // number of functions inside it (required steps are excluded). Replacing
    // its step with the extent expression lets slinky prove the single
    // iteration and fold the loop away entirely.
    const slinky::var loop_var = node.dims[0].dim;
    const bool elide_allowed =
        !meta.step_is_required && globals.is_pure_dim(loop_var);
    const slinky::expr once =
        is_single_iteration(globals, meta.extent, node.step);
    if (once.defined() &&
        slinky::prove_true(once, globals.fact_bounds, globals.fact_alignment)) {
      // Serial reduction ("k") dims additionally qualify for the
      // single-iteration elision below (but not for the widening elisions):
      // with provably one iteration there is no accumulation blocking to
      // preserve.
      const bool single_iteration_elide_allowed =
          elide_allowed ||
          (!meta.step_is_required && globals.is_reduction_dim(loop_var));
      if (single_iteration_elide_allowed) {
        node.step = slinky::max(slinky::simplify(meta.extent), 1);
      }
      node.max_workers = slinky::loop::serial;
      tasks[i] = tasks_above;
      tasks_lb[i] = tasks_above_lb;
    } else if (max_threads == 1 || globals.is_reduction_dim(loop_var)) {
      node.max_workers = slinky::loop::serial;
      // Reduction loops are left alone even when they contain a single
      // function: their step controls accumulation blocking, not just task
      // granularity.
      if (elide_allowed && meta.funcs_count <= 1) {
        node.step = slinky::max(meta.extent, 1);
      }
      tasks[i] = tasks_above;
      tasks_lb[i] = tasks_above_lb;
    } else {
      // The loop is provably serial iff the loops above it always produce
      // enough tasks: w = ceil_div(target, tasks_above) <= 1 iff
      // tasks_above >= target.
      if (tasks_above_lb >= target_task_count) {
        node.max_workers = slinky::loop::serial;
        if (elide_allowed && meta.funcs_count <= 1) {
          node.step = slinky::max(meta.extent, 1);
        }
      } else {
        slinky::expr w = globals.get(
            slinky::ceil_div(slinky::expr(target_task_count), tasks_above),
            "w");
        node.max_workers = slinky::simplify(slinky::select::make(
            w > 1, slinky::loop::parallel, slinky::loop::serial));
      }
      tasks[i] = slinky::simplify(tasks_above *
                                  slinky::ceil_div(meta.extent, node.step));
      tasks_lb[i] = tasks_above_lb * min_iterations(l, meta);
    }
  }
}

using source_region_map = std::map<std::pair<slinky::var, int>, int>;

int get_source_region(const source_region_map& source_regions, slinky::var buf,
                      int dim) {
  auto it = source_regions.find(std::make_pair(buf, dim));
  return it != source_regions.end() ? it->second : -1;
}

// The source region of `(f, v)`, or -1 if unknown.
int loop_source_region(const slinky::func* f, slinky::var v,
                       const source_region_map& source_regions) {
  if (f == nullptr || !v.defined()) return -1;
  for (const auto& out : f->outputs()) {
    for (int i = 0; i < static_cast<int>(out.dims.size()); ++i) {
      if (out.dims[i] == v) {
        return get_source_region(source_regions, out.sym(),
                                 physical_output_dim(*f, i));
      }
    }
  }
  return -1;
}

// Whether `f` produces an external output of the pipeline. Such a function
// can't be computed inside a loop of its consumers, see `external_output_syms`
// in schedule().
bool produces_external_output(
    const slinky::func& f, const std::set<slinky::var>& external_output_syms) {
  return std::any_of(f.outputs().begin(), f.outputs().end(),
                     [&](const slinky::func::output& o) {
                       return external_output_syms.count(o.buffer->sym()) > 0;
                     });
}

// Sharing a loop between the function that created it and a function being
// fused into it is two decisions, made in order:
// may this function's split cover the loop at all (find_matching_split), and
// what step does the shared loop end up with (reconcile_step).

// The product of the extents of the reduction dimensions of `f`.
slinky::expr reduction_extent(ynn::slinky_globals& globals,
                              const slinky::func& f) {
  slinky::expr result = 1;
  for (const slinky::func::output& out : f.outputs()) {
    for (int d = 0; d < static_cast<int>(out.dims.size()); ++d) {
      if (globals.is_pure_dim(out.dims[d])) continue;
      result = result * out.buffer->dim(d).bounds.extent();
    }
  }
  return result;
}

// Find the split of `f` that covers `consumer_source_region`, or -1 if none
// does. `split_matched` marks the splits already used by outer loops.
//
// The splits don't have to be matched in their declared order: loops over pure
// dims carry no state across iterations, so they can be freely reordered, and
// splits which are not matched simply remain the function's own inner loops.
// Non-pure (reduction) splits do carry state across iterations (they
// accumulate into the same output), so they act as a fence: nothing is matched
// at or beyond the first one.
int find_matching_split(ynn::slinky_globals& globals, const slinky::func& f,
                        const std::vector<ynn::scheduling_split>& loop_splits,
                        const std::vector<bool>& split_matched,
                        int consumer_source_region,
                        const source_region_map& source_regions) {
  if (consumer_source_region == -1) return -1;

  // Whether the search has passed over an unmatched (and non-trivial) split,
  // i.e. matching a later split would reorder the function's loops.
  bool out_of_order = false;

  // Whether matching `split_i` here preserves this function's required
  // blocking order, i.e. no required split before it is still unmatched.
  auto keeps_required_order = [&](int split_i) {
    for (int prev = 0; prev < split_i; ++prev) {
      if (!split_matched[prev] && loop_splits[prev].step_is_required) {
        return false;
      }
    }
    return true;
  };
  // Whether the product of this function's reduction extents is provably below
  // a threshold. The value works for the benchmarks we have; it may well need
  // tuning, or replacing by something derived from the shapes.
  constexpr slinky::index_t small_reduction_threshold = 256;
  auto has_small_reduction = [&]() {
    std::optional<slinky::index_t> k_product =
        slinky::as_constant(slinky::simplify(reduction_extent(globals, f)));
    return k_product && *k_product <= small_reduction_threshold;
  };

  for (int split_i = 0; split_i < loop_splits.size(); ++split_i) {
    if (split_matched[split_i]) continue;
    const ynn::scheduling_split& split = loop_splits[split_i];
    if (!globals.is_pure_dim(split.var)) {
      // We don't want to fuse a reduction dimension because it is likely being
      // broadcasted here, and we don't reorder other splits across it either.
      break;
    }
    if (split.step_is_required && out_of_order &&
        !(keeps_required_order(split_i) || has_small_reduction())) {
      // Matching a required split out of order is fine for any step, but it
      // must not reorder the function's required loops relative to each
      // other: that blocking was chosen deliberately, and inverting it makes
      // the function re-read its inputs. The exception is a function whose
      // reduction is small enough that re-reading costs nothing.
      continue;
    }
    // Instead of comparing forward extents (which causes false positives for
    // unrelated constant extents), we check if both loops share the exact same
    // inferred source region identifier.
    if (loop_source_region(&f, split.var, source_regions) ==
        consumer_source_region) {
      return split_i;
    }
    out_of_order = true;
  }
  return -1;
}

// Splits with extent 1 create no loop and are unavailable for matching. Treat
// them as already matched so they cannot block matching later splits or become
// degenerate levels of the global loop nest.
std::vector<bool> initial_split_matches(
    const ynn::slinky_globals& globals,
    const std::vector<ynn::scheduling_split>& splits) {
  std::vector<bool> matched(splits.size());
  for (int i = 0; i < splits.size(); ++i) {
    matched[i] = prove_true(splits[i].extent == 1, globals.fact_bounds,
                            globals.fact_alignment);
  }
  return matched;
}

// Decide the step of a loop now shared by its owner and `split`. Each side
// makes a claim on the step, and the stronger claim wins:
//
//   required + required : reconcile with the lcm, so the loop is an integer
//                         number of *both* tiles (two producers can require
//                         different tiles for a shared loop, e.g. the two
//                         attention matmuls pick different query tiles).
//   required + anything : the required step, which is a kernel's blocking.
//   chosen   + no claim : propose the chosen step, see below.
//   otherwise           : keep the loop's step. When both sides computed a
//                         real split, each is only meaningful within its own
//                         cache-budget allocation, so combining them (e.g. by
//                         taking the min) degenerates to tiny steps.
//
// "No claim" means the loop runs a single full-extent iteration, see
// is_single_iteration(). `f` is the matching function, and `loop_splits` are
// all of its splits, used to size its iteration space.
void reconcile_step(ynn::slinky_globals& globals, slinky::loop_nest::loop& loop,
                    loop_metadata& meta, const slinky::func& f,
                    const ynn::scheduling_split& split,
                    const std::vector<ynn::scheduling_split>& loop_splits) {
  const slinky::expr& cur_step = loop.step;
  if (split.step_is_required) {
    if (meta.step_is_required &&
        !prove_true(split.step == cur_step, globals.fact_bounds,
                    globals.fact_alignment)) {
      // If the LCM overflows, it clamps at max index_t (assuming no
      // splitting).
      loop.step = lcm_sat(globals, cur_step, split.step);
    } else {
      if (std::optional<slinky::var> v = slinky::as_variable(cur_step)) {
        // This is a special variable which defines partial reduction bounds,
        // so we need to override to match the loop step.
        if (globals.symbols.name(*v).rfind("pr_split", 0) == 0) {
          globals.update_let(*v, split.step);
        }
      }
      loop.step = split.step;
    }
    meta.step_is_required = true;
    // A required step is a kernel's blocking, so it outranks any step an
    // earlier function proposed for this loop.
    meta.proposed_step = slinky::expr();
    return;
  }
  if (meta.step_is_required || !cur_step.defined() || !split.step.defined()) {
    return;
  }
  // Only a pure dim's step is free to change. A partial reduction's "r" loop
  // is steppable, but its step is coupled to the reduction buffer's
  // fold_factor (the kernel's accumulation chunk), so a different step would
  // desync the accumulation and corrupt results.
  if (!globals.is_pure_dim(split.var) ||
      !globals.is_pure_dim(loop.dims[0].dim)) {
    return;
  }
  // This function computed a real split for a loop its owner left as a single
  // task. Propose the split, so that a producer fused into the loop keeps its
  // parallelism instead of running serially - without this, a norm's reduce
  // stages serialize the whole chain they fuse into.
  //
  // Several functions can match the same loop, and the last proposal that
  // applies wins. A function whose split is already the loop's step is asking
  // for nothing, so drop it rather than let it displace an earlier real
  // proposal.
  if (prove_true(split.step == cur_step, globals.fact_bounds,
                 globals.fact_alignment)) {
    return;
  }
  // The function's pure splits and its reduction dimensions span its whole
  // iteration space, so their extent product is the work that would be divided
  // up. Below this threshold the extra task dispatches cost more than the split
  // saves; the value was found experimentally.
  constexpr slinky::index_t min_work_per_split = 128 * 1024;
  slinky::expr work = reduction_extent(globals, f);
  for (const ynn::scheduling_split& ls : loop_splits) {
    if (globals.is_pure_dim(ls.var)) work = work * ls.extent;
  }
  // Both conditions can depend on the runtime shape, so they go into the
  // proposal as a select, which folds away for static shapes. If this
  // function's proposal doesn't apply, it falls back to an earlier proposal (if
  // any) rather than displacing it, e.g. a small producer fused into the loop
  // after a large consumer.
  meta.proposed_step = slinky::select(
      is_single_iteration(globals, meta.extent, cur_step) &&
          min_work_per_split <= work,
      split.step, meta.proposed_step.defined() ? meta.proposed_step : cur_step);
}

}  // namespace

// Logically this function has multiple separate blocks:
// 1) infer symbolic source regions for all buffers to ensure loops are only
//    fused if they share a common source region origin.
// 2) compute the candidate consumer loop path (root to consumer LCA) for a
//    given function.
// 3) walk that path from outermost inwards, matching compatible splits of the
//    function and reconciling loop steps along shared loops.
// 4) append any unmatched splits of the function as new loops in
//    `slinky::loop_nest` (or explicitly `place` untiled functions), and record
//    its innermost scope for upstream producers.
// 5) finalize loop worker counts and elide single-iteration pure loops once
//    the whole nest is built.
slinky::loop_nest ynn_runtime::schedule() {
  // Running LCA of all consumer execution scopes for each buffer.
  std::map<slinky::var, slinky::loop_ref> buffer_consumer_lca;
  slinky::loop_nest loops;
  // Indexed by `loop_ref::index()`; index 0 corresponds to `loops.root()`.
  std::vector<loop_metadata> loop_meta(1);

  // Maps {buffer_sym, dim_index} to its inferred source region unique
  // identifier.
  source_region_map source_regions = infer_source_regions(funcs);

  // Slinky doesn't allocate the pipeline's output buffers, and as a result it
  // also never crops them to the region a loop iteration needs. Fusing the
  // producer of an external output into a loop of one of its consumers would
  // therefore run the producer (and everything it depends on) over the
  // whole output on *every* iteration of that loop.
  std::set<slinky::var> external_output_syms;
  for (const ynn_runtime_value& value : values) {
    if (value.is_valid() && value.is_external_output()) {
      external_output_syms.insert(value.symbol);
    }
  }

  for (int i = static_cast<int>(funcs.size()) - 1; i >= 0; --i) {
    const slinky::func& f = funcs[i];
    // Find the least common ancestor across all consumers of `f`'s outputs.
    slinky::loop_ref consumer_lca = loops.root();
    bool has_consumers = false;
    for (const auto& output : f.outputs()) {
      auto it = buffer_consumer_lca.find(output.buffer->sym());
      if (it == buffer_consumer_lca.end()) continue;
      consumer_lca =
          has_consumers ? loops.lca(consumer_lca, it->second) : it->second;
      has_consumers = true;
    }

    const auto* sched = static_cast<const ynn::scheduling_info*>(f.user_data());
    std::vector<ynn::scheduling_split> loop_splits;
    std::vector<bool> split_matched;
    slinky::loop_ref place_at = consumer_lca;

    // If they are not matching and the extents of f are smaller than parent
    // we are risking over-compute. If it doesn't have schedule info then we
    // just compute at the innermost location.
    if (sched && !sched->loop_splits.empty()) {
      // Reverse to match from outermost inwards.
      loop_splits.assign(sched->loop_splits.rbegin(),
                         sched->loop_splits.rend());
      split_matched = initial_split_matches(globals, loop_splits);

      // Collect the candidate consumer loop path from `root()` down to
      // `consumer_lca`.
      std::vector<slinky::loop_ref> candidate_loops;
      for (slinky::loop_ref l = consumer_lca; !l.is_root();
           l = loops[l].parent) {
        candidate_loops.push_back(l);
      }
      std::reverse(candidate_loops.begin(), candidate_loops.end());

      // Walk the loop nest from the outermost loop inwards, sharing each loop
      // with a split of this function that covers the same source region (see
      // find_matching_split and reconcile_step). We must stop at the first
      // loop of the nest we can't cover: computing the function inside a loop
      // which doesn't slice its output would recompute the function on every
      // iteration of that loop.
      place_at = loops.root();
      for (slinky::loop_ref candidate : candidate_loops) {
        const int consumer_source_region =
            loop_meta[candidate.index()].source_region;
        const int matched_split =
            find_matching_split(globals, f, loop_splits, split_matched,
                                consumer_source_region, source_regions);
        if (matched_split == -1) {
          break;
        }
        split_matched[matched_split] = true;
        reconcile_step(globals, loops[candidate], loop_meta[candidate.index()],
                       f, loop_splits[matched_split], loop_splits);
        place_at = candidate;
      }
    }

    // A function producing an external output cannot be fused into a loop of
    // its consumers without recomputing all of it on every iteration, see
    // `external_output_syms` above.
    if ((sched && sched->force_root) ||
        (!place_at.is_root() &&
         produces_external_output(f, external_output_syms))) {
      place_at = loops.root();
      split_matched.assign(loop_splits.size(), false);
    }

    // Append any splits that did not match an existing consumer loop as new
    // inner loops under `place_at`.
    // NOTE: potentially we could also track how much specific loop are
    // computing by keeping a sum of compute amounts for each of the functions
    // inside of this loop and only schedule loops which have more
    // computations than certain threshold.
    slinky::loop_ref cur = place_at;
    for (size_t j = 0; j < loop_splits.size(); ++j) {
      if (split_matched[j]) continue;
      const ynn::scheduling_split& dim = loop_splits[j];
      assert(!f.outputs().empty());
      cur = loops.add_loop(cur, {f.outputs()[0].buffer, dim.var}, dim.step);
      loop_meta.push_back({dim.extent, dim.step_is_required, /*funcs_count=*/0,
                           loop_source_region(&f, dim.var, source_regions)});
    }

    // If `f` added its own inner loops (`cur != place_at`), Slinky already
    // anchors `f` at `cur` via its tiled output buffer. Otherwise, explicitly
    // pin `f` at `place_at` so Slinky does not sink `f` into unmatched inner
    // consumer loops.
    if (cur == place_at && !f.outputs().empty()) {
      loops.place(f.outputs()[0].buffer, place_at);
    }

    // Count `f` in every enclosing loop so `compute_workers` knows which loops
    // contain multiple functions vs. a single elidable function.
    for (slinky::loop_ref l = cur; !l.is_root(); l = loops[l].parent) {
      loop_meta[l.index()].funcs_count++;
    }

    // Record `cur` in the running consumer LCA of each input buffer.
    for (const auto& input : f.inputs()) {
      auto [it, inserted] =
          buffer_consumer_lca.try_emplace(input.buffer->sym(), cur);
      if (!inserted) {
        it->second = loops.lca(it->second, cur);
      }
    }
  }

  // `thread_count()` reports the number of background worker threads. The
  // thread that invokes the runtime also participates as a worker (it runs
  // tasks while waiting in `thread_pool::wait_for`), so the effective
  // parallelism is one more than the reported count. Without this `+ 1`, a
  // pool with a single background thread (two threads of execution in total)
  // would be scheduled serially, and every other size would be sized one
  // worker short.
  const int max_threads = threadpool() ? threadpool()->thread_count() + 1 : 1;
  compute_workers(globals, max_threads, loops, loop_meta);
  return loops;
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
      return ynn_status_invalid_parameter;
    }

    for (auto& i : runtime->values) {
      if (!i.is_valid()) continue;
      if (i.is_external_output()) {
        assert(i.data);
        assert(i.data->rank == i.rank());
        for (size_t d = 0; d < i.rank(); ++d) {
          slinky::index_t extent_d = evaluate(i.physical_extent(d), ctx);
          i.data->dims[d].set_min_extent(0, extent_d);
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

// Slinky will automatically place allocates on the stack if the allocation is
// smaller than this threshold.
constexpr size_t auto_stack_threshold = 64 * 1024;

void init_context(slinky::eval_context& ctx) {
  // `get_dot_cost_models` is expensive due to a syscall to learn the current
  // CPU core. To avoid this, do it when we create a new context.
  ctx.user_data =
      const_cast<ynn::dot_cost_models*>(&ynn::get_dot_cost_models());
}

}  // namespace

extern "C" {

ynn_runtime::ynn_runtime(ynn::ref_count<const ynn_subgraph> subgraph,
                         slinky::thread_pool* threadpool, uint32_t flags)
    : subgraph(subgraph), flags(flags), globals(subgraph->globals) {
  eval_config.thread_pool = threadpool;
  // Slinky's default check failure handler calls std::abort(), don't let that
  // happen here.
  eval_config.check_failed = [](const slinky::expr& e) {
    YNN_LOG_ERROR() << "Check failed";
  };
  eval_config.call_failed = [](const slinky::call_stmt* c) {
    // This output can be restored after we update slinky past
    // https://github.com/dsharlet/slinky/pull/874
    // YNN_LOG_ERROR() << c->attrs->name << " failed";
  };
  eval_config.base_alignment = YNN_ALLOCATION_ALIGNMENT;
  eval_config.auto_stack_threshold = auto_stack_threshold;

  eval_config.init_context = [](slinky::eval_context& ctx,
                                const slinky::eval_context& parent) {
    init_context(ctx);
  };

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
#ifdef YNN_ENABLE_PERFETTO
    if (ynn::perfetto_session::global()) {
      YNN_LOG_WARNING()
          << "tsl::profiler tracing is overriding perfetto tracing.";
    }
#endif
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
      value.buffer =
          slinky::buffer_expr::make_constant(value.symbol, value.data);
    } else if (value.is_external()) {
      value.make_buffer(*this);

      for (size_t d = 0; d < value.extents.size(); ++d) {
        if (!value.extents[d].defined()) {
          value.buffer->dim(d).bounds = slinky::point(0);
        } else if (const auto v = as_constant(i.physical_extent(d))) {
          value.buffer->dim(d).bounds = slinky::range(0, *v);
        }
      }

      if (value.data) {
        assert(!value.data->base);
        auto new_data = slinky::raw_buffer::make(value.data->rank);
        new_data->elem_size = value.data->elem_size;
        std::copy_n(value.data->dims, value.data->rank, new_data->dims);
        value.data = std::move(new_data);
      } else {
        value.data =
            slinky::raw_buffer::make(i.rank(), ynn::type_size_bytes(i.type));
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
      if (value.is_external() &&
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

      // This should be assert(value.data), but let's do that in a follow-up.
      if (value.data) {
        for (size_t d = 0;
             d < std::min(value.extents.size(), value.data->rank); ++d) {
          if (!value.extents[d].defined() ||
              slinky::is_constant(value.extents[d], 1)) {
            value.data->mutable_dim(d) = slinky::dim::broadcast();
          }
        }
      }
    }
  }

  slinky::loop_nest loops;
  if ((flags & YNN_RUNTIME_FLAG_NO_SCHEDULE) == 0) {
    loops = schedule();
  }

  slinky::build_options options;
  if ((flags & YNN_FLAG_ENABLE_SLINKY_TRACE) != 0) {
    options.trace = true;
  }
#ifdef YNN_ENABLE_PERFETTO
  options.trace = options.trace || get_trace_filename() != nullptr;
#endif
#ifdef YNN_ENABLE_TSL_PROFILER
  options.trace = options.trace || ynn_traceme_enabled();
#endif
#ifdef NDEBUG
  options.no_checks = true;
#endif

  pipeline = slinky::build_pipeline(globals.symbols, {}, inputs, outputs, loops,
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
  slinky::index_t result = slinky::evaluate(reshape_impl, eval_context);
  static_assert(ynn_status_success == 0, "");
  return static_cast<ynn_status>(result);
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
#ifdef YNN_ENABLE_PERFETTO
  ynn::scoped_perfetto_trace trace("ynn_create_runtime");
#endif
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
  if (!pipeline.body.defined()) {
    // This pipeline is a no-op.
    return ynn_status_success;
  }
  init_context(eval_context);
  slinky::index_t result = pipeline.evaluate(eval_context, /*is_set_up=*/true);
  // Heap blocks are reused within an evaluation, but not kept between invokes.
  eval_context.free_pool();
  return result ? ynn_status_error : ynn_status_success;
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
#ifdef YNN_ENABLE_PERFETTO
  ynn::scoped_perfetto_trace trace("ynn_reshape_runtime");
#endif
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
#ifdef YNN_ENABLE_PERFETTO
  ynn::scoped_perfetto_trace trace("ynn_invoke_runtime");
#endif
  YNN_RETURN_IF_ERROR(ynn::validate_runtime(runtime));
  return runtime->invoke();
}
namespace {

int32_t get_max_concurrency(const ynn_runtime& runtime) {
  // Traverse the pipeline body for any loops. If we find a parallel loop, we
  // return `max_int32`. Otherwise, we return 1.
  class visitor : public slinky::recursive_node_visitor {
   public:
    explicit visitor(const ynn_runtime& runtime) : runtime_(runtime) {}

    int32_t result = 1;
    void visit(const slinky::loop* op) override {
      if (!slinky::prove_true(op->max_workers == 1,
                              runtime_.globals.fact_bounds,
                              runtime_.globals.fact_alignment)) {
        result = std::numeric_limits<int32_t>::max();
      }
      slinky::recursive_node_visitor::visit(op);
    }

   private:
    const ynn_runtime& runtime_;
  } v(runtime);
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
