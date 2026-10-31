// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/copy.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/static_transpose.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/depends_on.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

using reshape_group = ynn_node::reshape::group;

namespace {

std::vector<slinky::expr> to_exprs(const std::vector<size_t>& dims) {
  std::vector<slinky::expr> output_extents(dims.size());
  for (size_t d = 0; d < dims.size(); ++d) {
    if (dims[d] == 1) {
      // Treat static extent 1 as a broadcast.
      output_extents[d] = {};
    } else {
      output_extents[d] = static_cast<slinky::index_t>(dims[d]);
    }
  }
  return output_extents;
}

// Replaces '0' in the output extents of a reshape with the deduced extent.
void deduce_reshape_extent(ynn_node& node, int input_idx,
                           ynn_subgraph& subgraph,
                           const std::vector<slinky::expr>& input_extents,
                           std::vector<slinky::expr>& output_extents) {
  slinky::expr num_elements = 1;
  for (const slinky::expr& extent : input_extents) {
    if (extent.defined()) {
      num_elements *= extent;
    }
  }

  int deduce_dim = -1;
  slinky::expr current_elements = 1;
  for (size_t d = 0; d < output_extents.size(); ++d) {
    slinky::expr extent_d = output_extents[d].defined() ? output_extents[d] : 1;
    if (is_constant(extent_d, 0)) {
      assert(deduce_dim == -1);
      deduce_dim = d;
    } else {
      current_elements *= extent_d;
    }
  }

  if (deduce_dim != -1) {
    slinky::expr deduced_extent = simplify(num_elements / current_elements);
    if (is_pure(deduced_extent)) {
      output_extents[deduce_dim] = deduced_extent;
    } else {
      // This ends up being a pretty complicated expression, so hoist it into
      // a global variable.
      output_extents[deduce_dim] =
          subgraph.globals.get(deduced_extent, "deduced_extent");
    }

    node.add_check(num_elements % current_elements == 0,
                   {"invalid deduced reshape in dimension ", deduce_dim, " of ",
                    ynn_node::input_idx{input_idx}});
  } else {
    node.add_check(num_elements == current_elements,
                   {"invalid reshape of ", ynn_node::input_idx{input_idx}});
  }
}

slinky::expr extent_or_one(const std::vector<slinky::expr>& extents, size_t d) {
  return d < extents.size() && extents[d].defined() ? extents[d] : 1;
}

// If reshaping `input_extents` to `output_extents` only adds or removes
// dimensions of extent 1, returns the permutation of the equivalent transpose.
std::optional<std::vector<int32_t>> reshape_as_transpose(
    const std::vector<slinky::expr>& input_extents,
    const std::vector<slinky::expr>& output_extents) {
  auto in = [&](int i) { return extent_or_one(input_extents, i); };
  auto out = [&](int o) { return extent_or_one(output_extents, o); };
  auto is_one = [](const slinky::expr& x) {
    return slinky::prove_true(x == 1);
  };
  const int in_rank = input_extents.size();
  const int out_rank = output_extents.size();
  std::vector<int32_t> permutation;
  int i = 0;
  int o = 0;
  while (o < out_rank) {
    if (i < in_rank && slinky::prove_true(out(o) == in(i))) {
      permutation.push_back(i++);
      ++o;
    } else if (is_one(out(o))) {
      // Add a new dimension of extent 1.
      permutation.push_back(in_rank);
      ++o;
    } else if (i < in_rank && is_one(in(i))) {
      // Remove this dimension of extent 1.
      ++i;
    } else {
      return std::nullopt;
    }
  }
  for (; i < in_rank; ++i) {
    if (!is_one(in(i))) return std::nullopt;
  }
  return permutation;
}

// Splits dimensions that are not modified by the reshape from the innermost
// and outermost dimensions of each group into their own groups.
std::vector<reshape_group> simplify_reshape_groups(
    const std::vector<slinky::expr>& input_extents,
    const std::vector<slinky::expr>& output_extents,
    const std::vector<reshape_group>& groups) {
  auto same_extent = [&](int i, int o) {
    slinky::expr a = extent_or_one(input_extents, i);
    slinky::expr b = extent_or_one(output_extents, o);
    return slinky::prove_true(a == b);
  };
  std::vector<reshape_group> result;
  int i = 0;
  int o = 0;
  for (reshape_group g : groups) {
    while (g.input_dims > 1 && g.output_dims > 1 && same_extent(i, o)) {
      result.push_back({1, 1});
      ++i;
      ++o;
      --g.input_dims;
      --g.output_dims;
    }
    int outer = 0;
    while (g.input_dims > 1 && g.output_dims > 1 &&
           same_extent(i + g.input_dims - 1, o + g.output_dims - 1)) {
      --g.input_dims;
      --g.output_dims;
      ++outer;
    }
    result.push_back(g);
    result.insert(result.end(), outer, {1, 1});
    i += g.input_dims + outer;
    o += g.output_dims + outer;
  }
  return result;
}

// Requires the first `bounds_rank` dimensions of `buf` to contain the whole
// extent of the dimension, and that there is no padding between dimensions.
void require_dense(slinky::buffer_expr& buf,
                   const std::vector<slinky::expr>& extents, int bounds_rank) {
  for (int d = 0; d < bounds_rank; ++d) {
    if (extents[d].defined()) {
      buf.dim(d).bounds.min = 0;
      buf.dim(d).bounds.max = extents[d] - 1;
    } else {
      buf.dim(d).bounds = slinky::point(0);
    }
  }
  require_contiguous(buf);
}

slinky::func make_reshape(ynn_runtime& runtime, const ynn_node& node) {
  const ynn_node::reshape& op = std::get<ynn_node::reshape>(node.op);
  const ynn_runtime_value& input = runtime.value(node.inputs[0]);
  const ynn_runtime_value& output = runtime.value(node.outputs[0]);
  slinky::buffer_expr_ptr input_buf = input.buffer;
  slinky::buffer_expr_ptr output_buf = output.buffer;
  const std::vector<slinky::expr> input_extents = input.physical_extents();
  const std::vector<slinky::expr> output_extents = output.physical_extents();
  const int in_rank = input_extents.size();
  const int out_rank = output_extents.size();

  // If the outermost dimension is not reshaped, a crop of that dimension is a
  // contiguous range of both the input and output, and the reshape of that
  // range is still a memcpy. This allows the reshape to be computed in loops
  // over this dimension. If the input has no producer, there is no reason to
  // compute this anywhere but the root.
  const bool outer_dim_not_reshaped =
      !op.groups.empty() && op.groups.back().input_dims == 1 &&
      op.groups.back().output_dims == 1 && output_extents.back().defined() &&
      !slinky::prove_true(output_extents.back() == 1) &&
      runtime.subgraph->get_producer(node.inputs[0]);
  const int outer_dims = outer_dim_not_reshaped ? 1 : 0;

  std::vector<slinky::var> dims = runtime.globals.make_dims(out_rank);
  slinky::box_expr bounds(in_rank);

  // Compute the "flat" index of the coordinates in the output.
  slinky::expr flat_out(0);
  slinky::expr output_stride = 1;
  for (int d = 0; d < out_rank - outer_dims; ++d) {
    if (output_extents[d].defined()) {
      flat_out += dims[d] * output_stride;
      output_stride *= output_extents[d];
    }
  }
  slinky::expr input_stride = 1;
  for (int d = 0; d < in_rank - outer_dims; ++d) {
    if (input_extents[d].defined()) {
      bounds[d] = slinky::point((flat_out / input_stride) % input_extents[d]);
      input_stride *= input_extents[d];
    } else {
      bounds[d] = slinky::point(0);
    }
  }
  if (outer_dims > 0) {
    bounds.back() = slinky::point(dims.back());
  }

  // We need all of the dimensions other than the outer dimension, and
  // reshape's definition assumes that there is no padding between dimensions.
  require_dense(*input_buf, input_extents, in_rank - outer_dims);
  require_dense(*output_buf, output_extents, out_rank - outer_dims);

  const int inner_rank = out_rank - outer_dims;
  auto fn = [inner_rank](
                const slinky::buffer<const void>& input,
                const slinky::buffer<void>& output) -> slinky::index_t {
    // `input` contains the same elements as `output` in the same order. The
    // outer dimension of the input might not start at 0, but it has the same
    // bounds as the outer dimension of the output.
    slinky::buffer<const void, max_tensor_rank> input_as_output = output;
    input_as_output.raw_buffer::base = input.raw_buffer::base;
    const int rank = std::min<int>(input_as_output.rank, inner_rank);
    for (int d = 0; d < rank; ++d) {
      input_as_output.mutable_dim(d).set_bounds(0,
                                                input_as_output.dim(d).max());
    }
    slinky::copy(input_as_output, output);
    return 0;
  };

  slinky::call_stmt::attributes attrs;
  attrs.name = "memcpy";
  // Hopefully this just aliases and is a no-op.
  attrs.allow_in_place = 0x1;
  slinky::func result =
      slinky::func::make(std::move(fn), {{input_buf, std::move(bounds)}},
                         {{output_buf, dims}}, std::move(attrs));
  auto sched = std::make_unique<scheduling_info>();
  if (outer_dims > 0) {
    sched->loop_splits.push_back(
        {dims.back(), output_extents.back(), output_extents.back()});
  } else {
    sched->force_root = true;
  }
  result.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));

  return result;

  // TODO(https://github.com/dsharlet/slinky/issues/187): When slinky can
  // properly optimize reshapes, we can change this to use slinky::copy
  // instead, which is less constrained.
}

ynn_status validate_new_shape(const char* node, size_t rank,
                              const size_t* new_dims) {
  YNN_RETURN_IF_ERROR(validate_rank(node, "new_dims", rank));
  if (new_dims == nullptr && rank > 0) {
    YNN_LOG_ERROR() << "For node `" << node
                    << "`, new_dims must be non-null for rank > 0";
    return ynn_status_invalid_parameter;
  }
  return ynn_status_success;
}

// Make a list of groups for a reshape that only modifies the dimensions
// `[begin, end)` of the input.
std::vector<reshape_group> make_reshape_groups(int input_rank, int begin,
                                               int end, int output_rank) {
  std::vector<reshape_group> groups;
  groups.reserve(std::max(input_rank, output_rank));
  for (int d = 0; d < begin; ++d) {
    groups.push_back({1, 1});
  }
  groups.push_back({end - begin, output_rank});
  for (int d = end; d < input_rank; ++d) {
    groups.push_back({1, 1});
  }
  return groups;
}

}  // namespace

std::vector<reshape_group> compose_reshape_groups(
    const std::vector<reshape_group>& a, const std::vector<reshape_group>& b) {
  // Each group of the result is the smallest run of groups of `a` and groups
  // of `b` that covers the same intermediate dimensions (the outputs of `a`
  // and the inputs of `b`).
  std::vector<reshape_group> result;
  size_t i = 0;
  size_t j = 0;
  while (i < a.size() || j < b.size()) {
    reshape_group g = {0, 0};
    int a_dims = 0;
    int b_dims = 0;
    do {
      if (i < a.size() && (a_dims <= b_dims || j == b.size())) {
        g.input_dims += a[i].input_dims;
        a_dims += a[i].output_dims;
        ++i;
      } else {
        assert(j < b.size());
        g.output_dims += b[j].output_dims;
        b_dims += b[j].input_dims;
        ++j;
      }
    } while (a_dims != b_dims);
    result.push_back(g);
  }
  return result;
}

void define_reshape(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                    uint32_t* output_id,
                    std::vector<slinky::expr> output_extents,
                    std::vector<reshape_group> groups) {
  const ynn_value& input = subgraph.value(input_id);

  if (std::optional<std::vector<int32_t>> permutation =
          reshape_as_transpose(input.extents, output_extents)) {
    define_static_transpose(subgraph, node, std::move(*permutation), input_id,
                            output_id);
    return;
  }
  ynn_value& output = subgraph.get_output_value(output_id, input);
  output.extents = std::move(output_extents);

  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = ynn_node::reshape{
      simplify_reshape_groups(input.extents, output.extents, groups)};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    runtime.funcs.push_back(make_reshape(runtime, node));
    return ynn_status_success;
  };
}

void define_copy(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                 uint32_t output_id, uint32_t flags) {
  const ynn_value& input = subgraph.value(input_id);

  // Propagate shape.
  ynn_value& output = subgraph.get_output_value(&output_id, input);
  output.extents = input.extents;

  node.inputs = {input_id};
  node.outputs = {output_id};
  node.op = ynn_node::copy{};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());
    slinky::box_expr bounds =
        make_elementwise_bounds(dims, input.physical_extents());
    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

void define_static_broadcast(ynn_subgraph& subgraph, ynn_node& node,
                             std::vector<size_t> new_dims, uint32_t input_id,
                             uint32_t* output_id) {
  // Propagate shape.
  const ynn_value& input = subgraph.value(input_id);
  ynn_node::static_broadcast op;
  op.new_dims = std::move(new_dims);
  std::vector<slinky::expr> output_extents = input.extents;
  bool noop = true;
  for (size_t d = 0; d < std::min(output_extents.size(), op.new_dims.size());
       ++d) {
    const slinky::index_t new_dim_d = op.new_dims[d];
    if (new_dim_d == 0) {
      continue;
    }
    if ((!output_extents[d].defined() && new_dim_d == 1) ||
        slinky::prove_true(output_extents[d] == new_dim_d)) {
      // This dimension is a no-op.
      op.new_dims[d] = 0;
      continue;
    }

    noop = false;
    if (new_dim_d == 1) {
      output_extents[d] = slinky::expr{};
    } else {
      output_extents[d] = new_dim_d;
    }
    if (d < input.rank() && input.extents[d].defined()) {
      node.add_check(input.extents[d] == 1 || input.extents[d] == new_dim_d,
                     {"invalid broadcast in dimension ", d, " of ",
                      ynn_node::input_idx{0}});
    }
  }
  for (size_t d = output_extents.size(); d < op.new_dims.size(); ++d) {
    // This is a new trailing dimension.
    if (op.new_dims[d] == 1) {
      output_extents.push_back(slinky::expr{});
    } else {
      output_extents.push_back(op.new_dims[d]);
    }
    noop = false;
  }

  if (noop && *output_id == YNN_INVALID_VALUE_ID) {
    // This node is a no-op, skip it.
    *output_id = input_id;
    return;
  }

  // Remove no-op trailing broadcasts.
  while (!op.new_dims.empty() && op.new_dims.back() == 0) {
    op.new_dims.pop_back();
  }

  ynn_value& output = subgraph.get_output_value(output_id, input);
  output.extents = std::move(output_extents);

  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int input_id = node.inputs[0];
    const ynn_runtime_value& input = runtime.value(input_id);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());
    slinky::box_expr bounds = make_broadcast_bounds(
        dims, input.physical_extents(), output.physical_extents());

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
}

extern "C" {

ynn_status ynn_define_copy(ynn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("copy", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("copy", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("copy", subgraph, "output_id", output_id));
  const ynn_value& input = subgraph->value(input_id);

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);

  ynn_node node;
  define_copy(*subgraph, node, input_id, output.id, flags);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_static_reshape(ynn_subgraph_t subgraph, size_t rank,
                                     const size_t* new_dims, uint32_t input_id,
                                     uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_reshape", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("static_reshape", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("static_reshape", subgraph,
                                             "output_id", output_id));
  YNN_RETURN_IF_ERROR(validate_new_shape("static_reshape", rank, new_dims));

  const ynn_value& input = subgraph->value(input_id);

  std::vector<size_t> op_new_dims(new_dims, new_dims + rank);
  std::reverse(op_new_dims.begin(), op_new_dims.end());

  ynn_node node;

  // Propagate shape.
  std::vector<slinky::expr> output_extents = to_exprs(op_new_dims);
  deduce_reshape_extent(node, 0, *subgraph, input.extents, output_extents);

  // A general reshape is one group of all the dimensions.
  define_reshape(*subgraph, node, input_id, output_id,
                 std::move(output_extents),
                 {{input.rank(), static_cast<int32_t>(rank)}});
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

ynn_status ynn_define_static_broadcast(ynn_subgraph_t subgraph, size_t rank,
                                       const size_t* new_dims,
                                       uint32_t input_id, uint32_t* output_id,
                                       uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("static_broadcast", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("static_broadcast", subgraph,
                                            "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("static_broadcast", subgraph,
                                             "output_id", output_id));
  YNN_RETURN_IF_ERROR(validate_new_shape("static_broadcast", rank, new_dims));

  std::vector<size_t> op_new_dims(new_dims, new_dims + rank);
  std::reverse(op_new_dims.begin(), op_new_dims.end());
  ynn_node node;
  define_static_broadcast(*subgraph, node, std::move(op_new_dims), input_id,
                          output_id);
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

ynn_status ynn_define_fuse_dim(ynn_subgraph_t subgraph, int32_t axis,
                               size_t axes_count, uint32_t input_id,
                               uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("fuse_dim", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("fuse_dim", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("fuse_dim", subgraph, "output_id", output_id));
  if (axes_count == 0) {
    YNN_LOG_ERROR() << "For node `fuse_dim`, axes_count must be greater than 0";
    return ynn_status_invalid_parameter;
  }
  const ynn_value& input = subgraph->value(input_id);
  const int rank = input.rank();
  YNN_RETURN_IF_ERROR(validate_axis("fuse_dim", "input", input.rank(), axis));

  auto clamped_axis_to_slinky_dim = [](int rank, int axis) {
    axis = axis < 0 ? axis + rank : axis;
    return std::max(0, std::min(rank - 1 - axis, rank - 1));
  };
  // Handle the reversed dimension range by finding the first and last dimension
  // with clamping.
  const int first = clamped_axis_to_slinky_dim(rank, axis + axes_count - 1);
  const int last = clamped_axis_to_slinky_dim(rank, axis);

  // Propagate shape.
  slinky::expr fused_elements = 1;
  for (int d = first; d <= last; ++d) {
    if (input.extents[d].defined()) {
      fused_elements *= input.extents[d];
    }
  }

  std::vector<slinky::expr> output_extents;
  output_extents.reserve(rank - (last - first));
  for (int d = 0; d < rank; ++d) {
    if (first <= d && d <= last) {
      // This is a fused dimension.
      if (fused_elements.defined()) {
        output_extents.push_back(fused_elements);
        fused_elements = {};
      }
    } else {
      output_extents.push_back(input.extents[d]);
    }
  }

  ynn_node node;
  define_reshape(*subgraph, node, input_id, output_id,
                 std::move(output_extents),
                 make_reshape_groups(rank, first, last + 1, 1));
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

ynn_status ynn_define_split_dim(ynn_subgraph_t subgraph, int32_t axis,
                                size_t num_splits, const size_t* splits,
                                uint32_t input_id, uint32_t* output_id,
                                uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("split_dim", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("split_dim", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("split_dim", subgraph, "output_id", output_id));
  if (splits == nullptr && num_splits > 0) {
    YNN_LOG_ERROR()
        << "For node `split_dim`, splits must be non-null for num_splits > 0";
    return ynn_status_invalid_parameter;
  }
  const ynn_value& input = subgraph->value(input_id);
  YNN_RETURN_IF_ERROR(
      validate_rank("split_dim", "output", input.rank() + num_splits - 1));
  YNN_RETURN_IF_ERROR(validate_axis("split_dim", "input", input.rank(), axis));

  const int slinky_axis = axis_to_slinky_dim(input.rank(), axis);
  std::vector<size_t> new_dims(splits, splits + num_splits);
  std::reverse(new_dims.begin(), new_dims.end());

  ynn_node node;

  // Propagate shape.
  // Split dims is essentially a standard reshape of 1 dimension into N.
  std::vector<slinky::expr> split_extents = to_exprs(new_dims);
  deduce_reshape_extent(node, 0, *subgraph, {input.extents[slinky_axis]},
                        split_extents);

  // And the rest of the dimensions are treated as batch dimensions.
  std::vector<slinky::expr> output_extents;
  output_extents.reserve(input.rank() + new_dims.size() - 1);
  std::copy_n(input.extents.begin(), slinky_axis,
              std::back_inserter(output_extents));
  std::copy(split_extents.begin(), split_extents.end(),
            std::back_inserter(output_extents));
  std::copy(input.extents.begin() + slinky_axis + 1, input.extents.end(),
            std::back_inserter(output_extents));

  define_reshape(*subgraph, node, input_id, output_id,
                 std::move(output_extents),
                 make_reshape_groups(input.rank(), slinky_axis, slinky_axis + 1,
                                     static_cast<int>(new_dims.size())));
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

ynn_status ynn_define_fuse_dims(ynn_subgraph_t subgraph, size_t num_axes,
                                const int32_t* axes, uint32_t input_id,
                                uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("fuse_dims", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("fuse_dims", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("fuse_dims", subgraph, "output_id", output_id));
  if (axes == nullptr && num_axes > 0) {
    YNN_LOG_ERROR()
        << "For node `fuse_dims`, axes must be non-null for num_axes > 0";
    return ynn_status_invalid_parameter;
  }

  const ynn_value& input = subgraph->value(input_id);
  const int rank = input.rank();

  // The set of dimensions to fuse with the next dimension.
  axes_set fuse_axes;
  for (size_t i = 0; i < num_axes; ++i) {
    // Since we are reversing the axes, the first dimension to fuse is actually
    // the next dimension.
    int axis = axis_to_slinky_dim(rank, axes[i] + 1);
    if (axis + 1 < rank) {
      fuse_axes[axis] = true;
    } else {
      // Fusing a broadcast dimension is a no-op.
    }
  }

  // Propagate shape.
  std::vector<slinky::expr> output_extents = input.extents;
  std::vector<reshape_group> groups(rank, {1, 1});
  for (int i = rank - 2; i >= 0; --i) {
    if (!fuse_axes[i]) continue;
    output_extents[i] = slinky::simplify(extent_or_one(output_extents, i) *
                                         extent_or_one(output_extents, i + 1));
    output_extents.erase(output_extents.begin() + i + 1);
    groups[i].input_dims += groups[i + 1].input_dims;
    groups.erase(groups.begin() + i + 1);
  }

  ynn_node node;
  define_reshape(*subgraph, node, input_id, output_id,
                 std::move(output_extents), std::move(groups));
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

ynn_status ynn_define_split_dims(ynn_subgraph_t subgraph, size_t num_axes,
                                 const int32_t* axes, const size_t* splits,
                                 uint32_t input_id, uint32_t* output_id,
                                 uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("split_dims", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("split_dims", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("split_dims", subgraph, "output_id", output_id));
  if (num_axes > 0 && (axes == nullptr || splits == nullptr)) {
    YNN_LOG_ERROR() << "For node `split_dims`, axes and splits must be "
                       "non-null when num_axes > 0";
    return ynn_status_invalid_parameter;
  }
  const ynn_value& input = subgraph->value(input_id);
  const int rank = input.rank();
  struct split {
    int axis;
    size_t factor;
  };
  std::vector<split> op_splits;
  for (size_t i = 0; i < num_axes; ++i) {
    YNN_RETURN_IF_ERROR(validate_axis("split_dims", "input", rank, axes[i]));
    op_splits.push_back({axis_to_slinky_dim(rank, axes[i]), splits[i]});
  }

  std::sort(op_splits.begin(), op_splits.end(),
            [](const split& a, const split& b) { return a.axis > b.axis; });

  ynn_node node;

  // Propagate shape.
  std::vector<slinky::expr> output_extents = input.extents;
  std::vector<reshape_group> groups(rank, {1, 1});
  for (const split& split : op_splits) {
    slinky::expr extent = extent_or_one(output_extents, split.axis);
    output_extents.insert(output_extents.begin() + split.axis, split.factor);
    node.add_check(extent % split.factor == 0,
                   {"invalid split by ", split.factor, " in dimension ",
                    split.axis, " (", extent, ") of ", ynn_node::input_idx{0}});
    output_extents[split.axis + 1] = extent / split.factor;
    groups[split.axis].output_dims++;
  }

  define_reshape(*subgraph, node, input_id, output_id,
                 std::move(output_extents), std::move(groups));
  if (node.is_valid()) {
    subgraph->add_node(std::move(node));
  }
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
