// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/builder/simplify.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/depends_on.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

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
    slinky::expr deduced_extent = num_elements / current_elements;
    if (is_pure(deduced_extent)) {
      output_extents[deduce_dim] = deduced_extent;
    } else {
      // This ends up being a pretty complicated expression, so hoist it into
      // a global variable.
      output_extents[deduce_dim] =
          subgraph.make_global_variable(deduced_extent, "deduced_extent");
    }

    node.checks.push_back({
        num_elements % current_elements == 0,
        {"invalid deduced reshape in dimension ", deduce_dim, " of ",
         ynn_node::input_idx{input_idx}},
    });
  }
}

slinky::func make_reshape(ynn_runtime& runtime,
                          slinky::buffer_expr_ptr input_buf,
                          const std::vector<slinky::expr>& input_extents,
                          slinky::buffer_expr_ptr output_buf,
                          const std::vector<slinky::expr>& output_extents) {
  const size_t in_rank = input_buf->rank();
  const size_t out_rank = output_extents.size();
  assert(out_rank == output_buf->rank());

  std::vector<slinky::var> dims = make_dims(out_rank, runtime.symbols);
  slinky::box_expr bounds(in_rank);

  // Compute the "flat" index of the coordinates in the output.
  slinky::expr flat_out(0);
  for (size_t d = 0; d < out_rank; ++d) {
    slinky::expr current_term = dims[d];

    for (size_t d1 = 0; d1 < d; ++d1) {
      if (output_extents[d1].defined()) {
        current_term *= output_extents[d1];
      }
    }

    flat_out += current_term;
  }

  slinky::expr input_extents_cum_prod = 1;
  for (size_t d = 0; d < in_rank; ++d) {
    if (input_extents[d].defined()) {
      bounds[d] =
          slinky::point((flat_out / input_extents_cum_prod) % input_extents[d]);
      input_extents_cum_prod *= input_extents[d];
    } else {
      bounds[d] = slinky::point(0);
    }
  }

  // Reshape's definition assumes that there is no padding between dimensions.
  require_contiguous(*input_buf);
  require_contiguous(*output_buf);

  // We also need to assume that we compute at least the min = 0... element of
  // the input.
  for (size_t d = 0; d < in_rank; ++d) {
    if (input_extents[d].defined()) {
      input_buf->dim(d).bounds.min = 0;
      input_buf->dim(d).bounds.max = input_extents[d] - 1;
    } else {
      input_buf->dim(d).bounds = slinky::point(0);
    }
  }

  slinky::func::input input{input_buf, bounds};
  slinky::func::output output{output_buf, dims};

  auto fn = [](const slinky::buffer<const void>& input,
               const slinky::buffer<void>& output) -> slinky::index_t {
    for (int d = 0; d < input.rank; ++d) {
      assert(input.dim(d).min() == 0);
    }
    slinky::buffer<const void, YNN_MAX_TENSOR_RANK> input_as_output = output;
    input_as_output.raw_buffer::base = input.raw_buffer::base;
    for (size_t d = 0; d < input_as_output.rank; ++d) {
      input_as_output.dim(d).set_bounds(0, input_as_output.dim(d).max());
    }
    slinky::copy(input_as_output, output);
    return 0;
  };

  slinky::call_stmt::attributes attrs;
  attrs.name = "memcpy";
  // Hopefully this just aliases and is a no-op.
  attrs.allow_in_place = 0x1;
  slinky::func result = slinky::func::make(std::move(fn), {std::move(input)},
                                           {std::move(output)}, attrs);
  auto sched = std::make_unique<scheduling_info>();
  sched->force_root = true;
  result.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));

  return result;

  // TODO(https://github.com/dsharlet/slinky/issues/187): When slinky can
  // properly optimize reshapes, we can change this to use slinky::copy
  // instead, which is less constrained (see comments above). return
  // slinky::func::make_copy(std::move(input), std::move(output));
}

}  // namespace

extern "C" {

ynn_status ynn_define_copy(ynn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  const ynn_value& input = subgraph->value(input_id);

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = input.extents;

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = ynn_node::copy{};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    std::vector<slinky::var> dims = make_dims(output.rank(), runtime.symbols);
    slinky::box_expr bounds = make_elementwise_bounds(dims, input.extents);
    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_static_reshape(ynn_subgraph_t subgraph, size_t rank,
                                     const size_t* new_dims, uint32_t input_id,
                                     uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  assert(rank == 0 || new_dims);
  const ynn_value& input = subgraph->value(input_id);
  ynn_value& output = subgraph->get_output_value(output_id, input);

  ynn_node::static_reshape op;
  op.new_dims.assign(new_dims, new_dims + rank);
  std::reverse(op.new_dims.begin(), op.new_dims.end());

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};

  // Propagate shape.
  output.extents = to_exprs(op.new_dims);
  deduce_reshape_extent(node, 0, *subgraph, input.extents, output.extents);

  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = make_reshape(runtime, input.buffer, input.extents,
                             output.buffer, output.extents);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_static_broadcast(ynn_subgraph_t subgraph, size_t rank,
                                       const size_t* new_dims,
                                       uint32_t input_id, uint32_t* output_id,
                                       uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  assert(rank == 0 || new_dims);
  const ynn_value& input = subgraph->value(input_id);

  ynn_node::static_broadcast op;
  op.new_dims.assign(new_dims, new_dims + rank);
  std::reverse(op.new_dims.begin(), op.new_dims.end());

  ynn_node node;

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  const std::vector<slinky::expr>& input_extents = input.extents;
  output.extents.clear();
  output.extents.resize(op.new_dims.size());
  for (size_t d = 0; d < output.rank(); ++d) {
    if (d < input.rank() && op.new_dims[d] == 0) {
      output.extents[d] = input_extents[d];
    } else {
      output.extents[d] = static_cast<slinky::index_t>(op.new_dims[d]);
      if (d < input.rank() && input_extents[d].defined()) {
        node.checks.push_back({
            input_extents[d] == 1 || input_extents[d] == op.new_dims[d],
            {"invalid broadcast in dimension ", d, " of ",
             ynn_node::input_idx{0}},
        });
      }
    }
  }

  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int input_id = node.inputs[0];
    const ynn_runtime_value& input = runtime.value(input_id);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    std::vector<slinky::var> dims = make_dims(output.rank(), runtime.symbols);
    slinky::box_expr bounds =
        make_broadcast_bounds(dims, input.extents, output.extents);

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_static_expand_dims(ynn_subgraph_t subgraph,
                                         size_t num_new_axes,
                                         const int32_t* new_axes,
                                         uint32_t input_id, uint32_t* output_id,
                                         uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  assert(num_new_axes == 0 || new_axes);
  const ynn_value& input = subgraph->value(input_id);
  ynn_value& output = subgraph->get_output_value(output_id, input);

  const int new_rank = input.rank() + num_new_axes;
  ynn_node::static_expand_dims op;
  for (size_t i = 0; i < num_new_axes; ++i) {
    op.new_axes[axis_to_slinky_dim(new_rank, new_axes[i])] = true;
  }

  // Propagate shape.
  output.extents.resize(new_rank);
  auto input_d = input.extents.begin();
  for (size_t d = 0; d < output.rank() && input_d != input.extents.end(); ++d) {
    if (op.new_axes[d]) {
      output.extents[d] = {};
    } else {
      output.extents[d] = *input_d++;
    }
  }

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn::axes_set& new_axes =
        std::get<ynn_node::static_expand_dims>(node.op).new_axes;
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);
    assert(input.rank() == input.extents.size());
    assert(output.rank() == input.rank() + new_axes.count());

    std::vector<slinky::var> dims = make_dims(output.rank(), runtime.symbols);
    slinky::box_expr bounds(input.rank());

    auto bounds_d = bounds.begin();
    for (size_t d = 0; d < output.rank() && bounds_d != bounds.end(); ++d) {
      if (!new_axes[d]) {
        *bounds_d++ = slinky::point(dims[d]);
      }
    }

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = slinky::func::make_copy({input.buffer, std::move(bounds)},
                                        {output.buffer, std::move(dims)});
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_fuse_dim(ynn_subgraph_t subgraph, int32_t axis,
                               size_t axes_count, uint32_t input_id,
                               uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  assert(axes_count > 0);
  const ynn_value& input = subgraph->value(input_id);

  ynn_node::fuse_dim op;
  // Since the first axis was specified with the dims in reverse order, we
  // actually want the last dim here.
  op.axis = axis_to_slinky_dim(input.rank(), axis) + 1 - axes_count;
  op.axes_count = axes_count;

  // Propagate shape.
  slinky::expr fused_elements = 1;
  for (size_t d = op.axis; d < op.axis + op.axes_count; ++d) {
    if (input.extents[d].defined()) {
      fused_elements *= input.extents[d];
    }
  }

  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents.clear();
  output.extents.reserve(input.rank() - op.axes_count + 1);

  for (size_t d = 0; d < input.rank(); ++d) {
    if (op.axis <= d && d < op.axis + op.axes_count) {
      // This is a fused dimension.
      if (fused_elements.defined()) {
        output.extents.push_back(fused_elements);
        fused_elements = {};
      }
    } else {
      output.extents.push_back(input.extents[d]);
    }
  }

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = make_reshape(runtime, input.buffer, input.extents,
                             output.buffer, output.extents);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_split_dim(ynn_subgraph_t subgraph, int32_t axis,
                                size_t num_splits, const size_t* splits,
                                uint32_t input_id, uint32_t* output_id,
                                uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(output_id);
  assert(num_splits == 0 || splits);
  const ynn_value& input = subgraph->value(input_id);
  ynn_value& output = subgraph->get_output_value(output_id, input);

  ynn_node::split_dim op;
  op.axis = axis_to_slinky_dim(input.rank(), axis);
  op.new_dims.assign(splits, splits + num_splits);
  std::reverse(op.new_dims.begin(), op.new_dims.end());

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};

  // Propagate shape.
  // Split dims is essentially a standard reshape of 1 dimension into N.
  std::vector<slinky::expr> split_extents = to_exprs(op.new_dims);
  deduce_reshape_extent(node, 0, *subgraph, {input.extents[op.axis]},
                        split_extents);

  // And the rest of the dimensions are treated as batch dimensions.
  output.extents.clear();
  output.extents.reserve(input.rank() + op.new_dims.size() - 1);
  std::copy_n(input.extents.begin(), op.axis,
              std::back_inserter(output.extents));
  std::copy(split_extents.begin(), split_extents.end(),
            std::back_inserter(output.extents));
  std::copy(input.extents.begin() + op.axis + 1, input.extents.end(),
            std::back_inserter(output.extents));

  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());

    auto func = make_reshape(runtime, input.buffer, input.extents,
                             output.buffer, output.extents);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_fuse_dims(ynn_subgraph_t subgraph, size_t num_axes,
                                const int32_t* axes, uint32_t input_id,
                                uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(subgraph->is_valid_value(*output_id));
  const ynn_value& input = subgraph->value(input_id);
  ynn_value& output = subgraph->get_output_value(output_id, input);

  ynn_node::fuse_dims op;
  for (size_t i = 0; i < num_axes; ++i) {
    // Since we are reversing the axes, the first dimension to fuse is actually
    // the next dimension.
    op.axes[axis_to_slinky_dim(input.rank(), axes[i] + 1)] = true;
  }

  // Propagate shape.
  output.extents = input.extents;
  for (int i = op.axes.size() - 1; i >= 0; --i) {
    if (!op.axes[i]) continue;
    output.extents[i] *= output.extents[i + 1];
    output.extents.erase(output.extents.begin() + i + 1);
  }

  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = make_reshape(runtime, input.buffer, input.extents,
                             output.buffer, output.extents);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_split_dims(ynn_subgraph_t subgraph, size_t num_axes,
                                 const int32_t* axes, const size_t* splits,
                                 uint32_t input_id, uint32_t* output_id,
                                 uint32_t flags) {
  // Validate arguments.
  assert(subgraph);
  assert(subgraph->is_valid_value(input_id));
  assert(subgraph->is_valid_value(*output_id));
  assert(num_axes == 0 || axes);
  assert(num_axes == 0 || splits);
  const ynn_value& input = subgraph->value(input_id);

  using split = ynn_node::split_dims::split;
  ynn_node::split_dims op;
  for (size_t i = 0; i < num_axes; ++i) {
    op.splits.push_back({axis_to_slinky_dim(input.rank(), axes[i]), splits[i]});
  }

  std::sort(op.splits.begin(), op.splits.end(),
            [](const split& a, const split& b) { return a.axis > b.axis; });

  ynn_node node;

  // Propagate shape.
  ynn_value& output = subgraph->get_output_value(output_id, input);
  output.extents = input.extents;
  for (const ynn_node::split_dims::split& split : op.splits) {
    output.extents.insert(output.extents.begin() + split.axis, split.factor);
    node.checks.push_back({
        output.extents[split.axis] % split.factor == 0,
        {"invalid split by ", split.factor, " in dimension ", split.axis, " (",
         output.extents[split.axis], ") of ", ynn_node::input_idx{0}},
    });
    output.extents[split.axis + 1] /= split.factor;
  }

  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = std::move(op);
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const int input_id = node.inputs[0];
    const ynn_runtime_value& input = runtime.value(input_id);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size());
    auto func = make_reshape(runtime, input.buffer, input.extents,
                             output.buffer, output.extents);
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
