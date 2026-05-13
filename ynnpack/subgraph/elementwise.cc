// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"
#include "ynnpack/kernels/lut/lut.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/kernels/unary/unary.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/utils.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

using ynn::operator<<;  // NOLINT(misc-unused-using-decls)

namespace ynn {

namespace {

// Call a unary kernel.
auto make_unary_elementwise_impl(unary_kernel_fn kernel, unary_params params) {
  return [kernel, params](slinky::raw_buffer a,
                          slinky::raw_buffer x) -> slinky::index_t {
    slinky::dim a_dims[2], x_dims[2];

    if (!fuse_and_slice_leading_dims<2>(&x_dims[0], x, &a_dims[0], a)) {
      return 0;
    }

    // We don't support broadcasting of `a` here in the innermost
    // dimension (and it would waste computation).
    assert(is_contiguous(a_dims[0], a.elem_size));

    const slinky::index_t x_n_extent = x_dims[0].extent();
    const slinky::index_t a_m_stride = a_dims[1].stride();
    const slinky::index_t x_m_extent = x_dims[1].extent();
    const slinky::index_t x_m_stride = x_dims[1].stride();

    slinky::for_each_element(
        [=, &params](void* x, const void* a) {
          kernel(x_m_extent, x_n_extent, a_m_stride, a, x_m_stride, x, &params);
        },
        x, a);
    return 0;
  };
}

// Call a lut kernel.
auto make_lut_impl(lut_kernel_fn kernel) {
  return [kernel](slinky::raw_buffer a, slinky::raw_buffer lut,
                  slinky::raw_buffer x) -> slinky::index_t {
    slinky::dim a_dims[1], x_dims[1];

    if (!fuse_and_slice_leading_dims<1>(&x_dims[0], x, &a_dims[0], a)) {
      return 0;
    }

    // We don't support broadcasting of `a` here in the innermost
    // dimension (and it would waste computation).
    assert(is_contiguous(a_dims[0], a.elem_size));
    assert(is_contiguous(x_dims[0], x.elem_size));

    const slinky::index_t x_n_extent = x_dims[0].extent();

    slinky::for_each_element(
        [=](void* x, const void* a) { kernel(x_n_extent, a, lut.base, x); }, x,
        a);
    return 0;
  };
}

// Call a binary kernel.
auto make_binary_elementwise_impl(binary_kernel_fn kernel) {
  return [kernel](slinky::raw_buffer a, slinky::raw_buffer b,
                  slinky::raw_buffer x) -> slinky::index_t {
    slinky::dim a_dims[2], b_dims[2], x_dims[2];

    if (!fuse_and_slice_leading_dims<2>(&x_dims[0], x, &a_dims[0], a,
                                        &b_dims[0], b)) {
      return 0;
    }

    const slinky::index_t x_m_extent = x_dims[1].extent();
    const slinky::index_t x_n_extent = x_dims[0].extent();
    const slinky::index_t a_m_stride = a_dims[1].stride();
    const slinky::index_t a_n_stride = a_dims[0].stride();
    const slinky::index_t b_m_stride = b_dims[1].stride();
    const slinky::index_t b_n_stride = b_dims[0].stride();
    const slinky::index_t x_m_stride = x_dims[1].stride();

    slinky::for_each_element(
        [=](void* x, const void* a, const void* b) {
          kernel(x_m_extent, x_n_extent, a_m_stride, a_n_stride, a, b_m_stride,
                 b_n_stride, b, x_m_stride, x, nullptr);
        },
        x, a, b);
    return 0;
  };
}

int compute_allow_in_place(const ynn_node& node, const ynn_subgraph& subgraph) {
  assert(node.outputs.size() == 1);
  int result = 0;
  for (int i = 0; i < node.inputs.size(); ++i) {
    if (allow_in_place(node.inputs[i], node.outputs[0], subgraph)) {
      result |= 1 << i;
    }
  }
  return result;
}

auto make_ternary_elementwise_impl(ternary_kernel_fn kernel) {
  return
      [kernel](slinky::raw_buffer a, slinky::raw_buffer b, slinky::raw_buffer c,
               slinky::raw_buffer x) -> slinky::index_t {
        slinky::dim a_dims[2], b_dims[2], c_dims[2], x_dims[2];

        if (!fuse_and_slice_leading_dims<2>(&x_dims[0], x, &a_dims[0], a,
                                            &b_dims[0], b, &c_dims[0], c)) {
          return 0;
        }

        const slinky::index_t x_m_extent = x_dims[1].extent();
        const slinky::index_t x_n_extent = x_dims[0].extent();
        const slinky::index_t a_m_stride = a_dims[1].stride();
        const slinky::index_t a_n_stride = a_dims[0].stride();
        const slinky::index_t b_m_stride = b_dims[1].stride();
        const slinky::index_t b_n_stride = b_dims[0].stride();
        const slinky::index_t c_m_stride = c_dims[1].stride();
        const slinky::index_t c_n_stride = c_dims[0].stride();
        const slinky::index_t x_m_stride = x_dims[1].stride();

        slinky::for_each_element(
            [=](void* x, const void* a, const void* b, const void* c) {
              kernel(x_m_extent, x_n_extent, a_m_stride, a_n_stride, a,
                     b_m_stride, b_n_stride, b, c_m_stride, c_n_stride, c,
                     x_m_stride, x, nullptr);
            },
            x, a, b, c);
        return 0;
      };
}

auto make_dequantize_dot_impl(dequantize_dot_kernel_fn kernel,
                              dequantize_dot_params params) {
  return [kernel, params](slinky::raw_buffer dot, slinky::raw_buffer a_offset,
                          slinky::raw_buffer b_offset,
                          slinky::raw_buffer a_scale,
                          slinky::raw_buffer b_scale, slinky::raw_buffer offset,
                          slinky::raw_buffer output) -> slinky::index_t {
    using slinky::index_t;

    index_t n_extent = 1;
    // These are intentionally left uninitialized, if the extent is 1 they
    // should be unused.
    index_t b_offset_n_stride, b_scale_n_stride, offset_n_stride;
    if (is_contiguous(dot.dim(0), dot.elem_size) &&
        is_broadcast(a_offset.dim(0)) && is_broadcast(a_scale.dim(0))) {
      const slinky::dim& n = slice_dim0(output);
      assert(is_contiguous(n, output.elem_size));
      const slinky::in_bounds n_min{n.min()};
      n_extent = n.extent();

      dot.slice(0, n_min);
      a_offset.slice(0);
      a_scale.slice(0);
      b_offset_n_stride = slice_dim0(b_offset, n_min).stride();
      b_scale_n_stride = slice_dim0(b_scale, n_min).stride();
      offset_n_stride = slice_dim0(offset, n_min).stride();
    }

    index_t m_extent = 1;
    // These are intentionally left uninitialized, if the extent is 1 they
    // should be unused.
    index_t m_stride, dot_m_stride, a_offset_m_stride, a_scale_m_stride;
    if (is_broadcast(b_offset.dim(0)) && is_broadcast(b_scale.dim(0)) &&
        is_broadcast(offset.dim(0))) {
      const slinky::dim& m = slice_dim0(output);
      const slinky::in_bounds m_min{m.min()};
      m_extent = m.extent();
      m_stride = m.stride();
      dot_m_stride = slice_dim0(dot, m_min).stride();
      a_offset_m_stride = slice_dim0(a_offset, m_min).stride();
      a_scale_m_stride = slice_dim0(a_scale, m_min).stride();
      b_offset.slice(0);
      b_scale.slice(0);
      offset.slice(0);
    }

    if (n_extent <= 0 || m_extent <= 0) {
      return 0;
    }

    slinky::for_each_element(
        [=, &params](void* output, const void* dot, const void* a_offset,
                     const void* b_offset, const void* offset,
                     const void* a_scale, const void* b_scale) {
          kernel(m_extent, n_extent, dot_m_stride, dot, a_offset_m_stride,
                 a_offset, b_offset_n_stride, b_offset, offset_n_stride, offset,
                 a_scale_m_stride, a_scale, b_scale_n_stride, b_scale, m_stride,
                 output, &params);
        },
        output, dot, a_offset, b_offset, offset, a_scale, b_scale);

    return 0;
  };
}

ynn_status create_unary(const ynn_node& node, ynn_runtime& runtime,
                        unary_kernel_fn kernel) {
  assert(node.inputs.size() == 1);
  assert(node.outputs.size() == 1);

  const unary_params& params =
      std::get<ynn_node::unary_elementwise>(node.op).params;

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);
  x.make_buffer(runtime);
  std::vector<slinky::var> dims = runtime.globals.make_dims(x.rank());
  slinky::box_expr bounds = make_elementwise_bounds(dims, a.physical_extents());

  slinky::call_stmt::attributes attrs;
  attrs.name = to_string(std::get<ynn_node::unary_elementwise>(node.op).op);
  attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);

  slinky::func func = slinky::func::make(
      make_unary_elementwise_impl(kernel, params),
      {{a.buffer, std::move(bounds)}}, {{x.buffer, dims}}, std::move(attrs));

  auto sched =
      runtime.make_schedule(dims, x.physical_extents(), x.buffer->elem_size());
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));

  runtime.funcs.push_back(std::move(func));
  return ynn_status_success;
}

ynn_status create_lut(const ynn_node& node, ynn_runtime& runtime,
                      lut_kernel_fn kernel) {
  assert(node.inputs.size() == 2);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  const ynn_runtime_value& lut = runtime.value(node.inputs[1]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);
  std::vector<slinky::var> dims = runtime.globals.make_dims(x.rank());
  slinky::box_expr bounds = make_elementwise_bounds(dims, a.physical_extents());

  slinky::box_expr lut_bounds = {
      slinky::interval_expr(0, 1 << type_size_bytes(a.type))};

  slinky::call_stmt::attributes attrs;
  attrs.name = "lut";
  attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);

  auto func = slinky::func::make(
      make_lut_impl(kernel),
      {{a.buffer, std::move(bounds)}, {lut.buffer, std::move(lut_bounds)}},
      {{x.buffer, dims}}, std::move(attrs));

  auto sched =
      runtime.make_schedule(dims, x.physical_extents(), x.buffer->elem_size());
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
}

ynn_status create_binary(const ynn_node& node, ynn_runtime& runtime,
                         binary_kernel_fn kernel) {
  assert(node.inputs.size() == 2);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  const ynn_runtime_value& b = runtime.value(node.inputs[1]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);

  slinky::call_stmt::attributes attrs;
  attrs.name = to_string(std::get<ynn_node::binary_elementwise>(node.op).op);
  attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);

  // Make the dims and bounds for this operation (does not depend on the
  // specific operation.)
  std::vector<slinky::var> dims = runtime.globals.make_dims(x.rank());
  slinky::box_expr a_bounds =
      make_elementwise_bounds(dims, a.physical_extents());
  slinky::box_expr b_bounds =
      make_elementwise_bounds(dims, b.physical_extents());
  a_bounds.resize(a.rank());
  b_bounds.resize(b.rank());
  auto func = slinky::func::make(
      make_binary_elementwise_impl(kernel),
      {{a.buffer, std::move(a_bounds)}, {b.buffer, std::move(b_bounds)}},
      {{x.buffer, dims}}, std::move(attrs));

  auto sched =
      runtime.make_schedule(dims, x.physical_extents(), x.buffer->elem_size());
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
}

ynn_status create_ternary(const ynn_node& node, ynn_runtime& runtime,
                          ternary_kernel_fn kernel) {
  assert(node.inputs.size() == 3);
  assert(node.outputs.size() == 1);

  const ynn_runtime_value& a = runtime.value(node.inputs[0]);
  const ynn_runtime_value& b = runtime.value(node.inputs[1]);
  const ynn_runtime_value& c = runtime.value(node.inputs[2]);
  ynn_runtime_value& x = runtime.value(node.outputs[0]);

  x.make_buffer(runtime);

  slinky::call_stmt::attributes attrs;
  attrs.name = to_string(std::get<ynn_node::ternary_elementwise>(node.op).op);
  attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);

  // Make the dims and bounds for this operation (does not depend on the
  // specific operation.)
  std::vector<slinky::var> dims = runtime.globals.make_dims(x.rank());
  slinky::box_expr a_bounds =
      make_elementwise_bounds(dims, a.physical_extents());
  slinky::box_expr b_bounds =
      make_elementwise_bounds(dims, b.physical_extents());
  slinky::box_expr c_bounds =
      make_elementwise_bounds(dims, c.physical_extents());
  a_bounds.resize(a.rank());
  b_bounds.resize(b.rank());
  c_bounds.resize(c.rank());
  auto func = slinky::func::make(make_ternary_elementwise_impl(kernel),
                                 {{a.buffer, std::move(a_bounds)},
                                  {b.buffer, std::move(b_bounds)},
                                  {c.buffer, std::move(c_bounds)}},
                                 {{x.buffer, dims}}, attrs);

  auto sched =
      runtime.make_schedule(dims, x.physical_extents(), x.buffer->elem_size());
  func.user_data() = sched.get();
  runtime.scheduling_info_storage.push_back(std::move(sched));
  runtime.funcs.push_back(std::move(func));

  return ynn_status_success;
}

void infer_shape(ynn_node& node, ynn_subgraph& subgraph) {
  ynn_value& x = subgraph.value(node.outputs[0]);
  int output_rank = 0;
  for (uint32_t i = 0; i < node.inputs.size(); ++i) {
    if (node.inputs[i] == YNN_INVALID_VALUE_ID) continue;
    output_rank = std::max(output_rank, subgraph.value(node.inputs[i]).rank());
  }
  x.extents.resize(output_rank);
  for (uint32_t i = 0; i < node.inputs.size(); ++i) {
    if (node.inputs[i] == YNN_INVALID_VALUE_ID) continue;
    for (size_t d = 0; d < x.rank(); ++d) {
      subgraph.infer_elementwise_shape(node, i, 0, d, d);
    }
  }
}

}  // namespace

void define_unary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                  uint32_t output_id, ynn_unary_operator op,
                  unary_kernel_fn kernel, const unary_params& params) {
  // Make the node.
  node.inputs = {input_a_id};
  node.outputs = {output_id};
  node.op = ynn_node::unary_elementwise{op, params};
  infer_shape(node, subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_unary(node, runtime, kernel);
  };
}

void define_binary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                   uint32_t input_b_id, uint32_t output_id,
                   ynn_binary_operator op, binary_kernel_fn kernel) {
  node.inputs = {input_a_id, input_b_id};
  node.outputs = {output_id};
  node.op = ynn_node::binary_elementwise{op};
  infer_shape(node, subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_binary(node, runtime, kernel);
  };
}

void define_ternary(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_a_id,
                    uint32_t input_b_id, uint32_t input_c_id,
                    uint32_t output_id, ternary_op op,
                    ternary_kernel_fn kernel) {
  node.inputs = {input_a_id, input_b_id, input_c_id};
  node.outputs = {output_id};
  node.op = ynn_node::ternary_elementwise{op};
  infer_shape(node, subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_ternary(node, runtime, kernel);
  };
}

void define_lut(ynn_subgraph& subgraph, ynn_node& node, uint32_t input_id,
                uint32_t lut_id, uint32_t& output_id) {
  const ynn_value& a = subgraph.value(input_id);
  ynn_value& x = subgraph.get_output_value(&output_id, a);

  // Find kernel.
  lut_kernel_fn kernel = get_lut_kernel(a.type, x.type);
  assert(kernel);

  node.inputs = {input_id, lut_id};
  node.outputs = {output_id};
  node.op = ynn_node::lut{};

  // Propagate shape from A only.
  x.extents.resize(a.rank());
  for (size_t d = 0; d < x.rank(); ++d) {
    subgraph.infer_elementwise_shape(node, /*input_idx=*/0, /*output_idx=*/0,
                                     /*input_dim=*/d, /*output_dim=*/d);
  }

  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_lut(node, runtime, kernel);
  };
}

bool define_dequantize_dot(ynn_subgraph& subgraph, ynn_node& node,
                           ynn_type output_type, uint32_t dot_id,
                           uint32_t a_offset_id, uint32_t b_offset_id,
                           uint32_t a_scale_id, uint32_t b_scale_id,
                           uint32_t offset_id, uint32_t& output_id,
                           const dequantize_dot_params& params) {
  dequantize_dot_kernel_fn kernel = get_dequantize_dot_kernel(output_type);
  if (kernel == nullptr) {
    return false;
  }

  const ynn_value& dot = subgraph.value(dot_id);
  ynn_value& output = subgraph.get_output_value(&output_id, output_type);

  // Propagate shape from dot.
  output.extents = dot.extents;

  node.inputs = {dot_id,     a_offset_id, b_offset_id,
                 a_scale_id, b_scale_id,  offset_id};
  node.outputs = {output_id};
  node.op = ynn_node::dequantize_dot{params};

  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::dequantize_dot& op =
        std::get<ynn_node::dequantize_dot>(node.op);
    const ynn_runtime_value& dot = runtime.value(node.inputs[0]);
    const ynn_runtime_value& a_offset = runtime.value(node.inputs[1]);
    const ynn_runtime_value& b_offset = runtime.value(node.inputs[2]);
    const ynn_runtime_value& a_scale = runtime.value(node.inputs[3]);
    const ynn_runtime_value& b_scale = runtime.value(node.inputs[4]);
    const ynn_runtime_value& offset = runtime.value(node.inputs[5]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime);

    std::vector<slinky::var> dims = runtime.globals.make_dims(output.rank());

    slinky::box_expr bounds;
    for (size_t i = 0; i < dims.size(); ++i) {
      bounds.push_back(slinky::point(dims[i]));
    }

    slinky::call_stmt::attributes attrs;
    attrs.name = "dequantize_dot";
    attrs.allow_in_place = compute_allow_in_place(node, *runtime.subgraph);
    auto func = slinky::func::make(make_dequantize_dot_impl(kernel, op.params),
                                   {{dot.buffer, bounds},
                                    {a_offset.buffer, bounds},
                                    {b_offset.buffer, bounds},
                                    {a_scale.buffer, bounds},
                                    {b_scale.buffer, bounds},
                                    {offset.buffer, bounds}},
                                   {{output.buffer, dims}}, std::move(attrs));

    auto sched = runtime.make_schedule(dims, output.physical_extents(),
                                       output.buffer->elem_size());
    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  return true;
}

ynn_status define_unary(ynn_subgraph_t subgraph, ynn_unary_operator op,
                        uint32_t input_a_id, unary_params params,
                        uint32_t* output_id, uint32_t flags) {
  const ynn_value& a = subgraph->value(input_a_id);
  ynn_value& x = subgraph->get_output_value(output_id, a);

  // Find the kernel.
  unary_kernel_fn kernel = get_unary_kernel(op, a.type, x.type);
  if (!kernel) {
    unary_kernel_fn float_kernel =
        get_unary_kernel(op, ynn_type_fp32, ynn_type_fp32);
    if (float_kernel) {
      uint32_t a_float_id = YNN_INVALID_VALUE_ID;
      ynn_status status =
          ynn_define_dequantize(subgraph, input_a_id, a.zero_point_id,
                                a.scale_id, ynn_type_fp32, &a_float_id,
                                /*flags=*/0);
      if (status != ynn_status_success) {
        return status;
      }

      uint32_t x_float_id = YNN_INVALID_VALUE_ID;
      status =
          define_unary(subgraph, op, a_float_id, params, &x_float_id, flags);
      if (status != ynn_status_success) {
        return status;
      }

      return ynn_define_quantize(subgraph, x_float_id, x.type, x.zero_point_id,
                                 x.scale_id, output_id, /*flags=*/0);
    }

    YNN_LOG_ERROR() << "Unsupported unary operator " << op << " for input type "
                    << a.type << " and output type " << x.type;
    return ynn_status_unsupported_parameter;
  }

  // Make the node.
  ynn_node node;
  ynn::define_unary(*subgraph, node, input_a_id, *output_id, op, kernel,
                    params);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

extern "C" {

ynn_status ynn_define_unary(ynn_subgraph_t subgraph, ynn_unary_operator op,
                            uint32_t input_a_id, uint32_t* output_id,
                            uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("unary", subgraph));
  if (op == ynn_unary_invalid) {
    YNN_LOG_ERROR() << "invalid unary operator of input " << input_a_id;
    return ynn_status_invalid_parameter;
  }
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("unary", subgraph, "input_a_id", input_a_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("unary", subgraph, "output_id", output_id));

  if (op == ynn_unary_convert) {
    if (*output_id == YNN_INVALID_VALUE_ID) {
      YNN_LOG_ERROR() << "For node `unary` with operator `ynn_unary_convert`, "
                         "output_id must be valid";
      return ynn_status_invalid_parameter;
    }

    const ynn_value& x = subgraph->value(*output_id);
    return ynn_define_convert(subgraph, input_a_id, x.type, x.zero_point_id,
                              x.scale_id, output_id, flags);
  }

  return define_unary(subgraph, op, input_a_id, get_unary_params(op), output_id,
                      flags);
}

ynn_status ynn_define_unary_polynomial(ynn_subgraph_t subgraph,
                                       uint32_t input_id, size_t degree,
                                       const float* coefficients,
                                       uint32_t* output_id, uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("unary_polynomial", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("unary_polynomial", subgraph,
                                            "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("unary_polynomial", subgraph,
                                             "output_id", output_id));

  if (degree > 3) {
    YNN_LOG_ERROR() << "Only degree 3 polynomials are supported.";
    return ynn_status_unsupported_parameter;
  }

  unary_params params;
  params.poly3.c0 = coefficients[0];
  params.poly3.c1 = 1 <= degree ? coefficients[1] : 0.0f;
  params.poly3.c2 = 2 <= degree ? coefficients[2] : 0.0f;
  params.poly3.c3 = 3 <= degree ? coefficients[3] : 0.0f;

  return define_unary(subgraph, ynn_unary_poly3, input_id, params, output_id,
                      flags);
}

ynn_status ynn_define_convert(ynn_subgraph_t subgraph, uint32_t input_id,
                              ynn_type output_type, uint32_t zero_point_id,
                              uint32_t scale_id, uint32_t* output_id,
                              uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("unary", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("unary", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("unary", subgraph, "output_id", output_id));

  const ynn_value& a = subgraph->value(input_id);
  ynn_value& x = subgraph->get_output_value(output_id, output_type,
                                            zero_point_id, scale_id);

  const uint32_t x_scale_id =
      scale_id != YNN_INVALID_VALUE_ID ? scale_id : x.scale_id;
  const uint32_t x_zero_point_id =
      zero_point_id != YNN_INVALID_VALUE_ID ? zero_point_id : x.zero_point_id;

  const bool a_is_quantized = a.scale_id != YNN_INVALID_VALUE_ID ||
                              a.zero_point_id != YNN_INVALID_VALUE_ID;
  const bool x_is_quantized = x_scale_id != YNN_INVALID_VALUE_ID ||
                              x_zero_point_id != YNN_INVALID_VALUE_ID;

  if (type_is_integral(x.type) && x_is_quantized &&
      type_is_floating_point(a.type)) {
    return ynn_define_quantize(subgraph, input_id, x.type, x_zero_point_id,
                               x_scale_id, output_id, flags);
  }

  if (type_is_integral(a.type) && a_is_quantized &&
      type_is_floating_point(x.type)) {
    return ynn_define_dequantize(subgraph, input_id, a.zero_point_id,
                                 a.scale_id, output_type, output_id, flags);
  }

  // We can use a convert kernel if quantization parameters match, or if there
  // are no quantization parameters.
  unary_kernel_fn kernel =
      (a.scale_id == x_scale_id && a.zero_point_id == x_zero_point_id)
          ? get_unary_kernel(ynn_unary_convert, a.type, x.type)
          : nullptr;
  if (!kernel) {
    // We either have quantization data to handle for a requantization, or we
    // don't have a kernel for this conversion. Handle it by converting to an
    // intermediate float.
    uint32_t intermediate_id = YNN_INVALID_VALUE_ID;
    ynn_status status =
        ynn_define_tensor(subgraph, ynn_type_fp32, /*rank=*/0, /*dims=*/nullptr,
                          /*data=*/nullptr, /*flags=*/0, &intermediate_id);
    if (status != ynn_status_success) {
      return status;
    }

    status = ynn_define_convert_v2(subgraph, input_id, ynn_type_fp32,
                                   &intermediate_id,
                                   /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }

    return ynn_define_convert(subgraph, intermediate_id, x.type,
                              x_zero_point_id, x_scale_id, output_id,
                              /*flags=*/0);
  }

  // Make the node.
  ynn_node node;
  node.inputs = {input_id};
  node.outputs = {*output_id};
  node.op = ynn_node::unary_elementwise{ynn_unary_convert, unary_params{}};
  infer_shape(node, *subgraph);
  node.create = [kernel](const ynn_node& node, ynn_runtime& runtime) {
    return create_unary(node, runtime, kernel);
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_convert_v2(ynn_subgraph_t subgraph, uint32_t input_id,
                                 ynn_type output_type, uint32_t* output_id,
                                 uint32_t flags) {
  return ynn_define_convert(subgraph, input_id, output_type,
                            YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
                            output_id, flags);
}

ynn_status ynn_define_quantize(ynn_subgraph_t subgraph, uint32_t input_id,
                               ynn_type output_type, uint32_t zero_point_id,
                               uint32_t scale_id, uint32_t* output_id,
                               uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("quantize", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("quantize", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("quantize", subgraph,
                                            "zero_point_id", zero_point_id,
                                            /*optional=*/true));
  YNN_RETURN_IF_ERROR(validate_input_tensor("quantize", subgraph, "scale_id",
                                            scale_id,
                                            /*optional=*/true));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("quantize", subgraph, "output_id", output_id));

  const ynn_value& a = subgraph->value(input_id);
  ynn_value& x = subgraph->get_output_value(output_id, output_type,
                                            zero_point_id, scale_id);

  uint32_t x_scale_id =
      scale_id != YNN_INVALID_VALUE_ID ? scale_id : x.scale_id;
  uint32_t x_zero_point_id =
      zero_point_id != YNN_INVALID_VALUE_ID ? zero_point_id : x.zero_point_id;
  if (x_scale_id == YNN_INVALID_VALUE_ID &&
      x_zero_point_id == YNN_INVALID_VALUE_ID) {
    return ynn_define_convert_v2(subgraph, input_id, output_type, output_id,
                                 flags);
  }

  if (x_scale_id == YNN_INVALID_VALUE_ID) {
    x_scale_id = subgraph->get_scalar_value_id(1.0f);
  }
  if (x_zero_point_id == YNN_INVALID_VALUE_ID) {
    x_zero_point_id = subgraph->get_scalar_value_id(0);
  }

  ternary_op op = x.type == ynn_type_int8 ? ternary_op::quantize_int8
                                          : ternary_op::quantize_uint8;
  ternary_kernel_fn kernel =
      get_ternary_kernel(op, a.type, ynn_type_fp32, ynn_type_int32, x.type);
  if (!kernel && a.type != ynn_type_fp32) {
    YNN_LOG_DEBUG() << "No ternary kernel for operator " << op
                    << ", input type " << a.type << " and output type "
                    << x.type << ", attempting to convert to fp32.";

    // Try converting a to fp32 first.
    kernel = get_ternary_kernel(op, ynn_type_fp32, ynn_type_fp32,
                                ynn_type_int32, x.type);
    assert(kernel);
    uint32_t input_float_id = YNN_INVALID_VALUE_ID;
    ynn_status status = ynn_define_convert_v2(subgraph, input_id, ynn_type_fp32,
                                              &input_float_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    input_id = input_float_id;
  }

  if (!kernel) {
    YNN_LOG_ERROR() << "No ternary kernel found for quantize operator " << op
                    << " with input type " << a.type << " and output type "
                    << x.type;
    return ynn_status_unsupported_parameter;
  }

  ynn_node node;
  define_ternary(*subgraph, node, input_id, x_scale_id, x_zero_point_id,
                 *output_id, op, kernel);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_dequantize(ynn_subgraph_t subgraph, uint32_t input_id,
                                 uint32_t zero_point_id, uint32_t scale_id,
                                 ynn_type output_type, uint32_t* output_id,
                                 uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("dequantize", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("dequantize", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("dequantize", subgraph,
                                            "zero_point_id", zero_point_id,
                                            /*optional=*/true));
  YNN_RETURN_IF_ERROR(validate_input_tensor("dequantize", subgraph, "scale_id",
                                            scale_id,
                                            /*optional=*/true));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("dequantize", subgraph, "output_id", output_id));

  const ynn_value& a = subgraph->value(input_id);
  ynn_value& x = subgraph->get_output_value(output_id, output_type);

  uint32_t a_scale_id =
      scale_id != YNN_INVALID_VALUE_ID ? scale_id : a.scale_id;
  uint32_t a_zero_point_id =
      zero_point_id != YNN_INVALID_VALUE_ID ? zero_point_id : a.zero_point_id;
  if (a_scale_id == YNN_INVALID_VALUE_ID &&
      a_zero_point_id == YNN_INVALID_VALUE_ID) {
    return ynn_define_convert_v2(subgraph, input_id, output_type, output_id,
                                 flags);
  }

  if (a_scale_id == YNN_INVALID_VALUE_ID) {
    a_scale_id = subgraph->get_scalar_value_id(1.0f);
  }
  if (a_zero_point_id == YNN_INVALID_VALUE_ID) {
    // First try to find a multiply kernel for this.
    binary_kernel_fn kernel =
        get_binary_kernel(ynn_binary_multiply, a.type, ynn_type_fp32, x.type);
    if (kernel) {
      ynn_node node;
      define_binary(*subgraph, node, input_id, a_scale_id, *output_id,
                    ynn_binary_multiply, kernel);
      subgraph->add_node(std::move(node));
      return ynn_status_success;
    }

    // Try converting from fp32.
    kernel = get_binary_kernel(ynn_binary_multiply, a.type, ynn_type_fp32,
                               ynn_type_fp32);
    if (kernel) {
      uint32_t output_float_id = YNN_INVALID_VALUE_ID;
      ynn_status status = ynn_define_tensor(
          subgraph, ynn_type_fp32, /*rank=*/0, /*dims=*/nullptr,
          /*data=*/nullptr, /*flags=*/0, &output_float_id);
      if (status != ynn_status_success) {
        return status;
      }

      ynn_node node;
      define_binary(*subgraph, node, input_id, a_scale_id, output_float_id,
                    ynn_binary_multiply, kernel);
      subgraph->add_node(std::move(node));
      return ynn_define_convert_v2(subgraph, output_float_id, x.type, output_id,
                                   flags);
    }

    // We didn't handle it with a multiply, try to do it with a dequantize
    // kernel.
    a_zero_point_id = subgraph->get_scalar_value_id(0);
  }

  ternary_kernel_fn kernel = get_ternary_kernel(
      ternary_op::dequantize, a.type, ynn_type_int32, ynn_type_fp32, x.type);

  if (kernel) {
    ynn_node node;
    define_ternary(*subgraph, node, input_id, a_zero_point_id, a_scale_id,
                   *output_id, ternary_op::dequantize, kernel);
    subgraph->add_node(std::move(node));
    return ynn_status_success;
  } else if (x.type != ynn_type_fp32) {
    YNN_LOG_DEBUG() << "No ternary kernel for operator dequantize, input type "
                    << a.type << ", output type " << x.type
                    << "; attempting to dequantize to fp32.";

    kernel = get_ternary_kernel(ternary_op::dequantize, a.type, ynn_type_int32,
                                ynn_type_fp32, ynn_type_fp32);
    assert(kernel);

    uint32_t output_float_id = YNN_INVALID_VALUE_ID;
    ynn_status status =
        ynn_define_tensor(subgraph, ynn_type_fp32, /*rank=*/0, /*dims=*/nullptr,
                          /*data=*/nullptr, /*flags=*/0, &output_float_id);
    if (status != ynn_status_success) {
      return status;
    }

    ynn_node node;
    define_ternary(*subgraph, node, input_id, a_zero_point_id, a_scale_id,
                   output_float_id, ternary_op::dequantize, kernel);
    subgraph->add_node(std::move(node));

    return ynn_define_convert_v2(subgraph, output_float_id, x.type, output_id,
                                 flags);
  } else {
    YNN_LOG_ERROR() << "No ternary kernel found for dequantize operator with "
                       "input type "
                    << a.type << " and output type " << x.type;
    return ynn_status_unsupported_parameter;
  }
}

ynn_status ynn_define_binary(ynn_subgraph_t subgraph, ynn_binary_operator op,
                             uint32_t input_a_id, uint32_t input_b_id,
                             uint32_t* output_id, uint32_t flags) {
  // Validate arguments.
  YNN_RETURN_IF_ERROR(validate_subgraph("binary", subgraph));
  if (op == ynn_binary_invalid) {
    YNN_LOG_ERROR() << "For node `binary`, invalid binary operator of inputs "
                    << input_a_id << ", " << input_b_id;
    return ynn_status_invalid_parameter;
  }
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("binary", subgraph, "input_a_id", input_a_id));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("binary", subgraph, "input_b_id", input_b_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("binary", subgraph, "output_id", output_id));
  const ynn_value& a = subgraph->value(input_a_id);
  const ynn_value& b = subgraph->value(input_b_id);
  ynn_value& x = subgraph->get_output_value(output_id, a);

  // Find the kernel.
  const bool is_quantized = a.scale_id != YNN_INVALID_VALUE_ID ||
                            a.zero_point_id != YNN_INVALID_VALUE_ID ||
                            b.scale_id != YNN_INVALID_VALUE_ID ||
                            b.zero_point_id != YNN_INVALID_VALUE_ID ||
                            x.scale_id != YNN_INVALID_VALUE_ID ||
                            x.zero_point_id != YNN_INVALID_VALUE_ID;
  binary_kernel_fn kernel =
      is_quantized ? nullptr : get_binary_kernel(op, a.type, b.type, x.type);
  if (!kernel) {
    YNN_LOG_DEBUG() << "No binary kernel for operator " << op
                    << ", input types " << a.type << ", " << b.type
                    << ", output type " << x.type
                    << "; attempting to convert to fp32.";

    if (!(a.type == ynn_type_fp32 && b.type == ynn_type_fp32)) {
      uint32_t a_fp32_id = YNN_INVALID_VALUE_ID;
      if (a.type != ynn_type_fp32) {
        ynn_status status = ynn_define_dequantize(
            subgraph, input_a_id, a.zero_point_id, a.scale_id, ynn_type_fp32,
            &a_fp32_id, /*flags=*/0);
        if (status != ynn_status_success) return status;
      } else {
        a_fp32_id = input_a_id;
      }
      uint32_t b_fp32_id = YNN_INVALID_VALUE_ID;
      if (b.type != ynn_type_fp32) {
        ynn_status status = ynn_define_dequantize(
            subgraph, input_b_id, b.zero_point_id, b.scale_id, ynn_type_fp32,
            &b_fp32_id, /*flags=*/0);
        if (status != ynn_status_success) return status;
      } else {
        b_fp32_id = input_b_id;
      }

      if (x.type == ynn_type_fp32) {
        return ynn_define_binary(subgraph, op, a_fp32_id, b_fp32_id, output_id,
                                 flags);
      } else {
        uint32_t x_fp32_id = YNN_INVALID_VALUE_ID;
        ynn_status status = ynn_define_binary(subgraph, op, a_fp32_id,
                                              b_fp32_id, &x_fp32_id, flags);
        if (status != ynn_status_success) return status;

        return ynn_define_quantize(subgraph, x_fp32_id, x.type, x.zero_point_id,
                                   x.scale_id, output_id, /*flags=*/0);
      }
    }

    YNN_LOG_ERROR() << "Unsupported binary operator " << op
                    << " for input types " << a.type << ", " << b.type
                    << " and output type " << x.type;
    return ynn_status_unsupported_parameter;
  }

  ynn_node node;
  ynn::define_binary(*subgraph, node, input_a_id, input_b_id, *output_id, op,
                     kernel);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

ynn_status ynn_define_lut(ynn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t lut_id, uint32_t* output_id,
                          uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("lut", subgraph));
  YNN_RETURN_IF_ERROR(
      validate_input_tensor("lut", subgraph, "input_id", input_id));
  YNN_RETURN_IF_ERROR(validate_input_tensor("lut", subgraph, "lut_id", lut_id));
  YNN_RETURN_IF_ERROR(
      validate_output_tensor("lut", subgraph, "output_id", output_id));

  const ynn_value& a = subgraph->value(input_id);
  const ynn_value& lut = subgraph->value(lut_id);

  if (!ynn::type_is_integral(a.type)) {
    YNN_LOG_ERROR() << "For node `lut`, input must be integral, got " << a.type;
    return ynn_status_invalid_parameter;
  }
  if (!ynn::type_is_integral(lut.type)) {
    YNN_LOG_ERROR() << "For node `lut`, lut must be integral, got " << lut.type;
    return ynn_status_invalid_parameter;
  }
  if (lut.rank() != 1) {
    YNN_LOG_ERROR() << "For node `lut`, lut must be 1D, got " << lut.rank();
    return ynn_status_invalid_parameter;
  }

  ynn_node node;
  define_lut(*subgraph, node, input_id, lut_id, *output_id);
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
