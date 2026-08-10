// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/grouped_dot/grouped_dot.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/subgraph/dot.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

namespace {

auto make_grouped_dot_impl(size_t E, size_t D_in, size_t D_out,
                           const dot_kernel& kernel) {
  return
      [E, D_in, D_out, kernel](
          slinky::raw_buffer input_a, slinky::raw_buffer packed_b,
          slinky::buffer<const int32_t, 1> expert_counts,
          slinky::buffer<const int32_t, 1> expert_offsets,
          slinky::raw_buffer output) -> slinky::index_t {
        size_t n_start = output.dim(0).min();
        size_t n_count = output.dim(0).extent();
        size_t m_start = output.dim(1).min();
        size_t m_count = output.dim(1).extent();

        if (m_count == 0 || n_count == 0) {
          return 0;
        }

        slinky::raw_buffer a = input_a;
        a.slice(1, slinky::in_bounds{static_cast<slinky::index_t>(m_start)});

        slinky::raw_buffer c = output;
        c.slice(0, slinky::in_bounds{static_cast<slinky::index_t>(n_start)});
        c.slice(0, slinky::in_bounds{static_cast<slinky::index_t>(m_start)});

        const void* a_ptr = a.base;
        const void* b_ptr = packed_b.base;
        void* c_ptr = c.base;

        size_t block_n = packed_b.dim(1).extent();
        size_t blocks_n_stride = packed_b.dim(3).stride();

        size_t packed_b_stride = packed_b.dim(2).stride();
        size_t expert_packed_stride = packed_b.dim(4).stride();

        size_t a_stride_m = input_a.dim(1).stride();
        size_t c_stride_m = output.dim(1).stride();

        size_t elem_size_a = input_a.elem_size;
        size_t elem_size_c = output.elem_size;

        ynn::grouped_dot(E, expert_counts.base(), expert_offsets.base(), a_ptr,
                         b_ptr, c_ptr, m_start, m_count, n_start, n_count,
                         D_in, D_out, a_stride_m, c_stride_m, packed_b_stride,
                         expert_packed_stride, kernel, block_n,
                         blocks_n_stride, elem_size_a, elem_size_c);
        return 0;
      };
}

}  // namespace

}  // namespace ynn

extern "C" {

ynn_status ynn_define_grouped_dot(
    ynn_subgraph_t subgraph, uint32_t input_a_id, uint32_t input_b_id,
    uint32_t expert_counts_id, uint32_t expert_offsets_id, uint32_t* output_id,
    uint32_t flags) {
  YNN_RETURN_IF_ERROR(ynn::validate_subgraph("grouped_dot", subgraph));
  YNN_RETURN_IF_ERROR(ynn::validate_input_tensor("grouped_dot", subgraph,
                                                 "input_a", input_a_id));
  YNN_RETURN_IF_ERROR(ynn::validate_input_tensor("grouped_dot", subgraph,
                                                 "input_b", input_b_id));
  YNN_RETURN_IF_ERROR(ynn::validate_input_tensor(
      "grouped_dot", subgraph, "expert_counts", expert_counts_id));
  YNN_RETURN_IF_ERROR(ynn::validate_input_tensor(
      "grouped_dot", subgraph, "expert_offsets", expert_offsets_id));
  YNN_RETURN_IF_ERROR(ynn::validate_output_tensor("grouped_dot", subgraph,
                                                  "output", output_id));

  const ynn_value& input_a = subgraph->value(input_a_id);
  const ynn_value& input_b = subgraph->value(input_b_id);
  const ynn_value& expert_counts = subgraph->value(expert_counts_id);
  const ynn_value& expert_offsets = subgraph->value(expert_offsets_id);

  if (!ynn::type_is_floating_point(input_a.type)) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, input_a must be a floating point type, "
           "got "
        << input_a.type;
    return ynn_status_unsupported_parameter;
  }
  if (!ynn::type_is_floating_point(input_b.type)) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, input_b must be a floating point type, "
           "got "
        << input_b.type;
    return ynn_status_unsupported_parameter;
  }
  if (expert_counts.type != ynn_type_int32 ||
      expert_offsets.type != ynn_type_int32) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, expert_counts and expert_offsets must be "
           "int32, got "
        << expert_counts.type << " and " << expert_offsets.type;
    return ynn_status_unsupported_parameter;
  }

  if (input_a.rank() != 2) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, input_a must have rank 2, got "
        << input_a.rank();
    return ynn_status_invalid_parameter;
  }
  if (input_b.rank() != 3) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, input_b must have rank 3, got "
        << input_b.rank();
    return ynn_status_invalid_parameter;
  }
  if (expert_counts.rank() != 1) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, expert_counts must have rank 1, got "
        << expert_counts.rank();
    return ynn_status_invalid_parameter;
  }
  if (expert_offsets.rank() != 1) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, expert_offsets must have rank 1, got "
        << expert_offsets.rank();
    return ynn_status_invalid_parameter;
  }

  auto opt_E = slinky::as_constant(expert_counts.extents[0]);
  auto opt_D_out = slinky::as_constant(input_b.extents[0]);
  auto opt_D_in = slinky::as_constant(input_b.extents[1]);
  auto opt_b_E = slinky::as_constant(input_b.extents[2]);
  auto opt_a_D_in = slinky::as_constant(input_a.extents[0]);
  auto opt_offsets_len = slinky::as_constant(expert_offsets.extents[0]);

  if (!opt_E.has_value() || !opt_D_out.has_value() || !opt_D_in.has_value() ||
      !opt_a_D_in.has_value() || *opt_a_D_in != *opt_D_in) {
    return ynn_status_invalid_parameter;
  }
  if (opt_b_E.has_value() && *opt_b_E != *opt_E) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, expert dimension in input_b (" << *opt_b_E
        << ") must match expert_counts (" << *opt_E << ")";
    return ynn_status_invalid_parameter;
  }
  if (opt_offsets_len.has_value() && *opt_offsets_len != *opt_E + 1) {
    YNN_LOG_ERROR()
        << "For node `grouped_dot`, expert_offsets size (" << *opt_offsets_len
        << ") must be E + 1 (" << *opt_E + 1 << ")";
    return ynn_status_invalid_parameter;
  }

  ynn_type type_a = input_a.type;
  ynn_type type_b = input_b.type;
  // TODO: Support non-floating-point / quantized types (e.g. int8 -> int32).
  ynn_type type_c = ynn_type_fp32;

  if (*output_id != YNN_INVALID_VALUE_ID) {
    const ynn_value& out_val = subgraph->value(*output_id);
    if (!ynn::type_is_floating_point(out_val.type)) {
      YNN_LOG_ERROR()
          << "For node `grouped_dot`, output must be a floating point type, "
             "got "
          << out_val.type;
      return ynn_status_unsupported_parameter;
    }
    if (out_val.rank() != 0 && out_val.rank() != 2) {
      YNN_LOG_ERROR()
          << "For node `grouped_dot`, output must have rank 2, got "
          << out_val.rank();
      return ynn_status_invalid_parameter;
    }
    type_c = out_val.type;
  }

  size_t E = *opt_E;
  size_t D_out = *opt_D_out;
  size_t D_in = *opt_D_in;

  // Pack B.
  ynn::dot_type dot_type_val = {type_a, type_b, type_c};
  ynn::dot_shape dot_shape_val = {/*m=*/ynn::unknown_dot_extent,
                                  /*n=*/D_out,
                                  /*k1=*/D_in};
  ynn::dot_kernel kernel =
      ynn::get_dot_kernel(dot_type_val, dot_shape_val, {}, 0, false);
  if (kernel.kernel == nullptr) {
    YNN_LOG_ERROR() << "For node `grouped_dot`, no kernel found for type "
                    << type_a << "x" << type_b << "->" << type_c;
    return ynn_status_unsupported_parameter;
  }
  if (D_in % kernel.tile_k != 0) {
    YNN_LOG_ERROR() << "For node `grouped_dot`, D_in (" << D_in
                    << ") must be a multiple of kernel.tile_k ("
                    << kernel.tile_k << ")";
    return ynn_status_unsupported_parameter;
  }

  uint32_t packed_b_id =
      ynn::define_pack_b(*subgraph, dot_type_val, kernel, /*num_k_dims=*/1,
                         /*consistent_arithmetic=*/false, input_b_id);

  ynn_value& output = subgraph->get_output_value(output_id, type_c);
  output.extents = {D_out,
                    input_a.extents[1]};  // [NK, D_out] -> Slinky: [D_out, NK]

  ynn_node node;
  node.inputs = {input_a_id, packed_b_id, expert_counts_id, expert_offsets_id};
  node.outputs = {*output_id};
  node.op = ynn_node::grouped_dot{};

  node.create = [E, D_in, D_out, kernel](const ynn_node& node,
                                         ynn_runtime& runtime) {
    ynn_runtime_value& input_a = runtime.value(node.inputs[0]);
    ynn_runtime_value& packed_b = runtime.value(node.inputs[1]);
    ynn_runtime_value& expert_counts = runtime.value(node.inputs[2]);
    ynn_runtime_value& expert_offsets = runtime.value(node.inputs[3]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    ynn::require_contiguous(*input_a.buffer, 1);
    ynn::require_contiguous(*packed_b.buffer, 3);
    output.make_buffer(runtime);
    ynn::require_contiguous(*output.buffer, 1);

    std::vector<slinky::var> dims = runtime.globals.make_dims(2, "gdot");

    slinky::box_expr a_bounds = {
        ynn::all_bounds(D_in),
        ynn::elementwise_bounds(dims[1], input_a.physical_extent(1))};
    slinky::box_expr b_bounds = {
        ynn::all_bounds(packed_b.physical_extents()[0]),
        ynn::all_bounds(packed_b.physical_extents()[1]),
        ynn::all_bounds(packed_b.physical_extents()[2]),
        ynn::all_bounds(packed_b.physical_extents()[3]),
        ynn::all_bounds(packed_b.physical_extents()[4])};
    slinky::box_expr counts_bounds = {ynn::all_bounds(E)};
    slinky::box_expr offsets_bounds = {ynn::all_bounds(E + 1)};

    slinky::call_stmt::attributes attrs;
    attrs.name = "grouped_dot";

    slinky::func func =
        slinky::func::make(ynn::make_grouped_dot_impl(E, D_in, D_out, kernel),
                           {{input_a.buffer, std::move(a_bounds)},
                            {packed_b.buffer, std::move(b_bounds)},
                            {expert_counts.buffer, std::move(counts_bounds)},
                            {expert_offsets.buffer, std::move(offsets_bounds)}},
                           {{output.buffer, dims}}, std::move(attrs));

    slinky::expr m = output.physical_extent(1);
    slinky::expr n = output.physical_extent(0);
    slinky::expr block_n = packed_b.physical_extents()[1];

    slinky::expr split_n = slinky::min(n, block_n);
    slinky::expr split_m = slinky::min(m, 16);

    std::vector<slinky::expr> splits = {split_n, split_m};

    auto sched = runtime.make_schedule(
        dims, output.physical_extents(), output.buffer->elem_size(), splits);

    for (size_t dim_idx = 0; dim_idx < dims.size(); ++dim_idx) {
      slinky::var sym = dims[dim_idx];
      for (size_t s = 0; s < sched->loop_splits.size(); ++s) {
        if (sched->loop_splits[s].var == sym) {
          sched->loop_splits[s].step_is_required = true;
          break;
        }
      }
    }

    ynn::scheduled_buffer sched_output_buffer = {output.buffer, 0};
    sched->scheduled_buffers.push_back(std::move(sched_output_buffer));

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));
    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };

  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"
