// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {
namespace {

// This is a near-duplicate of that found in src/xnnpack/quantization.h
template <typename T>
inline std::pair<float, int32_t> compute_qd8_params(
    T min, T max, int32_t output_zero_point = 0) {
  const float rmin = std::min(0.0f, static_cast<float>(min));
  const float rmax = std::max(0.0f, static_cast<float>(max));
  const float rdiff = rmax - rmin;
  if (rdiff == 0.0f) {
    return {1.0f, output_zero_point - 128};
  }
  const float scale = 255.0f / rdiff;
  const float inv_scale = rdiff / 255.0f;
  const float zero_point = std::clamp(-rmin * scale, 0.0f, 255.0f);
  // We compute the zero point for uint8, but we want the result to be for int8.
  // This means we can assume the (float) zero point is positive.
  const int32_t nudged_zero_point =
      static_cast<int32_t>(zero_point + 0.5f) + (output_zero_point - 128);
  return {inv_scale, nudged_zero_point};
}

// Call compute_qd8_params for each element.
template <typename T>
auto make_compute_qd8_params_impl(int32_t output_zero_point) {
  return [=](slinky::raw_buffer min_max, slinky::raw_buffer scale,
             slinky::raw_buffer zero_point) -> slinky::index_t {
    assert(min_max.rank > 0);
    const slinky::index_t index_stride_bytes =
        min_max.dim(min_max.rank - 1).stride();
    assert(index_stride_bytes % sizeof(T) == 0);
    const slinky::index_t index_stride = index_stride_bytes / sizeof(T);

    slinky::dim scale_n[1];
    slinky::dim zero_point_n[1];
    slinky::dim min_max_n[1];

    if (!fuse_and_slice_leading_dims<1>(scale_n, scale, zero_point_n,
                                        zero_point, min_max_n, min_max)) {
      return 0;
    }

    assert(is_contiguous(scale_n[0], scale.elem_size));
    assert(is_contiguous(zero_point_n[0], zero_point.elem_size));
    assert(is_contiguous(min_max_n[0], min_max.elem_size));
    const slinky::index_t n = scale_n[0].extent();

    slinky::for_each_element(
        [=](void* scale, void* zero_point, const void* min_max) {
          float* scale_ptr = reinterpret_cast<float*>(scale);
          int32_t* zero_point_ptr = reinterpret_cast<int32_t*>(zero_point);
          const T* min_ptr = reinterpret_cast<const T*>(min_max);
          const T* max_ptr = min_ptr + index_stride;
          for (slinky::index_t i = 0; i < n; ++i) {
            std::tie(scale_ptr[i], zero_point_ptr[i]) =
                compute_qd8_params(min_ptr[i], max_ptr[i], output_zero_point);
          }
        },
        scale, zero_point, min_max);
    return 0;
  };
}

}  // namespace

extern "C" {

ynn_status ynn_define_dynamic_quantization(ynn_subgraph_t subgraph,
                                           uint32_t min_max_id,
                                           enum ynn_type type,
                                           uint32_t* zero_point_id,
                                           uint32_t* scale_id, uint32_t flags) {
  YNN_RETURN_IF_ERROR(validate_subgraph("dynamic_quantization", subgraph));
  YNN_RETURN_IF_ERROR(validate_input_tensor("dynamic_quantization", subgraph,
                                            "min_max_id", min_max_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("dynamic_quantization", subgraph,
                                             "zero_point_id", zero_point_id));
  YNN_RETURN_IF_ERROR(validate_output_tensor("dynamic_quantization", subgraph,
                                             "scale_id", scale_id));

  if (type != ynn_type_int8 && type != ynn_type_uint8) {
    YNN_LOG_ERROR() << "For node `dynamic_quantization`, target type must be "
                       "int8 or uint8, got "
                    << type;
    return ynn_status_invalid_parameter;
  }

  const ynn_value& min_max = subgraph->value(min_max_id);
  if (min_max.rank() == 0) {
    YNN_LOG_ERROR()
        << "For node `dynamic_quantization`, min_max must not be scalar";
    return ynn_status_invalid_parameter;
  }

  ynn_value& scale = subgraph->get_output_value(scale_id, ynn_type_fp32);
  ynn_value& zero_point =
      subgraph->get_output_value(zero_point_id, ynn_type_int32);

  scale.extents = min_max.extents;
  scale.extents.pop_back();
  zero_point.extents = scale.extents;

  int32_t output_zero_point = 0;
  if (type == ynn_type_uint8) {
    output_zero_point = 128;
  }

  ynn_node node;
  node.inputs = {min_max_id};
  node.outputs = {*scale_id, *zero_point_id};
  node.op = ynn_node::dynamic_quantization{output_zero_point};
  node.create = [](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_node::dynamic_quantization& op =
        std::get<ynn_node::dynamic_quantization>(node.op);
    int32_t output_zero_point = op.output_zero_point;

    const ynn_runtime_value& min_max = runtime.value(node.inputs[0]);
    ynn_runtime_value& scale = runtime.value(node.outputs[0]);
    ynn_runtime_value& zero_point = runtime.value(node.outputs[1]);

    scale.make_buffer(runtime);
    zero_point.make_buffer(runtime);

    std::vector<slinky::var> dims = runtime.globals.make_dims(scale.rank());
    slinky::box_expr min_max_bounds =
        make_elementwise_bounds(dims, min_max.extents);
    min_max_bounds.push_back(slinky::bounds(0, 1));

    slinky::func func;
    slinky::call_stmt::attributes attrs;
    attrs.name = "compute_qd8_params";
    switch (min_max.type) {
      case ynn_type_fp32:
        func = slinky::func::make(
            make_compute_qd8_params_impl<float>(output_zero_point),
            {{min_max.buffer, std::move(min_max_bounds)}},
            {{scale.buffer, dims}, {zero_point.buffer, dims}}, attrs);
        break;
      case ynn_type_fp16:
        func = slinky::func::make(
            make_compute_qd8_params_impl<half>(output_zero_point),
            {{min_max.buffer, std::move(min_max_bounds)}},
            {{scale.buffer, dims}, {zero_point.buffer, dims}}, attrs);
        break;
      case ynn_type_bf16:
        func = slinky::func::make(
            make_compute_qd8_params_impl<bfloat16>(output_zero_point),
            {{min_max.buffer, std::move(min_max_bounds)}},
            {{scale.buffer, dims}, {zero_point.buffer, dims}}, attrs);
        break;
      default:
        YNN_UNREACHABLE;
    }

    auto sched =
        runtime.make_schedule(dims, scale.extents, scale.buffer->elem_size());

    func.user_data() = sched.get();
    runtime.scheduling_info_storage.push_back(std::move(sched));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph->add_node(std::move(node));
  return ynn_status_success;
}

}  // extern "C"

}  // namespace ynn
