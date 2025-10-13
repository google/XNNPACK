// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/subgraph.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/build_config.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/tensor.h"
#include "external/+_repo_rules+slinky/base/thread_pool.h"
#include "external/+_repo_rules+slinky/builder/simplify.h"
#include "external/+_repo_rules+slinky/builder/substitute.h"
#include "external/+_repo_rules+slinky/runtime/buffer.h"
#include "external/+_repo_rules+slinky/runtime/evaluate.h"
#include "external/+_repo_rules+slinky/runtime/expr.h"

std::string ynn_value::name() const {
  return name_prefix() + std::to_string(id);
}

ynn_status ynn_value::set_external_shape(size_t rank, const size_t* dims) {
  if (!is_external_input()) {
    YNN_LOG_ERROR() << "value " << id
                    << " is not an external input, cannot set shape.";
    return ynn_status_invalid_parameter;
  }
  assert(data);
  if (this->rank() != rank) {
    YNN_LOG_ERROR() << "new shape rank " << rank
                    << " does not match existing rank " << this->rank();
    return ynn_status_invalid_parameter;
  }

  size_t physical_dims[YNN_MAX_TENSOR_RANK];
  ynn_status status = ynn::to_physical_shape(type, rank, dims, physical_dims);
  if (status != ynn_status_success) {
    return status;
  }

  for (int i = 0; i < rank; ++i) {
    auto extent = slinky::as_constant(extents[rank - 1 - i]);
    if (extent && *extent != physical_dims[i]) {
      YNN_LOG_ERROR() << "value " << id << " has fixed shape " << *extent
                      << " in dimension " << i << " and cannot be reshaped to "
                      << physical_dims[i];
      return ynn_status_invalid_parameter;
    }
  }

  ynn::init_buffer(*data, ynn::type_size_bytes(type), rank, physical_dims,
                   data->base);
  return ynn_status_success;
}

ynn_status ynn_value::get_external_shape(size_t* rank, size_t* dims) const {
  assert(rank);
  if (*rank < data->rank) {
    YNN_LOG_ERROR() << "ynn_get_external_value_shape called with rank ("
                    << *rank << ") < value " << id << " rank (" << data->rank
                    << ")";
    return ynn_status_invalid_parameter;
  }
  *rank = data->rank;
  for (size_t i = 0; i < *rank; ++i) {
    dims[i] = data->dim(*rank - 1 - i).extent();
  }
  if (*rank > 0) {
    dims[*rank - 1] *= ynn::type_element_count(type);
  }
  return ynn_status_success;
}

ynn_subgraph::ynn_subgraph(uint32_t external_value_ids)
    : external_value_ids(external_value_ids) {
  for (size_t i = 0; i < external_value_ids; ++i) {
    values.push_back(ynn_value(i));
  }
}

ynn_value& ynn_subgraph::new_internal_value() {
  values.push_back(ynn_value(values.size()));
  return values.back();
}

ynn_value& ynn_subgraph::new_internal_value(ynn_type type) {
  ynn_value& value = new_internal_value();
  value.type = type;
  return value;
}

ynn_value& ynn_subgraph::new_internal_value(const ynn_value& template_value) {
  ynn_value& value = new_internal_value();
  value.type = template_value.type;
  value.scale_id = template_value.scale_id;
  value.zero_point_id = template_value.zero_point_id;
  return value;
}

ynn_value& ynn_subgraph::get_output_value(uint32_t* output_id,
                                          const ynn_value& template_value) {
  if (*output_id == YNN_INVALID_VALUE_ID) {
    ynn_value& new_output = new_internal_value(template_value);
    *output_id = new_output.id;
    return new_output;
  } else {
    return value(*output_id);
  }
}

ynn_value& ynn_subgraph::get_output_value(uint32_t* output_id, ynn_type type) {
  if (*output_id == YNN_INVALID_VALUE_ID) {
    ynn_value& new_output = new_internal_value(type);
    *output_id = new_output.id;
    return new_output;
  } else {
    return value(*output_id);
  }
}

void ynn_subgraph::add_node(ynn_node node) { nodes.push_back(std::move(node)); }

const ynn_node* ynn_subgraph::get_producer(uint32_t id) const {
  if (id == YNN_INVALID_VALUE_ID) {
    return nullptr;
  }
  for (const ynn_node& node : nodes) {
    if (!node.is_valid()) continue;
    for (uint32_t i : node.outputs) {
      if (i == id) {
        return &node;
      }
    }
  }
  return nullptr;
}

uint32_t ynn_subgraph::get_scalar_value_id(ynn_type type,
                                           uint32_t zero_point_id,
                                           uint32_t scale_id, float value_f32) {
  // TODO(dsharlet): We should have a cache of scalars and re-use them.
  return get_static_value_id(type, /*rank=*/0, /*dims=*/nullptr, zero_point_id,
                             scale_id, &value_f32);
}

uint32_t ynn_subgraph::get_static_value_id(ynn_type type, size_t rank,
                                           const size_t* dims,
                                           uint32_t zero_point_id,
                                           uint32_t scale_id,
                                           float* value_f32) {
  const size_t size =
      std::accumulate(dims, dims + rank, 1, std::multiplies<size_t>());
  assert(size > 0);

  float scale = 1.0f;
  if (std::all_of(value_f32, value_f32 + size,
                  [](float x) { return x == 0.0f; })) {
    if (zero_point_id != YNN_INVALID_VALUE_ID) {
      // We need to just convert zero to the (quantized) type we want.
      uint32_t id = YNN_INVALID_VALUE_ID;
      uint32_t zero_id =
          get_static_value_id(type, rank, dims, YNN_INVALID_VALUE_ID,
                              YNN_INVALID_VALUE_ID, value_f32);
      ynn_define_tensor_value(this, type, /*num_dims=*/0, /*dims=*/nullptr,
                              /*data=*/nullptr, zero_point_id, scale_id,
                              /*flags=*/0, &id);
      ynn_define_unary(this, ynn_unary_convert, zero_id, &id,
                       /*flags=*/0);
      return id;
    } else {
      // If we want a 0, we don't care about the scale, even if it's not a
      // scalar.
    }
  } else if (scale_id != YNN_INVALID_VALUE_ID) {
    scale = value(scale_id).GetStaticScalarValue<float>();
  }

  int32_t zero_point = 0;
  if (zero_point_id != YNN_INVALID_VALUE_ID) {
    zero_point = value(zero_point_id).GetStaticScalarValue<int32_t>();
  }

  std::vector<char> value(size * ynn::type_size_bytes(type));
  switch (type) {
    case ynn_type_fp32:
      std::copy_n(value_f32, size, reinterpret_cast<float*>(value.data()));
      break;
    case ynn_type_fp16:
      std::copy_n(value_f32, size, reinterpret_cast<ynn::half*>(value.data()));
      break;
    case ynn_type_bf16:
      std::copy_n(value_f32, size,
                  reinterpret_cast<ynn::bfloat16*>(value.data()));
      break;
    case ynn_type_int32:
      ynn::quantize(value_f32, reinterpret_cast<int32_t*>(value.data()), size,
                    1.0f / scale, zero_point);
      break;
    case ynn_type_int8:
      ynn::quantize(value_f32, reinterpret_cast<int8_t*>(value.data()), size,
                    1.0f / scale, zero_point);
      break;
    case ynn_type_uint8:
      ynn::quantize(value_f32, reinterpret_cast<uint8_t*>(value.data()), size,
                    1.0f / scale, zero_point);
      break;
    default:
      YNN_UNREACHABLE;
  }

  uint32_t id = YNN_INVALID_VALUE_ID;
  ynn_define_tensor_value(this, type, rank, dims, value.data(), zero_point_id,
                          scale_id, YNN_VALUE_FLAG_COPY_DATA, &id);
  return id;
}

void ynn_subgraph::infer_elementwise_shape(ynn_node& node, int input_idx,
                                           int output_idx, int input_dim,
                                           int output_dim,
                                           int input_type_element_count,
                                           int output_type_element_count) {
  const int input_id = node.inputs[input_idx];
  if (input_id == YNN_INVALID_VALUE_ID) {
    return;
  }
  const int output_id = node.outputs[output_idx];
  const ynn_value& input = value(input_id);
  if (input_dim >= input.extents.size()) {
    // We allow implicit broadcasting of dimensions that don't exist on inputs.
    return;
  }
  slinky::expr input_i = input.extents[input_dim];
  if (!input_i.defined()) {
    // This input is a broadcast, we don't learn anything from it.
    return;
  }
  if (input_type_element_count != 1) input_i *= input_type_element_count;
  ynn_value& output = value(output_id);
  assert(output_dim < output.extents.size());
  slinky::expr& output_i = output.extents[output_dim];
  if (output_i.defined()) {
    // If we already have an extent here, it must match the new extent.
    node.checks.push_back(ynn_node::check{
        output_i * output_type_element_count == input_i,
        {"dimension ", output_dim, " (", output_i * output_type_element_count,
         ") of ", ynn_node::output_idx{output_idx},
         " does not match dimension ", input_dim, " (", input_i, ") of ",
         ynn_node::input_idx{input_idx}},
    });
  }
  if (!output_i.defined() || !as_constant(output_i)) {
    // We don't have an extent, or it wasn't constant. Maybe the new extent is
    // constant?
    output_i = input_i;
    if (input_type_element_count != 1) output_i /= output_type_element_count;
  }
}

slinky::var ynn_subgraph::make_global_variable(slinky::expr value,
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

// Find parts of the graph that can be executed independently of any inputs.
ynn_status ynn_subgraph::fold_constants(slinky::thread_pool* threadpool) {
  // The number of nodes could be large, and we don't frequently read/write this
  // vector, so using std::vector<bool> makes sense.
  std::vector<bool> value_is_static(values.size(), false);

  // First, mark all values that are static by construction.
  for (uint32_t i = 0; i < values.size(); ++i) {
    if (values[i].is_static()) {
      value_is_static[i] = true;
    }
  }

  // Make a copy of this subgraph. We'll remove any non-constant nodes from
  // `constants`, and any constant nodes from `this`.
  ynn_subgraph constants(*this);
  for (uint32_t i = 0; i < nodes.size(); ++i) {
    ynn_node& node = nodes[i];
    if (std::all_of(node.inputs.begin(), node.inputs.end(), [&](uint32_t i) {
          return i == YNN_INVALID_VALUE_ID || value_is_static[i];
        })) {
      // Remove the node from the subgraph.
      node.invalidate();
      for (uint32_t i : node.outputs) {
        if (i == YNN_INVALID_VALUE_ID) continue;
        value_is_static[i] = true;
      }
    } else {
      // Remove the node (and its outputs) from the constant subgraph.
      for (uint32_t i : node.outputs) {
        if (i == YNN_INVALID_VALUE_ID) continue;
        constants.values[i].invalidate();
      }
      constants.nodes[i].invalidate();
    }
  }

  // Find all the values that need to be outputs of `constants`, and taken as
  // static values to `this`
  std::set<uint32_t> to_fold;
  for (uint32_t i = 0; i < nodes.size(); ++i) {
    ynn_node& node = nodes[i];
    if (!node.is_valid()) continue;

    for (uint32_t i : node.inputs) {
      if (i == YNN_INVALID_VALUE_ID) continue;
      if (value_is_static[i] && !values[i].is_static() &&
          !values[i].is_external_output()) {
        to_fold.insert(i);
      }
    }
  }

  if (to_fold.empty()) {
    return ynn_status_success;
  }

  // Mark these values as external outputs in `constants` so we can reshape and
  // learn the shape of the constants.
  for (uint32_t i : to_fold) {
    constants.values[i].flags |= YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  constants.invalidate_dead_values();

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
  YNN_LOG_DEBUG() << "constant subgraph:\n";
  constants.dump(std::cout);
#endif

  ynn_runtime runtime(constants, threadpool, 0);
  ynn_status status = runtime.build();
  if (status != ynn_status_success) {
    return status;
  }

  status = runtime.reshape();
  if (status != ynn_status_success) {
    return status;
  }

  // Use the results of reshape to allocate static buffers for the constants.
  for (uint32_t i : to_fold) {
    ynn_runtime_value& value = runtime.value(i);
    values[i].data =
        slinky::raw_buffer::make(value.data->rank, value.data->elem_size,
                                 value.data->dims, YNN_ALLOCATION_ALIGNMENT);
    value.data = values[i].data;
    values[i].extents = value.extents;
  }

  status = runtime.setup();
  if (status != ynn_status_success) {
    return status;
  }

  return runtime.invoke();
}

void ynn_subgraph::invalidate_dead_values() {
  std::vector<int> value_uses(values.size(), 0);

  // Count all the consumers of values.
  for (ynn_node& node : nodes) {
    if (!node.is_valid()) continue;
    for (uint32_t i : node.inputs) {
      if (i == YNN_INVALID_VALUE_ID) continue;
      value_uses[i]++;
    }
  }

  // Count external values as used.
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i].is_valid() && values[i].is_external()) {
      value_uses[i]++;
    }
  }

  // Going in reverse order, remove dead nodes, and discount the uses of the
  // inputs to newly dead nodes.
  for (auto i = nodes.rbegin(); i != nodes.rend(); ++i) {
    ynn_node& node = *i;
    if (!node.is_valid()) continue;
    bool dead =
        std::all_of(node.outputs.begin(), node.outputs.end(), [&](uint32_t i) {
          return i == YNN_INVALID_VALUE_ID || value_uses[i] == 0;
        });
    if (dead) {
      for (uint32_t i : node.inputs) {
        if (i == YNN_INVALID_VALUE_ID) continue;
        value_uses[i]--;
      }
      node.invalidate();
    }
  }

  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i].is_valid() && value_uses[i] == 0) {
      values[i].invalidate();
    }
  }
}

ynn_status ynn_subgraph::optimize(slinky::thread_pool* threadpool) {
  ynn_status status;

  status = fusion();
  if (status != ynn_status_success) {
    return status;
  }

  status = fold_constants(threadpool);
  if (status != ynn_status_success) {
    return status;
  }

  invalidate_dead_values();

  return ynn_status_success;
}

namespace {

const char* name_of(const ynn_node::invalid&) { return "invalid"; }
const char* name_of(const ynn_node::opaque&) { return "opaque"; }
const char* name_of(const ynn_node::binary_elementwise&) {
  return "binary_elementwise";
}
const char* name_of(const ynn_node::unary_elementwise&) {
  return "unary_elementwise";
}
const char* name_of(const ynn_node::reduce&) { return "reduce"; }
const char* name_of(const ynn_node::broadcast&) { return "broadcast"; }
const char* name_of(const ynn_node::broadcast_like&) {
  return "broadcast_like";
}
const char* name_of(const ynn_node::concatenate&) { return "concatenate"; }
const char* name_of(const ynn_node::stack&) { return "stack"; }
const char* name_of(const ynn_node::even_split&) { return "even_split"; }
const char* name_of(const ynn_node::copy&) { return "copy"; }
const char* name_of(const ynn_node::fuse_dim&) { return "fuse_dim"; }
const char* name_of(const ynn_node::fuse_dims&) { return "fuse_dims"; }
const char* name_of(const ynn_node::split_dim&) { return "split_dim"; }
const char* name_of(const ynn_node::split_dims&) { return "split_dims"; }
const char* name_of(const ynn_node::static_reshape&) {
  return "static_reshape";
}
const char* name_of(const ynn_node::static_broadcast&) {
  return "static_broadcast";
}
const char* name_of(const ynn_node::static_expand_dims&) {
  return "static_expand_dims";
}
const char* name_of(const ynn_node::static_pad&) { return "static_pad"; }
const char* name_of(const ynn_node::static_slice&) { return "static_slice"; }
const char* name_of(const ynn_node::static_transpose&) {
  return "static_transpose";
}
const char* name_of(const ynn_node::stencil_copy&) { return "stencil_copy"; }
const char* name_of(const ynn_node::dot&) { return "dot"; }
const char* name_of(const ynn_node::pack_b&) { return "pack_b"; }
const char* name_of(const ynn_node::get_tensor_shape&) {
  return "get_tensor_shape";
}

using ynn::operator<<;  // NOLINT(misc-unused-using-decls)

std::ostream& operator<<(std::ostream& os,
                         const ynn_node::split_dims::split& new_dim) {
  os << "{axis=" << new_dim.axis << ",factor=" << new_dim.factor << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const ynn_node::static_pad::padding& padding) {
  os << "{axis=" << padding.axis << ",pre_padding=" << padding.pre_padding
     << ",post_padding=" << padding.post_padding << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const ynn_node::stencil_copy::stencil& stencil) {
  os << "{axis=" << stencil.axis << ",new_axis=" << stencil.new_axis
     << ",extent=" << stencil.extent << ",stride=" << stencil.stride
     << ",dilation=" << stencil.dilation << "}";
  return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "{";
  const char* sep = "";
  for (const T& v : vec) {
    os << sep << v;
    sep = ",";
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ynn::axes_set& axes) {
  os << "{";
  const char* sep = "";
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i]) {
      os << sep << i;
      sep = ",";
    }
  }
  os << "}";
  return os;
}

void print(std::ostream& os, const ynn_node::invalid&) { YNN_UNREACHABLE; }

void print(std::ostream& os, const ynn_node::opaque& op) {
  if (op.name) {
    os << op.name;
  }
}

void print(std::ostream& os, const ynn_node::binary_elementwise& op) {
  os << "op=" << op.op;
}

void print(std::ostream& os, const ynn_node::unary_elementwise& op) {
  os << "op=" << op.op;
}

void print(std::ostream& os, const ynn_node::reduce& op) {
  os << "op=" << op.op << " axes=" << op.k_dims;
  if (op.keep_dims) {
    os << " keep_dims";
  }
}

void print(std::ostream& os, const ynn_node::broadcast& op) {
  os << "axes=" << op.axes;
}

void print(std::ostream& os, const ynn_node::broadcast_like& op) {
  os << "axes=" << op.axes;
}

void print(std::ostream& os, const ynn_node::concatenate& op) {
  os << "axis=" << op.axis;
}

void print(std::ostream& os, const ynn_node::stack& op) {
  os << "axis=" << op.axis;
}

void print(std::ostream& os, const ynn_node::even_split& op) {
  os << "axis=" << op.axis;
}

void print(std::ostream& os, const ynn_node::copy& op) { os << "copy"; }

void print(std::ostream& os, const ynn_node::fuse_dim& op) {
  os << "axis=" << op.axis << " axes_count=" << op.axes_count;
}

void print(std::ostream& os, const ynn_node::fuse_dims& op) {
  os << "axes=" << op.axes;
}

void print(std::ostream& os, const ynn_node::split_dim& op) {
  os << "axis=" << op.axis << " " << op.new_dims;
}

void print(std::ostream& os, const ynn_node::split_dims& op) {
  os << "splits=" << op.splits;
}

void print(std::ostream& os, const ynn_node::static_reshape& op) {
  os << "new_dims=" << op.new_dims;
}

void print(std::ostream& os, const ynn_node::static_broadcast& op) {
  os << "new_dims=" << op.new_dims;
}

void print(std::ostream& os, const ynn_node::static_expand_dims& op) {
  os << "new_axes=" << op.new_axes;
}

void print(std::ostream& os, const ynn_node::static_pad& op) {
  os << "paddings=" << op.paddings;
}

void print(std::ostream& os, const ynn_node::static_slice& op) {
  os << "slices={";
  for (size_t i = 0; i < op.slices.size(); ++i) {
    const ynn_node::static_slice::slice& slice = op.slices[i];
    if (op.slice_dims) {
      os << "{axis=" << slice.axis << ",at=" << slice.begin << "}";
    } else {
      os << "{axis=" << slice.axis << ",begin=" << slice.begin
         << ",end=" << slice.end << ",stride=" << slice.stride << "}";
    }
    if (i + 1 < op.slices.size()) {
      os << ",";
    }
  }
  os << "}";
}

void print(std::ostream& os, const ynn_node::static_transpose& op) {
  os << "permutation=" << op.permutation;
  if (op.alias) {
    os << " alias";
  }
}

void print(std::ostream& os, const ynn_node::stencil_copy& op) {
  os << "stencils=" << op.stencils;
}

void print(std::ostream& os, const ynn_node::dot& op) {
  os << "num_k_dims=" << op.num_k_dims;
}

void print(std::ostream& os, const ynn_node::pack_b& op) {}

void print(std::ostream& os, const ynn_node::get_tensor_shape& op) {
  os << "axes=" << op.axes;
  if (op.reshape_1d) {
    os << " reshape_1d";
  }
}

std::string value_ids_to_string(const std::vector<uint32_t>& value_ids) {
  std::stringstream ss;
  ss << "{";
  const char* sep = "";
  for (uint32_t value_id : value_ids) {
    ss << sep;
    if (value_id != YNN_INVALID_VALUE_ID) {
      ss << value_id;
    }
    sep = ",";
  }
  ss << "}";
  return ss.str();
}

}  // namespace

const char* ynn_node::name() const {
  return std::visit([&](const auto& op) { return name_of(op); }, op);
}

std::string ynn_node::to_string() const {
  std::stringstream ss;
  ss << name() << " ";
  std::visit([&](const auto& op) { print(ss, op); }, op);
  return ss.str();
}

void ynn_subgraph::dump(std::ostream& os) const {
  // Values header.
  constexpr int id_width = 10;
  constexpr int rank_width = 4;
  constexpr int type_width = 6;
  constexpr int zero_point_id_width = 13;
  constexpr int scale_id_width = 8;
  os << std::setw(id_width) << "value id" << " ";
  os << std::setw(rank_width) << "rank" << " ";
  os << std::setw(type_width) << "type" << " ";
  os << std::setw(zero_point_id_width) << "zero_point_id" << " ";
  os << std::setw(scale_id_width) << "scale_id" << std::endl;

  os << std::string(id_width, '-') << " ";
  os << std::string(rank_width, '-') << " ";
  os << std::string(type_width, '-') << " ";
  os << std::string(zero_point_id_width, '-') << " ";
  os << std::string(scale_id_width, '-') << std::endl;

  // Values
  int values_count = 0;
  for (const ynn_value& value : values) {
    if (!value.is_valid()) continue;
    os << std::setw(id_width) << value.id << " ";
    os << std::setw(rank_width) << value.rank() << " ";
    os << std::setw(type_width) << value.type << " ";
    if (value.zero_point_id != YNN_INVALID_VALUE_ID) {
      os << std::setw(zero_point_id_width) << value.zero_point_id << " ";
    } else {
      os << std::setw(zero_point_id_width) << " " << " ";
    }
    if (value.scale_id != YNN_INVALID_VALUE_ID) {
      os << std::setw(scale_id_width) << value.scale_id << " ";
    } else {
      os << std::setw(scale_id_width) << " " << " ";
    }
    if (value.flags & YNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
      os << "external_output ";
    }
    if (value.flags & YNN_VALUE_FLAG_EXTERNAL_INPUT) {
      os << "external_input ";
    }
    if (value.is_static()) {
      os << "static ";
    }
    os << std::endl;
    ++values_count;
  }

  // Nodes header
  const int inputs_width = (std::ceil(std::log10(values.size())) + 1) * 4 + 3;
  constexpr int outputs_width = 10;
  os << std::setw(inputs_width) << "node inputs" << " ";
  os << std::setw(outputs_width) << "outputs" << std::endl;
  os << std::string(inputs_width, '-') << " ";
  os << std::string(outputs_width, '-') << std::endl;

  // Nodes
  int nodes_count = 0;
  for (const ynn_node& node : nodes) {
    if (!node.is_valid()) continue;
    os << std::setw(inputs_width) << value_ids_to_string(node.inputs) << " ";
    os << std::setw(outputs_width) << value_ids_to_string(node.outputs) << " ";
    os << node.to_string() << std::endl;
    ++nodes_count;
  }
  os << "subgraph contains " << values_count << " values and " << nodes_count
     << " nodes." << std::endl;
}

extern "C" {

ynn_status ynn_create_subgraph(uint32_t external_value_ids, uint32_t flags,
                               ynn_subgraph_t* subgraph) {
  *subgraph = new ynn_subgraph(external_value_ids);
  return ynn_status_success;
}

void ynn_delete_subgraph(ynn_subgraph_t subgraph) { delete subgraph; }

}  // extern "C"
