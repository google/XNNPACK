// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_SUBGRAPH_H_
#define XNNPACK_YNNPACK_SUBGRAPH_SUBGRAPH_H_

#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include "ynnpack/subgraph/slinky.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/evaluate.h"
#include "slinky/runtime/expr.h"

namespace ynn {

// dot packing splits + transposes 2 dimensions.
constexpr size_t ynn_internal_extra_dims = 2;

struct axes_set : std::bitset<YNN_MAX_TENSOR_RANK + ynn_internal_extra_dims> {
  using bitset::bitset;
};

inline bool operator<(const axes_set& a, const axes_set& b) {
  return a.to_ulong() < b.to_ulong();
}

// Define a transpose node, optionally using a slinky copy that may alias even
// if dimension 0 is not stride 1 in the result.
ynn_status define_static_transpose(ynn_subgraph_t subgraph,
                                   std::vector<int32_t> permutation,
                                   uint32_t input_id, uint32_t* output_id,
                                   bool alias = false);

}  // namespace ynn

struct ynn_node;

// Abstraction for a collections of elements produced and consumed by nodes.
struct ynn_value {
  explicit ynn_value(uint32_t id = YNN_INVALID_VALUE_ID) : id(id) {}

  uint32_t id;
  ynn_type type = ynn_type_invalid;
  uint32_t flags = 0;

  // The data we have for this value, if any.
  slinky::raw_buffer_ptr data;

  // Tensor IDs for quantization data, if any.
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;

  // The inferred shape of this value.
  // TODO: We need an absl::InlinedVector for things like this.
  std::vector<slinky::expr> extents;

  // The symbol we've named this value in the Slinky pipeline.
  slinky::var symbol;

  int rank() const { return extents.size(); }

  bool is_external() const {
    return (flags & (YNN_VALUE_FLAG_EXTERNAL_INPUT |
                     YNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0;
  }

  bool is_external_output() const {
    return (flags & YNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0;
  }

  bool is_external_input() const {
    return (flags & YNN_VALUE_FLAG_EXTERNAL_INPUT) != 0;
  }

  bool is_static() const { return !is_external() && data && data->base; }

  bool is_static_scalar() const {
    return is_static() && data->size_bytes() == data->elem_size;
  }

  bool is_valid() const {
    return id != YNN_INVALID_VALUE_ID && type != ynn_type_invalid;
  }

  ynn_status set_external_shape(size_t rank, const size_t* dims);
  ynn_status get_external_shape(size_t* rank, size_t* dims) const;

  void invalidate() { id = YNN_INVALID_VALUE_ID; }

  const char* name_prefix() const {
    if (is_static()) {
      return "const";
    } else if (is_external_input()) {
      return "in";
    } else if (is_external_output()) {
      return "out";
    } else {
      return "v";
    }
  }

  std::string name() const;

  // Get the extent of a dimension, or 1 if it is implicitly broadcasted.
  slinky::expr extent(size_t i) const {
    return i < extents.size() && extents[i].defined() ? extents[i] : 1;
  }

  // Asserting that the value is reshapable to a static scalar value of type T,
  // returns that value.
  template <typename T>
  T static_scalar_value() const {
    assert(is_static_scalar());
    assert(sizeof(T) == data->elem_size);
    T result;
    memcpy(&result, data->base, sizeof(T));
    return result;
  }

  // If the value is reshape-able to a scalar, returns the value converted to
  // a float, otherwise returns nullopt.
  std::optional<float> as_scalar_float() const;
};

struct ynn_node {
  struct invalid {
    friend bool operator==(const invalid&, const invalid&) { return true; }
    friend bool operator<(const invalid&, const invalid&) { return false; }
  };
  struct opaque {
    const char* name = "opaque";
    friend bool operator==(const opaque& a, const opaque& b) {
      return std::string_view(a.name) == std::string_view(b.name);
    }
    friend bool operator<(const opaque& a, const opaque& b) {
      return std::string_view(a.name) < std::string_view(b.name);
    }
  };
  struct broadcast {
    // The dimensions to broadcast.
    ynn::axes_set axes;
    friend bool operator==(const broadcast& a, const broadcast& b) {
      return a.axes == b.axes;
    }
    friend bool operator<(const broadcast& a, const broadcast& b) {
      return a.axes < b.axes;
    }
  };
  struct broadcast_like {
    // The dimensions to attempt to broadcast.
    ynn::axes_set axes;
    friend bool operator==(const broadcast_like& a, const broadcast_like& b) {
      return a.axes == b.axes;
    }
    friend bool operator<(const broadcast_like& a, const broadcast_like& b) {
      return a.axes < b.axes;
    }
  };
  struct concatenate {
    int32_t axis;
    friend bool operator==(const concatenate& a, const concatenate& b) {
      return a.axis == b.axis;
    }
    friend bool operator<(const concatenate& a, const concatenate& b) {
      return a.axis < b.axis;
    }
  };
  struct stack {
    int32_t axis;
    friend bool operator==(const stack& a, const stack& b) {
      return a.axis == b.axis;
    }
    friend bool operator<(const stack& a, const stack& b) {
      return a.axis < b.axis;
    }
  };
  struct even_split {
    int32_t axis;
    friend bool operator==(const even_split& a, const even_split& b) {
      return a.axis == b.axis;
    }
    friend bool operator<(const even_split& a, const even_split& b) {
      return a.axis < b.axis;
    }
  };
  struct unary_elementwise {
    ynn_unary_operator op;
    friend bool operator==(const unary_elementwise& a,
                           const unary_elementwise& b) {
      return a.op == b.op;
    }
    friend bool operator<(const unary_elementwise& a,
                          const unary_elementwise& b) {
      return a.op < b.op;
    }
  };
  struct lut {
    friend bool operator==(const lut&, const lut&) { return true; }
    friend bool operator<(const lut&, const lut&) { return false; }
  };
  struct binary_elementwise {
    ynn_binary_operator op;
    friend bool operator==(const binary_elementwise& a,
                           const binary_elementwise& b) {
      return a.op == b.op;
    }
    friend bool operator<(const binary_elementwise& a,
                          const binary_elementwise& b) {
      return a.op < b.op;
    }
  };
  struct ternary_elementwise {
    ynn::ternary_op op;
    friend bool operator==(const ternary_elementwise& a,
                           const ternary_elementwise& b) {
      return a.op == b.op;
    }
    friend bool operator<(const ternary_elementwise& a,
                          const ternary_elementwise& b) {
      return a.op < b.op;
    }
  };
  struct copy {
    friend bool operator==(const copy&, const copy&) { return true; }
    friend bool operator<(const copy&, const copy&) { return false; }
  };
  struct fuse_dim {
    // Fuse `axes_count` dimensions starting at `axis` into one dimension.
    int32_t axis;
    size_t axes_count;
    friend bool operator==(const fuse_dim& a, const fuse_dim& b) {
      return a.axis == b.axis && a.axes_count == b.axes_count;
    }
    friend bool operator<(const fuse_dim& a, const fuse_dim& b) {
      return std::tie(a.axis, a.axes_count) < std::tie(b.axis, b.axes_count);
    }
  };
  struct fuse_dims {
    // Set of dimensions to fuse with the next dimension.
    ynn::axes_set axes;
    friend bool operator==(const fuse_dims& a, const fuse_dims& b) {
      return a.axes == b.axes;
    }
    friend bool operator<(const fuse_dims& a, const fuse_dims& b) {
      return a.axes < b.axes;
    }
  };
  struct split_dim {
    // Split `axis` into new axes of extent `new_dims`.
    int32_t axis;
    std::vector<size_t> new_dims;
    friend bool operator==(const split_dim& a, const split_dim& b) {
      return a.axis == b.axis && a.new_dims == b.new_dims;
    }
    friend bool operator<(const split_dim& a, const split_dim& b) {
      return std::tie(a.axis, a.new_dims) < std::tie(b.axis, b.new_dims);
    }
  };
  struct split_dims {
    struct split {
      // Axis to split.
      int32_t axis;
      size_t factor;
      friend bool operator==(const split& a, const split& b) {
        return a.axis == b.axis && a.factor == b.factor;
      }
      friend bool operator<(const split& a, const split& b) {
        return std::tie(a.axis, a.factor) < std::tie(b.axis, b.factor);
      }
    };
    std::vector<split> splits;
    friend bool operator==(const split_dims& a, const split_dims& b) {
      return a.splits == b.splits;
    }
    friend bool operator<(const split_dims& a, const split_dims& b) {
      return a.splits < b.splits;
    }
  };
  struct static_reshape {
    // Extents of the new dimensions after a reshape. '0' is replaced with the
    // extent that preserves the total number of elements.
    std::vector<size_t> new_dims;
    friend bool operator==(const static_reshape& a, const static_reshape& b) {
      return a.new_dims == b.new_dims;
    }
    friend bool operator<(const static_reshape& a, const static_reshape& b) {
      return a.new_dims < b.new_dims;
    }
  };
  struct static_broadcast {
    // Extents of the new dimensions after a broadcast. '0' is replaced with the
    // original extent of the input.
    std::vector<size_t> new_dims;
    friend bool operator==(const static_broadcast& a,
                           const static_broadcast& b) {
      return a.new_dims == b.new_dims;
    }
    friend bool operator<(const static_broadcast& a,
                          const static_broadcast& b) {
      return a.new_dims < b.new_dims;
    }
  };
  struct static_expand_dims {
    ynn::axes_set new_axes;
    friend bool operator==(const static_expand_dims& a,
                           const static_expand_dims& b) {
      return a.new_axes == b.new_axes;
    }
    friend bool operator<(const static_expand_dims& a,
                          const static_expand_dims& b) {
      return a.new_axes < b.new_axes;
    }
  };
  struct static_pad {
    struct padding {
      int32_t axis;
      int64_t pre_padding;
      int64_t post_padding;
      friend bool operator==(const padding& a, const padding& b) {
        return a.axis == b.axis && a.pre_padding == b.pre_padding &&
               a.post_padding == b.post_padding;
      }
      friend bool operator<(const padding& a, const padding& b) {
        return std::tie(a.axis, a.pre_padding, a.post_padding) <
               std::tie(b.axis, b.pre_padding, b.post_padding);
      }
    };
    std::vector<padding> paddings;
    friend bool operator==(const static_pad& a, const static_pad& b) {
      return a.paddings == b.paddings;
    }
    friend bool operator<(const static_pad& a, const static_pad& b) {
      return a.paddings < b.paddings;
    }
  };
  struct static_slice {
    struct slice {
      int32_t axis;
      int64_t begin;
      int64_t end;
      int64_t stride;
      friend bool operator==(const slice& a, const slice& b) {
        return a.axis == b.axis && a.begin == b.begin && a.end == b.end &&
               a.stride == b.stride;
      }
      friend bool operator<(const slice& a, const slice& b) {
        return std::tie(a.axis, a.begin, a.end, a.stride) <
               std::tie(b.axis, b.begin, b.end, b.stride);
      }
    };
    std::vector<slice> slices;
    bool slice_dims;
    friend bool operator==(const static_slice& a, const static_slice& b) {
      return a.slices == b.slices && a.slice_dims == b.slice_dims;
    }
    friend bool operator<(const static_slice& a, const static_slice& b) {
      return std::tie(a.slices, a.slice_dims) <
             std::tie(b.slices, b.slice_dims);
    }
  };
  struct static_transpose {
    std::vector<int32_t> permutation;
    bool alias;
    friend bool operator==(const static_transpose& a,
                           const static_transpose& b) {
      return a.permutation == b.permutation && a.alias == b.alias;
    }
    friend bool operator<(const static_transpose& a,
                          const static_transpose& b) {
      return std::tie(a.permutation, a.alias) <
             std::tie(b.permutation, b.alias);
    }
  };
  struct stencil_copy {
    struct stencil {
      int32_t axis;
      int32_t new_axis;
      size_t extent;
      size_t stride;
      size_t dilation;
      friend bool operator==(const stencil& a, const stencil& b) {
        return a.axis == b.axis && a.new_axis == b.new_axis &&
               a.extent == b.extent && a.stride == b.stride &&
               a.dilation == b.dilation;
      }
      friend bool operator<(const stencil& a, const stencil& b) {
        return std::tie(a.axis, a.new_axis, a.extent, a.stride, a.dilation) <
               std::tie(b.axis, b.new_axis, b.extent, b.stride, b.dilation);
      }
    };
    std::vector<stencil> stencils;
    friend bool operator==(const stencil_copy& a, const stencil_copy& b) {
      return a.stencils == b.stencils;
    }
    friend bool operator<(const stencil_copy& a, const stencil_copy& b) {
      return a.stencils < b.stencils;
    }
  };
  struct dot {
    size_t num_k_dims;
    friend bool operator==(const dot& a, const dot& b) {
      return a.num_k_dims == b.num_k_dims;
    }
    friend bool operator<(const dot& a, const dot& b) {
      return a.num_k_dims < b.num_k_dims;
    }
  };
  struct pack_b {
    friend bool operator==(const pack_b&, const pack_b&) { return true; }
    friend bool operator<(const pack_b&, const pack_b&) { return false; }
  };
  struct transpose_a {
    size_t tile_k;
    int32_t m_dim;
    friend bool operator==(const transpose_a& a, const transpose_a& b) {
      return a.tile_k == b.tile_k && a.m_dim == b.m_dim;
    }
    friend bool operator<(const transpose_a& a, const transpose_a& b) {
      return std::tie(a.tile_k, a.m_dim) < std::tie(b.tile_k, b.m_dim);
    }
  };
  struct get_tensor_shape {
    std::vector<int32_t> axes;
    bool reshape_1d;
    friend bool operator==(const get_tensor_shape& a,
                           const get_tensor_shape& b) {
      return a.axes == b.axes && a.reshape_1d == b.reshape_1d;
    }
    friend bool operator<(const get_tensor_shape& a,
                          const get_tensor_shape& b) {
      return std::tie(a.axes, a.reshape_1d) < std::tie(b.axes, b.reshape_1d);
    }
  };
  struct reduce {
    ynn::axes_set k_dims;
    ynn_reduce_operator op;
    bool keep_dims;
    friend bool operator==(const reduce& a, const reduce& b) {
      return a.k_dims == b.k_dims && a.op == b.op && a.keep_dims == b.keep_dims;
    }
    friend bool operator<(const reduce& a, const reduce& b) {
      return std::tie(a.op, a.keep_dims, a.k_dims) <
             std::tie(b.op, b.keep_dims, b.k_dims);
    }
  };

  // Value IDs for node inputs and outputs.
  // TODO: We need an absl::InlinedVector for things like this.
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
  std::variant<invalid, opaque, broadcast, broadcast_like, concatenate,
               even_split, copy, split_dim, fuse_dim, fuse_dims, split_dims,
               stack, static_reshape, static_broadcast, static_expand_dims,
               static_pad, static_slice, static_transpose, stencil_copy,
               unary_elementwise, lut, binary_elementwise, ternary_elementwise,
               dot, pack_b, transpose_a, get_tensor_shape, reduce>
      op;

  const char* name() const;
  std::string to_string() const;

  bool is_valid() const { return op.index() != 0; }

  void invalidate() { op = invalid{}; }

  // This function should create the Slinky funcs and buffers that implement the
  // pipeline
  std::function<ynn_status(const ynn_node&, ynn_runtime&)> create;

  struct input_idx {
    int idx;
  };
  struct output_idx {
    int idx;
  };

  struct check {
    // A condition that must evaluate to true.
    slinky::expr condition;

    // The error message to emit if the condition is not true. The message is
    // formed by concatenating all of the parts of the message, evaluating the
    // expressions if needed.
    std::vector<std::variant<const char*, slinky::expr, input_idx, output_idx>>
        message;
  };
  std::vector<check> checks;
};

struct ynn_subgraph {
  explicit ynn_subgraph(uint32_t external_value_ids, uint32_t flags);

  // Number of Value IDs reserved for communication with external graph
  // representation. Values created during subgraph transformation avoid using
  // IDs in [0, reserved_value_ids-1] range.
  uint32_t external_value_ids;

  uint32_t flags;

  // We use std::deque, so we can push_back without invalidating pointers to
  // these objects.
  std::deque<ynn_value> values;
  std::deque<ynn_node> nodes;

  ynn::slinky_globals globals;

  bool is_valid_value(uint32_t id) const { return id < values.size(); }

  const ynn_value& value(uint32_t id) const {
    assert(id < values.size());
    return values[id];
  }
  ynn_value& value(uint32_t id) {
    assert(id < values.size());
    return values[id];
  }

  // Returns a newly allocated internal value.
  ynn_value& new_internal_value(ynn_type type = ynn_type_invalid);
  void add_node(ynn_node node);

  // Find the node that produces `id`.
  const ynn_node* get_producer(uint32_t id) const;

  // If `output_id` is `YNN_INVALID_VALUE_ID`, makes a new value like
  // `template_value`, and updates `output_id` with the new value ID.
  // Otherwise, returns the existing value.
  ynn_value& get_output_value(uint32_t* output_id,
                              const ynn_value& template_value);
  ynn_value& get_output_value(uint32_t* output_id, ynn_type type);
  ynn_value& get_output_value(uint32_t* output_id, ynn_type type,
                              uint32_t zero_point_id, uint32_t scale_id);

  // Get a scalar value of the given type and quantization parameters.
  uint32_t get_scalar_value_id(ynn_type type, uint32_t zero_point_id,
                               uint32_t scale_id, float value_f32);
  // Get a constant value of the given type and quantization parameters.
  uint32_t get_static_value_id(ynn_type type, size_t rank, const size_t* dims,
                               uint32_t zero_point_id, uint32_t scale_id,
                               float* value_f32);

  template <typename T>
  uint32_t get_scalar_value_id(T value) {
    return get_scalar_value_id(ynn::type_of<T>(),
                               /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                               /*scale_id=*/YNN_INVALID_VALUE_ID, value);
  }

  void infer_elementwise_shape(ynn_node& node, int input_idx, int output_idx,
                               int input_dim, int output_dim,
                               int input_type_element_count = 1,
                               int output_type_element_count = 1);

  // Find parts of the graph that can be executed independently of any inputs.
  ynn_status fold_constants(slinky::thread_pool* threadpool);

  // Rewrite parts of the graph that we have optimized patterns for.
  ynn_status fusion();

  // Common subexpression elimination.
  ynn_status eliminate_common_subgraphs();

  // Invalidate unused values.
  void invalidate_dead_values();

  ynn_status optimize(slinky::thread_pool* threadpool);

  void dump(std::ostream& os) const;
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_SUBGRAPH_H_
