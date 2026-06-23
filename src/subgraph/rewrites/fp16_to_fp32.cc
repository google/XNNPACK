#include "src/subgraph/rewrites/fp16_to_fp32.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/fp16.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

namespace xnnpack {

namespace {

// Replaces all instances `old_value` with `new_value` in the given array.
//
// Returns `true` if at least a value was replaced.
bool ReplaceInSet(uint32_t* arr, uint32_t size, uint32_t old_value,
                  uint32_t new_value) {
  bool replaced = false;
  for (uint32_t i = 0; i < size; i++) {
    if (arr[i] == old_value) {
      arr[i] = new_value;
      replaced = true;
    }
  }
  return replaced;
}

enum class OpAction {
  kNone,             // Don't do anything, skip the node.
  kRewrite,          // Force outputs to fp32, insert converts from fp16 input
                     // to fp32
  kNeedsFP16Inputs,  // The inputs must be converted back to fp16 if they were
                     // rewritten.
  kTransparent,  // If the inputs have been rewritten, the outputs must be also.
  kElide,        // This op should be removed (eg. convert(fp32, fp32)).
};

template <class T, class = void>
struct HasUKernel : std::false_type {};

template <class T>
struct HasUKernel<T, std::void_t<decltype(T::ukernel)>> : std::true_type {};

template <class Config>
bool IsValid(const Config* config) {
  if constexpr (HasUKernel<Config>::value) {
    return config && config->ukernel;
  } else if constexpr (std::is_same_v<Config, xnn_dwconv_config>) {
    return config && config->minmax;
  } else if constexpr (std::is_same_v<Config, xnn_gemm_config>) {
    return config && config->minmax.gemm[0].function[0];
  } else {
    return config && config->op_ukernel;
  }
}

#define XNN_EXPAND(x) x

#define XNN_OP_ACTION_CASE(prefix, ...)                                    \
  XNN_EXPAND(XNN_OP_ACTION_CASE_IMPL_OVERLOAD(__VA_ARGS__, 5, 4, 3, 2, 1)( \
      prefix, __VA_ARGS__))

#define XNN_OP_ACTION_CASE_IMPL_OVERLOAD(op, i, i2, i3, i4, count, ...) \
  XNN_OP_ACTION_CASE_IMPL_##count

#define XNN_OP_ACTION_CASE_IMPL_1(prefix, op) \
  XNN_OP_ACTION_CASE_IMPL_2(prefix, op, op)

#define XNN_OP_ACTION_CASE_IMPL_2(prefix, op, impl) \
  case xnn_##prefix##_##op: {                       \
    if (IsValid(xnn_init_f16_##impl##_config())) {  \
      return OpAction::kTransparent;                \
    }                                               \
  } break

#define XNN_OP_ACTION_CASE_IMPL_3(prefix, op, impl1, impl2) \
  case xnn_##prefix##_##op: {                               \
    if (IsValid(xnn_init_f16_##impl1##_config()) &&         \
        IsValid(xnn_init_f16_##impl2##_config())) {         \
      return OpAction::kTransparent;                        \
    }                                                       \
  } break

#define XNN_OP_ACTION_CASE_IMPL_4(prefix, op, impl1, impl2, impl3) \
  case xnn_##prefix##_##op: {                                      \
    if (IsValid(xnn_init_f16_##impl1##_config()) &&                \
        IsValid(xnn_init_f16_##impl2##_config()) &&                \
        IsValid(xnn_init_f16_##impl3##_config())) {                \
      return OpAction::kTransparent;                               \
    }                                                              \
  } break

// Is this op supported when fp16 hardware is missing (allow-list).
OpAction GetOpAction(const xnn_subgraph_t subgraph, const xnn_node& node) {
  switch (node.type) {
    case xnn_node_type_unary_elementwise:
      switch (node.unary_operator) {
        XNN_OP_ACTION_CASE(unary, abs);
        XNN_OP_ACTION_CASE(unary, clamp);
        XNN_OP_ACTION_CASE(unary, elu);
        XNN_OP_ACTION_CASE(unary, approxgelu);
        XNN_OP_ACTION_CASE(unary, cosine);
        XNN_OP_ACTION_CASE(unary, exp);
        XNN_OP_ACTION_CASE(unary, gelu);
        XNN_OP_ACTION_CASE(unary, hardswish, hswish);
        XNN_OP_ACTION_CASE(unary, leaky_relu, lrelu);
        XNN_OP_ACTION_CASE(unary, log);
        XNN_OP_ACTION_CASE(unary, negate, neg);
        XNN_OP_ACTION_CASE(unary, sigmoid);
        XNN_OP_ACTION_CASE(unary, sine);
        XNN_OP_ACTION_CASE(unary, square, sqr);
        XNN_OP_ACTION_CASE(unary, square_root, sqrt);
        XNN_OP_ACTION_CASE(unary, tanh);
        XNN_OP_ACTION_CASE(unary, reciprocal_square_root, rsqrt);
        XNN_OP_ACTION_CASE(unary, ceiling, rndu);
        XNN_OP_ACTION_CASE(unary, floor, rndd);
        XNN_OP_ACTION_CASE(unary, bankers_rounding, rndne);
        case xnn_unary_convert: {
          const xnn_value& output = subgraph->values[node.outputs[0]];
          const xnn_value& input = subgraph->values[node.inputs[0]];
          if (output.datatype == input.datatype &&
              (output.datatype == xnn_datatype_fp16 ||
               output.datatype == xnn_datatype_fp32)) {
            // Elide converts from T to T. These are no-ops that may be
            // introduced by the rewrite.
            return OpAction::kElide;
          }
          if (output.datatype == xnn_datatype_fp16 ||
              input.datatype == xnn_datatype_fp16) {
            return OpAction::kNone;
          }
        } break;
        default:
          break;
      }
      break;
    case xnn_node_type_binary_elementwise:
      switch (node.binary_operator) {
        XNN_OP_ACTION_CASE(binary, add, vadd);
        XNN_OP_ACTION_CASE(binary, subtract, vsub);
        XNN_OP_ACTION_CASE(binary, multiply, vmul);
        XNN_OP_ACTION_CASE(binary, divide, vdiv);
        XNN_OP_ACTION_CASE(binary, maximum, vmax);
        XNN_OP_ACTION_CASE(binary, minimum, vmin);
        XNN_OP_ACTION_CASE(binary, prelu, vprelu);
        XNN_OP_ACTION_CASE(binary, squared_difference, vsqrdiff);
        default:
          break;
      }
      break;

      XNN_OP_ACTION_CASE(node_type, softmax, rmax, raddstoreexpminusmax, vmul);
      XNN_OP_ACTION_CASE(node_type, static_resize_bilinear_2d, ibilinear,
                         ibilinear_chw);
      XNN_OP_ACTION_CASE(node_type, rope, cmul);
      XNN_OP_ACTION_CASE(node_type, average_pooling_2d, avgpool);
      XNN_OP_ACTION_CASE(node_type, global_average_pooling_1d, avgpool);
      XNN_OP_ACTION_CASE(node_type, global_average_pooling_2d, avgpool);
      XNN_OP_ACTION_CASE(node_type, max_pooling_2d, maxpool);
      XNN_OP_ACTION_CASE(node_type, static_mean, f32acc_rsum);
      XNN_OP_ACTION_CASE(node_type, static_sum, f32acc_rsum);
      XNN_OP_ACTION_CASE(node_type, global_sum_pooling_1d, f32acc_rsum);
      XNN_OP_ACTION_CASE(node_type, global_sum_pooling_2d, f32acc_rsum);
      XNN_OP_ACTION_CASE(node_type, static_reduce_max, rmax);
      XNN_OP_ACTION_CASE(node_type, static_reduce_min, rmin);
    case xnn_node_type_fully_connected:
    case xnn_node_type_batch_matrix_multiply:
    case xnn_node_type_deconvolution_2d: {
      if (node.flags & XNN_FLAG_INLINE_LHS_PACKING) {
        if (IsValid(xnn_init_pf16_gemm_config())) {
          return OpAction::kTransparent;
        }
      } else {
        if (IsValid(xnn_init_f16_gemm_config())) {
          return OpAction::kTransparent;
        }
      }
      break;
    }
    case xnn_node_type_convolution_2d: {
      if (node.flags & XNN_FLAG_INLINE_LHS_PACKING) {
        if (IsValid(xnn_init_pf16_gemm_config()) &&
            IsValid(xnn_init_f16_vmulcaddc_config()) &&
            IsValid(xnn_init_f16_dwconv_config())) {
          return OpAction::kTransparent;
        }
      } else {
        if (IsValid(xnn_init_f16_gemm_config()) &&
            IsValid(xnn_init_f16_vmulcaddc_config()) &&
            IsValid(xnn_init_f16_dwconv_config())) {
          return OpAction::kTransparent;
        }
      }
      break;
    }
      XNN_OP_ACTION_CASE(node_type, depthwise_convolution_2d, dwconv);
      XNN_OP_ACTION_CASE(node_type, fully_connected_sparse, spmm);
    case xnn_node_type_convert: {
      const xnn_value& input = subgraph->values[node.inputs[0]];
      const xnn_value& output = subgraph->values[node.outputs[0]];
      if (input.datatype == output.datatype &&
          (input.datatype == xnn_datatype_fp16 ||
           input.datatype == xnn_datatype_fp32)) {
        return OpAction::kElide;
      }
      if (input.datatype == xnn_datatype_fp16 &&
          output.datatype == xnn_datatype_qdint8) {
        if (IsValid(xnn_init_f16_to_qs8_cvt_config()) &&
            IsValid(xnn_init_f16_rminmax_config()) &&
            IsValid(xnn_init_qs8_rsum_config())) {
          return OpAction::kTransparent;
        }
      } else if (input.datatype == xnn_datatype_fp16 &&
                 output.datatype == xnn_datatype_qduint8) {
        if (IsValid(xnn_init_f16_to_qu8_cvt_config()) &&
            IsValid(xnn_init_f16_rminmax_config()) &&
            IsValid(xnn_init_qu8_rsum_config())) {
          return OpAction::kTransparent;
        }
      } else if (input.datatype == xnn_datatype_fp32 &&
                 output.datatype == xnn_datatype_fp16) {
        if (IsValid(xnn_init_f32_to_f16_cvt_config())) {
          return OpAction::kTransparent;
        }
      } else if (input.datatype == xnn_datatype_fp16 &&
                 output.datatype == xnn_datatype_fp32) {
        if (IsValid(xnn_init_f16_to_f32_cvt_config())) {
          return OpAction::kTransparent;
        }
      }
      break;
    }
    case xnn_node_type_static_reshape:
    case xnn_node_type_static_transpose:
    case xnn_node_type_even_split:
    case xnn_node_type_concatenate:
    case xnn_node_type_copy:
    case xnn_node_type_static_slice:
    case xnn_node_type_depth_to_space_2d:
    case xnn_node_type_space_to_depth_2d:
    case xnn_node_type_static_expand_dims:
    case xnn_node_type_fuse_dims:
    case xnn_node_type_split_dims:
    case xnn_node_type_static_broadcast:
    case xnn_node_type_static_constant_pad:
    case xnn_node_type_pack_lh:
      return OpAction::kTransparent;
    default:
      break;
  }
  return OpAction::kRewrite;
}

// Checks if an op has an fp16 input or output and whether we currently
// support that.
bool HasFp16Values(const xnn_subgraph_t subgraph, const xnn_node& node) {
  auto IsFp16Value = [subgraph](uint32_t id) {
    assert(id < subgraph->num_values);
    return subgraph->values[id].datatype == xnn_datatype_fp16 ||
           subgraph->values[id].fp16_to_fp32_fallback.was_overwritten;
  };
  return std::any_of(node.inputs, node.inputs + node.num_inputs, IsFp16Value) ||
         std::any_of(node.outputs, node.outputs + node.num_outputs,
                     IsFp16Value);
}

void RemoveFlag(uint32_t& bitfield, uint32_t flag) {
  const uint32_t mask = 0xFFFFFFFF ^ flag;
  bitfield &= mask;
}

static xnn_value kInvalidValue{};

struct CloneValueRet {
  xnn_status status;
  xnn_value& value;
};

// Creates a clone of the specified value in the subgraph and returns the
// original and the new value.
//
// We use this to avoid getting dangling references to the original value when
// the subgraph value array is reallocated when adding a new value.
//
// Warning: `value` will be updated if expanding the internal value array
// requires a reallocation.
//
// Note: we don't provide an overload for anything other than an lvalue pointer
// reference to avoid inadvertently dangling a reference.
CloneValueRet CloneValue(xnn_subgraph_t subgraph, const xnn_value*& value) {
  assert(value);
  assert(value->id < subgraph->num_values);
  const int value_id = value->id;
  if (xnn_status status = xnn_subgraph_add_internal_values(subgraph, 1);
      status != xnn_status_success) {
    return {status, kInvalidValue};
  }
  value = &subgraph->values[value_id];
  xnn_value_copy(&subgraph->values[subgraph->num_values - 1], value);
  return {xnn_status_success, subgraph->values[subgraph->num_values - 1]};
}

CloneValueRet CloneValue(xnn_subgraph_t subgraph, xnn_value*& value) {
  return CloneValue(subgraph, const_cast<const xnn_value*&>(value));
}

}  // namespace

}  // namespace xnnpack

extern "C" enum xnn_status xnn_subgraph_fallback_from_fp16_to_fp32(
    xnn_subgraph_t subgraph, int optimization_flags) {
  xnn_subgraph_analyze_consumers_and_producers(subgraph);
  // Maps fp16 value ids to the corresponding fp32 value id if a conversion
  // has been inserted.
  std::vector<uint32_t> fp16_id_to_fp32_id(subgraph->num_values,
                                           XNN_INVALID_VALUE_ID);
  // Maps fp32 value ids to the corresponding fp16 value id if a conversion
  // has been inserted.
  std::vector<uint32_t> fp32_id_to_fp16_id(subgraph->num_values,
                                           XNN_INVALID_VALUE_ID);

  xnn_log_debug("Running fp16 analysis and falling back to fp32.");

  // Count changes that are made to the graph.
  size_t changes = 0;

  // Go through the graph. Count nodes that will need to be converted.
  const uint32_t original_num_nodes = subgraph->num_nodes;
  for (uint32_t node_id = 0; node_id < original_num_nodes; ++node_id) {
    // Editing the subgraph may reallocate nodes, we need to access the
    // current node through the array each time.
    auto CurrentNode = [=]() -> xnn_node& { return subgraph->nodes[node_id]; };
    if (CurrentNode().type == xnn_node_type_invalid) {
      continue;
    }

    if (!xnnpack::HasFp16Values(subgraph, CurrentNode())) {
      xnn_log_debug("node %d doesn't have fp16 values", node_id);
      continue;
    }

    xnnpack::OpAction op_action = xnnpack::GetOpAction(subgraph, CurrentNode());
    if (op_action == xnnpack::OpAction::kNone) {
      continue;
    }

    if (op_action == xnnpack::OpAction::kNeedsFP16Inputs) {
      // Check for overwritten inputs that need to be converted back to fp16.
      for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
        // The value is copied because adding new values may invalidate
        // references.
        const xnn_value value = subgraph->values[CurrentNode().inputs[i]];
        if (value.datatype == xnn_datatype_fp32 &&
            value.fp16_to_fp32_fallback.was_overwritten) {
          if (fp32_id_to_fp16_id[value.id] == XNN_INVALID_VALUE_ID) {
            const xnn_value* value_ptr =
                subgraph->values + CurrentNode().inputs[i];
            XNN_ASSIGN_OR_RETURN_CXX(xnn_value & fp16_value,
                                     xnnpack::CloneValue(subgraph, value_ptr),
                                     "Failed to clone value.");
            fp16_value.datatype = xnn_datatype_fp16;
            fp16_value.size = xnn_tensor_get_size(&fp16_value);
            fp16_value.allocation_type = xnn_allocation_type_workspace;
            xnn_log_debug("Adding a convert[fp32, fp16](%d, %d) node.",
                          value.id, fp16_value.id);
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             value.id, fp16_value.id,
                             /*flags=*/0);
            fp32_id_to_fp16_id[value.id] = fp16_value.id;
            ++changes;
          } else {
            xnn_log_debug("Reusing convert[fp32, fp16](%d, %d) node.", value.id,
                          fp32_id_to_fp16_id[value.id]);
          }
          CurrentNode().inputs[i] = fp32_id_to_fp16_id[value.id];
        }
      }
      continue;
    }

    if (op_action == xnnpack::OpAction::kElide) {
      // If the inputs and outputs cannot be directly mapped, we don't elide.
      // This log an error because that would mean that the elision strategy
      // needs to be specified when marking a node as elidable.
      if (CurrentNode().num_inputs != CurrentNode().num_outputs) {
        xnn_log_error("Node %" PRIu32
                      " should be elided but it doesn't have that same number "
                      "of inputs and outputs.",
                      node_id);
      } else {
        xnn_node& node = CurrentNode();
        // To elide an operation, we need to either:
        //
        // - remove the inputs and directly write to the outputs;
        // - remove the outputs and directly read from the inputs.
        //
        // The choice of which action to take depends on whether some
        // inputs/outputs are external as these cannot be orphaned.
        bool can_write_to_outputs = true;
        bool can_read_from_inputs = true;
        for (uint32_t i = 0; i < node.num_inputs; ++i) {
          xnn_value& output = subgraph->values[node.outputs[i]];
          if (xnn_value_is_external_output(output.flags)) {
            can_read_from_inputs = false;
          }
          xnn_value& input = subgraph->values[node.inputs[i]];
          if (xnn_value_is_external(input.flags) ||
              input.producer == XNN_INVALID_NODE_ID) {
            can_write_to_outputs = false;
          }
        }

        if (!can_read_from_inputs && !can_write_to_outputs) {
          xnn_log_debug("Node %" PRIu32
                        " cannot be elided because it connects an external "
                        "input to an external output.",
                        node_id);
          // If we cannot elide a T->T convert, we leave it as is.
          continue;
        } else {
          xnn_log_debug("Eliding node %" PRIu32, node_id);
          for (uint32_t i = 0; i < node.num_inputs; ++i) {
            xnn_value& output = subgraph->values[node.outputs[i]];
            xnn_value& input = subgraph->values[node.inputs[i]];
            if (can_read_from_inputs) {
              // Remove the outputs and update the consumers to directly reuse
              // the inputs.
              uint32_t candidate_consumer_id =
                  std::min(node_id, output.first_consumer);
              for (int consumers_updated_count = 0;
                   consumers_updated_count < output.num_consumers &&
                   candidate_consumer_id < subgraph->num_nodes;
                   ++candidate_consumer_id) {
                xnn_node& node_k = subgraph->nodes[candidate_consumer_id];
                consumers_updated_count += xnnpack::ReplaceInSet(
                    node_k.inputs, node_k.num_inputs, output.id, input.id);
              }
              input.num_consumers += output.num_consumers - 1;
              input.first_consumer =
                  std::min(input.first_consumer, output.first_consumer);
            } else {  // can_write_to_outputs == true
              // Remove the inputs and update the producers to directly reuse
              // the outputs.
              //
              // Update the input producer to write to the output directly.
              xnn_node& producer = subgraph->nodes[input.producer];
              xnnpack::ReplaceInSet(producer.outputs, producer.num_outputs,
                                    input.id, output.id);
              // Update all of the input consumers to reuse the output value.
              uint32_t candidate_consumer_id =
                  std::min(node_id, input.first_consumer);
              for (int consumers_updated_count = 0;
                   consumers_updated_count < input.num_consumers &&
                   candidate_consumer_id < subgraph->num_nodes;
                   ++candidate_consumer_id) {
                xnn_node& node_k = subgraph->nodes[candidate_consumer_id];
                consumers_updated_count += xnnpack::ReplaceInSet(
                    node_k.inputs, node_k.num_inputs, input.id, output.id);
              }
              output.producer = input.producer;
              output.num_consumers += input.num_consumers - 1;
              output.first_consumer =
                  std::min(input.first_consumer, output.first_consumer);
            }
            node.type = xnn_node_type_invalid;
            ++changes;
          }
          continue;
        }
      }
    }

    if (op_action == xnnpack::OpAction::kTransparent) {
      // If an input has been rewritten from fp16 to fp32, the outputs should
      // also be rewritten.
      bool needs_output_rewrite = false;
      for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
        xnn_value& value = subgraph->values[CurrentNode().inputs[i]];
        if (value.datatype == xnn_datatype_fp32 &&
            value.fp16_to_fp32_fallback.was_overwritten) {
          needs_output_rewrite = true;
          break;
        }
      }
      if (!needs_output_rewrite) {
        continue;
      }
      xnn_log_debug("Node %" PRIu32
                    " is transparent and it's inputs have been rewritten.",
                    node_id);
    }

    // Force node outputs to be fp32.
    for (uint32_t i = 0; i < CurrentNode().num_outputs; i++) {
      if (subgraph->values[CurrentNode().outputs[i]].datatype !=
          xnn_datatype_fp16) {
        continue;
      }

      if (xnn_value_is_external(
              subgraph->values[CurrentNode().outputs[i]].flags)) {
        // External values can't be overwritten, so we insert a value to get
        // the fp32 output and a conversion to the original external tensor.
        struct xnn_value* value = subgraph->values + CurrentNode().outputs[i];
        XNN_ASSIGN_OR_RETURN_CXX(xnn_value & fp32_value,
                                 xnnpack::CloneValue(subgraph, value),
                                 "Failed to clone value");
        fp32_value.datatype = xnn_datatype_fp32;
        fp32_value.size = xnn_tensor_get_size(&fp32_value);
        xnnpack::RemoveFlag(
            fp32_value.flags,
            XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
        fp32_value.fp16_to_fp32_fallback.was_overwritten = true;

        if (fp32_value.data != nullptr) {
          fp32_value.fp16_to_fp32_fallback.original_data = fp32_value.data;
          fp32_value.data =
              xnn_allocate_zero_memory(fp32_value.size + XNN_EXTRA_BYTES);
          fp32_value.flags |= XNN_VALUE_FLAG_NEEDS_CLEANUP;
        } else {
          fp32_value.allocation_type = xnn_allocation_type_workspace;
        }

        CurrentNode().outputs[i] = fp32_value.id;

        // Update internal nodes that may be reusing the output node to point to
        // the fp32 value.
        uint32_t k = std::min(node_id, value->first_consumer);
        // Note: value is an external output, which means that num_consumers is
        // offset by one.
        for (int j = 0; j < value->num_consumers - 1 && k < subgraph->num_nodes;
             ++k) {
          xnn_node& node_k = subgraph->nodes[k];
          j += xnnpack::ReplaceInSet(node_k.inputs, node_k.num_inputs,
                                     value->id, fp32_value.id);
        }

        xnn_log_debug("Adding a convert[fp32, fp16](%d, %d) node.",
                      fp32_value.id, value->id);
        xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                         fp32_value.id, value->id,
                         /*flags=*/0);
      } else {
        xnn_value& value = subgraph->values[CurrentNode().outputs[i]];
        xnn_log_debug("Overriding value %d from fp16 to fp32.", value.id);
        value.datatype = xnn_datatype_fp32;
        value.size = xnn_tensor_get_size(&value);
        value.fp16_to_fp32_fallback.was_overwritten = true;
      }
      ++changes;
    }

    // Insert conversions to fp32 for fp16 inputs.
    for (uint32_t i = 0; i < CurrentNode().num_inputs; i++) {
      const uint32_t input_id = CurrentNode().inputs[i];
      if (input_id == XNN_INVALID_VALUE_ID) {
        continue;
      }
      assert(input_id < subgraph->num_values);
      // The value is copied because adding new values may invalidate
      // references.
      const xnn_value value = subgraph->values[input_id];
      if (value.datatype == xnn_datatype_fp16) {
        if (fp16_id_to_fp32_id[value.id] == XNN_INVALID_VALUE_ID) {
          const xnn_value* value_ptr =
              subgraph->values + CurrentNode().inputs[i];
          XNN_ASSIGN_OR_RETURN_CXX(xnn_value & fp32_value,
                                   xnnpack::CloneValue(subgraph, value_ptr),
                                   "Failed to clone value.");
          fp32_value.datatype = xnn_datatype_fp32;
          fp32_value.size = xnn_tensor_get_size(&fp32_value);
          xnnpack::RemoveFlag(
              fp32_value.flags,
              XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT);
          if (xnn_value_is_static(value.allocation_type)) {
            xnn_log_debug("Converting static value %d to new fp32 value %d.",
                          value.id, fp32_value.id);
            // We convert static values directly to the new value without
            // inserting a convert node.
            fp32_value.data =
                xnn_allocate_zero_memory(fp32_value.size + XNN_EXTRA_BYTES);
            fp32_value.flags |= XNN_VALUE_FLAG_NEEDS_CLEANUP;
            fp32_value.fp16_to_fp32_fallback.original_data = value.data;
            xnn_run_unary_elementwise_nc(
                xnn_unary_convert, xnn_datatype_fp16, xnn_datatype_fp32,
                /*params=*/nullptr, /*input_quantization=*/nullptr,
                /*output_quantization=*/nullptr, /*flags=*/0,
                /*batch_size=*/xnn_shape_multiply_all_dims(&value.shape),
                /*channels=*/1,
                /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr,
                /*input=*/value.data, /*output=*/fp32_value.data);
          } else {
            xnn_log_debug("Adding a convert[fp16, fp32](%d, %d) node.",
                          value.id, fp32_value.id);
            fp32_value.allocation_type = xnn_allocation_type_workspace;
            xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                             value.id, fp32_value.id,
                             /*flags=*/0);
          }
          fp16_id_to_fp32_id[value.id] = fp32_value.id;
        }
        CurrentNode().inputs[i] = fp16_id_to_fp32_id[value.id];
        ++changes;
      }
    }

    // Handle node parameters that may have been stored as fp16.
    switch (CurrentNode().type) {
      case xnn_node_type_static_constant_pad: {
        uint32_t fp16_val = CurrentNode().params.static_pad.padding_value;
        float fp32_float = fp16_ieee_to_fp32_value((uint16_t)fp16_val);
        CurrentNode().params.static_pad.padding_value =
            fp32_to_bits(fp32_float);
        xnn_log_debug(
            "Rewriting static_constant_pad padding_value from FP16 bits to "
            "FP32 bits: %04X -> %08X",
            fp16_val, CurrentNode().params.static_pad.padding_value);
      } break;
      default:
        break;
    }
  }

  if (changes) {
    xnn_subgraph_clean_up(subgraph);
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_alias_fp16_fp32_fallback_data(
    xnn_subgraph_t subgraph, xnn_weights_cache_t cache) {
  if (cache) {
    for (uint32_t i = 0; i < subgraph->num_values; ++i) {
      const xnn_value& value = subgraph->values[i];
      if (value.fp16_to_fp32_fallback.original_data) {
        XNN_RETURN_IF_ERROR(xnn_weights_cache_alias_data(
            cache, value.data, value.fp16_to_fp32_fallback.original_data));
      }
    }
  }
  return xnn_status_success;
}
