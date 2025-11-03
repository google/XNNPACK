// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/xnnpack/utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "include/xnnpack.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

namespace ynn {

ynn_status define_tensor_value_like(ynn_subgraph_t subgraph, uint32_t type_id,
                                    size_t rank, uint32_t* id_out) {
  const ynn_value& type_value = subgraph->value(type_id);
  assert(*id_out == YNN_INVALID_VALUE_ID);
  return ynn_define_tensor_value(subgraph, type_value.type, rank,
                                 /*dims=*/nullptr, /*data=*/nullptr,
                                 type_value.zero_point_id, type_value.scale_id,
                                 /*flags=*/0, id_out);
}

ynn_status define_tensor_value_like(ynn_subgraph_t subgraph, uint32_t id,
                                    uint32_t* id_out) {
  return define_tensor_value_like(subgraph, id, rank_of_value(subgraph, id),
                                  id_out);
}

ynn_status define_scalar_value_like(ynn_subgraph_t subgraph, uint32_t id,
                                    float value_fp32, uint32_t* id_out) {
  assert(*id_out == YNN_INVALID_VALUE_ID);
  const ynn_value& id_value = subgraph->value(id);
  *id_out = subgraph->get_scalar_value_id(id_value.type, id_value.zero_point_id,
                                          id_value.scale_id, value_fp32);
  return ynn_status_success;
}

namespace {

bool type_promotes_to_float(ynn_type type) {
  switch (type) {
    case ynn_type_fp32:
    case ynn_type_fp16:
    case ynn_type_bf16:
      return true;
    default:
      return false;
  }
}

ynn_type product_type(ynn_type a, ynn_type b) {
  if (type_promotes_to_float(a) || type_promotes_to_float(b)) {
    return ynn_type_fp32;
  } else {
    return ynn_type_int32;
  }
}

ynn_status slice_dims(ynn_subgraph_t subgraph, size_t num_dims,
                      const int32_t* dims, uint32_t* scale_id,
                      uint32_t* zero_point_id, uint32_t flags) {
  int64_t zero[YNN_MAX_TENSOR_RANK];
  int64_t one[YNN_MAX_TENSOR_RANK];
  std::fill_n(zero, YNN_MAX_TENSOR_RANK, 0);
  std::fill_n(one, YNN_MAX_TENSOR_RANK, 1);

  for (uint32_t* id : {scale_id, zero_point_id}) {
    if (!id || *id == YNN_INVALID_VALUE_ID) continue;

    size_t num_dims_i = num_dims;
    const int32_t* dims_i = dims;

    const int rank = rank_of_value(subgraph, *id);
    while (num_dims_i > 0) {
      // Don't try to slice dimensions beyond the rank of this value.
      // TODO: This seems like a workaround for a problem with static_slice, we
      // should be able to just slice these and have it do nothing. The problem
      // is `axis_to_slinky_dim` makes it hard to name the right dim.
      if (std::min(dims_i[0], dims_i[0] + rank) >= rank) {
        num_dims_i--;
        dims_i++;
      } else if (std::max(dims_i[0], dims_i[0] + rank) < 0) {
        num_dims_i--;
      } else {
        break;
      }
    }
    if (num_dims_i == 0) {
      continue;
    }

    uint32_t sliced_id = YNN_INVALID_VALUE_ID;
    ynn_status status = ynn_define_static_slice(subgraph, num_dims_i, dims_i,
                                                /*begins=*/zero,
                                                /*ends=*/one, /*strides=*/one,
                                                *id, &sliced_id, flags);
    if (status != ynn_status_success) {
      return status;
    }

    *id = sliced_id;
  }
  return ynn_status_success;
}

// Computes output = output - dot(a, b), assuming that a is a broadcast of a
// scalar in the k-dims, so it is implemented as slice(a)*sum(b). The dot would
// be more general (it would handle blockwise quantization), but it's harder for
// the subgraph optimization to see that sum(b) is a constant and can be
// constant folded in that case.
ynn_status subtract_a_times_sum_b(ynn_subgraph_t subgraph, size_t num_k_dims,
                                  const int32_t* a_k_dims,
                                  const int32_t* b_k_dims, uint32_t a_id,
                                  uint32_t b_id, uint32_t* output_id) {
  if (a_id == YNN_INVALID_VALUE_ID) {
    return ynn_status_success;
  }

  // Get the sum of b.
  uint32_t init_sum_id = YNN_INVALID_VALUE_ID;
  ynn_status status =
      ynn::define_scalar_value_like(subgraph, a_id, 0.0f, &init_sum_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t sum_sliced_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_tensor_value(
      subgraph, accumulator_for_type(ynn::type_of_value(subgraph, b_id)),
      ynn::rank_of_value(subgraph, b_id) - num_k_dims, /*dims=*/nullptr,
      /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID, /*scale_id=*/YNN_INVALID_VALUE_ID,
      /*flags=*/0, &sum_sliced_id);
  if (status != ynn_status_success) {
    return status;
  }

  status = ynn_define_reduce(subgraph, ynn_reduce_sum, num_k_dims, b_k_dims,
                             b_id, init_sum_id, &sum_sliced_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  // Put one of the k dims back (to be broadcasted).
  uint32_t sum_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_expand_dims(subgraph, 1, &b_k_dims[0],
                                         sum_sliced_id, &sum_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t zero_times_sum_id = YNN_INVALID_VALUE_ID;
  status =
      define_binary_with_broadcasting(subgraph, ynn_binary_multiply, a_id,
                                      sum_id, &zero_times_sum_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  if (*output_id != YNN_INVALID_VALUE_ID) {
    // Add the product of the zero point and the sum to the output.
    uint32_t sub_id = YNN_INVALID_VALUE_ID;
    status = define_binary_with_broadcasting(subgraph, ynn_binary_subtract,
                                             *output_id, zero_times_sum_id,
                                             &sub_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }

    *output_id = sub_id;
  } else {
    // The output is 0, just use the (negated) result as the output.
    status = ynn_define_unary(subgraph, ynn_unary_negate, zero_times_sum_id,
                              output_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }
  return ynn_status_success;
}

// This function implements the logic for propagating the scale and zero points
// of quantized dot products.
//
// We have the following:
//
//   ((a - a_zp)*a_s).(b - b_zp)*b_s
//
// First, observe we can reassociate the scales:
//
//   (a_s*b_s)*((a - a_zp).(b - b_zp))
//
// Distributing:
//
//   (a_s*b_s)*(a.b - a_zp.b - b_zp.a + a_zp.b_zp)
//
// The terms here are:
// - a_s*b_s: Elementwise broadcasted multiply
// - a.b: the actual dot we need to compute.
// - a_zp.b, b_zp.a: if the zero points are broadcasted scalars, these can be
//   implemented as a_zp*sum(b) and b_zp*sum(a) instead.
// - a_zp.b_zp: Elementwise broadcasted multiply
//
// In this form, we can see we can get the result we want by:
// - Initializing the accumulators with the sum of the last 3 terms
// - Using a_s*b_s as the scale of the result,
//
// Note that many of these operations have interesting broadcasting patterns
// and constant folding opportunities where a naive implementation will be
// inefficient. This motivates some special case operators:
// - (a_s*b_s)*x is a basically an outer product multiplied by x. We should
//   implement this with a 3 way multiply that does the two broadcasts at the
//   same time.
// - The sum of the 3 zero point products have a similar pattern, but with
//   an add instead of a multiply.
// - The final product of two zero points also has this pattern.
//
// The technique above can also accomodate an addition of the bias:
//
//  ((a - a_zp)*a_s).(b - b_zp)*b_s + (bias - bias_zp)*bias_s
//
// The scale of the bias is a product of scales of a and b which is an
// assumption that matched the behavior of XNNPACK.
//
//   bias_s = a_s * b_s
//
// so
//
//   ((a - a_zp)*a_s).(b - b_zp)*b_s + (bias - bias_zp)*a_s*b_s
//
// and
//
//   (a_s*b_s)*(a.b - a_zp.b - b_zp.a + a_zp.b_zp + bias - bias_zp)
//
ynn_status define_xnn_accumulator_for_dot(
    ynn_subgraph_t subgraph, size_t num_k_dims, uint32_t a_id, uint32_t b_id,
    uint32_t* init_output_id, uint32_t* output_id, uint32_t* add_to_output_id,
    bool allow_reuse) {
  const ynn_value& output_value = subgraph->value(*output_id);
  const ynn_value& a = subgraph->value(a_id);
  const ynn_value& b = subgraph->value(b_id);
  ynn_type type = accumulator_for_type(product_type(a.type, b.type));

  *add_to_output_id = YNN_INVALID_VALUE_ID;
  if (init_output_id && *init_output_id != YNN_INVALID_VALUE_ID) {
    if (type_of_value(subgraph, *init_output_id) != type) {
      // We have a bias of a different type than either the accumulators or the
      // output, we need to just save it for later.
      *add_to_output_id = *init_output_id;
      *init_output_id = YNN_INVALID_VALUE_ID;
    }
  }

  uint32_t a_zero_point_id = a.zero_point_id;
  uint32_t a_scale_id = a.scale_id;
  uint32_t b_zero_point_id = b.zero_point_id;
  uint32_t b_scale_id = b.scale_id;

  // We need a list of the k dims for computing reductions.
  // We would also need to slice k-dims of zero points and scales of a and b,
  // but define_xnn_stencil doesn't insert dims corresponding to stencil into
  // them.
  assert(a_zero_point_id == YNN_INVALID_VALUE_ID ||
         (rank_of_value(subgraph, a_zero_point_id) <=
          (rank_of_value(subgraph, a_id) - num_k_dims + 1)));
  assert(a_scale_id == YNN_INVALID_VALUE_ID ||
         (rank_of_value(subgraph, a_scale_id) <=
          (rank_of_value(subgraph, a_id) - num_k_dims + 1)));
  int32_t a_k_dims[YNN_MAX_TENSOR_RANK];
  int32_t b_k_dims[YNN_MAX_TENSOR_RANK];
  std::iota(a_k_dims, a_k_dims + num_k_dims, -static_cast<int>(num_k_dims));
  std::iota(b_k_dims, b_k_dims + num_k_dims, -static_cast<int>(num_k_dims) - 1);
  std::reverse(a_k_dims, a_k_dims + num_k_dims);
  std::reverse(b_k_dims, b_k_dims + num_k_dims);

  ynn_status status = ynn_status_success;

  if (a_zero_point_id != YNN_INVALID_VALUE_ID &&
      b_zero_point_id != YNN_INVALID_VALUE_ID) {
    // We need to add a_zero_point * b_zero_point to the accumulator
    // initialization. Even if a_zero_point and b_zero_point are scalars
    // this is conceptually a dot-product of broadcasted vectors, so we
    // need to additionally multiply by k.
    uint32_t a_times_b_zero_point_no_k_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_binary(subgraph, ynn_binary_multiply, a_zero_point_id,
                               b_zero_point_id, &a_times_b_zero_point_no_k_id,
                               /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }

    uint32_t k_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_get_tensor_shape(subgraph, num_k_dims, b_k_dims,
                                         ynn_type_int32,
                                         /*rank=*/0, b_id, &k_id,
                                         /*flags=*/YNN_NODE_FLAG_RESHAPE_1D);
    if (status != ynn_status_success) {
      return status;
    }

    uint32_t a_times_b_zero_point_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_binary(subgraph, ynn_binary_multiply,
                               a_times_b_zero_point_no_k_id, k_id,
                               &a_times_b_zero_point_id,
                               /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }

    if (*init_output_id != YNN_INVALID_VALUE_ID) {
      uint32_t new_init_output_id = YNN_INVALID_VALUE_ID;

      // This is to reset the scale of the bias, so it has the same scale as all
      // other operands. The assumption here is that the bias scale is equal to
      // the product of the input scales (which matches XNNPACK behavior) and
      // since we multiply by the input scales product later (see the
      // function-level comment for details) we need to remove it here.
      ynn_value& bias = subgraph->value(*init_output_id);
      bias.scale_id = YNN_INVALID_VALUE_ID;

      status = define_binary_with_broadcasting(
          subgraph, ynn_binary_add, *init_output_id, a_times_b_zero_point_id,
          &new_init_output_id,
          /*flags=*/0);
      if (status != ynn_status_success) {
        return status;
      }
      *init_output_id = new_init_output_id;
    } else {
      *init_output_id = a_times_b_zero_point_id;
    }
  }

  // We need to add a_zero_point * sum(b) to the accumulator initialization.
  // This product is missing dimension 0 from the result of the dot product.
  status = subtract_a_times_sum_b(subgraph, num_k_dims, a_k_dims, b_k_dims,
                                  a_zero_point_id, b_id, init_output_id);
  if (status != ynn_status_success) {
    return status;
  }

  // We need to add b_zero_point * sum(a) to the accumulator initialization.
  // This product is missing dimension 1 from the result of the dot product.
  status = subtract_a_times_sum_b(subgraph, num_k_dims, b_k_dims, a_k_dims,
                                  b_zero_point_id, a_id, init_output_id);
  if (status != ynn_status_success) {
    return status;
  }

  // The scale of the accumulator is the product of the scales of the inputs.
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  if (a_scale_id != YNN_INVALID_VALUE_ID &&
      b_scale_id != YNN_INVALID_VALUE_ID) {
    status = define_binary_with_broadcasting(subgraph, ynn_binary_multiply,
                                             a_scale_id, b_scale_id, &scale_id,
                                             /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  } else if (a_scale_id != YNN_INVALID_VALUE_ID) {
    scale_id = a_scale_id;
  } else if (b_scale_id != YNN_INVALID_VALUE_ID) {
    scale_id = b_scale_id;
  }

  if (type == output_value.type && allow_reuse) {
    assert(scale_id == YNN_INVALID_VALUE_ID);
  } else {
    *output_id = YNN_INVALID_VALUE_ID;
    ynn_status status =
        ynn_define_tensor_value(subgraph, type, /*rank=*/0,
                                /*dims=*/nullptr, /*data=*/nullptr,
                                /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                                /*scale_id=*/scale_id,
                                /*flags=*/0, output_id);
    if (status != ynn_status_success) {
      return status;
    }
  }

  if (*init_output_id != YNN_INVALID_VALUE_ID) {
    if (type_of_value(subgraph, *init_output_id) != type) {
      uint32_t converted_init_output_id = YNN_INVALID_VALUE_ID;
      // The init_output_id we were given does not match the type of the
      // accumulator we want to use, convert it.
      ynn_status status = ynn_define_tensor_value(
          subgraph, type, rank_of_value(subgraph, *init_output_id),
          /*dims=*/nullptr, /*data=*/nullptr,
          /*zero_point_id=*/YNN_INVALID_VALUE_ID,
          /*scale_id=*/scale_id,
          /*flags=*/0, &converted_init_output_id);
      if (status != ynn_status_success) {
        return status;
      }

      status = ynn_define_unary(subgraph, ynn_unary_convert, *init_output_id,
                                &converted_init_output_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return status;
      }

      *init_output_id = converted_init_output_id;
    }
  }

  return ynn_status_success;
}

ynn_status convert_to(ynn_subgraph_t subgraph, uint32_t* value_id,
                      ynn_type type) {
  if (*value_id == YNN_INVALID_VALUE_ID ||
      type_of_value(subgraph, *value_id) == type) {
    return ynn_status_success;
  }

  uint32_t converted_id = YNN_INVALID_VALUE_ID;
  ynn_status status =
      ynn_define_tensor_value(subgraph, type, /*rank=*/0,
                              /*dims=*/nullptr, /*data=*/nullptr,
                              /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                              /*scale_id=*/YNN_INVALID_VALUE_ID,
                              /*flags=*/0, &converted_id);
  if (status != ynn_status_success) {
    return status;
  }
  status = ynn_define_unary(subgraph, ynn_unary_convert, *value_id,
                            &converted_id, /*flags=*/0);
  *value_id = converted_id;
  return status;
}

ynn_status define_convert_dot_inputs(ynn_subgraph_t subgraph,
                                     uint32_t input_a_id, uint32_t* input_b_id,
                                     uint32_t* bias_id) {
  if (type_of_value(subgraph, *input_b_id) == ynn_type_uint8) {
    const ynn_value& b = subgraph->value(*input_b_id);
    ynn_status status;
    // Convert uint8 to int8
    uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
    if (b.zero_point_id != YNN_INVALID_VALUE_ID) {
      status =
          ynn::define_binary_scalar_b(subgraph, ynn_binary_subtract,
                                      b.zero_point_id, 128.0f, &zero_point_id);
      if (status != ynn_status_success) {
        return status;
      }
    } else {
      zero_point_id = subgraph->get_scalar_value_id<int32_t>(-128);
    }

    uint32_t b_int8_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_tensor_value(subgraph, ynn_type_int8, /*rank=*/0,
                                     /*dims=*/nullptr, /*data=*/nullptr,
                                     zero_point_id, b.scale_id, /*flags=*/0,
                                     &b_int8_id);
    if (status != ynn_status_success) {
      return status;
    }

    status = ynn_define_unary(subgraph, ynn_unary_convert, *input_b_id,
                              &b_int8_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    *input_b_id = b_int8_id;
  } else if (!type_is_integral(type_of_value(subgraph, input_a_id))) {
    // XNNPACK allows a mix of fp16 and fp32 inputs, and it always converts the
    // weights and bias to the same type as the input.
    ynn_type a_type = type_of_value(subgraph, input_a_id);
    // TODO(dsharlet): XNNPACK also supports fp input, quantized weights, but
    // that support is questionably correct/useful, so leaving it for later.
    assert(!type_is_integral(type_of_value(subgraph, *input_b_id)));
    ynn_status status = convert_to(subgraph, input_b_id, a_type);
    if (status != ynn_status_success) {
      return status;
    }

    // We need biases to be fp32, so we can initialize the accumulators, which
    // are always fp32 for floating point inputs.
    status = convert_to(subgraph, bias_id, ynn_type_fp32);
    if (status != ynn_status_success) {
      return status;
    }
  }
  return ynn_status_success;
}

}  // namespace

ynn_type accumulator_for_type(ynn_type type) {
  if (type_promotes_to_float(type)) {
    return ynn_type_fp32;
  } else {
    return ynn_type_int32;
  }
}

ynn_status define_xnn_dot(ynn_subgraph_t subgraph, size_t num_k_dims,
                          uint32_t a_id, uint32_t b_id, uint32_t bias_id,
                          uint32_t output_id) {
  uint32_t init_accumulator_id = bias_id;
  ynn_status status =
      define_convert_dot_inputs(subgraph, a_id, &b_id, &init_accumulator_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t accumulator_id = output_id;
  status = define_xnn_accumulator_for_dot(subgraph, num_k_dims, a_id, b_id,
                                          &init_accumulator_id, &accumulator_id,
                                          &bias_id,
                                          /*allow_reuse=*/true);
  if (status != ynn_status_success) {
    return status;
  }

  status = ynn_define_dot(subgraph, num_k_dims, a_id, b_id, init_accumulator_id,
                          &accumulator_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  // If `bias_id` exists, we need to add it to the result. It might
  // have a different type than either the accumulator or the output (e.g.
  // qd8-f16 could have a bias of fp32, accumulator type of int32, and output
  // type fp16). So, we need to potentially convert the accumulator to the type
  // of the bias, add, and then convert to the output type.
  // TODO: To make this fast, we probably need to pattern match this whole
  // sequence into one elementwise op:
  // output = f16(f32(accumulators_i32) + bias_f32)
  uint32_t output_unconverted_id = accumulator_id;
  if (bias_id != YNN_INVALID_VALUE_ID) {
    // Convert the accumulator to the type of the bias.
    uint32_t accumulator_converted_id = YNN_INVALID_VALUE_ID;
    status = ynn::define_tensor_value_like(subgraph, bias_id,
                                           &accumulator_converted_id);
    if (status != ynn_status_success) {
      return status;
    }

    status = ynn_define_unary(subgraph, ynn_unary_convert, accumulator_id,
                              &accumulator_converted_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }

    if (type_of_value(subgraph, output_id) !=
        type_of_value(subgraph, accumulator_converted_id)) {
      // The bias and output have different types, we'll have to convert again.
      output_unconverted_id = YNN_INVALID_VALUE_ID;
    } else {
      output_unconverted_id = output_id;
    }

    status =
        ynn_define_binary(subgraph, ynn_binary_add, accumulator_converted_id,
                          bias_id, &output_unconverted_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }

  if (output_unconverted_id != output_id) {
    status = ynn_define_unary(subgraph, ynn_unary_convert,
                              output_unconverted_id, &output_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }

  return ynn_status_success;
}

ynn_status define_binary_scalar_a(ynn_subgraph_t subgraph,
                                  ynn_binary_operator op, float scalar_a,
                                  uint32_t input_b_id, uint32_t* output_id) {
  uint32_t scalar_a_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn::define_scalar_value_like(subgraph, input_b_id,
                                                    scalar_a, &scalar_a_id);
  if (status != ynn_status_success) {
    return status;
  }

  return ynn_define_binary(subgraph, op, scalar_a_id, input_b_id, output_id,
                           /*flags=*/0);
}

ynn_status define_binary_scalar_b(ynn_subgraph_t subgraph,
                                  ynn_binary_operator op, uint32_t input_a_id,
                                  float scalar_b, uint32_t* output_id) {
  uint32_t scalar_b_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn::define_scalar_value_like(subgraph, input_a_id,
                                                    scalar_b, &scalar_b_id);
  if (status != ynn_status_success) {
    return status;
  }

  return ynn_define_binary(subgraph, op, input_a_id, scalar_b_id, output_id,
                           /*flags=*/0);
}

ynn_status implement_xnn_broadcasting(ynn_subgraph_t subgraph,
                                      uint32_t* input_a_id,
                                      uint32_t* input_b_id, uint32_t flags,
                                      size_t exclude_a, size_t exclude_b) {
  if (flags & XNN_FLAG_NO_BROADCAST) {
    // XNNPACK promises that no broadcasting is required in this case.
    return ynn_status_success;
  }
  const size_t rank_a = rank_of_value(subgraph, *input_a_id);
  const size_t rank_b = rank_of_value(subgraph, *input_b_id);

  ynn_status status;

  std::array<int32_t, YNN_MAX_TENSOR_RANK> all_axes;
  std::iota(all_axes.begin(), all_axes.end(), 0);

  uint32_t input_a_broadcasted_id = *input_a_id;
  if (rank_a > exclude_a) {
    input_a_broadcasted_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_broadcast_like(
        subgraph, /*num_axes=*/rank_a - exclude_a, /*axes=*/all_axes.data(),
        *input_a_id, *input_b_id, &input_a_broadcasted_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }

  uint32_t input_b_broadcasted_id = *input_b_id;
  if (rank_b > exclude_b) {
    input_b_broadcasted_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_broadcast_like(
        subgraph, /*num_axes=*/rank_b - exclude_b, /*axes=*/all_axes.data(),
        *input_b_id, *input_a_id, &input_b_broadcasted_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }

  *input_a_id = input_a_broadcasted_id;
  *input_b_id = input_b_broadcasted_id;

  return ynn_status_success;
}

ynn_status define_binary_with_broadcasting(
    ynn_subgraph_t subgraph, ynn_binary_operator op, uint32_t input_a_id,
    uint32_t input_b_id, uint32_t* output_id, uint32_t /*flags*/) {
  ynn_status status =
      implement_xnn_broadcasting(subgraph, &input_a_id, &input_b_id);
  if (status != ynn_status_success) {
    return status;
  }
  return ynn_define_binary(subgraph, op, input_a_id, input_b_id, output_id,
                           /*flags=*/0);
}

ynn_status implement_gelu(ynn_subgraph_t subgraph, uint32_t input_id,
                          uint32_t output_id) {
  ynn_type input_type = type_of_value(subgraph, input_id);

  if (ynn::type_is_integral(input_type)) {
    // Convert quantized inputs to float. We'll just convert this whole subgraph
    // into a LUT anyways.
    uint32_t input_float_id = YNN_INVALID_VALUE_ID;
    ynn_status status = ynn_define_tensor_value(
        subgraph, ynn_type_fp32, rank_of_value(subgraph, input_id),
        /*dims=*/nullptr, /*data=*/nullptr,
        /*zero_point_id=*/YNN_INVALID_VALUE_ID,
        /*scale_id=*/YNN_INVALID_VALUE_ID, /*flags=*/0, &input_float_id);
    if (status != ynn_status_success) {
      return status;
    }
    status = ynn_define_unary(subgraph, ynn_unary_convert, input_id,
                              &input_float_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    input_id = input_float_id;
  }

  uint32_t x_over_2_id = YNN_INVALID_VALUE_ID;
  ynn_status status = define_binary_scalar_b(subgraph, ynn_binary_multiply,
                                             input_id, 0.5f, &x_over_2_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t x_sqrt2_over_2_id = YNN_INVALID_VALUE_ID;
  status = define_binary_scalar_b(subgraph, ynn_binary_multiply, input_id,
                                  std::sqrt(2) / 2, &x_sqrt2_over_2_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t erf_x_sqrt2_over_2_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_unary(subgraph, ynn_unary_erf, x_sqrt2_over_2_id,
                            &erf_x_sqrt2_over_2_id,
                            /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t erf_x_sqrt2_over_2_id_times_x_over_2_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_binary(
      subgraph, ynn_binary_multiply, x_over_2_id, erf_x_sqrt2_over_2_id,
      &erf_x_sqrt2_over_2_id_times_x_over_2_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t output_float_id = output_id;
  if (ynn::type_is_integral(input_type)) {
    output_float_id = YNN_INVALID_VALUE_ID;
  }

  status = ynn_define_binary(subgraph, ynn_binary_add, x_over_2_id,
                             erf_x_sqrt2_over_2_id_times_x_over_2_id,
                             &output_float_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  if (ynn::type_is_integral(input_type)) {
    status = ynn_define_unary(subgraph, ynn_unary_convert, output_float_id,
                              &output_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }
  return ynn_status_success;
}

ynn_status define_clamp(ynn_subgraph_t subgraph, float min, float max,
                        uint32_t input_id, uint32_t* output_id) {
  ynn_status status;

  uint32_t min_id = YNN_INVALID_VALUE_ID;
  status = define_scalar_value_like(subgraph, input_id, min, &min_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t max_id = YNN_INVALID_VALUE_ID;
  status = define_scalar_value_like(subgraph, input_id, max, &max_id);
  if (status != ynn_status_success) {
    return status;
  }

  uint32_t maxed_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_binary(subgraph, ynn_binary_max, input_id, min_id,
                             &maxed_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  return ynn_define_binary(subgraph, ynn_binary_min, maxed_id, max_id,
                           output_id, /*flags=*/0);
}

ynn_status implement_clamp(ynn_subgraph_t subgraph, float min, float max,
                           uint32_t output_id) {
  if (min == -INFINITY && max == INFINITY) {
    return ynn_status_success;
  }

  uint32_t clamped_id = YNN_INVALID_VALUE_ID;
  ynn_status status = define_clamp(subgraph, min, max, output_id, &clamped_id);
  if (status != ynn_status_success) {
    return status;
  }

  // Now we need to swap the output_id and clamped_id values in the graph. This
  // is a bit of a hack: we want subsequent uses of output_id to use the clamped
  // value, but those don't exist yet, so we can't do something like
  // `replace_uses_of_with`.
  for (ynn_node& node : subgraph->nodes) {
    for (uint32_t& i : node.inputs) {
      if (i == output_id) {
        i = clamped_id;
      } else if (i == clamped_id) {
        i = output_id;
      }
    }
    for (uint32_t& i : node.outputs) {
      if (i == output_id) {
        i = clamped_id;
      } else if (i == clamped_id) {
        i = output_id;
      }
    }
  }
  return ynn_status_success;
}

namespace {

bool compute_same_padding(uint32_t kernel_size, uint32_t stride,
                          uint32_t dilation, uint32_t& padding_min,
                          uint32_t& padding_max) {
  assert(kernel_size > 0);
  assert(dilation > 0);
  assert(stride > 0);
  int dilated_kernel_size = (kernel_size - 1) * dilation + 1;

  // int output_extent = ceil_div(input_extent, stride);
  // int unpadded_extent = (output_extent - 1) * stride + dilated_kernel_size;
  // padding_min = max(unpadded_extent - input_extent, 0) / 2;

  if (stride != 1) {
    return false;
  }

  padding_min = (dilated_kernel_size - 1) / 2;
  padding_max = dilated_kernel_size - 1 - padding_min;
  return true;
}

}  // namespace

ynn_status define_xnn_stencil(
    ynn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, float padding_value, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t stride_height, uint32_t stride_width,
    uint32_t dilation_height, uint32_t dilation_width, uint32_t input_id,
    uint32_t* stencil_id, uint32_t flags) {
  bool same_padding = false;
  if (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    if (compute_same_padding(pooling_width, stride_width, dilation_width,
                             input_padding_left, input_padding_right) &&
        compute_same_padding(pooling_height, stride_height, dilation_height,
                             input_padding_top, input_padding_bottom)) {
    } else {
      input_padding_left = 0;
      input_padding_right = 0;
      input_padding_top = 0;
      input_padding_bottom = 0;
      same_padding = true;
    }
  }

  int32_t input_rank = rank_of_value(subgraph, input_id);
  uint32_t padding_id = YNN_INVALID_VALUE_ID;
  if (input_padding_top || input_padding_right || input_padding_bottom ||
      input_padding_left || same_padding) {
    // The padding should just be the zero point of the input, converted to the
    // same type as the input.
    const ynn_value& input = subgraph->value(input_id);
    if (input.zero_point_id == YNN_INVALID_VALUE_ID) {
      ynn_status status = define_scalar_value_like(subgraph, input_id,
                                                   padding_value, &padding_id);
      if (status != ynn_status_success) {
        return status;
      }
    } else {
      assert(padding_value == 0.0f);

      padding_id = subgraph->get_scalar_value_id(
          input.type, input.zero_point_id, input.scale_id, 0.0f);

      // Assume this is a dynamically quantized convolution, and broadcast
      // the non-batch dimensions.
      uint32_t padding_broadcast_id = YNN_INVALID_VALUE_ID;
      const int32_t nonbatch_axes[YNN_MAX_TENSOR_RANK] = {-1, -2, -3, -4,
                                                          -5, -6, -7, -8};
      ynn_status status = ynn_define_broadcast(
          subgraph, /*num_axes=*/input_rank - 1, nonbatch_axes, padding_id,
          &padding_broadcast_id, /*flags=*/0);

      if (status != ynn_status_success) {
        return status;
      }

      padding_id = padding_broadcast_id;
    }

    if (!same_padding) {
      uint32_t padded_id = YNN_INVALID_VALUE_ID;
      const int32_t padding_axes[2] = {1, 2};
      const int64_t pre_paddings[2] = {input_padding_top, input_padding_left};
      const int64_t post_paddings[2] = {input_padding_bottom,
                                        input_padding_right};
      ynn_status status = ynn_define_static_pad(
          subgraph, /*num_axes=*/2, padding_axes, pre_paddings, post_paddings,
          input_id, padding_id, &padded_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return status;
      }
      input_id = padded_id;
      padding_id = YNN_INVALID_VALUE_ID;
    }
  }

  // (n, y, x, c) -> (n, y, x, dy, dx, c)
  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {input_rank - 1, input_rank};
  const size_t stencil_dims[] = {pooling_height, pooling_width};
  const size_t stencil_strides[] = {stride_height, stride_width};
  const size_t stencil_dilations[] = {dilation_height, dilation_width};

  return ynn_define_stencil_copy(subgraph, /*num_stencils=*/2, stencil_axes,
                                 new_axes, stencil_dims, stencil_strides,
                                 stencil_dilations, input_id, padding_id,
                                 stencil_id, /*flags=*/0);
}

ynn_type type_of_value(ynn_subgraph_t subgraph, uint32_t id) {
  return subgraph->value(id).type;
}

size_t rank_of_value(ynn_subgraph_t subgraph, uint32_t id) {
  return subgraph->value(id).rank();
}

uint32_t value_flags_from_xnn(uint32_t flags) {
  uint32_t ynn = 0;
  if (flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) {
    ynn |= YNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  if (flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
    ynn |= YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }
  return ynn;
}

xnn_status xnn_status_from_ynn(ynn_status status) {
  switch (status) {
    case ynn_status_success:
      return xnn_status_success;
    case ynn_status_error:
      return xnn_status_invalid_state;
    case ynn_status_invalid_parameter:
      return xnn_status_invalid_parameter;
    case ynn_status_unsupported_parameter:
      return xnn_status_unsupported_parameter;
    case ynn_status_deprecated:
      return xnn_status_deprecated;
    default:
      return xnn_status_unsupported_parameter;
  }
}

ynn_binary_operator binary_operator_from_xnn(xnn_binary_operator op) {
  switch (op) {
    case xnn_binary_add:
      return ynn_binary_add;
    case xnn_binary_subtract:
      return ynn_binary_subtract;
    case xnn_binary_multiply:
      return ynn_binary_multiply;
    case xnn_binary_divide:
      return ynn_binary_divide;
    case xnn_binary_maximum:
      return ynn_binary_max;
    case xnn_binary_minimum:
      return ynn_binary_min;
    case xnn_binary_copysign:
      return ynn_binary_copysign;
    case xnn_binary_squared_difference:
      return ynn_binary_squared_difference;
    case xnn_binary_prelu:
      return ynn_binary_leaky_relu;
    case xnn_binary_pow:
      return ynn_binary_pow;
    case xnn_binary_invalid:
    case xnn_binary_modulus:
    case xnn_binary_atan2:
    case xnn_binary_bitwise_and:
    case xnn_binary_bitwise_or:
    case xnn_binary_bitwise_xor:
    case xnn_binary_shift_left:
    case xnn_binary_shift_right_logical:
    case xnn_binary_shift_right_arithmetic:
      break;
  }
  return ynn_binary_invalid;
}

ynn_unary_operator unary_operator_from_xnn(xnn_unary_operator op) {
  switch (op) {
    case xnn_unary_abs:
      return ynn_unary_abs;
    case xnn_unary_floor:
      return ynn_unary_floor;
    case xnn_unary_ceiling:
      return ynn_unary_ceil;
    case xnn_unary_bankers_rounding:
      return ynn_unary_round;
    case xnn_unary_negate:
      return ynn_unary_negate;
    case xnn_unary_square:
      return ynn_unary_square;
    case xnn_unary_square_root:
      return ynn_unary_square_root;
    case xnn_unary_cube_root:
      return ynn_unary_cube_root;
    case xnn_unary_reciprocal_square_root:
      return ynn_unary_reciprocal_square_root;
    case xnn_unary_log:
      return ynn_unary_log;
    case xnn_unary_exp:
      return ynn_unary_exp;
    case xnn_unary_tanh:
      return ynn_unary_tanh;
    case xnn_unary_convert:
      return ynn_unary_convert;
    case xnn_unary_sign:
      return ynn_unary_sign;
    case xnn_unary_cosine:
      return ynn_unary_cosine;
    case xnn_unary_sigmoid:
      return ynn_unary_sigmoid;
    case xnn_unary_sine:
      return ynn_unary_sine;
    case xnn_unary_invalid:
    case xnn_unary_approxgelu:
    case xnn_unary_clamp:
    case xnn_unary_elu:
    case xnn_unary_gelu:
    case xnn_unary_hardswish:
    case xnn_unary_leaky_relu:
    case xnn_unary_bitwise_not:
    case xnn_unary_count_leading_zeros:
    case xnn_unary_popcount:
      break;
  }
  return ynn_unary_invalid;
}

ynn_reduce_operator reduce_operator_from_xnn(xnn_reduce_operator op) {
  switch (op) {
    case xnn_reduce_sum:
      return ynn_reduce_sum;
    case xnn_reduce_sum_squared:
      return ynn_reduce_sum_squared;
    case xnn_reduce_max:
      return ynn_reduce_max;
    case xnn_reduce_min:
      return ynn_reduce_min;
    case xnn_reduce_mean:
    case xnn_reduce_mean_squared:
    case xnn_reduce_invalid:
      break;
  }
  return ynn_reduce_invalid;
}

ynn_type type_from_xnn(xnn_datatype type) {
  switch (type) {
    case xnn_datatype_fp32:
      return ynn_type_fp32;
    case xnn_datatype_fp16:
      return ynn_type_fp16;
    case xnn_datatype_qint8:
      return ynn_type_int8;
    case xnn_datatype_quint8:
      return ynn_type_uint8;
    case xnn_datatype_qint32:
      return ynn_type_int32;
    case xnn_datatype_qcint8:
      return ynn_type_int8;
    case xnn_datatype_qcint32:
      return ynn_type_int32;
    case xnn_datatype_qcint4:
      return ynn_type_int4;
    case xnn_datatype_qdint8:
      return ynn_type_int8;
    case xnn_datatype_int32:
      return ynn_type_int32;
    case xnn_datatype_qbint4:
      return ynn_type_int4;
    case xnn_datatype_bf16:
      return ynn_type_bf16;
    case xnn_datatype_qduint8:
      return ynn_type_uint8;
    case xnn_datatype_qpint8:
    case xnn_datatype_pfp32:
    case xnn_datatype_pfp16:
    case xnn_datatype_pqint8:
    case xnn_datatype_invalid:
      break;
  }
  return ynn_type_invalid;
}

xnn_datatype xnn_datatype_from_ynn(ynn_type type) {
  // YNNPACK quantization schemes are not part of the datatype, so we just need
  // to guess.
  switch (type) {
    case ynn_type_fp32:
      return xnn_datatype_fp32;
    case ynn_type_fp16:
      return xnn_datatype_fp16;
    case ynn_type_int8:
      return xnn_datatype_qint8;
    case ynn_type_uint8:
      return xnn_datatype_quint8;
    case ynn_type_int32:
      return xnn_datatype_qint32;
    case ynn_type_int4:
      return xnn_datatype_qcint4;
    case ynn_type_uint4:
      return xnn_datatype_qcint4;
    case ynn_type_bf16:
      return xnn_datatype_bf16;
    case ynn_type_opaque:
    case ynn_type_invalid:
      break;
  }
  return xnn_datatype_invalid;
}

}  // namespace ynn
