// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"

void xnn_normalize_slice(
    const size_t num_dims,
    const size_t offsets[XNN_MIN_ELEMENTS(1)],
    const size_t sizes[XNN_MIN_ELEMENTS(1)],
    const size_t input_shape[XNN_MIN_ELEMENTS(1)],
    size_t normalized_offsets[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t normalized_input_shape[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t normalized_output_shape[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t* num_normalized_dims)
{
  *num_normalized_dims = num_dims;
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    normalized_offsets[i] = 0;
    normalized_input_shape[i] = 1;
    normalized_output_shape[i] = 1;
  }

  // First normalization pass will remove all slices of size 1, by merging it to an adjacent inner dimension.
  size_t num_size_one = 0;
  for (size_t i = 0; i < num_dims; i++) {
    const size_t offset = offsets[num_dims - 1 - i];
    const size_t input_dim = input_shape[num_dims - 1 - i];
    const size_t size = sizes[num_dims - 1 - i] > 0 ? sizes[num_dims - 1 - i] : input_dim;

    // If the innermost dimension is size 1, we can't merge it anywhere, so skip it.
    if (size == 1 && i != 0) {
      normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - i + 1 + num_size_one] +=
          offset * normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i + 1 + num_size_one];
      normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i + 1 + num_size_one] *= input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i + 1 + num_size_one] *= size;
      num_size_one++;
    } else {
      normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - i + num_size_one] = offset;
      normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i + num_size_one] = input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i + num_size_one] = size;
    }
  }

  size_t new_num_dims = num_dims - num_size_one;
  size_t output_dims = new_num_dims;
  bool merge_previous_dim = false;
  size_t num_sliced_dims = 0;
  for (size_t i = 0; i < new_num_dims; i++) {
    const size_t offset = normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - i];
    const size_t size = normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - i];
    const size_t input_dim = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - i];

    const bool merge_current_dim = (offset == 0 && size == input_dim) ;
    if (merge_previous_dim) {
      normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] =
        offset * normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims];
      normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] *= input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] *= size;
      output_dims -= 1;
      if (!merge_current_dim) {
        num_sliced_dims += 1;
      }
    } else {
      normalized_offsets[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] = offset;
      normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] = input_dim;
      normalized_output_shape[XNN_MAX_TENSOR_DIMS - 1 - num_sliced_dims] = size;
      if (!merge_current_dim) {
        // If merge_current_dim, we can merge current dimension with the next dim, so don't advance num_sliced_dims.
        num_sliced_dims += 1;
      }
    }
    merge_previous_dim = merge_current_dim;
  }

  // new_num_dims <= num_dims due to merge of size == 1, so we are left with some extra values at the front of the
  // normalized values, set them to default values.
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS - output_dims; i++) {
    normalized_offsets[i] = 0;
    normalized_input_shape[i] = 1;
    normalized_output_shape[i] = 1;
  }

  *num_normalized_dims = output_dims;
}

// Returns true if input stride and output stride are NULL or the expected input/output stride matches the actual input/output stride.
static bool can_dimension_be_removed(
    const size_t* input_stride,
    const size_t* output_stride,
    const size_t* shape,
    const size_t* inverted_perm,
    size_t input_dim,
    const size_t num_dims) {
  const size_t output_dim = inverted_perm[input_dim];
  if (input_dim == 0 && output_dim == 0) {
    return true;
  }
  if (input_stride != NULL && input_dim > 0) {
    if (input_stride[input_dim - 1] != input_stride[input_dim] * shape[input_dim]) {
      return false;
    }
  }
  if (output_stride != NULL && output_dim > 0) {
    if (output_stride[output_dim - 1] != output_stride[output_dim] * shape[input_dim]) {
      return false;
    }
  }
  return true;
}

// Remove dimension perm[dim] from shape, perm, input & output strides.
static void fold_into_previous_dim(
    size_t* shape,
    size_t* perm,
    size_t* inverted_perm,
    size_t* input_stride,
    size_t* output_stride,
    size_t num_dims,
    size_t input_dim)
{
  const size_t perm_idx = inverted_perm[input_dim];
  // Update preceding dimension size.
  if (input_dim > 0) {
    shape[input_dim - 1] *= shape[input_dim];
  }
  // Shift shape to the left to overwrite the squashed dim.
  for (size_t j = input_dim; j + 1 < num_dims; ++j) {
    shape[j] = shape[j + 1];
  }
  // Shift strides to the left to overwrite the squashed dim.
  if (input_stride != NULL) {
    for (size_t j = max(1, input_dim) - 1; j + 1 < num_dims; ++j) {
      input_stride[j] = input_stride[j + 1];
    }
  }
  if (output_stride != NULL) {
    for (size_t j = max(1, perm_idx) - 1; j + 1 < num_dims; ++j) {
      output_stride[j] = output_stride[j + 1];
    }
  }
  // Update dimensions that were greater than the one removed.
  for (size_t j = 0; j < num_dims; ++j) {
    if (perm[j] > input_dim) {
      perm[j] -= 1;
    }
  }
  // Shift permutation.
  for (size_t j = perm_idx; j + 1 < num_dims; ++j) {
    perm[j] = perm[j + 1];
  }
  // Update the inverted perm.
  for (size_t j = 0; j + 1 < num_dims; ++j) {
    inverted_perm[perm[j]] = j;
  }
}

void xnn_normalize_transpose_permutation(
    const size_t num_dims,
    const size_t element_size,
    const size_t* perm,
    const size_t* shape,
    const size_t* input_stride,
    const size_t* output_stride,
    size_t* normalized_num_dims,
    size_t* normalized_element_size_out,
    size_t* normalized_perm,
    size_t* normalized_shape,
    size_t* normalized_input_stride,
    size_t* normalized_output_stride)
{
  size_t output_dims = num_dims;
  memcpy(normalized_perm, perm, num_dims * sizeof(size_t));
  memcpy(normalized_shape, shape, num_dims * sizeof(size_t));
  size_t* normalized_input_stride_ptr = NULL;
  size_t* normalized_output_stride_ptr = NULL;
  if (input_stride != NULL) {
    memcpy(normalized_input_stride, input_stride, num_dims * sizeof(size_t));
    normalized_input_stride_ptr = normalized_input_stride;
  }
  if (output_stride != NULL) {
    memcpy(normalized_output_stride, output_stride, num_dims * sizeof(size_t));
    normalized_output_stride_ptr = normalized_output_stride;
  }

  size_t normalized_inverted_perm[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < num_dims; ++i) {
    normalized_inverted_perm[perm[i]] = i;
  }

  size_t input_dim = 0;
  // Remove dimensions of size 1 and fold dimensions which are adjacent in both input and output tensors.
  while (input_dim < output_dims) {
    const bool has_size_1 = normalized_shape[input_dim] == 1;
    const bool previous_dim_in_output_is_previous_dim_in_input = input_dim > 0 &&
        normalized_inverted_perm[input_dim] == normalized_inverted_perm[input_dim-1] + 1;
    const bool strides_allow_fold_left = can_dimension_be_removed(
        normalized_input_stride_ptr, normalized_output_stride_ptr, normalized_shape,
        normalized_inverted_perm, input_dim, output_dims);
    if (strides_allow_fold_left && (has_size_1 || previous_dim_in_output_is_previous_dim_in_input)) {
      fold_into_previous_dim(normalized_shape, normalized_perm, normalized_inverted_perm,
                             normalized_input_stride_ptr, normalized_output_stride_ptr,
                             output_dims, input_dim);
      output_dims -= 1;
      // When a dimension has been removed, new folds may be possible so check
      // it again.
      if (input_dim > 0) {
        input_dim -= 1;
      }
    } else {
      input_dim += 1;
    }
  }

  // All dimensions are size 1.
  if (input_dim == 0) {
    *normalized_num_dims = 1;
    *normalized_element_size_out = element_size;
    normalized_perm[0] = 0;
    normalized_shape[0] = 1;
    normalized_input_stride[0] = element_size;
    normalized_output_stride[0] = element_size;
    return;
  }

  // If The last input and output dimensions are the same, treat it as one large
  // element.
  size_t normalized_element_size = element_size;
  if (normalized_perm[output_dims - 1] == output_dims - 1) {
    const size_t last_dim = output_dims - 1;
    normalized_element_size = element_size * normalized_shape[last_dim];
    if (output_dims > 1 && can_dimension_be_removed(normalized_input_stride_ptr,
                                                    normalized_output_stride_ptr, normalized_shape,
                                                    normalized_inverted_perm, last_dim, output_dims)) {
      output_dims -= 1;
    } else {
      if (normalized_input_stride != NULL) {
        normalized_input_stride[last_dim] *= normalized_shape[last_dim];
      }
      if (normalized_output_stride != NULL) {
        normalized_output_stride[normalized_perm[last_dim]] *= normalized_shape[last_dim];
      }
      normalized_shape[last_dim] = 1;
    }
  }
  // If input_strides is not provided, calculate it using normalized_shape and normalized_element_size.
  if (input_stride == NULL) {
    normalized_input_stride[output_dims - 1] = normalized_element_size;
    for (size_t i = output_dims - 1; i > 0; --i) {
      normalized_input_stride[i - 1] = normalized_input_stride[i] * normalized_shape[i];
    }
  } else {
    // Scale input_stride by element size.
    for (size_t i = 0; i < output_dims; ++i) {
      normalized_input_stride[i] *= element_size;
    }
  }
  // If output_strides is not provided, calculate it using normalized_shape and normalized_element_size.
  if (output_stride == NULL) {
    normalized_output_stride[output_dims - 1] = normalized_element_size;
    for (size_t i = output_dims - 1; i > 0; --i) {
      normalized_output_stride[i - 1] = normalized_output_stride[i] * normalized_shape[normalized_perm[i]];
    }
  } else {
    // Scale output_stride by element size.
    for (size_t i = 0; i < output_dims; ++i) {
      normalized_output_stride[i] *= element_size;
    }
  }
  *normalized_element_size_out = normalized_element_size;
  *normalized_num_dims = output_dims;
}

static int cmp_value_size_t(const void* a_ptr, const void* b_ptr) {
  const size_t a = *((const size_t*) a_ptr);
  const size_t b = *((const size_t*) b_ptr);
  return (b < a) - (b > a);
}

void xnn_normalize_reduction(
    size_t* num_reduction_axes_ptr,
    size_t* reduction_axes,
    size_t* num_input_dims_ptr,
    size_t* input_dims)
{
  size_t num_reduction_axes = *num_reduction_axes_ptr;
  qsort(reduction_axes, num_reduction_axes, sizeof(size_t), cmp_value_size_t);

  // The original number of input dimensions.
  const size_t num_input_dims = *num_input_dims_ptr;

  // Running variables for tracking sequences of adjacent axes, e.g. 1, 2, 3
  size_t axes_sequence_start = SIZE_MAX;
  size_t axes_sequence_length = 0;
  // Running product of input dimensions for a sequence of adjacent axes, e.g. input_dims[1] * input_dims[2] * ...
  size_t num_reduction_elements = 0;

  // Tracking variables for consumed and produced input dimensions.
  // Each consumed/produced input dimension is read/written only once.
  // Invariant num_consumed_input_dims <= num_produced_input_dims holds at each iteration.
  size_t num_consumed_input_dims = 0;
  size_t num_produced_input_dims = 0;

  // Tracking variables for consumed and produced reduction axes.
  // Each consumed/produced reduction axis is read/written only once.
  // Invariant consumed_reduction_axes <= &reduction_axes[num_produced_reduction_axes] holds at each iteration.
  const size_t* consumed_reduction_axes = reduction_axes;
  size_t num_produced_reduction_axes = 0;
  for (; num_reduction_axes != 0; num_reduction_axes -= 1) {
    const size_t axis = *consumed_reduction_axes++;
    if (axis == axes_sequence_start + axes_sequence_length) {
      // Continue a sequence of adjacent reduction axes.
      axes_sequence_length += 1;

      assert(axis == num_consumed_input_dims);
      num_reduction_elements *= input_dims[num_consumed_input_dims++];
      assert(num_consumed_input_dims <= num_input_dims);
    } else {
      if (axes_sequence_length != 0) {
        // Write out merged input dimensions of the last axes sequence.
        input_dims[num_produced_input_dims++] = num_reduction_elements;
      }

      // Start tracking a new sequence of adjacent reduction axes.
      axes_sequence_start = axis;
      axes_sequence_length = 1;

      assert(num_consumed_input_dims <= axis);
      if (num_consumed_input_dims != axis) {
        // Merge input dimensions in the [num_consumed_input_dims:axis] range.
        size_t normalized_dim = input_dims[num_consumed_input_dims++];
        while (num_consumed_input_dims != axis) {
          normalized_dim *= input_dims[num_consumed_input_dims++];
        }
        input_dims[num_produced_input_dims++] = normalized_dim;
        assert(num_produced_input_dims <= num_consumed_input_dims);
      }
      assert(num_consumed_input_dims == axis);

      // Adjust and write out the reduction axis.
      const size_t num_eliminated_input_dims = num_consumed_input_dims - num_produced_input_dims;
      reduction_axes[num_produced_reduction_axes++] = axis - num_eliminated_input_dims;

      // Reinitialize the running product of input dimensions.
      num_reduction_elements = input_dims[num_consumed_input_dims++];
      assert(num_consumed_input_dims <= num_input_dims);
    }
  }

  // If we're tracking a sequence of adjacent reduction axes, terminate it.
  if (num_consumed_input_dims == axes_sequence_start + axes_sequence_length) {
    input_dims[num_produced_input_dims++] = num_reduction_elements;
  }
  assert(num_produced_input_dims <= num_consumed_input_dims);
  assert(num_consumed_input_dims <= num_input_dims);

  // If there're input dims after the last reduction axis, normalize them.
  if (num_consumed_input_dims != num_input_dims) {
    // Merge input dimensions in the [num_consumed_input_dims:num_input_dims] range.
    size_t normalized_dim = input_dims[num_consumed_input_dims++];
    while (num_consumed_input_dims != num_input_dims) {
      normalized_dim *= input_dims[num_consumed_input_dims++];
    }
    input_dims[num_produced_input_dims++] = normalized_dim;
    assert(num_produced_input_dims <= num_consumed_input_dims);
  }
  assert(num_produced_input_dims <= num_consumed_input_dims);
  assert(num_consumed_input_dims == num_input_dims);

  *num_input_dims_ptr = num_produced_input_dims;
  *num_reduction_axes_ptr = num_produced_reduction_axes;
}
