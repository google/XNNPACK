// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include <xnnpack/math.h>

// Returns true if input stride and output stride are NULL or the expected input/output stride matches the actual input/output stride.
static bool can_dimension_be_removed(
    const size_t* input_stride,
    const size_t* output_stride,
    const size_t* shape,
    const size_t* perm,
    size_t dim) {
  if (dim == 0 && perm[dim] == 0) {
    return true;
  }
  if (input_stride != NULL && dim > 0) {
    if (input_stride[dim - 1] != input_stride[dim] * shape[dim]) {
      return false;
    }
  }
  if (output_stride != NULL && perm[dim] > 0) {
    if (output_stride[perm[dim] - 1] != output_stride[perm[dim]] * shape[dim]) {
      return false;
    }
  }
  return true;
}

// Remove dimension perm[dim] from shape, perm, input & output strides.
static void remove_dimension(
    size_t* shape,
    size_t* perm,
    size_t* input_stride,
    size_t* output_stride,
    size_t num_dims,
    size_t dim)
{
  for (size_t j = perm[dim]; j + 1 < num_dims; ++j) {
    shape[j] = shape[j + 1];
  }
  if (input_stride != NULL) {
    for (size_t j = max(1, perm[dim]) - 1; j + 1 < num_dims; ++j) {
      input_stride[j] = input_stride[j + 1];
    }
  }
  if (output_stride != NULL) {
    for (size_t j = max(1, dim) - 1; j + 1 < num_dims; ++j) {
      output_stride[j] = output_stride[j + 1];
    }
  }
  for (size_t j = 0; j < num_dims; ++j) {
    if (perm[j] > perm[dim]) {
      perm[j] -= 1;
    }
  }
  for (size_t j = dim; j + 1 < num_dims; ++j) {
    perm[j] = perm[j + 1];
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

  size_t output_pos = 0;
  // Remove dimensions of size 1 and fold dimensions which are adjacent in both input and output tensors.
  for (; output_pos < output_dims;) {
    if (can_dimension_be_removed(normalized_input_stride_ptr, normalized_output_stride_ptr, normalized_shape,
                                 normalized_perm, normalized_perm[output_pos])
        && ((normalized_shape[normalized_perm[output_pos]] == 1)
            || (output_pos > 0 && normalized_perm[output_pos] == normalized_perm[output_pos - 1] + 1))) {
      if (output_pos > 0) {
        normalized_shape[normalized_perm[output_pos - 1]] *= normalized_shape[normalized_perm[output_pos]];
      }
      remove_dimension(normalized_shape, normalized_perm, normalized_input_stride_ptr, normalized_output_stride_ptr,
                       output_dims, output_pos);
      output_dims -= 1;
      // When a dimension has been removed, new folds may be possible so check
      // it again.
      if (output_pos > 0) {
        output_pos -= 1;
      }
    } else {
      output_pos += 1;
    }
  }
  // All dimensions are size 1.
  if (output_pos == 0) {
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
    normalized_element_size = element_size * normalized_shape[output_dims - 1];
    if (output_dims > 1 && can_dimension_be_removed(normalized_input_stride_ptr, normalized_output_stride_ptr, normalized_shape,
                                 normalized_perm, output_dims - 1)) {
      output_dims -= 1;
    } else {
      if (normalized_input_stride != NULL) {
        normalized_input_stride[output_dims - 1] *= normalized_shape[output_dims - 1];
      }
      if (normalized_output_stride != NULL) {
        normalized_output_stride[normalized_perm[output_dims - 1]] *= normalized_shape[output_dims - 1];
      }
      normalized_shape[output_dims - 1] = 1;
    }
  }
  // If input_strides is not provided, calculate it using normalized_shape and normalized_element_size.
  if (input_stride == NULL) {
    normalized_input_stride[output_dims - 1] = normalized_element_size;
    for(size_t i = output_dims - 1; i > 0; --i) {
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
    for(size_t i = output_dims - 1; i > 0; --i) {
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
