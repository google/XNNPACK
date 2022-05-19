// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string.h>
#include <stddef.h>

static void remove_dimension(
    size_t* normalized_shape,
    size_t* normalized_perm,
    size_t num_dims,
    size_t i)
{
  for (size_t j = normalized_perm[i]; j + 1 < num_dims; ++j) {
    normalized_shape[j] = normalized_shape[j + 1];
  }
  for (size_t j = 0; j < num_dims; ++j) {
    if (normalized_perm[j] > normalized_perm[i]) {
      normalized_perm[j] -= 1;
    }
  }
  for (size_t j = i; j + 1 < num_dims; ++j) {
    normalized_perm[j] = normalized_perm[j + 1];
  }
}

void xnn_normalize_transpose_permutation(
    const size_t num_dims,
    const size_t element_size,
    const size_t* perm,
    const size_t* shape,
    size_t* normalized_num_dims,
    size_t* normalized_element_size_out,
    size_t* normalized_perm,
    size_t* normalized_shape)
{
  size_t output_dims = num_dims;
  memcpy(normalized_perm, perm, num_dims * sizeof(size_t));
  normalized_shape[normalized_perm[0]] = shape[perm[0]];
  size_t output_pos = 0;
  for (size_t input_pos = 0; input_pos < num_dims; ++input_pos) {
    if (shape[perm[input_pos]] == 1) {
      remove_dimension(normalized_shape, normalized_perm, output_dims, output_pos);
      output_dims -= 1;
    } else {
      normalized_shape[normalized_perm[output_pos]] = shape[perm[input_pos]];
      output_pos += 1;
    }
  }
  if (output_pos == 0) {
    *normalized_num_dims = 1;
    *normalized_element_size_out = element_size;
    normalized_perm[0] = 0;
    normalized_shape[0] = 1;
    return;
  }
  output_pos = 1;
  for (size_t input_pos = 1; input_pos < output_dims;) {
    if (normalized_perm[output_pos] == normalized_perm[output_pos - 1] + 1) {
      normalized_shape[normalized_perm[output_pos - 1]] *= normalized_shape[normalized_perm[output_pos]];
      remove_dimension(normalized_shape, normalized_perm, num_dims, output_pos);
      output_dims -= 1;
    } else {
      input_pos += 1;
      output_pos += 1;
    }
  }
  size_t normalized_element_size = element_size;
  if (normalized_perm[output_dims - 1] == output_dims - 1) {
    normalized_element_size = element_size * normalized_shape[output_dims - 1];
    normalized_shape[output_dims - 1] = 1;
    if (output_dims > 1) {
      output_dims -= 1;
    }
  }
  *normalized_element_size_out = normalized_element_size;
  *normalized_num_dims = output_dims;
}
