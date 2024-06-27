// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>

#include "xnnpack.h"
#include "xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Calculates normalized offsets, input_shape, and output_shape.
// Each value in offsets must be less than the corresponding dimension of input_shape.
// Each value in sizes must be >= 0 and less than or equals to the corresponding dimension of input_shape.
// This function merges dimensions dimensions that are full slices into the outermost dimension possible.
// If value in sizes is 0, it will be treated as value at same index from input_shape.
// E.g. Given input shape { 4, 5, 3 }, with offsets { 0, 2, 0 }, and sizes { 4, 1, 3 }, the innermost dimension is a
// full slice, and so can be merged with its outer dimension, to give normalized input shape of { 4, 15 },
// output shape { 4, 3 } with offsets { 0, 6 }.
void xnn_normalize_slice(
    size_t num_dims,
    const size_t offsets[XNN_MIN_ELEMENTS(1)],
    const size_t sizes[XNN_MIN_ELEMENTS(1)],
    const size_t input_shape[XNN_MIN_ELEMENTS(1)],
    size_t normalized_offsets[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t normalized_input_shape[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t normalized_output_shape[XNN_MIN_ELEMENTS(XNN_MAX_TENSOR_DIMS)],
    size_t* num_normalized_dims);

void xnn_normalize_transpose_permutation(
    size_t num_dims,
    size_t element_size,
    const size_t* perm,
    const size_t* shape,
    const size_t* input_stride,
    const size_t* output_stride,
    size_t* normalized_num_dims,
    size_t* normalized_element_size,
    size_t* normalized_perm,
    size_t* normalized_shape,
    size_t* normalized_input_stride,
    size_t* normalized_output_stride);

void xnn_normalize_reduction(
    size_t* num_reduction_axes_ptr,
    size_t* reduction_axes,
    size_t* num_input_dims_ptr,
    size_t* input_dims);

#ifdef __cplusplus
}  // extern "C"
#endif
