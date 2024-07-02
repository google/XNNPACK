// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"

class SliceOperatorTester {
 public:
  SliceOperatorTester& input_shape(std::initializer_list<size_t> input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  const std::vector<size_t>& input_shape() const {
    return input_shape_;
  }

  size_t input_dim(size_t i) const {
    return i < input_shape_.size() ? input_shape_[i] : 1;
  }

  size_t num_dims() const {
    return input_shape_.size();
  }

  size_t num_input_elements() const {
    return std::accumulate(
      input_shape_.cbegin(), input_shape_.cend(), size_t(1), std::multiplies<size_t>());
  }

  SliceOperatorTester& offsets(std::initializer_list<size_t> offsets) {
    assert(offsets.size() <= XNN_MAX_TENSOR_DIMS);
    offsets_ = std::vector<size_t>(offsets);
    return *this;
  }

  const std::vector<size_t>& offsets() const {
    return offsets_;
  }

  size_t offset(size_t i) const {
    return i < offsets_.size() ? offsets_[i] : 0;
  }

  size_t num_offsets() const {
    return offsets_.size();
  }

  SliceOperatorTester& sizes(std::initializer_list<size_t> sizes) {
    assert(sizes.size() <= XNN_MAX_TENSOR_DIMS);
    sizes_ = std::vector<size_t>(sizes);
    return *this;
  }

  const std::vector<size_t>& sizes() const {
    return sizes_;
  }

  size_t size(size_t i) const {
    return i < sizes_.size() ? sizes_[i] : 1;
  }

  size_t num_sizes() const {
    return sizes_.size();
  }

  size_t output_dim(size_t i) const {
    return size(i);
  }

  size_t num_output_elements() const {
    size_t elements = 1;
    for (size_t i = 0; i < num_dims(); i++) {
      elements *= output_dim(i);
    }
    return elements;
  }

  SliceOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestX8() const {
    ASSERT_EQ(num_dims(), num_offsets());
    ASSERT_EQ(num_dims(), num_sizes());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_offsets;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_sizes;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_offsets.begin(), input_offsets.end(), 0);
    std::fill(output_sizes.begin(), output_sizes.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_offsets[XNN_MAX_TENSOR_DIMS - num_dims() + i] = offset(i);
      output_sizes[XNN_MAX_TENSOR_DIMS - num_dims() + i] = size(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + num_input_elements());
    std::vector<uint8_t> output(num_output_elements());
    std::vector<uint8_t> output_ref(num_output_elements());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), UINT8_C(0));
      std::fill(output.begin(), output.end(), UINT32_C(0xAA));

      ComputeReference(input_dims, output_dims, input_offsets, input, output_ref);

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t slice_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_slice_nd_x8(
          0, &slice_op));
      ASSERT_NE(nullptr, slice_op);

      // Smart pointer to automatically delete slice_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_slice_op(slice_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_slice_nd_x8(
          slice_op,
          num_dims(),
          input_shape().data(), offsets().data(), sizes().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_slice_nd_x8(
          slice_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(slice_op, /*threadpool=*/nullptr));

      ASSERT_EQ(output, output_ref);
    }
  }

  void TestX16() const {
    ASSERT_EQ(num_dims(), num_offsets());
    ASSERT_EQ(num_dims(), num_sizes());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_offsets;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_sizes;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_offsets.begin(), input_offsets.end(), 0);
    std::fill(output_sizes.begin(), output_sizes.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_offsets[XNN_MAX_TENSOR_DIMS - num_dims() + i] = offset(i);
      output_sizes[XNN_MAX_TENSOR_DIMS - num_dims() + i] = size(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + num_input_elements());
    std::vector<uint16_t> output(num_output_elements());
    std::vector<uint16_t> output_ref(num_output_elements());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), UINT16_C(0));
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));

      ComputeReference(input_dims, output_dims, input_offsets, input, output_ref);

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t slice_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_slice_nd_x16(
          0, &slice_op));
      ASSERT_NE(nullptr, slice_op);

      // Smart pointer to automatically delete slice_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_slice_op(slice_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_slice_nd_x16(
          slice_op,
          num_dims(),
          input_shape().data(), offsets().data(), sizes().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_slice_nd_x16(
          slice_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(slice_op, /*threadpool=*/nullptr));

      ASSERT_EQ(output, output_ref);
    }
  }

  void TestX32() const {
    ASSERT_EQ(num_dims(), num_offsets());
    ASSERT_EQ(num_dims(), num_sizes());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_offsets;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_sizes;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_offsets.begin(), input_offsets.end(), 0);
    std::fill(output_sizes.begin(), output_sizes.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_offsets[XNN_MAX_TENSOR_DIMS - num_dims() + i] = offset(i);
      output_sizes[XNN_MAX_TENSOR_DIMS - num_dims() + i] = size(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    std::vector<uint32_t> input(XNN_EXTRA_BYTES / sizeof(uint32_t) + num_input_elements());
    std::vector<uint32_t> output(num_output_elements());
    std::vector<uint32_t> output_ref(num_output_elements());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), UINT32_C(0));
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));

      ComputeReference(input_dims, output_dims, input_offsets, input, output_ref);

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t slice_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_slice_nd_x32(
          0, &slice_op));
      ASSERT_NE(nullptr, slice_op);

      // Smart pointer to automatically delete slice_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_slice_op(slice_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_slice_nd_x32(
          slice_op,
          num_dims(),
          input_shape().data(), offsets().data(), sizes().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_slice_nd_x32(
          slice_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(slice_op, /*threadpool=*/nullptr));

      ASSERT_EQ(output, output_ref);
    }
  }

void TestRunX32() const {
    ASSERT_EQ(num_dims(), num_offsets());
    ASSERT_EQ(num_dims(), num_sizes());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_offsets;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_sizes;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_offsets.begin(), input_offsets.end(), 0);
    std::fill(output_sizes.begin(), output_sizes.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_offsets[XNN_MAX_TENSOR_DIMS - num_dims() + i] = offset(i);
      output_sizes[XNN_MAX_TENSOR_DIMS - num_dims() + i] = size(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    std::vector<uint32_t> input(XNN_EXTRA_BYTES / sizeof(uint32_t) + num_input_elements());
    std::vector<uint32_t> output(num_output_elements());
    std::vector<uint32_t> output_ref(num_output_elements());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), UINT32_C(0));
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));

      ComputeReference(input_dims, output_dims, input_offsets, input, output_ref);

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      ASSERT_EQ(xnn_status_success,
        xnn_run_slice_nd_x32(
          num_dims(),
          input_shape().data(), offsets().data(), sizes().data(),
          input.data(), output.data(),
          0,
          /*threadpool=*/nullptr));
      ASSERT_EQ(output, output_ref);
    }
  }

 private:
  template <typename T>
  void ComputeReference(
      const std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims,
      const std::array<size_t, XNN_MAX_TENSOR_DIMS> input_offsets,
      const std::vector<T>& input,
      std::vector<T>& output) const {
    for (size_t i = 0; i < output_dims[0]; i++) {
      for (size_t j = 0; j < output_dims[1]; j++) {
        for (size_t k = 0; k < output_dims[2]; k++) {
          for (size_t l = 0; l < output_dims[3]; l++) {
            for (size_t m = 0; m < output_dims[4]; m++) {
              for (size_t n = 0; n < output_dims[5]; n++) {
                const size_t output_index =
                    ((((i * output_dims[1]
                        + j) * output_dims[2]
                       + k) * output_dims[3]
                      + l) * output_dims[4]
                     + m) * output_dims[5]
                    + n;
                const size_t input_index =
                    (((((input_offsets[0] + i) * input_dims[1] +
                        (input_offsets[1] + j)) * input_dims[2] +
                       (input_offsets[2] + k)) * input_dims[3] +
                      (input_offsets[3] + l)) * input_dims[4] +
                     (input_offsets[4] + m)) * input_dims[5] +
                    (input_offsets[5] + n);
                output[output_index] = input[input_index];
              }
            }
          }
        }
      }
    }
  }

  std::vector<size_t> input_shape_;
  std::vector<size_t> offsets_;
  std::vector<size_t> sizes_;
  size_t iterations_{1};  // Use less iteration because we test a lot of dimensions.
};
