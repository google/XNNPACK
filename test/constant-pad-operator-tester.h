// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>


class ConstantPadOperatorTester {
 public:
  inline ConstantPadOperatorTester& input_shape(std::initializer_list<size_t> input_shape) {
    assert(input_shape.size() <= XNN_MAX_TENSOR_DIMS);
    input_shape_ = std::vector<size_t>(input_shape);
    return *this;
  }

  inline const std::vector<size_t>& input_shape() const {
    return input_shape_;
  }

  inline size_t input_dim(size_t i) const {
    return i < input_shape_.size() ? input_shape_[i] : 1;
  }

  inline size_t num_dims() const {
    return input_shape_.size();
  }

  inline size_t num_input_elements() const {
    return std::accumulate(
      input_shape_.cbegin(), input_shape_.cend(), size_t(1), std::multiplies<size_t>());
  }

  inline ConstantPadOperatorTester& pre_paddings(std::initializer_list<size_t> pre_paddings) {
    assert(pre_paddings.size() <= XNN_MAX_TENSOR_DIMS);
    pre_paddings_ = std::vector<size_t>(pre_paddings);
    return *this;
  }

  inline const std::vector<size_t>& pre_paddings() const {
    return pre_paddings_;
  }

  inline size_t pre_padding(size_t i) const {
    return i < pre_paddings_.size() ? pre_paddings_[i] : 0;
  }

  inline size_t num_pre_paddings() const {
    return pre_paddings_.size();
  }

  inline ConstantPadOperatorTester& post_paddings(std::initializer_list<size_t> post_paddings) {
    assert(post_paddings.size() <= XNN_MAX_TENSOR_DIMS);
    post_paddings_ = std::vector<size_t>(post_paddings);
    return *this;
  }

  inline const std::vector<size_t>& post_paddings() const {
    return post_paddings_;
  }

  inline size_t post_padding(size_t i) const {
    return i < post_paddings_.size() ? post_paddings_[i] : 0;
  }

  inline size_t num_post_paddings() const {
    return post_paddings_.size();
  }

  inline size_t output_dim(size_t i) const {
    return pre_padding(i) + input_dim(i) + post_padding(i);
  }

  inline size_t num_output_elements() const {
    size_t elements = 1;
    for (size_t i = 0; i < num_dims(); i++) {
      elements *= output_dim(i);
    }
    return elements;
  }

  inline ConstantPadOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestX8() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + num_input_elements());
    std::vector<uint8_t> output(num_output_elements());
    std::vector<uint8_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT32_C(0xAA));
      const uint8_t padding_value = u8dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t pad_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_constant_pad_nd_x8(
          &padding_value, 0, &pad_op));
      ASSERT_NE(nullptr, pad_op);

      // Smart pointer to automatically delete pad_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_pad_op(pad_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_constant_pad_nd_x8(
          pad_op,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_constant_pad_nd_x8(
          pad_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(pad_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

  void TestRunX8() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + num_input_elements());
    std::vector<uint8_t> output(num_output_elements());
    std::vector<uint8_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT32_C(0xAA));
      const uint8_t padding_value = u8dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_constant_pad_nd_x8(
          0 /* flags */,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          input.data(), output.data(),
          &padding_value,
          /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

  void TestX16() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<uint16_t> u16dist;

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + num_input_elements());
    std::vector<uint16_t> output(num_output_elements());
    std::vector<uint16_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u16dist(rng); });
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));
      const uint16_t padding_value = u16dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t pad_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_constant_pad_nd_x16(
          &padding_value, 0, &pad_op));
      ASSERT_NE(nullptr, pad_op);

      // Smart pointer to automatically delete pad_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_pad_op(pad_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_constant_pad_nd_x16(
          pad_op,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_constant_pad_nd_x16(
          pad_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(pad_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

  void TestRunX16() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<uint16_t> u16dist;

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + num_input_elements());
    std::vector<uint16_t> output(num_output_elements());
    std::vector<uint16_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u16dist(rng); });
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));
      const uint16_t padding_value = u16dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_constant_pad_nd_x16(
          0 /* flags */,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          input.data(), output.data(),
          &padding_value,
          /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

  void TestX32() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<uint32_t> u32dist;

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint32_t> input(XNN_EXTRA_BYTES / sizeof(uint32_t) + num_input_elements());
    std::vector<uint32_t> output(num_output_elements());
    std::vector<uint32_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u32dist(rng); });
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));
      const uint32_t padding_value = u32dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      // Create, setup, run, and destroy a binary elementwise operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t pad_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_constant_pad_nd_x32(
          &padding_value, 0, &pad_op));
      ASSERT_NE(nullptr, pad_op);

      // Smart pointer to automatically delete pad_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_pad_op(pad_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_constant_pad_nd_x32(
          pad_op,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_constant_pad_nd_x32(
          pad_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(pad_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

  void TestRunX32() const {
    ASSERT_EQ(num_dims(), num_pre_paddings());
    ASSERT_EQ(num_dims(), num_post_paddings());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<uint32_t> u32dist;

    // Compute generalized shapes.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_dims;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_pre_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_post_paddings;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;
    std::fill(input_dims.begin(), input_dims.end(), 1);
    std::fill(input_pre_paddings.begin(), input_pre_paddings.end(), 0);
    std::fill(input_post_paddings.begin(), input_post_paddings.end(), 0);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    for (size_t i = 0; i < num_dims(); i++) {
      input_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = input_dim(i);
      input_pre_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = pre_padding(i);
      input_post_paddings[XNN_MAX_TENSOR_DIMS - num_dims() + i] = post_padding(i);
      output_dims[XNN_MAX_TENSOR_DIMS - num_dims() + i] = output_dim(i);
    }

    // Compute generalized strides.
    std::array<size_t, XNN_MAX_TENSOR_DIMS> input_strides;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> output_strides;
    size_t input_stride = 1, output_stride = 1;
    for (size_t i = XNN_MAX_TENSOR_DIMS; i != 0; i--) {
      input_strides[i - 1] = input_stride;
      output_strides[i - 1] = output_stride;
      input_stride *= input_dims[i - 1];
      output_stride *= output_dims[i - 1];
    }

    std::vector<uint32_t> input(XNN_EXTRA_BYTES / sizeof(uint32_t) + num_input_elements());
    std::vector<uint32_t> output(num_output_elements());
    std::vector<uint32_t> output_ref(num_output_elements());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u32dist(rng); });
      std::fill(output.begin(), output.end(), UINT32_C(0xDEADBEEF));
      const uint32_t padding_value = u32dist(rng);

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), padding_value);
      for (size_t i = 0; i < input_dims[0]; i++) {
        for (size_t j = 0; j < input_dims[1]; j++) {
          for (size_t k = 0; k < input_dims[2]; k++) {
            for (size_t l = 0; l < input_dims[3]; l++) {
              for (size_t m = 0; m < input_dims[4]; m++) {
                for (size_t n = 0; n < input_dims[5]; n++) {
                  const size_t output_index =
                    (i + input_pre_paddings[0]) * output_strides[0] +
                    (j + input_pre_paddings[1]) * output_strides[1] +
                    (k + input_pre_paddings[2]) * output_strides[2] +
                    (l + input_pre_paddings[3]) * output_strides[3] +
                    (m + input_pre_paddings[4]) * output_strides[4] +
                    (n + input_pre_paddings[5]) * output_strides[5];
                  const size_t input_index =
                    i * input_strides[0] + j * input_strides[1] + k * input_strides[2] +
                    l * input_strides[3] + m * input_strides[4] + n * input_strides[5];
                  output_ref[output_index] = input[input_index];
                }
              }
            }
          }
        }
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_constant_pad_nd_x32(
          0 /* flags */,
          num_dims(),
          input_shape().data(), pre_paddings().data(), post_paddings().data(),
          input.data(), output.data(),
          &padding_value,
          /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < output_dims[0]; i++) {
        for (size_t j = 0; j < output_dims[1]; j++) {
          for (size_t k = 0; k < output_dims[2]; k++) {
            for (size_t l = 0; l < output_dims[3]; l++) {
              for (size_t m = 0; m < output_dims[4]; m++) {
                for (size_t n = 0; n < output_dims[5]; n++) {
                  const size_t index =
                    i * output_strides[0] + j * output_strides[1] + k * output_strides[2] +
                    l * output_strides[3] + m * output_strides[4] + n * output_strides[5];
                  EXPECT_EQ(output[index], output_ref[index])
                    << "(i, j, k, l, m, n) = ("
                    << i << ", " << j << ", " << k << ", " << l << ", " << m << ", " << n << ")"
                    << ", padding value = " << padding_value;
                }
              }
            }
          }
        }
      }
    }
  }

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> pre_paddings_;
  std::vector<size_t> post_paddings_;
  size_t iterations_{3};
};
