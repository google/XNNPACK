#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <numeric>
#include <vector>

#include <xnnpack.h>

#include <gtest/gtest.h>

// Delete before submit
// DO NOT SUBMIT
#include <iostream>

namespace dbg {

template <typename Container>
void print(const Container& c) {
  for (int i : c) {
    std::cerr << i << ", ";
  }
  std::cerr << "\n";
}

}  // namespace dbg
// End DO NOT SUBMIT

namespace xnnpack {

using create_bts_fn = decltype(&xnn_create_batch_to_space_nhwc_x32);
using setup_bts_fn = decltype(&xnn_setup_batch_to_space_nhwc_x32);

template<size_t Size>
struct BatchToSpaceFuncFromSize;

template<>
struct BatchToSpaceFuncFromSize<sizeof(int32_t)> {
  static constexpr create_bts_fn create = xnn_create_batch_to_space_nhwc_x32;
  static constexpr setup_bts_fn setup = xnn_setup_batch_to_space_nhwc_x32;
};

template<>
struct BatchToSpaceFuncFromSize<sizeof(int16_t)> {
  static constexpr create_bts_fn create = xnn_create_batch_to_space_nhwc_x16;
  static constexpr setup_bts_fn setup = xnn_setup_batch_to_space_nhwc_x16;
};

template<>
struct BatchToSpaceFuncFromSize<sizeof(int8_t)> {
  static constexpr create_bts_fn create = xnn_create_batch_to_space_nhwc_x8;
  static constexpr setup_bts_fn setup = xnn_setup_batch_to_space_nhwc_x8;
};

template<typename T>
using BatchToSpaceFunc = BatchToSpaceFuncFromSize<sizeof(T)>;


// Computes the coordinates matching given index and tensor shape.
template <typename T, size_t N>
std::array<T, N> CoordinatesFor(size_t idx, const std::array<T, N>& shape) {
  static_assert(N > 0, "Shape must have at least one element.");
  std::array<T, N> coordinates{};
  T i = N - 1;
  do {
    coordinates[i] = idx % shape[i];
    idx = (idx - coordinates[i]) / shape[i];
  } while (i-- > 0);
  return coordinates;
}

// Computes the index matching the given coordinates and tensor shape.
template <typename T, size_t N>
size_t IndexFor(const std::array<T, N> coords, const std::array<T, N> shape) {
  static_assert(N > 0, "Shape must have at least one element.");
  size_t idx = coords[N - 1];
  size_t stride = 1;
  for (int i = N - 2; i >= 0; --i) {
    stride *= shape[i + 1];
    idx += coords[i] * stride;
  }
  return idx;
}

// Computes the product of the items in the given array.
template <typename Container>
size_t Product(const Container& arr) {
  return std::accumulate(arr.begin(), arr.end(),
                         static_cast<typename Container::value_type>(1),
                         std::multiplies<typename Container::value_type>());
}

// Keeps tensor shape and data together.
template<typename T>
struct Tensor {
  std::array<size_t, 4> shape {};
  std::vector<T> data;

  size_t batch() const { return shape[0]; }
  size_t height() const { return shape[1]; }
  size_t width() const { return shape[2]; }
  size_t channels() const { return shape[3]; }
};

// Holds the test configuration for batch to space and space to batch.
//
// Note: padding and cropping parameters are mutually exclusive.
template<typename T>
class TestConfiguration {
 public:
  TestConfiguration& input(const std::vector<T>& data, const std::array<size_t, 4>& shape) {
    input_.data = data;
    input_.shape = shape;
  }

  const Tensor<T>& input() const {
    return input_;
  }

  TestConfiguration& input_shape(const std::array<size_t, 4>& shape) {
    input_.shape = shape;
    return *this;
  }

  const std::array<size_t, 4>& input_shape() { return input_.shape; }

  TestConfiguration& block_shape(const std::array<size_t, 2>& shape) {
    block_shape_ = shape;
    return *this;
  }

  const std::array<size_t, 2>& block_shape() const { return block_shape_; }

  size_t block_height() const { return block_shape_[0]; }

  size_t block_width() const { return block_shape_[1]; }

  TestConfiguration& padding(const std::array<size_t, 4>& padding) {
    assert(offset_kind_ != offset_kind::crop);
    offset_kind_ = offset_kind::pad;
    offset_ = padding;
    return *this;
  }

  const std::array<size_t, 4>& padding() const {
    assert(offset_kind_ != offset_kind::crop);
    return offset_;
  }

  size_t padding_top() const { return padding()[0]; }
  size_t padding_bottom() const { return padding()[1]; }
  size_t padding_left() const { return padding()[2]; }
  size_t padding_right() const { return padding()[3]; }

  TestConfiguration& crop(const std::array<size_t, 4>& crop) {
    assert(offset_kind_ != offset_kind::pad);
    offset_kind_ = offset_kind::crop;
    offset_ = crop;
    return *this;
  }

  const std::array<size_t, 4>& crop() const {
    assert(offset_kind_ != offset_kind::pad);
    return offset_;
  }

  size_t crop_top() const { return crop()[0]; }
  size_t crop_bottom() const { return crop()[1]; }
  size_t crop_left() const { return crop()[2]; }
  size_t crop_right() const { return crop()[3]; }

  TestConfiguration& Finalize() {
    const size_t input_size = Product(input_.shape);
    if (input_.data.empty()) {
      input_.data.resize(input_size);
      std::iota(input_.data.begin(), input_.data.end(), 0);
    }
    finalized_ = true;
    return *this;
  }

  void CheckPrerequisites() const {
    ASSERT_TRUE(finalized_);
    ASSERT_TRUE(!input_.data.empty());
    ASSERT_EQ(Product(input_.shape), input_.data.size());
  }

 private:
  enum class offset_kind {undecided, pad, crop};
  offset_kind offset_kind_ = offset_kind::undecided;
  std::array<size_t, 2> block_shape_ {};
  std::array<size_t, 4> offset_ {};
  Tensor<T> input_;
  bool finalized_ = false;
};

template<typename T>
void ApplyBatchToSpacePermutation(const Tensor<T>& input,
                                       const std::array<size_t, 2>& block_shape,
                                       Tensor<T>& result)
{
  // Reshape from input to intermediate tensor.
  const size_t bs_height = block_shape[0];
  const size_t bs_width = block_shape[1];
  const size_t input_batch_remain = input.batch() / bs_height / bs_width;
  std::array<size_t, 6> permutated_intermediate_shape{
      input_batch_remain, input.height(), bs_height,
      input.width(),      bs_width,       input.channels()};
  // Setup output data.
  result.shape = {input_batch_remain, input.height() * bs_height,
                  input.width() * bs_width, input.channels()};
  result.data.resize(input.data.size());
  // We only do this to catch errors in this reference implmeentation.
  std::fill(result.data.begin(), result.data.end(), 0xFC);

  const std::array<size_t, 3> reshape_prefix_shape {bs_height, bs_width, input_batch_remain};
  int in_idx = 0;
  for (size_t b = 0; b < input.shape[0]; ++b) {
    for (size_t i = 0; i < input.shape[1]; ++i) {
      for (size_t j = 0; j < input.shape[2]; ++j) {
        for (size_t c = 0; c < input.shape[3]; ++c) {
          const std::array<size_t, 3> batch_reshape = CoordinatesFor(b, reshape_prefix_shape);
          const std::array<size_t, 6> output_coords {
            batch_reshape[2], i, batch_reshape[0], j, batch_reshape[1], c};
          const size_t out_idx =
              IndexFor(output_coords, permutated_intermediate_shape);
          result.data[out_idx] = input.data[in_idx];
          ++in_idx;
        }
      }
    }
  }
}

template <typename T>
void Crop(const Tensor<T>& input, const std::array<size_t, 4>& crops, Tensor<T>& output)
{
  if (crops == decltype(crops){}) {
    output = input;
    return;
  }
  const std::vector<T>& in_data = input.data;
  const std::array<size_t, 4>& in_shape = input.shape;
  // Compute new shape.
  const size_t vertical_crop = crops[0] + crops[1];
  const size_t horizontal_crop = crops[2] + crops[3];
  const size_t output_height = vertical_crop < in_shape[1] ? in_shape[1] - vertical_crop : 0;
  const size_t output_width = horizontal_crop < in_shape[2] ? in_shape[2] - horizontal_crop : 0;

  output.shape = {
    in_shape[0],
    output_height,
    output_width,
    in_shape[3]};
  output.data.resize(Product(output.shape));
  // Copy data
  int out_idx = 0;
  for (size_t b = 0; b < in_shape[0]; ++b) {
    for (size_t i = crops[0]; i + crops[1] < in_shape[1]; ++i) {
      for (size_t j = crops[2]; j + crops[3] < in_shape[2]; ++j) {
        for (size_t c = 0; c < in_shape[3]; ++c) {
          const int in_idx = IndexFor({b,i,j,c}, in_shape);
          output.data[out_idx] = in_data[in_idx];
          ++out_idx;
        }
      }
    }
  }
}

template<typename T>
void ComputeBatchToSpaceReferenceData(TestConfiguration<T>& config, Tensor<T>& result) {
  config.CheckPrerequisites();
  ASSERT_GE(config.input().batch(), Product(config.block_shape()));
  ASSERT_EQ(config.input().batch() % Product(config.block_shape()), 0);
  Tensor<T> tmp;
  ApplyBatchToSpacePermutation(config.input(), config.block_shape(), tmp);
  Crop(tmp, config.crop(), result);
}

template<typename T>
void RunBatchToSpace(const TestConfiguration<T>& config, std::vector<T>& xnn_output_data) {
// #define IGNORE_OUTPUT_DATA_SIZE
#ifndef IGNORE_OUTPUT_DATA_SIZE
  const size_t horizontal_crop = config.crop_left() + config.crop_right();
  const size_t vertical_crop = config.crop_top() + config.crop_bottom();
  const size_t output_width_nocrop = config.input().width() * config.block_width();
  const size_t output_height_nocrop = config.input().height() * config.block_height();
  const size_t output_batch =
      config.input().batch() / config.block_height() / config.block_width();
  const size_t output_width = output_width_nocrop > horizontal_crop ? output_width_nocrop - horizontal_crop : 0;
  const size_t output_height = output_height_nocrop > vertical_crop ? output_height_nocrop - vertical_crop : 0;
  const size_t output_size =
      output_batch * output_height * output_width * config.input().channels();

  xnn_output_data.resize(output_size, -1);
#else
  xnn_output_data.resize(config.input().data.size(), -1);
#endif

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t batch_to_space_op = nullptr;

  ASSERT_EQ(xnn_status_success, BatchToSpaceFunc<T>::create(&batch_to_space_op));
  ASSERT_NE(nullptr, batch_to_space_op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
      batch_to_space_op_guard(batch_to_space_op, xnn_delete_operator);

  // ASAN is too smart and sometimes realises that XNNPACK will read outside of
  // the container... To prevent triggering it, we use another buffer sized
  // according to the maximum overreach that XNNPACK will allow.
  std::vector<T> input(config.input().data.size() + XNN_EXTRA_BYTES/sizeof(T));
  std::copy(config.input().data.begin(), config.input().data.end(), input.begin());

  ASSERT_EQ(xnn_status_success,
            BatchToSpaceFunc<T>::setup(
                batch_to_space_op,
                config.input().batch(),
                config.input().height(),
                config.input().width(),
                config.input().channels(),
                config.block_height(),
                config.block_width(),
                config.crop_top(),
                config.crop_bottom(),
                config.crop_left(),
                config.crop_right(),
                input.data(),
                xnn_output_data.data()
                ));

  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(batch_to_space_op, nullptr /* thread pool */));
}

}  //  namespace xnnpack
