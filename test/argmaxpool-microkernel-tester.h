// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/buffer.h"
#include "xnnpack/isa-checks.h"
#include "next_prime.h"
#include "replicable_random_device.h"

class ArgMaxPoolMicrokernelTester {
 public:
  ArgMaxPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const {
    return this->output_pixels_;
  }

  ArgMaxPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const {
    return this->step_;
  }

  ArgMaxPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const {
    return this->input_offset_;
  }

  ArgMaxPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  ArgMaxPoolMicrokernelTester& pooling_tile(size_t primary_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = 0;
    return *this;
  }

  ArgMaxPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  ArgMaxPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  ArgMaxPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  ArgMaxPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  ArgMaxPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  ArgMaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_argmaxpool_unipass_ukernel_fn argmaxpool) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    xnnpack::Buffer<float> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<uint32_t> index(output_pixels() * channels());
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    xnnpack::Buffer<uint32_t> index_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = indirect_input[x * step()][c + input_offset()];
          uint32_t max_index = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float value = indirect_input[x * step() + p][c + input_offset()];
            if (value > max_value) {
              max_value = value;
              max_index = p;
            }
          }
          output_ref[x * channels() + c] = max_value;
          index_ref[x * channels() + c] = max_index;
        }
      }

      // Call optimized micro-kernel.
      argmaxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), output.data(), index.data(),
        step() * sizeof(void*),
        (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(output_ref[x * channels() + c], output[x * output_stride() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(
              indirect_input[x * step() + index_ref[x * channels() + c]][c + input_offset()],
              indirect_input[x * step() + index[x * channels() + c]][c + input_offset()])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(index_ref[x * channels() + c], index[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_argmaxpool_multipass_ukernel_fn argmaxpool) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    xnnpack::Buffer<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    xnnpack::Buffer<float> output((output_pixels() - 1) * output_stride() + channels());
    xnnpack::Buffer<uint32_t> index(output_pixels() * channels());
    xnnpack::Buffer<uint32_t, XNN_ALLOCATION_ALIGNMENT> index_buffer(
        channels() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> output_buffer(
        channels() + XNN_EXTRA_BYTES / sizeof(float));
    xnnpack::Buffer<float> output_ref(output_pixels() * channels());
    xnnpack::Buffer<uint32_t> index_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = indirect_input[x * step()][c + input_offset()];
          uint32_t max_index = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            const float value = indirect_input[x * step() + p][c + input_offset()];
            if (value > max_value) {
              max_value = value;
              max_index = p;
            }
          }
          output_ref[x * channels() + c] = max_value;
          index_ref[x * channels() + c] = max_index;
        }
      }

      // Call optimized micro-kernel.
      argmaxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float),
        output_buffer.data(), index_buffer.data(),
        output.data(), index.data(),
        (step() - (packed_pooling_elements() - incremental_pooling_tile())) * sizeof(void*),
        (output_stride() - channels()) * sizeof(float));

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(output_ref[x * channels() + c], output[x * output_stride() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(
              indirect_input[x * step() + index_ref[x * channels() + c]][c + input_offset()],
              indirect_input[x * step() + index[x * channels() + c]][c + input_offset()])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(index_ref[x * channels() + c], index[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t step_{1};
  size_t primary_pooling_tile_{1};
  size_t incremental_pooling_tile_{1};
  size_t output_stride_{0};
  size_t iterations_{3};
};

void ArgmaxPoolUnipassTest(
  xnn_f32_argmaxpool_unipass_ukernel_fn ukernel,
  const std::string kernel_name,
  size_t _pooling_elements,
  size_t _primary_tile,
  size_t _incremental_tile,
  size_t _channels,
  size_t _input_offset)
{
  ArgMaxPoolMicrokernelTester tester;
  tester.pooling_elements(_pooling_elements);
  tester.pooling_tile(_primary_tile, (_incremental_tile != 0) ? _incremental_tile : 0);
  tester.channels(_channels);
  if (_input_offset != 0) {
    tester.input_offset(_input_offset);
  }
  if (kernel_name.find("scalar") != std::string::npos) {
    tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Native);
  }
  else {
    tester.Test(ukernel);
  }
}

void ArgmaxPoolMultipassTest(
  xnn_f32_argmaxpool_multipass_ukernel_fn ukernel,
  const std::string kernel_name,
  size_t _pooling_elements,
  size_t _primary_tile,
  size_t _incremental_tile,
  size_t _channels,
  size_t _input_offset)
{
  ArgMaxPoolMicrokernelTester tester;
  tester.pooling_elements(_pooling_elements);
  tester.pooling_tile(_primary_tile, (_incremental_tile != 0) ? _incremental_tile : 0);
  tester.channels(_channels);
  if (_input_offset != 0) {
    tester.input_offset(_input_offset);
  }
  if (kernel_name.find("scalar") != std::string::npos) {
    tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Native);
  }
  else {
    tester.Test(ukernel);
  }
}

#define XNN_TEST_ARGMAXPOOL_CHANNELS_EQ_UNIPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_eq_unipass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t _channels = channel_tile * get_batch_scale<datatype>();                                               \
    const std::string kernel_name = #ukernel;                                                                          \
    ArgmaxPoolUnipassTest(ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, 0);           \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_unipass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile + 1) : channel_scaled_tile + 1;          \
    const std::string kernel_name = #ukernel;                                                                          \
    ArgmaxPoolUnipassTest(                                                                                             \
      ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, channel_scaled_tile, _input_offset);         \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_unipass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      ArgmaxPoolUnipassTest(                                                                                           \
        ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, channel_scaled_tile, 0);              \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_unipass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile + 1) : channel_scaled_tile + 1;          \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      ArgmaxPoolUnipassTest(                                                                                           \
        ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, channel_scaled_tile, _input_offset);  \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_DIV_UNIPASS(                                                                      \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_div_unipass_fulltile)                                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;                 \
    for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {            \
      ArgmaxPoolUnipassTest(ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, 0);         \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_unipass_fulltile_with_input_offset)                                                       \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;                 \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 8) : channel_scaled_tile * 8;          \
    for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {            \
      ArgmaxPoolUnipassTest(                                                                                           \
        ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, _input_offset);                 \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_unipass_subtile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;                 \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolUnipassTest(ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, 0);  \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_unipass_subtile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;                 \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 8) : channel_scaled_tile * 8;          \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolUnipassTest(                                                                                         \
          ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, _input_offset);          \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_LT_UNIPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_lt_unipass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                         \
      ArgmaxPoolUnipassTest(ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, 0);         \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_unipass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile) : channel_scaled_tile;                  \
    for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                         \
      ArgmaxPoolUnipassTest(                                                                                           \
        ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, _input_offset);                 \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_unipass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                       \
        ArgmaxPoolUnipassTest(ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, 0);  \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_unipass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile != 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile) : channel_scaled_tile;                  \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                       \
        ArgmaxPoolUnipassTest(                                                                                         \
          ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, _input_offset);          \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_GT_UNIPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_gt_unipass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    const size_t channels_start = _channel_tile + 1;                                                                   \
    const size_t channels_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                         \
    for (size_t _channels = channels_start; _channels < channels_end; _channels++) {                                   \
      ArgmaxPoolUnipassTest(ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, 0);         \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_unipass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 2) : channel_scaled_tile * 2;          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    const size_t channels_start = _channel_tile + 1;                                                                   \
    const size_t channels_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                         \
    for (size_t _channels = channels_start; _channels < channels_end; _channels++) {                                   \
      ArgmaxPoolUnipassTest(                                                                                           \
        ukernel, kernel_name, primary_tile, primary_tile, incremental_tile, _channels, _input_offset);                 \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_unipass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    const size_t channels_start = _channel_tile + 1;                                                                   \
    const size_t channels_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                         \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = channels_start; _channels < channels_end; _channels++) {                                 \
        ArgmaxPoolUnipassTest(ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, 0);  \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_unipass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile != 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 2) : channel_scaled_tile * 2;          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    const size_t channels_start = _channel_tile + 1;                                                                   \
    const size_t channels_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                         \
    for (size_t _pooling_elements = 2; _pooling_elements < primary_tile; _pooling_elements++) {                        \
      for (size_t _channels = channels_start; _channels < channels_end; _channels++) {                                 \
        ArgmaxPoolUnipassTest(                                                                                         \
          ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, _input_offset);          \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_EQ_TWOPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_eq_twopass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t _channels = channel_tile * get_batch_scale<datatype>();                                               \
    const std::string kernel_name = #ukernel;                                                                          \
    ArgmaxPoolMultipassTest(                                                                                           \
      ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);            \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_twopass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile + 1) : channel_scaled_tile + 1;          \
    ArgmaxPoolMultipassTest(                                                                                           \
      ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, channel_scaled_tile,      \
      _input_offset);                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_twopass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, channel_scaled_tile, 0);              \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_twopass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile + 1) : channel_scaled_tile + 1;          \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, channel_scaled_tile, _input_offset);  \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_DIV_TWOPASS(                                                                      \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_div_twopass_fulltile)                                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {            \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);          \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_twopass_fulltile_with_input_offset)                                                       \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 5) : channel_scaled_tile * 5;          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {            \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,              \
        _input_offset);                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_twopass_subtile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, 0);                      \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_twopass_subtile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 8) : channel_scaled_tile * 8;          \
    const size_t _channel_tile = (channel_scaled_tile == channel_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, _pooling_elements, primary_tile, incremental_tile, _channels, _input_offset);          \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_LT_TWOPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_lt_twopass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                         \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);          \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_twopass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile) : channel_scaled_tile;                  \
    for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                         \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,              \
        _input_offset);                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_twopass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = channel_scaled_tile;                                                                  \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = 1; _channels < _channel_tile; _channels++) {                                             \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_twopass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile) : channel_scaled_tile;                  \
    const size_t _channel_tile = channel_scaled_tile;                                                                  \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = 1; _channels < _channel_tile; _channels++) {                                             \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,            \
          _input_offset);                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_GT_TWOPASS(                                                                       \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_gt_twopass_fulltile)                                                                          \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                     \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);          \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_twopass_fulltile_with_input_offset)                                                        \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 2) : channel_scaled_tile * 2;          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                     \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,              \
        _input_offset);                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_twopass_subtile)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                   \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_twopass_subtile_with_input_offset)                                                         \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 2) : channel_scaled_tile * 2;          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _pooling_elements = primary_tile + 1; _pooling_elements < primary_tile + incremental_tile;             \
         _pooling_elements++) {                                                                                        \
      for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                   \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,            \
          _input_offset);                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_EQ_MULTIPASS(                                                                     \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_eq_multipass)                                                                                 \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, channel_scaled_tile,    \
        0);                                                                                                            \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_eq_multipass_with_input_offset)                                                               \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile + 1) : channel_scaled_tile + 1;          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      ArgmaxPoolMultipassTest(                                                                                         \
        ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, channel_scaled_tile,    \
        _input_offset);                                                                                                \
    }                                                                                                                  \
  }


#define XNN_TEST_ARGMAXPOOL_CHANNELS_DIV_MULTIPASS(                                                                    \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_div_multipass)                                                                                \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = (channel_tile == channel_scaled_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_div_multipass_with_input_offset)                                                              \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 8) : channel_scaled_tile * 8;          \
    const size_t _channel_tile = (channel_tile == channel_scaled_tile) ? channel_tile : channel_scaled_tile;           \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = _channel_tile * 2; _channels < _channel_tile * 8; _channels += _channel_tile) {          \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,            \
          _input_offset);                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_LT_MULTIPASS(                                                                     \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_lt_multipass)                                                                                 \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                       \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_lt_multipass_with_input_offset)                                                               \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0 || (channel_tile <= 1 && channel_scaled_tile == channel_tile)) {                         \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = 1; _channels < channel_scaled_tile; _channels++) {                                       \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,            \
          channel_scaled_tile);                                                                                        \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_CHANNELS_GT_MULTIPASS(                                                                     \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, channels_gt_multipass)                                                                                 \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                   \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels, 0);        \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, channels_gt_multipass_with_input_offset)                                                               \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    if (incremental_tile == 0) {                                                                                       \
      GTEST_SKIP();                                                                                                    \
    }                                                                                                                  \
    const std::string kernel_name = #ukernel;                                                                          \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 2) : channel_scaled_tile * 2;          \
    const size_t _channel_tile = channel_scaled_tile == channel_tile ? channel_tile : channel_scaled_tile;             \
    const size_t channel_start = _channel_tile + 1;                                                                    \
    const size_t channel_end = (_channel_tile == 1) ? 10 : _channel_tile * 2;                                          \
    for (size_t _pooling_elements = primary_tile + incremental_tile + 1;                                               \
         _pooling_elements <= primary_tile + incremental_tile * 3; _pooling_elements += 3) {                           \
      for (size_t _channels = channel_start; _channels < channel_end; _channels++) {                                   \
        ArgmaxPoolMultipassTest(                                                                                       \
          ukernel, kernel_name, primary_tile + incremental_tile, primary_tile, incremental_tile, _channels,            \
          _input_offset);                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define XNN_TEST_ARGMAXPOOL_FEW_OUTPUT_PIXELS(                                                                         \
  ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params)  \
  TEST(ukernel, few_output_pixels)                                                                                     \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    const size_t min_pooling = (incremental_tile == 0) ? 2 : primary_tile + 1;                                         \
    const size_t max_pooling = (incremental_tile == 0) ? primary_tile : primary_tile + incremental_tile;               \
    const size_t loop_channel_tile = channel_tile == channel_scaled_tile ? channel_tile : channel_scaled_tile;         \
    for (size_t _output_pixels = 2; _output_pixels <= 5; _output_pixels++) {                                           \
      for (size_t _pooling_elements = min_pooling; _pooling_elements <= max_pooling; _pooling_elements++) {            \
        const size_t channel_step = std::max<size_t>(1, loop_channel_tile - 1);                                        \
        for (size_t _channels = 1; _channels <= loop_channel_tile * 5; _channels += channel_step) {                    \
          ArgMaxPoolMicrokernelTester tester;                                                                          \
          tester.output_pixels(_output_pixels);                                                                        \
          tester.pooling_elements(_pooling_elements);                                                                  \
          tester.pooling_tile(primary_tile, (incremental_tile != 0) ? incremental_tile : 0);                           \
          tester.channels(_channels);                                                                                  \
          if (kernel_name.find("scalar") != std::string::npos) {                                                       \
            tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Scalar);                                        \
          }                                                                                                            \
          else {                                                                                                       \
            tester.Test(ukernel);                                                                                      \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, few_output_pixels_with_input_offset)                                                                   \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _input_offset =                                                                                             \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 5 + 1) : channel_scaled_tile * 5 + 1;  \
    const size_t min_pooling = (incremental_tile == 0) ? 2 : primary_tile + 1;                                         \
    const size_t max_pooling = (incremental_tile == 0) ? primary_tile : primary_tile + incremental_tile;               \
    const size_t loop_channel_tile = channel_tile == channel_scaled_tile ? channel_tile : channel_scaled_tile;         \
    for (size_t _output_pixels = 2; _output_pixels <= 5; _output_pixels++) {                                           \
      for (size_t _pooling_elements = min_pooling; _pooling_elements <= max_pooling; _pooling_elements++) {            \
        const size_t channel_step = std::max<size_t>(1, loop_channel_tile - 1);                                        \
        for (size_t _channels = 1; _channels <= loop_channel_tile * 5; _channels += channel_step) {                    \
          ArgMaxPoolMicrokernelTester tester;                                                                          \
          tester.output_pixels(_output_pixels);                                                                        \
          tester.pooling_elements(_pooling_elements);                                                                  \
          tester.pooling_tile(primary_tile, (incremental_tile != 0) ? incremental_tile : 0);                           \
          tester.channels(_channels);                                                                                  \
          tester.input_offset(_input_offset);                                                                          \
          if (kernel_name.find("scalar") != std::string::npos) {                                                       \
            tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Scalar);                                        \
          }                                                                                                            \
          else {                                                                                                       \
            tester.Test(ukernel);                                                                                      \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, few_output_pixels_with_output_stride)                                                                  \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _output_stride =                                                                                            \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 5 + 1) : channel_scaled_tile * 5 + 1;  \
    const size_t min_pooling = (incremental_tile == 0) ? 2 : primary_tile + 1;                                         \
    const size_t max_pooling = (incremental_tile == 0) ? primary_tile : primary_tile + incremental_tile;               \
    const size_t loop_channel_tile = channel_tile == channel_scaled_tile ? channel_tile : channel_scaled_tile;         \
    for (size_t _output_pixels = 2; _output_pixels <= 5; _output_pixels++) {                                           \
      for (size_t _pooling_elements = min_pooling; _pooling_elements <= max_pooling; _pooling_elements++) {            \
        const size_t channel_step = std::max<size_t>(1, loop_channel_tile - 1);                                        \
        for (size_t _channels = 1; _channels <= loop_channel_tile * 5; _channels += channel_step) {                    \
          ArgMaxPoolMicrokernelTester tester;                                                                          \
          tester.output_pixels(_output_pixels);                                                                        \
          tester.pooling_elements(_pooling_elements);                                                                  \
          tester.pooling_tile(primary_tile, (incremental_tile != 0) ? incremental_tile : 0);                           \
          tester.channels(_channels);                                                                                  \
          tester.output_stride(_output_stride);                                                                        \
          if (kernel_name.find("scalar") != std::string::npos) {                                                       \
            tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Scalar);                                        \
          }                                                                                                            \
          else {                                                                                                       \
            tester.Test(ukernel);                                                                                      \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TEST(ukernel, few_output_pixels_with_step)                                                                           \
  {                                                                                                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                              \
    const std::string kernel_name = #ukernel;                                                                          \
    const size_t channel_scaled_tile = channel_tile * get_batch_scale<datatype>();                                     \
    size_t _output_stride =                                                                                            \
      (channel_tile == channel_scaled_tile) ? xnnpack::NextPrime(channel_tile * 5 + 1) : channel_scaled_tile * 5 + 1;  \
    const size_t min_pooling = (incremental_tile == 0) ? 2 : primary_tile + 1;                                         \
    const size_t max_pooling = (incremental_tile == 0) ? primary_tile : primary_tile + incremental_tile;               \
    const size_t loop_channel_tile = channel_tile == channel_scaled_tile ? channel_tile : channel_scaled_tile;         \
    for (size_t _output_pixels = 2; _output_pixels <= 5; _output_pixels++) {                                           \
      for (size_t _pooling_elements = min_pooling; _pooling_elements <= max_pooling; _pooling_elements++) {            \
        const size_t channel_step = std::max<size_t>(1, loop_channel_tile - 1);                                        \
        for (size_t _channels = 1; _channels <= loop_channel_tile * 5; _channels += channel_step) {                    \
          for (size_t _step = 2; _step <= _pooling_elements; _step++) {                                                \
            ArgMaxPoolMicrokernelTester tester;                                                                        \
            tester.output_pixels(_output_pixels);                                                                      \
            tester.pooling_elements(_pooling_elements);                                                                \
            tester.pooling_tile(primary_tile, (incremental_tile != 0) ? incremental_tile : 0);                         \
            tester.step(_step);                                                                                        \
            tester.channels(_channels);                                                                                \
            tester.output_stride(_output_stride);                                                                      \
            if (kernel_name.find("scalar") != std::string::npos) {                                                     \
              tester.Test(ukernel, ArgMaxPoolMicrokernelTester::Variant::Scalar);                                      \
            }                                                                                                          \
            else {                                                                                                     \
              tester.Test(ukernel);                                                                                    \
            }                                                                                                          \
          }                                                                                                            \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }
