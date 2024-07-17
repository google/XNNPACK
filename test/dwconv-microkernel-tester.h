// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/requantization.h"

class DWConvMicrokernelTester {
 public:
  DWConvMicrokernelTester& width(uint32_t width) {
    assert(width >= 1);
    this->width_ = width;
    return *this;
  }

  uint32_t width() const { return this->width_; }

  DWConvMicrokernelTester& step(uint32_t step) {
    assert(step >= 1);
    this->step_ = step;
    return *this;
  }

  uint32_t step() const { return this->step_; }

  DWConvMicrokernelTester& channels(uint32_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  uint32_t channels() const { return this->channels_; }

  DWConvMicrokernelTester& channel_tile(uint32_t channel_tile) {
    assert(channel_tile != 0);
    this->channel_tile_ = channel_tile;
    return *this;
  }

  uint32_t channel_tile() const { return this->channel_tile_; }

  DWConvMicrokernelTester& channel_subtile(uint32_t channel_subtile) {
    assert(channel_subtile != 0);
    this->channel_subtile_ = channel_subtile;
    return *this;
  }

  uint32_t channel_subtile() const { return this->channel_subtile_; }

  DWConvMicrokernelTester& channel_round(uint32_t channel_round) {
    assert(channel_round != 0);
    this->channel_round_ = channel_round;
    return *this;
  }

  uint32_t channel_round() const { return this->channel_round_; }

  DWConvMicrokernelTester& kernel_tile(uint32_t kernel_tile) {
    assert(kernel_tile != 0);
    this->kernel_tile_ = kernel_tile;
    return *this;
  }

  uint32_t kernel_tile() const { return this->kernel_tile_; }

  DWConvMicrokernelTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size != 0);
    this->kernel_size_ = kernel_size;
    return *this;
  }

  uint32_t kernel_size() const { return this->kernel_size_; }

  uint32_t packed_channels() const {
    return (channels() / channel_tile() + !!(channels() % channel_tile())) *
           channel_tile();
  }

  DWConvMicrokernelTester& output_stride(uint32_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  uint32_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  DWConvMicrokernelTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  uint8_t input_zero_point() const { return this->input_zero_point_; }

  DWConvMicrokernelTester& kernel_zero_point(uint8_t kernel_zero_point) {
    this->kernel_zero_point_ = kernel_zero_point;
    return *this;
  }

  uint8_t kernel_zero_point() const { return this->kernel_zero_point_; }

  DWConvMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  DWConvMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  DWConvMicrokernelTester& input_offset(size_t input_offset) {
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const { return this->input_offset_; }

  DWConvMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  size_t zero_index() const { return this->zero_index_; }

  DWConvMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  DWConvMicrokernelTester& first_pass_tile(size_t first_pass_tile) {
    this->first_pass_tile_ = first_pass_tile;
    return *this;
  }

  size_t first_pass_tile() const { return this->first_pass_tile_; }

  DWConvMicrokernelTester& middle_pass_tile(size_t middle_pass_tile) {
    this->middle_pass_tile_ = middle_pass_tile;
    return *this;
  }

  size_t middle_pass_tile() const { return this->middle_pass_tile_; }

  DWConvMicrokernelTester& last_pass_tile(size_t last_pass_tile) {
    this->last_pass_tile_ = last_pass_tile;
    return *this;
  }

  size_t last_pass_tile() const { return this->last_pass_tile_; }

  void Test(xnn_qu8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
            xnn_init_qu8_conv_minmax_params_fn init_params,
            xnn_qu8_requantize_fn requantize) const;

  void Test(xnn_qu8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
            xnn_init_qu8_conv_minmax_params_fn init_params,
            xnn_qu8_requantize_fn requantize) const;

  void Test(xnn_qs8_qc8w_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
            xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
            xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qs8_qc8w_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
            xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
            xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qs8_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
            xnn_init_qs8_conv_minmax_params_fn init_params,
            xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_qs8_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
            xnn_init_qs8_conv_minmax_params_fn init_params,
            xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
            xnn_init_f16_minmax_params_fn init_params) const;

  void Test(xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
            xnn_init_f16_minmax_params_fn init_params) const;

  void Test(xnn_f32_dwconv_unipass_ukernel_fn dwconv) const;

  void Test(xnn_f32_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
            xnn_init_f32_minmax_params_fn init_params) const;

  void Test(xnn_f32_dwconv_multipass_ukernel_fn dwconv) const;

  void Test(xnn_f32_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
            xnn_init_f32_minmax_params_fn init_params) const;

 private:
  uint32_t channels_{1};
  uint32_t channel_tile_{1};
  uint32_t channel_subtile_{1};
  uint32_t channel_round_{1};
  uint32_t kernel_tile_{1};
  uint32_t kernel_size_{1};
  uint32_t width_{1};
  uint32_t step_{1};
  uint32_t output_stride_{0};
  uint8_t input_zero_point_{127};
  uint8_t kernel_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t input_offset_{0};
  size_t zero_index_{SIZE_MAX};
  size_t iterations_{3};
  size_t first_pass_tile_{0};
  size_t middle_pass_tile_{0};
  size_t last_pass_tile_{0};
};

struct LoopParams {
  LoopParams() = default;
  explicit LoopParams(size_t from, size_t to, size_t step)
      : is_set(true), from(from), to(to), step(step) {}
  bool is_set = false;
  size_t from = 1;
  size_t to = 1;
  size_t step = 1;
};

struct DWConvTestParams {
  DWConvTestParams(std::string test_name, DWConvMicrokernelTester tester,
                 std::function<void(DWConvMicrokernelTester& tester)> test_func,
                 std::function<void(void)> isa_check = nullptr)
      : test_name(test_name),
        tester(tester),
        test_func(test_func),
        isa_check(isa_check) {}

  // Setters for the loops over `k`, `m`, and `n`.
  DWConvTestParams& loop_kernel_size(size_t from, size_t to, size_t step = 1) {
    loop_kernel_size_ = LoopParams(from, to, step);
    return *this;
  }
  DWConvTestParams& loop_channels(size_t from, size_t to, size_t step = 1) {
    loop_channels_ = LoopParams(from, to, step);
    return *this;
  }
  DWConvTestParams& loop_step(size_t from, size_t to, size_t step = 1) {
    loop_step_ = LoopParams(from, to, step);
    return *this;
  }
  DWConvTestParams& loop_zi(size_t from, size_t to, size_t step = 1) {
    loop_zi_ = LoopParams(from, to, step);
    return *this;
  }

  std::string test_name;
  DWConvMicrokernelTester tester;
  std::function<void(DWConvMicrokernelTester& tester)> test_func;
  std::function<void(void)> isa_check;
  LoopParams loop_kernel_size_;
  LoopParams loop_channels_;
  LoopParams loop_step_;
  LoopParams loop_zi_;
};

using DWConvTest = testing::TestWithParam<DWConvTestParams>;
