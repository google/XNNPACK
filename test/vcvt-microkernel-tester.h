// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"

class VCvtMicrokernelTester {
 public:
  VCvtMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VCvtMicrokernelTester& scale(float scale) {
    assert(scale > 0.0f);
    assert(std::isnormal(scale));
    this->scale_ = scale;
    return *this;
  }

  float scale() const { return this->scale_; }

  VCvtMicrokernelTester& input_zero_point(int16_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  int16_t input_zero_point() const { return this->input_zero_point_; }

  VCvtMicrokernelTester& output_zero_point(int16_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  int16_t output_zero_point() const { return this->output_zero_point_; }

  VCvtMicrokernelTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  int16_t qmin() const { return this->qmin_; }

  VCvtMicrokernelTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  int16_t qmax() const { return this->qmax_; }

  VCvtMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_f16_f32_vcvt_ukernel_fn vcvt) const;

  void Test(xnn_f32_f16_vcvt_ukernel_fn vcvt) const;

  void Test(xnn_f16_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_f16_qs8_cvt_params_fn init_params);

  void Test(xnn_f32_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_f32_qs8_cvt_params_fn init_params) const;

  void Test(xnn_f32_qu8_vcvt_ukernel_fn vcvt,
            xnn_init_f32_qu8_cvt_params_fn init_params) const;

  void Test(xnn_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_qs8_cvt_params_fn init_params) const;

  void Test(xnn_qs16_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_qs16_qs8_cvt_params_fn init_params) const;

  void Test(xnn_qs8_f16_vcvt_ukernel_fn vcvt,
            xnn_init_qs8_f16_cvt_params_fn init_params) const;

  void Test(xnn_qs8_f32_vcvt_ukernel_fn vcvt,
            xnn_init_qs8_f32_cvt_params_fn init_params) const;

  void Test(xnn_qu8_vcvt_ukernel_fn vcvt,
            xnn_init_qu8_cvt_params_fn init_params) const;

  void Test(xnn_qu8_f32_vcvt_ukernel_fn vcvt,
            xnn_init_qu8_f32_cvt_params_fn init_params) const;

 private:
  float scale_ = 1.75f;
  int16_t input_zero_point_ = 1;
  int16_t output_zero_point_ = 5;
  int16_t qmin_ = std::numeric_limits<int16_t>::min();
  int16_t qmax_ = std::numeric_limits<int16_t>::max();
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};
