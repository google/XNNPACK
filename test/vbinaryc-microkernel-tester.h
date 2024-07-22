// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/requantization.h"

class VBinaryCMicrokernelTester {
 public:
  enum class OpType {
    AddC,
    CopySignC,
    RCopySignC,
    DivC,
    RDivC,
    MaxC,
    MinC,
    MulC,
    SqrDiffC,
    SubC,
    RSubC,
  };

  VBinaryCMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VBinaryCMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const { return this->inplace_; }

  VBinaryCMicrokernelTester& a_scale(float a_scale) {
    assert(a_scale > 0.0f);
    assert(std::isnormal(a_scale));
    this->a_scale_ = a_scale;
    return *this;
  }

  float a_scale() const { return this->a_scale_; }

  VBinaryCMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  uint8_t a_zero_point() const { return this->a_zero_point_; }

  VBinaryCMicrokernelTester& b_scale(float b_scale) {
    assert(b_scale > 0.0f);
    assert(std::isnormal(b_scale));
    this->b_scale_ = b_scale;
    return *this;
  }

  float b_scale() const { return this->b_scale_; }

  VBinaryCMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  uint8_t b_zero_point() const { return this->b_zero_point_; }

  VBinaryCMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  float y_scale() const { return this->y_scale_; }

  VBinaryCMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  uint8_t y_zero_point() const { return this->y_zero_point_; }

  VBinaryCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  VBinaryCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  VBinaryCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_f16_vbinary_ukernel_fn vbinaryc, OpType op_type) const;

  void Test(xnn_f16_vbinary_minmax_ukernel_fn vbinaryc_minmax, OpType op_type,
            xnn_init_f16_minmax_params_fn init_params) const;

  void Test(xnn_f32_vbinary_ukernel_fn vbinaryc, OpType op_type,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vbinary_relu_ukernel_fn vbinaryc_relu,
            OpType op_type) const;

  void Test(xnn_f32_vbinary_minmax_ukernel_fn vbinaryc_minmax, OpType op_type,
            xnn_init_f32_minmax_params_fn init_params) const;

  void Test(xnn_qu8_vadd_minmax_ukernel_fn vaddc_minmax,
            xnn_init_qu8_add_minmax_params_fn init_params) const;

  void Test(xnn_qu8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qu8_mul_minmax_params_fn init_params,
            xnn_qu8_requantize_fn requantize) const;

  void Test(xnn_qs8_vadd_minmax_ukernel_fn vaddc_minmax,
            xnn_init_qs8_add_minmax_params_fn init_params) const;

  void Test(xnn_qs8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qs8_mul_minmax_params_fn init_params,
            xnn_qs8_requantize_fn requantize) const;

  void Test(xnn_s32_vbinary_ukernel_fn vbinaryc, OpType op_type,
            xnn_init_s32_default_params_fn init_params = nullptr) const;
 private:
  size_t batch_size_{1};
  bool inplace_{false};
  float a_scale_{0.75f};
  float b_scale_{1.25f};
  float y_scale_{0.96875f};
  uint8_t a_zero_point_{121};
  uint8_t b_zero_point_{127};
  uint8_t y_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
