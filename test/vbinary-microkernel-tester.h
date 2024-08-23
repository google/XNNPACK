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
#include <type_traits>

#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"

struct Float16 {
  uint16_t value;

  Float16() = default;
  Float16(float value) : value(fp16_ieee_from_fp32_value(value)) {}

  operator float() const { return fp16_ieee_to_fp32_value(value); }
};

class VBinaryMicrokernelTester {
 public:
  enum class OpType {
    Add,
    CopySign,
    RCopySign,
    Div,
    RDiv,
    Max,
    Min,
    Mul,
    Sub,
    RSub,
    SqrDiff,
  };

  template <typename A, typename B, typename Result>
  void reference_op_impl(const A* a, const B* b, Result* result, size_t n, OpType op_type) const {
    size_t stride_b = broadcast_b() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
      switch (op_type) {
        case OpType::Add:
          result[i] = a[i] + b[i * stride_b];
          break;
        case OpType::CopySign:
          result[i] = std::copysign(a[i], b[i * stride_b]);
          break;
        case OpType::RCopySign:
          result[i] = std::copysign(b[i * stride_b], a[i]);
          break;
        case OpType::Div:
          result[i] = a[i] / b[i * stride_b];
          break;
        case OpType::RDiv:
          result[i] = b[i * stride_b] / a[i];
          break;
        case OpType::Max:
          result[i] = std::max(a[i], b[i * stride_b]);
          break;
        case OpType::Min:
          result[i] = std::min(a[i], b[i * stride_b]);
          break;
        case OpType::Mul:
          if (std::is_integral<A>::value && std::is_integral<B>::value) {
            // Overflow is the expected behavior.
            int64_t result_wide = static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i * stride_b]);
            result[i] = result_wide & ((static_cast<int64_t>(1) << (sizeof(Result) * 8)) - 1);
          } else {
            result[i] = a[i] * b[i * stride_b];
          }
          break;
        case OpType::SqrDiff: {
          const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i * stride_b]);
          result[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          result[i] = a[i] - b[i * stride_b];
          break;
        case OpType::RSub:
          result[i] = b[i * stride_b] - a[i];
          break;
      }
    }
  }

  VBinaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VBinaryMicrokernelTester& inplace_a(bool inplace_a) {
    this->inplace_a_ = inplace_a;
    return *this;
  }

  bool inplace_a() const { return this->inplace_a_; }

  VBinaryMicrokernelTester& inplace_b(bool inplace_b) {
    this->inplace_b_ = inplace_b;
    return *this;
  }

  bool inplace_b() const { return this->inplace_b_; }

  VBinaryMicrokernelTester& broadcast_b(bool broadcast_b) {
    this->broadcast_b_ = broadcast_b;
    return *this;
  }

  bool broadcast_b() const { return this->broadcast_b_; }

  VBinaryMicrokernelTester& a_scale(float a_scale) {
    assert(a_scale > 0.0f);
    assert(std::isnormal(a_scale));
    this->a_scale_ = a_scale;
    return *this;
  }

  float a_scale() const { return this->a_scale_; }

  VBinaryMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  uint8_t a_zero_point() const { return this->a_zero_point_; }

  VBinaryMicrokernelTester& b_scale(float b_scale) {
    assert(b_scale > 0.0f);
    assert(std::isnormal(b_scale));
    this->b_scale_ = b_scale;
    return *this;
  }

  float b_scale() const { return this->b_scale_; }

  VBinaryMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  uint8_t b_zero_point() const { return this->b_zero_point_; }

  VBinaryMicrokernelTester& y_scale(float y_scale) {
    assert(y_scale > 0.0f);
    assert(std::isnormal(y_scale));
    this->y_scale_ = y_scale;
    return *this;
  }

  float y_scale() const { return this->y_scale_; }

  VBinaryMicrokernelTester& y_zero_point(uint8_t y_zero_point) {
    this->y_zero_point_ = y_zero_point;
    return *this;
  }

  uint8_t y_zero_point() const { return this->y_zero_point_; }

  VBinaryMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  VBinaryMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  VBinaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  void Test(xnn_f16_vbinary_ukernel_fn vbinary, OpType op_type) const;

  void Test(xnn_f16_vbinary_minmax_ukernel_fn vbinary_minmax, OpType op_type,
            xnn_init_f16_minmax_params_fn init_params) const;

  void Test(xnn_f32_vbinary_ukernel_fn vbinary, OpType op_type) const;

  void Test(xnn_f32_vbinary_minmax_ukernel_fn vbinary_minmax, OpType op_type,
            xnn_init_f32_minmax_params_fn init_params) const;

  void Test(xnn_s32_vbinary_ukernel_fn vbinary, OpType op_type) const;

  void Test(xnn_f32_vbinary_relu_ukernel_fn vbinary_relu, OpType op_type) const;

  void Test(xnn_qu8_vadd_minmax_ukernel_fn vadd_minmax,
            xnn_init_qu8_add_minmax_params_fn init_params) const;

  void Test(xnn_qu8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qu8_mul_minmax_params_fn init_params) const;

  void Test(xnn_qs8_vadd_minmax_ukernel_fn vadd_minmax,
            xnn_init_qs8_add_minmax_params_fn init_params) const;

  void Test(xnn_qs8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qs8_mul_minmax_params_fn init_params) const;

 private:
  size_t batch_size_{1};
  bool inplace_a_{false};
  bool inplace_b_{false};
  bool broadcast_b_{false};
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
