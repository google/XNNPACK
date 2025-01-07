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

#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"

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
    Prelu,
    RPrelu,
  };

  float compute_float(OpType op, float a, float b) const {
    switch (op) {
      case OpType::Add:
        return a + b;
      case OpType::CopySign:
        return std::copysign(a, b);
      case OpType::RCopySign:
        return std::copysign(b, a);
      case OpType::Div:
        return a / b;
      case OpType::RDiv:
        return b / a;
      case OpType::Max:
        return std::max(a, b);
      case OpType::Min:
        return std::min(a, b);
      case OpType::Mul:
        return a * b;
      case OpType::Prelu:
        return a < 0 ? a * b : a;
      case OpType::RPrelu:
        return b < 0 ? b * a : b;
      case OpType::SqrDiff:
        return (a - b) * (a - b);
      case OpType::Sub:
        return a - b;
      case OpType::RSub:
        return b - a;
    }
    XNN_UNREACHABLE;
    return 0.0;
  }

  template <typename T>
  void reference_op_impl(const T* a, const T* b, T* result, size_t n,
                         OpType op_type) const {
    size_t stride_b = broadcast_b() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
      float f_result = compute_float(op_type, a[i], b[i * stride_b]);
      // TODO: We shouldn't need to do this check for overflow.
      if (f_result < xnnpack::NumericLimits<T>::min()) {
        result[i] = -xnnpack::NumericLimits<T>::infinity();
      } else if (f_result > xnnpack::NumericLimits<T>::max()) {
        result[i] = xnnpack::NumericLimits<T>::infinity();
      } else {
        result[i] = compute_float(op_type, a[i], b[i * stride_b]);
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

  void Test(xnn_f16_vbinary_ukernel_fn vbinary, OpType op_type,
            xnn_init_f16_default_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vbinary_ukernel_fn vbinary, OpType op_type,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_qu8_vadd_minmax_ukernel_fn vadd_minmax,
            xnn_init_qu8_add_minmax_params_fn init_params) const;

  void Test(xnn_qu8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qu8_mul_minmax_params_fn init_params) const;

  void Test(xnn_qs8_vadd_minmax_ukernel_fn vadd_minmax,
            xnn_init_qs8_add_minmax_params_fn init_params) const;

  void Test(xnn_qs8_vmul_minmax_ukernel_fn vmul_minmax,
            xnn_init_qs8_mul_minmax_params_fn init_params) const;
  
  void Test(xnn_qs8_vprelu_ukernel_fn vprelu, OpType op_type,
           xnn_init_qs8_vprelu_params_fn init_params) const;

  void Test(xnn_qu8_vprelu_ukernel_fn vprelu, OpType op_type,
            xnn_init_qu8_vprelu_params_fn init_params) const;

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

#define XNN_TEST_BINARY_BATCH_EQ(ukernel, arch_flags, batch_tile, is_binaryc, \
                                 datatype, ...)                               \
  TEST(ukernel, batch_eq) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                     \
    const size_t batch_scale = get_batch_scale<datatype>();                   \
    VBinaryMicrokernelTester()                                                \
        .batch_size(batch_tile * batch_scale)                                 \
        .broadcast_b(is_binaryc)                                              \
        .Test(__VA_ARGS__);                                                   \
  }

#define XNN_TEST_BINARY_BATCH_DIV(ukernel, arch_flags, batch_tile, is_binaryc, \
                                  datatype, ...)                               \
  TEST(ukernel, batch_div) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    const size_t batch_scale = get_batch_scale<datatype>();                    \
    if (batch_tile == 1 && batch_scale == 1) return;                           \
    for (size_t batch_size = batch_tile * batch_scale * 2;                     \
         batch_size < batch_tile * batch_scale * 10;                           \
         batch_size += batch_tile * batch_scale) {                             \
      VBinaryMicrokernelTester()                                               \
          .batch_size(batch_size)                                              \
          .broadcast_b(is_binaryc)                                             \
          .Test(__VA_ARGS__);                                                  \
    }                                                                          \
  }
#define XNN_TEST_BINARY_BATCH_LT(ukernel, arch_flags, batch_tile, is_binaryc, \
                                 datatype, ...)                               \
  TEST(ukernel, batch_lt) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                     \
    const size_t batch_scale = get_batch_scale<datatype>();                   \
    if (batch_tile == 1 && batch_scale == 1) return;                          \
    for (size_t batch_size = batch_scale;                                     \
         batch_size < batch_tile * batch_scale; batch_size++) {               \
      VBinaryMicrokernelTester()                                              \
          .batch_size(batch_size)                                             \
          .broadcast_b(is_binaryc)                                            \
          .Test(__VA_ARGS__);                                                 \
    }                                                                         \
  }

#define XNN_TEST_BINARY_BATCH_GT(ukernel, arch_flags, batch_tile, is_binaryc, \
                                 datatype, ...)                               \
  TEST(ukernel, batch_gt) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                     \
    const size_t batch_scale = get_batch_scale<datatype>();                   \
    const size_t batch_end = batch_tile == 1 ? 10 : batch_tile * 2;           \
    const size_t batch_step = batch_scale == 1 ? 1 : batch_tile * 2;          \
    for (size_t batch_size = batch_tile + 1; batch_size < batch_end;          \
         batch_size += batch_step) {                                          \
      VBinaryMicrokernelTester()                                              \
          .batch_size(batch_size)                                             \
          .broadcast_b(is_binaryc)                                            \
          .Test(__VA_ARGS__);                                                 \
    }                                                                         \
  }

#define XNN_TEST_BINARY_INPLACE_A(ukernel, arch_flags, batch_tile, is_binaryc, \
                                  datatype, ...)                               \
  TEST(ukernel, inplace_a) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    const size_t batch_scale = get_batch_scale<datatype>();                    \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5;    \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {            \
      VBinaryMicrokernelTester()                                               \
          .batch_size(batch_size)                                              \
          .inplace_a(true)                                                     \
          .broadcast_b(is_binaryc)                                             \
          .Test(__VA_ARGS__);                                                  \
    }                                                                          \
  }

#define XNN_TEST_BINARY_INPLACE_B(ukernel, arch_flags, batch_tile, is_binaryc, \
                                  datatype, ...)                               \
  TEST(ukernel, inplace_b) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    const size_t batch_scale = get_batch_scale<datatype>();                    \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5;    \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {            \
      VBinaryMicrokernelTester()                                               \
          .batch_size(batch_size)                                              \
          .inplace_b(true)                                                     \
          .broadcast_b(is_binaryc)                                             \
          .Test(__VA_ARGS__);                                                  \
    }                                                                          \
  }

#define XNN_TEST_BINARY_INPLACE_A_AND_B(ukernel, arch_flags, batch_tile,    \
                                        is_binaryc, datatype, ...)          \
  TEST(ukernel, inplace_a_and_b) {                                          \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      VBinaryMicrokernelTester()                                            \
          .batch_size(batch_size)                                           \
          .inplace_a(true)                                                  \
          .inplace_b(true)                                                  \
          .broadcast_b(is_binaryc)                                          \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }

#define XNN_TEST_BINARY_A_ZERO_POINT(ukernel, arch_flags, batch_tile,       \
                                     is_binaryc, datatype, ...)             \
  TEST(ukernel, a_zero_point) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      for (int32_t a_zero_point = -128; a_zero_point <= 127;                \
           a_zero_point += 51) {                                            \
        VBinaryMicrokernelTester()                                          \
            .batch_size(batch_size)                                         \
            .a_zero_point(a_zero_point)                                     \
            .broadcast_b(is_binaryc)                                        \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }

#define XNN_TEST_BINARY_B_ZERO_POINT(ukernel, arch_flags, batch_tile,       \
                                     is_binaryc, datatype, ...)             \
  TEST(ukernel, b_zero_point) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      for (int32_t b_zero_point = -128; b_zero_point <= 127;                \
           b_zero_point += 51) {                                            \
        VBinaryMicrokernelTester()                                          \
            .batch_size(batch_size)                                         \
            .b_zero_point(b_zero_point)                                     \
            .broadcast_b(is_binaryc)                                        \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }

#define XNN_TEST_BINARY_Y_ZERO_POINT(ukernel, arch_flags, batch_tile,       \
                                     is_binaryc, datatype, ...)             \
  TEST(ukernel, y_zero_point) {                                             \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      for (int32_t y_zero_point = -128; y_zero_point <= 127;                \
           y_zero_point += 51) {                                            \
        VBinaryMicrokernelTester()                                          \
            .batch_size(batch_size)                                         \
            .y_zero_point(y_zero_point)                                     \
            .broadcast_b(is_binaryc)                                        \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }

#define XNN_TEST_BINARY_A_SCALE(ukernel, arch_flags, batch_tile, is_binaryc, \
                                datatype, ...)                               \
  TEST(ukernel, a_scale) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                    \
    const size_t batch_scale = get_batch_scale<datatype>();                  \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5;  \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {          \
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {       \
        VBinaryMicrokernelTester()                                           \
            .batch_size(batch_size)                                          \
            .a_scale(a_scale)                                                \
            .broadcast_b(is_binaryc)                                         \
            .Test(__VA_ARGS__);                                              \
      }                                                                      \
    }                                                                        \
  }

#define XNN_TEST_BINARY_B_SCALE(ukernel, arch_flags, batch_tile, is_binaryc, \
                                datatype, ...)                               \
  TEST(ukernel, b_scale) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                    \
    const size_t batch_scale = get_batch_scale<datatype>();                  \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5;  \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {          \
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {       \
        VBinaryMicrokernelTester()                                           \
            .batch_size(batch_size)                                          \
            .b_scale(b_scale)                                                \
            .broadcast_b(is_binaryc)                                         \
            .Test(__VA_ARGS__);                                              \
      }                                                                      \
    }                                                                        \
  }

#define XNN_TEST_BINARY_Y_SCALE(ukernel, arch_flags, batch_tile, is_binaryc, \
                                datatype, ...)                               \
  TEST(ukernel, y_scale) {                                                   \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                    \
    const size_t batch_scale = get_batch_scale<datatype>();                  \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5;  \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {          \
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {       \
        VBinaryMicrokernelTester()                                           \
            .batch_size(batch_size)                                          \
            .y_scale(y_scale)                                                \
            .broadcast_b(is_binaryc)                                         \
            .Test(__VA_ARGS__);                                              \
      }                                                                      \
    }                                                                        \
  }

#define XNN_TEST_BINARY_QMIN(ukernel, arch_flags, batch_tile, is_binaryc,   \
                             datatype, ...)                                 \
  TEST(ukernel, qmin) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      VBinaryMicrokernelTester()                                            \
          .batch_size(batch_size)                                           \
          .qmin(128)                                                        \
          .broadcast_b(is_binaryc)                                          \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }

#define XNN_TEST_BINARY_QMAX(ukernel, arch_flags, batch_tile, is_binaryc,   \
                             datatype, ...)                                 \
  TEST(ukernel, qmax) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    for (size_t batch_size = 1; batch_size <= batch_tile * batch_scale * 5; \
         batch_size += std::max(1, batch_tile - 1) * batch_scale) {         \
      VBinaryMicrokernelTester()                                            \
          .batch_size(batch_size)                                           \
          .qmax(128)                                                        \
          .broadcast_b(is_binaryc)                                          \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }
