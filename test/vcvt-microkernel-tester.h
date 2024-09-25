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

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/isa-checks.h"
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

  void Test(xnn_f16_f32_vcvt_ukernel_fn vcvt, const void* = nullptr) const;

  void Test(xnn_f32_f16_vcvt_ukernel_fn vcvt, const void* = nullptr) const;

  void Test(xnn_f16_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_f16_qs8_cvt_params_fn init_params);

  void Test(xnn_f32_qs8_vcvt_ukernel_fn vcvt,
            xnn_init_f32_qs8_cvt_params_fn init_params) const;

  void Test(xnn_f32_qu8_vcvt_ukernel_fn vcvt,
            xnn_init_f32_qu8_cvt_params_fn init_params) const;

  void Test(xnn_s32_f32_vcvt_ukernel_fn vcvt,
            xnn_init_s32_f32_cvt_params_fn init_params) const;

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

  void Test(xnn_u32_f32_vcvt_ukernel_fn vcvt,
            xnn_init_u32_f32_cvt_params_fn init_params) const;

 private:
  float scale_ = 1.75f;
  int16_t input_zero_point_ = 0;
  int16_t output_zero_point_ = 5;
  int16_t qmin_ = std::numeric_limits<int16_t>::min();
  int16_t qmax_ = std::numeric_limits<int16_t>::max();
  size_t batch_size_ = 1;
  size_t iterations_ = 15;
};

template <typename T>
VCvtMicrokernelTester make_vcvt_tester() {
  if (std::is_integral<T>::value) {
    return VCvtMicrokernelTester()
        .qmin(std::numeric_limits<T>::min())
        .qmax(std::numeric_limits<T>::max())
        .output_zero_point(std::numeric_limits<T>::min() / 2 +
                           std::numeric_limits<T>::max() / 2 + 1);
  } else {
    return VCvtMicrokernelTester();
  }
}

#define XNN_TEST_CVT_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype_in, \
                              datatype_out, ...)                            \
  TEST(ukernel, batch_eq) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype_in>();              \
    make_vcvt_tester<datatype_out>()                                        \
        .batch_size(batch_tile* batch_scale)                                \
        .Test(__VA_ARGS__);                                                 \
  }

#define XNN_TEST_CVT_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype_in, \
                               datatype_out, ...)                            \
  TEST(ukernel, batch_div) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                    \
    const size_t batch_scale = get_batch_scale<datatype_in>();               \
    if (batch_tile == 1 && batch_scale == 1) return;                         \
    for (size_t batch_size = batch_tile * batch_scale * 2;                   \
         batch_size < batch_tile * batch_scale * 10;                         \
         batch_size += batch_tile * batch_scale) {                           \
      make_vcvt_tester<datatype_out>()                                       \
          .batch_size(batch_size)                                            \
          .Test(__VA_ARGS__);                                                \
    }                                                                        \
  }

#define XNN_TEST_CVT_BATCH_LT(ukernel, arch_flags, batch_tile, datatype_in, \
                              datatype_out, ...)                            \
  TEST(ukernel, batch_lt) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype_in>();              \
    if (batch_tile == 1 && batch_scale == 1) return;                        \
    for (size_t batch_size = batch_scale;                                   \
         batch_size < batch_tile * batch_scale; batch_size++) {             \
      make_vcvt_tester<datatype_out>()                                      \
          .batch_size(batch_size)                                           \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }

#define XNN_TEST_CVT_BATCH_GT(ukernel, arch_flags, batch_tile, datatype_in, \
                              datatype_out, ...)                            \
  TEST(ukernel, batch_gt) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype_in>();              \
    const size_t batch_end = batch_tile == 1 ? 10 : batch_tile * 2;         \
    const size_t batch_step = batch_scale == 1 ? 1 : batch_tile * 2;        \
    for (size_t batch_size = batch_tile + 1; batch_size < batch_end;        \
         batch_size += batch_step) {                                        \
      make_vcvt_tester<datatype_out>()                                      \
          .batch_size(batch_size)                                           \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }

#define XNN_TEST_CVT_SCALE(ukernel, arch_flags, batch_tile, datatype_in, \
                           datatype_out, ...)                            \
  TEST(ukernel, scale) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                \
    const size_t batch_scale = get_batch_scale<datatype_in>();           \
    const size_t batch_end = batch_tile * batch_scale * 5;               \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;    \
    for (size_t batch_size = 1; batch_size <= batch_end;                 \
         batch_size += batch_step) {                                     \
      make_vcvt_tester<datatype_out>()                                   \
          .batch_size(batch_size)                                        \
          .scale(50)                                                     \
          .Test(__VA_ARGS__);                                            \
    }                                                                    \
  }

#define XNN_TEST_CVT_INPUT_ZERO_POINT(ukernel, arch_flags, batch_tile, \
                                      datatype_in, datatype_out, ...)  \
  TEST(ukernel, input_zero_point) {                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                              \
    const size_t batch_scale = get_batch_scale<datatype_in>();         \
    const size_t batch_end = batch_tile * batch_scale * 5;             \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;  \
    for (int16_t input_zero_point = 0; input_zero_point < 5;           \
         input_zero_point += 2) {                                      \
      for (size_t batch_size = 1; batch_size <= batch_end;             \
           batch_size += batch_step) {                                 \
        make_vcvt_tester<datatype_out>()                               \
            .batch_size(batch_size)                                    \
            .input_zero_point(input_zero_point)                        \
            .Test(__VA_ARGS__);                                        \
      }                                                                \
    }                                                                  \
  }

#define XNN_TEST_CVT_OUTPUT_ZERO_POINT(ukernel, arch_flags, batch_tile, \
                                       datatype_in, datatype_out, ...)  \
  TEST(ukernel, output_zero_point) {                                    \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                               \
    const size_t batch_scale = get_batch_scale<datatype_in>();          \
    const size_t batch_end = batch_tile * batch_scale * 5;              \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;   \
    for (int16_t output_zero_point = 0; output_zero_point < 5;          \
         output_zero_point += 2) {                                      \
      for (size_t batch_size = 1; batch_size <= batch_end;              \
           batch_size += batch_step) {                                  \
        make_vcvt_tester<datatype_out>()                                \
            .batch_size(batch_size)                                     \
            .output_zero_point(output_zero_point)                       \
            .Test(__VA_ARGS__);                                         \
      }                                                                 \
    }                                                                   \
  }

#define XNN_TEST_CVT_SATURATION(ukernel, arch_flags, batch_tile, datatype_in, \
                                datatype_out, ...)                            \
  TEST(ukernel, saturation) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                     \
    const size_t batch_scale = get_batch_scale<datatype_in>();                \
    const size_t batch_end = batch_tile * batch_scale * 5;                    \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;         \
    for (size_t batch_size = 1; batch_size <= batch_end;                      \
         batch_size += batch_step) {                                          \
      make_vcvt_tester<datatype_out>()                                        \
          .batch_size(batch_size)                                             \
          .scale(500)                                                         \
          .Test(__VA_ARGS__);                                                 \
    }                                                                         \
  }

#define XNN_TEST_CVT_OVERFLOW(ukernel, arch_flags, batch_tile, datatype_in, \
                              datatype_out, ...)                            \
  TEST(ukernel, overflow) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype_in>();              \
    const size_t batch_end = batch_tile * batch_scale * 5;                  \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;       \
    for (size_t batch_size = 1; batch_size <= batch_end;                    \
         batch_size += batch_step) {                                        \
      make_vcvt_tester<datatype_out>()                                      \
          .batch_size(batch_size)                                           \
          .scale(4294967296.0f)                                             \
          .Test(__VA_ARGS__);                                               \
    }                                                                       \
  }

#define XNN_TEST_CVT_QMIN(ukernel, arch_flags, batch_tile, datatype_in, \
                          datatype_out, ...)                            \
  TEST(ukernel, qmin) {                                                 \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                               \
    const size_t batch_scale = get_batch_scale<datatype_in>();          \
    const size_t batch_end = batch_tile * batch_scale * 5;              \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;   \
    for (int32_t qmin = std::numeric_limits<datatype_out>::min();       \
         qmin < std::numeric_limits<datatype_out>::max(); qmin += 51) { \
      for (size_t batch_size = 1; batch_size <= batch_end;              \
           batch_size += batch_step) {                                  \
        make_vcvt_tester<datatype_out>()                                \
            .batch_size(batch_size)                                     \
            .scale(500)                                                 \
            .qmin(qmin)                                                 \
            .Test(__VA_ARGS__);                                         \
      }                                                                 \
    }                                                                   \
  }

#define XNN_TEST_CVT_QMAX(ukernel, arch_flags, batch_tile, datatype_in,  \
                          datatype_out, ...)                             \
  TEST(ukernel, qmax) {                                                  \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                \
    const size_t batch_scale = get_batch_scale<datatype_in>();           \
    const size_t batch_end = batch_tile * batch_scale * 5;               \
    const size_t batch_step = std::max<size_t>(2, batch_end / 8) - 1;    \
    for (int32_t qmax = std::numeric_limits<datatype_out>::min() + 1;    \
         qmax <= std::numeric_limits<datatype_out>::max(); qmax += 51) { \
      for (size_t batch_size = 1; batch_size <= batch_end;               \
           batch_size += batch_step) {                                   \
        make_vcvt_tester<datatype_out>()                                 \
            .batch_size(batch_size)                                      \
            .scale(500)                                                  \
            .qmax(qmax)                                                  \
            .Test(__VA_ARGS__);                                          \
      }                                                                  \
    }                                                                    \
  }
