// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <ostream>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/pack.h"
#include "xnnpack/requantization.h"
#include "next_prime.h"

class GemmMicrokernelTester {
 public:
  GemmMicrokernelTester clone() const {
    return *this;
  }

  GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  size_t mr() const {
    return this->mr_;
  }

  GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  size_t nr() const {
    return this->nr_;
  }


  GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  size_t kr() const {
    return this->kr_;
  }

  GemmMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  size_t sr() const {
    return this->sr_;
  }

  GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  size_t m() const {
    return this->m_;
  }

  GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  size_t n() const {
    return this->n_;
  }

  GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  size_t k() const {
    return this->k_;
  }

  GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  size_t ks() const {
    return this->ks_;
  }

  inline GemmMicrokernelTester& bl(size_t bl) {
    this->bl_ = bl;
    return *this;
  }

  inline size_t bl() const {
    return this->bl_;
  }

  size_t packed_k() const {
    return round_up_po2(k(), kr() * sr());
  }

  size_t packed_n() const {
    return round_up(n(), nr());
  }

  GemmMicrokernelTester& a_stride(size_t a_stride) {
    this->a_stride_ = a_stride;
    return *this;
  }

  size_t a_stride() const {
    return this->a_stride_ == 0 ? k() : this->a_stride_;
  }

  GemmMicrokernelTester& cm_stride(size_t cm_stride) {
    this->cm_stride_ = cm_stride;
    return *this;
  }

  size_t cm_stride() const {
    return this->cm_stride_ == 0 ? cn_stride() * ((n() - 1) / nr()) + (n() - 1) % nr() + 1 : this->cm_stride_;
  }

  GemmMicrokernelTester& cn_stride(size_t cn_stride) {
    this->cn_stride_ = cn_stride;
    return *this;
  }

  size_t cn_stride() const {
    return this->cn_stride_ == 0 ? nr() : this->cn_stride_;
  }

  GemmMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  GemmMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  GemmMicrokernelTester& a_offset(size_t a_offset) {
    this->a_offset_ = a_offset;
    return *this;
  }

  size_t a_offset() const {
    return this->a_offset_;
  }

  GemmMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  size_t zero_index() const {
    return this->zero_index_;
  }

  GemmMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  GemmMicrokernelTester& known_nc_mod_nr(bool known_nc_mod_nr) {
    this->known_nc_mod_nr_ = known_nc_mod_nr;
    return *this;
  }

  bool known_nc_mod_nr() const {
    return known_nc_mod_nr_;
  }

  GemmMicrokernelTester& relu(bool relu) {
    this->relu_ = relu;
    return *this;
  }

  bool relu() const {
    return relu_;
  }

  GemmMicrokernelTester& mr_packed(size_t mr_packed) {
    this->mr_packed_ = mr_packed;
    return *this;
  }

  size_t mr_packed() const {
    if (this->mr_packed_ == 0) {
      return this->mr_;
    }
    return this->mr_packed_;
  }

  size_t nc_mod_nr() const {
    return known_nc_mod_nr() ? n() % nr() : SIZE_MAX;
  }

  void Test(
    xnn_qd8_f16_qc8w_igemm_ukernel_fn igemm,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack) const;

  void Test(
    xnn_qd8_f32_qc8w_igemm_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack) const;

  void Test(
    xnn_qu8_gemm_minmax_ukernel_fn gemm,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_pack_qu8_gemm_fn pack,
    xnn_qu8_requantize_fn requantize) const;

  void Test(
    xnn_qu8_igemm_minmax_ukernel_fn igemm,
    xnn_init_qu8_conv_minmax_params_fn init_params,
    xnn_pack_qu8_igemm_fn pack,
    xnn_qu8_requantize_fn requantize);

  void Test(
    xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;

  void Test(
    xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;

  void Test(
    xnn_qs8_gemm_minmax_ukernel_fn gemm,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;

  void Test(
    xnn_qd8_f16_qc8w_gemm_ukernel_fn gemm,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack) const;

  void Test(
    xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack) const;

  void Test(
    xnn_qd8_f16_qc4w_gemm_ukernel_fn gemm,
    xnn_init_f16_qc4w_minmax_params_fn init_params,
    xnn_pack_qs8_qc4w_gemm_fn pack) const;

  void Test(
    xnn_qd8_f16_qb4w_gemm_ukernel_fn gemm,
    xnn_init_f16_qb4w_minmax_params_fn init_params,
    xnn_pack_qs8_qb4w_gemm_fn pack) const;

  void Test(
    xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
    xnn_init_f32_qc4w_minmax_params_fn init_params,
    xnn_pack_qs8_qc4w_gemm_fn pack) const;

  void Test(
    xnn_qd8_f32_qb4w_gemm_ukernel_fn gemm,
    xnn_init_f32_qb4w_minmax_params_fn init_params,
    xnn_pack_qs8_qb4w_gemm_fn pack) const;

  void Test(
    xnn_qs8_igemm_minmax_ukernel_fn igemm,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;

  void Test(
    xnn_bf16_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_bf16_minmax_params_fn init_params,
    xnn_pack_f16_gemm_fn pack) const;

  void Test(
    xnn_f16_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_gemm_fn pack) const;

  void Test(
    xnn_f16_igemm_minmax_ukernel_fn igemm_minmax,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_igemm_fn pack) const;

  void Test(
    xnn_f32_ppmm_minmax_ukernel_fn ppmm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack) const;

  void Test(
    xnn_f32_gemm_ukernel_fn gemm,
    xnn_pack_f32_gemm_fn pack) const;

  void Test(
    xnn_f32_gemm_relu_ukernel_fn gemm_relu,
    xnn_pack_f32_gemm_fn pack) const;

  void Test(
    xnn_f32_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack) const;

  void Test(
    xnn_f32_gemm_goi_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params) const;

  void Test(
    xnn_f32_qc4w_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_qc4w_minmax_params_fn init_params,
    xnn_pack_f32_qc4w_gemm_fn pack) const;

  void Test(
      xnn_f32_qc8w_gemm_ukernel_fn gemm,
      xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(
      xnn_f32_qc8w_gemm_relu_ukernel_fn gemm_relu,
      xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(
    xnn_f32_qc8w_gemm_minmax_ukernel_fn gemm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_qs8w_gemm_fn pack) const;

  void Test(
    xnn_f32_gemminc_minmax_ukernel_fn gemminc,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemminc_fn pack) const;

  void Test(
      xnn_f32_igemm_ukernel_fn igemm,
      xnn_pack_f32_igemm_fn pack) const;

  void Test(
      xnn_f32_igemm_relu_ukernel_fn igemm_relu,
      xnn_pack_f32_igemm_fn pack) const;

  void Test(
    xnn_f32_igemm_minmax_ukernel_fn igemm_minmax,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_igemm_fn pack) const;

  void Test(xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn gemm,
            xnn_init_f32_minmax_params_fn init_minmax_params,
            xnn_pack_weights_and_biases_fn pack,
            xnn_packed_stride_weights_and_biases_fn packed_stride);

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t bl_{SIZE_MAX};
  size_t a_stride_{0};
  size_t cm_stride_{0};
  size_t cn_stride_{0};
  uint8_t a_zero_point_{127};
  uint8_t b_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t a_offset_{0};
  size_t zero_index_{SIZE_MAX};
  size_t iterations_{15};
  bool known_nc_mod_nr_{true};
  bool relu_{false};
  size_t mr_packed_{0};
};

enum class LoopStepType {
  Linear,
  NextPrime
};

struct LoopParams {
  LoopParams() = default;
  explicit LoopParams(size_t from, size_t to, size_t step, LoopStepType step_type)
      : is_set(true), from(from), to(to), step(step), step_type(step_type) {}
  bool is_set = false;
  size_t from = 1;
  size_t to = 1;
  size_t step = 1;
  LoopStepType step_type = LoopStepType::Linear;

  size_t next(size_t n) const {
    switch (step_type) {
      case LoopStepType::Linear:
        return n + step;
      case LoopStepType::NextPrime:
        return xnnpack::NextPrime(n + step);
      default:
        std::cerr << "Unknown loop step type " << static_cast<int>(step_type) << std::endl;
        std::abort();
    }
  }
};

struct GemmTestParams {
  GemmTestParams(std::string test_name, GemmMicrokernelTester tester,
                 std::function<void(GemmMicrokernelTester& tester)> test_func,
                 std::function<void(void)> isa_check = nullptr)
      : test_name(test_name),
        tester(tester),
        test_func(test_func),
        isa_check(isa_check) {}

  // Setters for the loops over `k`, `m`, and `n`.
  GemmTestParams& loop_k(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::NextPrime) {
    loop_k_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_m(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::Linear) {
    loop_m_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_n(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::NextPrime) {
    loop_n_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_zi(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::Linear) {
    loop_zi_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_bzp(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::Linear) {
    loop_bzp_ = LoopParams(from, to, step, step_type);
    return *this;
  }
  GemmTestParams& loop_bl(size_t from, size_t to, size_t step = 1, LoopStepType step_type = LoopStepType::Linear) {
    loop_bl_ = LoopParams(from, to, step, step_type);
    return *this;
  }

  std::string test_name;
  GemmMicrokernelTester tester;
  std::function<void(GemmMicrokernelTester& tester)> test_func;
  std::function<void(void)> isa_check;
  LoopParams loop_k_;
  LoopParams loop_m_;
  LoopParams loop_n_;
  LoopParams loop_zi_;
  LoopParams loop_bzp_;
  LoopParams loop_bl_;
};

using GemmTest = testing::TestWithParam<GemmTestParams>;
