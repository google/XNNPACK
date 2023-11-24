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
#include <vector>

#include <xnnpack/microfnptr.h>
#include <xnnpack/pack.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/requantization.h>


class GemmMicrokernelTester {
 public:
  inline GemmMicrokernelTester& mr(size_t mr) {
    this->mr_ = mr;
    return *this;
  }

  inline size_t mr() const {
    return this->mr_;
  }

  inline GemmMicrokernelTester& nr(size_t nr) {
    this->nr_ = nr;
    return *this;
  }

  inline size_t nr() const {
    return this->nr_;
  }


  inline GemmMicrokernelTester& kr(size_t kr) {
    this->kr_ = kr;
    return *this;
  }

  inline size_t kr() const {
    return this->kr_;
  }

  inline GemmMicrokernelTester& sr(size_t sr) {
    this->sr_ = sr;
    return *this;
  }

  inline size_t sr() const {
    return this->sr_;
  }

  inline GemmMicrokernelTester& m(size_t m) {
    this->m_ = m;
    return *this;
  }

  inline size_t m() const {
    return this->m_;
  }

  inline GemmMicrokernelTester& n(size_t n) {
    this->n_ = n;
    return *this;
  }

  inline size_t n() const {
    return this->n_;
  }

  inline GemmMicrokernelTester& k(size_t k) {
    this->k_ = k;
    return *this;
  }

  inline size_t k() const {
    return this->k_;
  }

  inline GemmMicrokernelTester& ks(size_t ks) {
    this->ks_ = ks;
    return *this;
  }

  inline size_t ks() const {
    return this->ks_;
  }

  inline size_t packed_k() const {
    return round_up_po2(k(), kr() * sr());
  }

  inline size_t packed_n() const {
    return round_up(n(), nr());
  }

  inline GemmMicrokernelTester& a_stride(size_t a_stride) {
    this->a_stride_ = a_stride;
    return *this;
  }

  inline size_t a_stride() const {
    return this->a_stride_ == 0 ? k() : this->a_stride_;
  }

  inline GemmMicrokernelTester& cm_stride(size_t cm_stride) {
    this->cm_stride_ = cm_stride;
    return *this;
  }

  inline size_t cm_stride() const {
    return this->cm_stride_ == 0 ? cn_stride() * ((n() - 1) / nr()) + (n() - 1) % nr() + 1 : this->cm_stride_;
  }

  inline GemmMicrokernelTester& cn_stride(size_t cn_stride) {
    this->cn_stride_ = cn_stride;
    return *this;
  }

  inline size_t cn_stride() const {
    return this->cn_stride_ == 0 ? nr() : this->cn_stride_;
  }

  inline GemmMicrokernelTester& a_zero_point(uint8_t a_zero_point) {
    this->a_zero_point_ = a_zero_point;
    return *this;
  }

  inline uint8_t a_zero_point() const {
    return this->a_zero_point_;
  }

  inline GemmMicrokernelTester& b_zero_point(uint8_t b_zero_point) {
    this->b_zero_point_ = b_zero_point;
    return *this;
  }

  inline uint8_t b_zero_point() const {
    return this->b_zero_point_;
  }

  inline GemmMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GemmMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GemmMicrokernelTester& a_offset(size_t a_offset) {
    this->a_offset_ = a_offset;
    return *this;
  }

  inline size_t a_offset() const {
    return this->a_offset_;
  }

  inline GemmMicrokernelTester& zero_index(size_t zero_index) {
    this->zero_index_ = zero_index;
    return *this;
  }

  inline size_t zero_index() const {
    return this->zero_index_;
  }

  inline GemmMicrokernelTester& extended_weights(bool extended_weights) {
    this->extended_weights_ = extended_weights;
    return *this;
  }

  inline bool extended_weights() const {
    return this->extended_weights_;
  }

  inline GemmMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
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
    xnn_qd8_f32_qc4w_gemm_ukernel_fn gemm,
    xnn_init_f32_qc4w_minmax_params_fn init_params,
    xnn_pack_qs8_qc4w_gemm_fn pack) const;

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

#if XNN_PLATFORM_JIT
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_gemm_fn pack) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_igemm_fn pack) const;
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_igemm_fn pack) const;
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_pack_qs8_gemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_qs8_conv_minmax_params_fn init_params,
    xnn_pack_qs8_igemm_fn pack,
    xnn_qs8_requantize_fn requantize) const;
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack,
    const std::vector<xnn_post_operation>& fused_operators) const;
  void Test(
    xnn_jit_igemm_code_generator_fn gemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_igemm_fn pack,
    const std::vector<xnn_post_operation>& fused_operators) const;

  // Test that JIT generated code matches assembly.
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_gemm_fn pack,
    xnn_f16_gemm_minmax_ukernel_fn gemm_minmax) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_f16_minmax_params_fn init_params,
    xnn_pack_f16_igemm_fn pack,
    xnn_f16_igemm_minmax_ukernel_fn igemm_minmax) const;
  void Test(
    xnn_jit_gemm_code_generator_fn gemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_gemm_fn pack,
    xnn_f32_gemm_minmax_ukernel_fn gemm_minmax) const;
  void Test(
    xnn_jit_igemm_code_generator_fn igemm_generator,
    xnn_init_f32_minmax_params_fn init_params,
    xnn_pack_f32_igemm_fn pack,
    xnn_f32_igemm_minmax_ukernel_fn igemm_minmax) const;
#endif  // XNN_PLATFORM_JIT

 private:
  size_t mr_{1};
  size_t nr_{1};
  size_t kr_{1};
  size_t sr_{1};
  size_t m_{1};
  size_t n_{1};
  size_t k_{1};
  size_t ks_{1};
  size_t a_stride_{0};
  size_t cm_stride_{0};
  size_t cn_stride_{0};
  uint8_t a_zero_point_{127};
  uint8_t b_zero_point_{127};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t a_offset_{0};
  size_t zero_index_{SIZE_MAX};
  bool extended_weights_{false};
  size_t iterations_{15};
  bool known_nc_mod_nr_{true};
  bool relu_{false};
};
