// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f32-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/packw.h"
#include "xnnpack/ppmm.h"
#include "xnnpack/requantization.h"
#include "gemm-microkernel-tester.h"
#include "next_prime.h"

namespace {

std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cn_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            .b_zero_point(8)
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block).iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        tester.clone()
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      tester.clone()
          .m(mr).n(nr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr)
            .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
            .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      tester.clone()
          .iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        tester.clone()
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      tester.clone()
          .m(mr)
          .cn_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      tester.clone()
          .iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_strided_cn",
      tester.clone()
          .m(mr)
          .cn_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      tester.clone()
          .iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        tester.clone()
            .ks(3).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      tester.clone()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .iterations(1)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "a_offset",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmin(128)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      tester.clone()
          .m(mr).n(nr).k(k_block).qmax(128)
          .b_zero_point(8)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, isa_check));

  return gemm_tests;
}

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  std::vector<GemmTestParams> CreateTests2(
      size_t k_block, size_t adj_k_block,
      size_t mr, size_t nr, size_t kr, size_t sr,
      bool is_igemm,
      std::function<void(GemmMicrokernelTester& tester)> test_func,
      std::function<void()> isa_check = nullptr) {
    std::string kbs = std::to_string(k_block);
    std::string kb2s = std::to_string(k_block * 2);
    std::string akbs = std::to_string(adj_k_block);
    nr = nr * xnn_init_hardware_config()->vlenb / sizeof(int32_t);
    std::string nrs = std::to_string(nr);

    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr);

    std::vector<GemmTestParams> gemm_tests;
    gemm_tests.reserve(42);

    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs,
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .b_zero_point(8)
        , test_func, isa_check));
    gemm_tests.push_back(GemmTestParams(
        "strided_cn",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .cn_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
      , test_func, isa_check));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .a_stride(xnnpack::NextPrime(k_block + 1))
              .b_zero_point(8)
          , test_func, isa_check));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile",
        tester.clone()
            .k(k_block).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile_m",
        tester.clone()
            .n(nr).k(k_block).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile_n",
        tester.clone()
            .m(mr).k(k_block).iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(1, nr));
    if (k_block > 1) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs,
          tester.clone()
              .m(mr).n(nr)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(adj_k_block + 1))
                .b_zero_point(8)
            , test_func, isa_check)
            .loop_k(1, adj_k_block - 1));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_subtile",
          tester.clone()
              .iterations(1)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
              .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_subtile",
        tester.clone()
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
    if (k_block > 1) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs,
          tester.clone()
              .m(mr).n(nr)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 5, k_block));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                .b_zero_point(8)
            , test_func, isa_check)
            .loop_k(adj_k_block + k_block, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_subtile",
          tester.clone()
              .iterations(1)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 5, k_block)
          .loop_n(1, nr)
          .loop_m(1, mr));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs,
        tester.clone()
            .m(mr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_cn",
        tester.clone()
            .m(mr)
            .cn_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_strided_a",
          tester.clone()
              .m(mr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_n(nr + 1, nr * 2 - 1, 4)
          .loop_k(1, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_subtile",
        tester.clone()
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs,
        tester.clone()
            .m(mr)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_cn",
        tester.clone()
            .m(mr)
            .cn_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_strided_a",
          tester.clone()
              .m(mr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_subtile",
        tester.clone()
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_m(1, mr));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "small_kernel",
          tester.clone()
              .m(mr).n(nr).ks(3)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "small_kernel_subtile",
          tester.clone()
              .ks(3).iterations(1)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_small_kernel",
          tester.clone()
              .m(mr).ks(3)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_n(nr + 1, nr * 2 - 1, 4)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_small_kernel",
          tester.clone()
              .m(mr).ks(3)
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "strided_cm_subtile",
        tester.clone()
            .mr(mr).nr(nr).kr(kr).sr(sr)
            .cm_stride(xnnpack::NextPrime(nr + 1))
            .iterations(1)
            .b_zero_point(8)
        , test_func, isa_check)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "a_offset",
          tester.clone()
              .m(mr).n(nr).ks(3)
              .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "zero",
          tester.clone()
              .m(mr).n(nr).ks(3)
              .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, isa_check)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_zi(0, mr - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "qmin",
        tester.clone()
            .m(mr).n(nr).k(k_block).qmin(128)
            .b_zero_point(8)
        , test_func, isa_check));
    gemm_tests.push_back(GemmTestParams(
        "qmax",
        tester.clone()
            .m(mr).n(nr).k(k_block).qmax(128)
            .b_zero_point(8)
        , test_func, isa_check));
    gemm_tests.push_back(GemmTestParams(
        "strided_cm",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .cm_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
        , test_func, isa_check));

    return gemm_tests;
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

}  // namespace


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX256SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256skx_madd,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X8C8__AVX256SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256skx_madd,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVX256SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X8C8__AVX256SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C8__AVX2_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX2_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X8C8__AVX2_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX2_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_28X16C4__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/28, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_28x16c4__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C4__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C4__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C4__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C8__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C8__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C8__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_madd_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X64C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/1, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x64c4__avx512amx,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512AMX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X64C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/7, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x64c4__avx512amx,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512AMX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_16X16C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512AMX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_16X16C4__AVX512AMX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/16, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x16c4__avx512amx_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512AMX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_28X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/28, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_28x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c8__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C4__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_avx512vnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX512VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX256VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx256vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X8C8__AVX256VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256VNNIGFNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm,
                        xnn_init_f32_qc4w_minmax_avxvnni_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVXVNNI;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2,
                        xnn_init_f32_qc4w_minmax_avx_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx2,
                        xnn_init_f32_qc4w_minmax_avx_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx,
                        xnn_init_f32_qc4w_minmax_avx_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx256skx,
                        xnn_init_f32_qc4w_minmax_avx_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx,
                        xnn_init_f32_qc4w_minmax_avx_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX256SKX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X4C8__AVX_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X4C8__AVX_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X4C8__SSE41_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE41;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X4C8__SSE2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64,
                        xnn_init_f32_qc4w_minmax_sse_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_SSE2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4v__rvv,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_RISCV_VECTOR;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/8, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x4v__rvv,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_RISCV_VECTOR;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_f32_qc4w_minmax_wasmsimd_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QC4W_GEMM_MINMAX_1X1__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/1, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar,
                      xnn_init_f32_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QC4W_GEMM_MINMAX_2X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar,
                      xnn_init_f32_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X4__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8__WASM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
