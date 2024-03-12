// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gemm-jit.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>
#include <xnnpack/ppmm.h>
#include <xnnpack/requantization.h>

#include "gemm-microkernel-tester.h"
#include <gtest/gtest.h>

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

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cn_stride(NextPrime(nr + 1))
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).n(nr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(adj_k_block + 1))
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
            .a_stride(NextPrime(adj_k_block == 1 ? 10 : adj_k_block * 2 + 1))
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 10, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(k_block * 10 + 1))
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 10, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 10, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "unknown_nc_mod_nr",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).known_nc_mod_nr(false)
      , test_func, isa_check)
      .loop_n(1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "relu",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).relu(true)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 5 + 1))
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 5 + 1))
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).ks(3).iterations(1)
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(NextPrime(nr + 1))
          .iterations(1)
      , test_func, isa_check)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "a_offset",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 5 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 5 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmin(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmax(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cm_stride(NextPrime(nr + 1))
      , test_func, isa_check));

  return gemm_tests;
}

std::vector<GemmTestParams> CreateTests2(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cn_stride(NextPrime(nr + 1))
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).n(nr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).k(k_block).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kb2s,
      GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block * 2)
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kb2s + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block * 2)
            .a_stride(NextPrime(k_block * 2 + 1))
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kb2s + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block * 2).iterations(1)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(adj_k_block + 1))
          , test_func, isa_check)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
        , test_func, isa_check)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
            .a_stride(NextPrime(adj_k_block == 1 ? 10 : adj_k_block * 2 + 1))
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_k(adj_k_block + 1, adj_k_block * (adj_k_block == 1 ? 10 : 2) - 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 10, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          GemmMicrokernelTester()
              .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr)
              .a_stride(NextPrime(k_block * 10 + 1))
          , test_func, isa_check)
          .loop_k(adj_k_block + k_block, k_block * 10, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
        , test_func, isa_check)
        .loop_k(adj_k_block + k_block, k_block * 10, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "unknown_nc_mod_nr",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).known_nc_mod_nr(false)
      , test_func, isa_check)
      .loop_n(1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "relu",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).relu(true)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 5 + 1))
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
          .cn_stride(NextPrime(nr + 1))
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr)
            .a_stride(NextPrime(k_block * 5 + 1))
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).iterations(1)
      , test_func, isa_check)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).ks(3).iterations(1)
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
        , test_func, isa_check)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).ks(3)
        , test_func, isa_check)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 5, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(NextPrime(nr + 1))
          .iterations(1)
      , test_func, isa_check)
      .loop_k(1, k_block * 5, k_block + 1)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "a_offset",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 5 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).ks(3)
            .a_offset(NextPrime(mr * k_block * 5 + 1))
        , test_func, isa_check)
        .loop_k(1, k_block * 5, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "qmin",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmin(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "qmax",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block).qmax(128)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cm_stride(NextPrime(nr + 1))
      , test_func, isa_check));

  return gemm_tests;
}

}  // namespace


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH32_NEON_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A7, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A7, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A7, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A7, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A7, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A55, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A55, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH32_NEON_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_LD64, hardswish) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_LD64, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH32_NEON_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_LD64, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 1; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 4; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_LD64, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .nr(8)
      .n(8)
      .k(2)
      .Test(
          xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_1X8__AARCH64_NEONFMA_LD64, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .nr(8)
        .n(8)
        .k(2)
        .Test(
            xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_LD128, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 4; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_LD128, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .m(4)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_LD128, hardswish_max_mr_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 4; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_4X8__AARCH64_NEONFMA_LD128, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .m(4)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A53_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/4,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A55, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/8,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 16; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(8)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_CORTEX_A75_PRFM, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(8)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_FMA;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_LD128, subtile_m_upto_mr) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_LD128, hardswish) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_LD128, hardswish_max_mr_lt_6) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }

  #if XNN_ENABLE_ASSEMBLY
    TEST(GENERATE_F32_GEMM_6X8__AARCH64_NEONFMA_LD128, matches_assembly) {
      TEST_REQUIRES_ARM_NEON_FMA;
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .m(6)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            &xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128);
    }
  #endif // XNN_ENABLE_ASSEMBLY
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMSIMD32_X86_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMSIMD32_X86_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_FMA_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/1,
          /*adj_k_block=*/1,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 2; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(1)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_LOADSPLAT_X8, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(1)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8__WASMRELAXEDSIMD32_X86_SPLAT_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_FMA_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_XINF, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X1, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X2, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT


#if XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
  INSTANTIATE_TEST_SUITE_P(
      GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4,
                        xnn_init_f32_minmax_scalar_params,
                        xnn_pack_f32_gemm_goi_w);
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, subtile_m_upto_mr) {
    for (uint32_t max_mr = 1; max_mr <= 6; max_mr++) {
      for (uint32_t m = 1; m <= max_mr; m++) {
        for (size_t k = 1; k <= 8; k += 1) {
          GemmMicrokernelTester()
            .mr(max_mr)
            .nr(8)
            .sr(4)
            .m(m)
            .n(8)
            .k(k)
            .iterations(1)
            .Test(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w);
        }
      }
    }
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, hardswish) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .sr(4)
      .m(6)
      .n(8)
      .k(4)
      .Test(
          xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
          fused_operators);
  }

  TEST(GENERATE_F32_GEMM_6X8S4__WASMRELAXEDSIMD32_X86_X4, hardswish_max_mr_lt_6) {
    const std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
    for (uint32_t max_mr = 1; max_mr < 6; max_mr++) {
      GemmMicrokernelTester()
        .mr(max_mr)
        .nr(8)
        .sr(4)
        .m(max_mr)
        .n(8)
        .k(4)
        .Test(
            xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4, xnn_init_f32_minmax_scalar_params, xnn_pack_f32_gemm_goi_w,
            fused_operators);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD && XNN_PLATFORM_JIT
