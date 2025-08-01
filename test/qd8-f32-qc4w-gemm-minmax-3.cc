// clang-format off
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
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/pack-lh.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/ppmm.h"
#include "src/xnnpack/requantization.h"
#include "test/gemm-microkernel-tester.h"
#include "test/next_prime.h"

namespace {

struct ConstantOrFunction {
  ConstantOrFunction(size_t x) : fn([x]() { return x; }) {}  //NOLINT
  ConstantOrFunction(int x) : fn([x]() { return x; }) {}  //NOLINT
  template <typename Fn>
  ConstantOrFunction(Fn fn) : fn(std::move(fn)) {}  //NOLINT

  std::function<size_t()> fn;

  operator size_t() const { return fn(); }  //NOLINT
};

}  // namespace


namespace {

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
    bool is_igemm,
    bool unsigned_inputs,
    uint8_t planes,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    uint64_t arch_flags = 0) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  const GemmMicrokernelTester tester = GemmMicrokernelTester()
      .mr(mr).nr(nr).kr(kr).sr(sr).unsigned_inputs(unsigned_inputs).planes(planes);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, arch_flags));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .a_stride(xnnpack::NextPrime(k_block + 1))
            .b_zero_point(8)
        , test_func, arch_flags));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      tester.clone()
          .k(k_block)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      tester.clone()
          .n(nr).k(k_block)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      tester.clone()
          .m(mr).k(k_block)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(1, nr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(1, adj_k_block - 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block + 1))
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, adj_k_block - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_lt_" + akbs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(1, adj_k_block - 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs,
      tester.clone()
          .m(mr).n(nr)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_strided_a",
        tester.clone()
            .m(mr).n(nr)
            .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
            .b_zero_point(8)
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_gt_" + akbs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
      .loop_n(1, nr)
      .loop_m(1, mr));
  if (k_block > 1) {
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(adj_k_block + k_block, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_div_" + kbs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(adj_k_block + k_block, k_block * 5, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_gt_" + nrs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(nr + 1, nr * 2 - 1)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs,
      tester.clone()
          .m(mr)
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_strided_a",
        tester.clone()
            .m(mr)
            .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block));
  }
  gemm_tests.push_back(GemmTestParams(
      "n_div_" + nrs + "_subtile",
      tester.clone()
          .b_zero_point(8)
      , test_func, arch_flags)
      .loop_n(nr * 2, nr * 3, nr)
      .loop_k(1, k_block * 3, k_block + 1)
      .loop_m(1, mr));
  if (is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "small_kernel",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "small_kernel_subtile",
        tester.clone()
            .ks(3)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_small_kernel",
        tester.clone()
            .m(mr).ks(3)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "strided_cm_subtile",
      tester.clone()
          .mr(mr).nr(nr).kr(kr).sr(sr)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, arch_flags)
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
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1));
    gemm_tests.push_back(GemmTestParams(
        "zero",
        tester.clone()
            .m(mr).n(nr).ks(3)
            .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_zi(0, mr - 1));
  }
  gemm_tests.push_back(GemmTestParams(
      "min",
      tester.clone()
          .m(mr).n(nr).k(k_block).min(0.0f)
          .b_zero_point(8)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "max",
      tester.clone()
          .m(mr).n(nr).k(k_block).max(0.0f)
          .b_zero_point(8)
      , test_func, arch_flags));
  gemm_tests.push_back(GemmTestParams(
      "strided_cm",
      tester.clone()
          .m(mr).n(nr).k(k_block)
          .cm_stride(xnnpack::NextPrime(nr + 1))
          .b_zero_point(8)
      , test_func, arch_flags));

  return gemm_tests;
}

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  // NOLINTNEXTLINE(clang-diagnostic-unused-function)
  std::vector<GemmTestParams> CreateTests2(
      size_t k_block, size_t adj_k_block,
      ConstantOrFunction mr, ConstantOrFunction nr, size_t kr, size_t sr,
      bool is_igemm,
      bool unsigned_inputs,
      uint8_t planes,
      std::function<void(GemmMicrokernelTester& tester)> test_func,
      uint64_t arch_flags = 0) {
    std::string kbs = std::to_string(k_block);
    std::string kb2s = std::to_string(k_block * 2);
    std::string akbs = std::to_string(adj_k_block);
    nr = nr * xnn_init_hardware_config()->vlenb / sizeof(int32_t);
    std::string nrs = std::to_string(nr);

    const GemmMicrokernelTester tester = GemmMicrokernelTester()
        .mr(mr).nr(nr).kr(kr).sr(sr).unsigned_inputs(unsigned_inputs).planes(planes);

    std::vector<GemmTestParams> gemm_tests;
    gemm_tests.reserve(42);

    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs,
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .b_zero_point(8)
        , test_func, arch_flags));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_eq_" + kbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr).k(k_block)
              .a_stride(xnnpack::NextPrime(k_block + 1))
              .b_zero_point(8)
          , test_func, arch_flags));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile",
        tester.clone()
            .k(k_block)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(1, nr)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile_m",
        tester.clone()
            .n(nr).k(k_block)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_subtile_n",
        tester.clone()
            .m(mr).k(k_block)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(1, nr));
    if (k_block > 1) {
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs,
          tester.clone()
              .m(mr).n(nr)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, adj_k_block - 1));
      if (!is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_lt_" + akbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(adj_k_block + 1))
                .b_zero_point(8)
            , test_func, arch_flags)
            .loop_k(1, adj_k_block - 1));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_lt_" + akbs + "_subtile",
          tester.clone()
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, adj_k_block - 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs,
        tester.clone()
            .m(mr).n(nr)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "k_gt_" + akbs + "_strided_a",
          tester.clone()
              .m(mr).n(nr)
              .a_stride(xnnpack::NextPrime(adj_k_block * 2 + 1))
              .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "k_gt_" + akbs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_k(adj_k_block + 1, adj_k_block * 2 - 1, k_block)
        .loop_n(1, nr)
        .loop_m(1, mr));
    if (k_block > 1) {
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs,
          tester.clone()
              .m(mr).n(nr)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(adj_k_block + k_block, k_block * 5, k_block));
      if (is_igemm) {
        gemm_tests.push_back(GemmTestParams(
            "k_div_" + kbs + "_strided_a",
            tester.clone()
                .m(mr).n(nr)
                .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
                .b_zero_point(8)
            , test_func, arch_flags)
            .loop_k(adj_k_block + k_block, k_block * 3, k_block));
      }
      gemm_tests.push_back(GemmTestParams(
          "k_div_" + kbs + "_subtile",
          tester.clone()
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(adj_k_block + k_block, k_block * 5, k_block)
          .loop_n(1, nr)
          .loop_m(1, mr));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs,
        tester.clone()
            .m(mr)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_strided_a",
          tester.clone()
              .m(mr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_n(nr + 1, nr * 2 - 1, 4)
          .loop_k(1, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_gt_" + nrs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr + 1, nr * 2 - 1, 4)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_m(1, mr));
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs,
        tester.clone()
            .m(mr)
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1));
    if (!is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_strided_a",
          tester.clone()
              .m(mr)
              .a_stride(xnnpack::NextPrime(k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block));
    }
    gemm_tests.push_back(GemmTestParams(
        "n_div_" + nrs + "_subtile",
        tester.clone()
            .b_zero_point(8)
        , test_func, arch_flags)
        .loop_n(nr * 2, nr * 3, nr)
        .loop_k(1, k_block * 3, k_block + 1)
        .loop_m(1, mr));
    if (is_igemm) {
      gemm_tests.push_back(GemmTestParams(
          "small_kernel",
          tester.clone()
              .m(mr).n(nr).ks(3)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "small_kernel_subtile",
          tester.clone()
              .ks(3)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_n(1, nr)
          .loop_m(1, mr));
      gemm_tests.push_back(GemmTestParams(
          "n_gt_" + nrs + "_small_kernel",
          tester.clone()
              .m(mr).ks(3)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_n(nr + 1, nr * 2 - 1, 4)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "n_div_" + nrs + "_small_kernel",
          tester.clone()
              .m(mr).ks(3)
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_n(nr * 2, nr * 3, nr)
          .loop_k(1, k_block * 3, k_block + 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "strided_cm_subtile",
        tester.clone()
            .mr(mr).nr(nr).kr(kr).sr(sr)
            .cm_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
        , test_func, arch_flags)
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
          , test_func, arch_flags)
          .loop_k(1, k_block * 3, k_block + 1));
      gemm_tests.push_back(GemmTestParams(
          "zero",
          tester.clone()
              .m(mr).n(nr).ks(3)
              .a_offset(xnnpack::NextPrime(mr * k_block * 3 + 1))
              .b_zero_point(8)
          , test_func, arch_flags)
          .loop_k(1, k_block * 3, k_block + 1)
          .loop_zi(0, mr - 1));
    }
    gemm_tests.push_back(GemmTestParams(
        "min",
        tester.clone()
            .m(mr).n(nr).k(k_block).min(0.0f)
            .b_zero_point(8)
        , test_func, arch_flags));
    gemm_tests.push_back(GemmTestParams(
        "max",
        tester.clone()
            .m(mr).n(nr).k(k_block).max(0.0f)
            .b_zero_point(8)
        , test_func, arch_flags));
    gemm_tests.push_back(GemmTestParams(
        "strided_cm",
        tester.clone()
            .m(mr).n(nr).k(k_block)
            .cm_stride(xnnpack::NextPrime(nr + 1))
            .b_zero_point(8)
        , test_func, arch_flags));

    return gemm_tests;
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

}  // namespace


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X8C8__AVX256SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx256skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X8C8__AVX2_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx2_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C8__AVX2_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X8C8__AVX2_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X16C4__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X16C4__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/12, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c4__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C8__AVX512SKX_MADD, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512skx_madd,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X16C8__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X16C8__AVX512SKX_MADD_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x16c8__avx512skx_madd_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4uw_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_16X64C4__AVX512AMX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/16, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x64c4__avx512amx_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512amx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_16X32C4__AVX512AMX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512amx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_16X32C4__AVX512AMX_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/64,
          /*adj_k_block=*/64,
          /*mr=*/16, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_16x32c4__avx512amx_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512amx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512AMX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C4__ASM_AARCH64_NEONDOT_LD32_2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld32_2,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X8C4__ASM_AARCH64_NEONDOT_LD64_2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__asm_aarch64_neondot_ld64_2,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X16C4__ASM_AARCH64_NEONDOT_LD128_2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__asm_aarch64_neondot_ld128_2,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X16C4__ASM_AARCH64_NEONDOT_LD128_2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__asm_aarch64_neondot_ld128_2,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_aarch64);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X16C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X8C4__NEONDOT, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_dot)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon_i8mm)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X16__NEON_MLAL_LANE_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X16__NEON_MLAL_LANE, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_arm_neon)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C8__AVX512SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C8__AVX512SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C4__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X16C4__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/10, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x16c4__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X16C8__AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C8__AVX512VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/7, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C4__AVX512VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C4__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/9, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c4__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X16C4__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/14, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c4__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X16C8__AVX512VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x16c8__avx512vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx512vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X8C8__AVX256VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/10, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_12X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/12, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_12x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X8C8__AVX256VNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X8C8__AVX256VNNIGFNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX256VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X8C8__AVX256VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/9, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x8c8__avx256vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_14X8C8__AVX256VNNIGFNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/14, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_14x8c8__avx256vnnigfni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256vnnigfni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256VNNIGFNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X8C8__AVXVNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVXVNNI_PRFM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avxvnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X16C8__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_8X16C8__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_11X16C8__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/11, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_11x16c8__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X32C4__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/1, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_5X32C4__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/5, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c4__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_9X32C4__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/9, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_9x32c4__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_10X32C4__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/10, /*nr=*/32, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_10x32c4__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X64C4__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/4,
          /*adj_k_block=*/4,
          /*mr=*/2, /*nr=*/64, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_4X32C8__ASM_AMD64_AVX512VNNI, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/true,
          /*planes=*/1,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__asm_amd64_avx512vnni,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w_non_planar_avx512);
          },
          xnn_arch_x86_avx512vnni)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX512VNNI && XNN_ARCH_X86_64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx2)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx256skx,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_7X8C8__AVX256SKX, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx256skx,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_avx256skx)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X4C8__SSE41_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_x86_sse4_1)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_3X4C8__SSE2_LD128, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_2X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4v__rvv,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_6X4V__RVV, GemmTest,
      testing::ValuesIn(CreateTests2(
          /*k_block=*/2,
          /*adj_k_block=*/2,
          /*mr=*/6, /*nr=*/4, /*kr=*/1, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x4v__rvv,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          xnn_arch_riscv_vector)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      QD8_F32_QC4W_GEMM_MINMAX_1X4C8__WASMSIMD_DOT16X2_LD64, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          /*unsigned_inputs=*/false,
          /*planes=*/2,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
                        xnn_init_f32_qc4w_minmax_scalar_params,
                        xnn_pack_qs8_qc4w_gemm_goi_w);
          },
          0)),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QC4W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/2,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f32_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F32_QC4W_GEMM_MINMAX_2X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        /*unsigned_inputs=*/false,
        /*planes=*/2,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar,
                      xnn_init_f32_qc4w_minmax_scalar_params,
                      xnn_pack_qs8_qc4w_gemm_goi_w);
        },
        0)),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });

