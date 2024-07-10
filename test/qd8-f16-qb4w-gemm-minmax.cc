// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qb4w-gemm-minmax.yaml
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
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cn_stride(NextPrime(nr + 1))
          .b_zero_point(8)
          .bl(kr * sr * 2)
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
            .b_zero_point(8)
            .bl(kr * sr * 2)
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).n(nr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_n(1, nr));
  gemm_tests.push_back(GemmTestParams(
      "bl",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block * 12)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
      .loop_bl(2 * kr * sr, k_block * 12, 2 * kr * sr));

  return gemm_tests;
}

}  // namespace


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_5X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_5X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/5, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_6X8C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/8, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_6X16C4__NEONDOTFP16ARITH, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/8,
          /*adj_k_block=*/8,
          /*mr=*/6, /*nr=*/16, /*kr=*/4, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_DOT_FP16_ARITH;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_5X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_5X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_5X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/5, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_5x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_6X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_6X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_6X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/6, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_6x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_7X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_7X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_7X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/7, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_7x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_8X8C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x8c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_8X16C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x16c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });

  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_8X32C8__NEONI8MM, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/8, /*nr=*/32, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_8x32c8__neoni8mm,
                        xnn_init_f16_qb4w_minmax_scalar_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_ARM_NEON_I8MM;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ENABLE_ARM_I8MM && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
