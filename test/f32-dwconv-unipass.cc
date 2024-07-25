// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-unipass.yaml
//   Generator: tools/generate-dwconv-unipass-test.py


#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/requantization.h"
#include "dwconv-microkernel-tester.h"
#include "next_prime.h"

namespace {

std::vector<DWConvTestParams> CreateTests1(
    size_t c_block, size_t adj_c_block, size_t cr, size_t kr,
    std::function<void(DWConvMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  const std::string cbs = std::to_string(c_block);
  const std::string acbs = std::to_string(adj_c_block);

  std::vector<DWConvTestParams> tests;
  tests.reserve(18);

  tests.push_back(DWConvTestParams(
      "c_eq_" + cbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .channels(c_block)
      , test_func, isa_check));


  if (c_block > 1) {
    tests.push_back(DWConvTestParams(
        "c_div_" + cbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func, isa_check)
        .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));


    tests.push_back(DWConvTestParams(
        "c_lt_" + acbs,
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
        , test_func, isa_check)
      .loop_channels(1, adj_c_block - 1));
  }

  tests.push_back(DWConvTestParams(
      "c_gt_" + acbs,
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
      , test_func, isa_check)
      .loop_channels(adj_c_block + 1, (c_block == 1 ? 10 : adj_c_block + c_block) - 1));


  tests.push_back(DWConvTestParams(
      "multipixel",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(3)
      , test_func, isa_check)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));

  tests.push_back(DWConvTestParams(
      "multipixel_with_step",
        DWConvMicrokernelTester()
            .channel_tile(cr)
            .kernel_tile(kr)
            .width(3)
        , test_func, isa_check)
        .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1))
        .loop_step(2, kr));

  tests.push_back(DWConvTestParams(
      "multipixel_with_output_stride",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .width(5)
          .output_stride(xnnpack::NextPrime(cr * 5 + 1))
      , test_func, isa_check)
      .loop_channels(1, c_block * 5, std::max(size_t(1), c_block - 1)));



  tests.push_back(DWConvTestParams(
      "input_offset",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  tests.push_back(DWConvTestParams(
      "zero",
      DWConvMicrokernelTester()
          .channel_tile(cr)
          .kernel_tile(kr)
          .input_offset(xnnpack::NextPrime(cr + 1) * 16)
      , test_func, isa_check)
      .loop_zi(0, kr - 1)
      .loop_channels(adj_c_block + c_block, cr * 16 - 1, cr * 3));

  return tests;
}

}  // namespace


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_3P4C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_3P8C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_4P4C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_4P8C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P4C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P4C__WASMSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P8C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P8C__WASMSIMD_ACC2, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_25P4C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_25P8C__WASMSIMD, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/3,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/4,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/9,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/4, /*adj_c_block=*/4, /*cr=*/4, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  INSTANTIATE_TEST_SUITE_P(
      F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, DWConvTest,
      testing::ValuesIn(CreateTests1(
          /*c_block=*/8, /*adj_c_block=*/8, /*cr=*/8, /*kr=*/25,
          [](DWConvMicrokernelTester& tester) {
            tester.Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
          })),
      [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_WASMRELAXEDSIMD


INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_3P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_3P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_3P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_3P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/3,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_4P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_4P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_4P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_4P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/4,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_9P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_9P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_9P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_9P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/9,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_25P1C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_25P1C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/1, /*adj_c_block=*/1, /*cr=*/1, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_25P2C__SCALAR, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });

INSTANTIATE_TEST_SUITE_P(
    F32_DWCONV_25P2C__SCALAR_ACC2, DWConvTest,
    testing::ValuesIn(CreateTests1(
        /*c_block=*/2, /*adj_c_block=*/2, /*cr=*/2, /*kr=*/25,
        [](DWConvMicrokernelTester& tester) {
          tester.Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
        })),
    [](const testing::TestParamInfo<DWConvTest::ParamType>& info) {
      return info.param.test_name;
    });