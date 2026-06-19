# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_lut STATIC lut.cc)
TARGET_LINK_LIBRARIES(ynnpack_kernels_lut PUBLIC ynnpack_base)

IF(YNNPACK_BUILD_TESTS)
  ADD_EXECUTABLE(ynnpack_kernels_lut_test test.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_lut_test ynnpack_kernels_lut ynnpack_base ynnpack_base_test_util gmock gtest gtest_main)
  ADD_TEST(NAME ynnpack_kernels_lut_test COMMAND ynnpack_kernels_lut_test)
ENDIF()

IF(YNNPACK_BUILD_BENCHMARKS)
  ADD_EXECUTABLE(ynnpack_kernels_lut_bench bench.cc)
  TARGET_LINK_LIBRARIES(ynnpack_kernels_lut_bench ynnpack_kernels_lut ynnpack_base benchmark::benchmark_main)
ENDIF()
