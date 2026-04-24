# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_base_test_buffer INTERFACE)
TARGET_INCLUDE_DIRECTORIES(ynnpack_base_test_buffer INTERFACE .)

ADD_LIBRARY(ynnpack_base_test_tensor INTERFACE)
TARGET_INCLUDE_DIRECTORIES(ynnpack_base_test_tensor INTERFACE .)
TARGET_LINK_LIBRARIES(ynnpack_base_test_tensor INTERFACE ynnpack_base_test_buffer ynnpack_base)

ADD_LIBRARY(ynnpack_base_test_random INTERFACE)
TARGET_INCLUDE_DIRECTORIES(ynnpack_base_test_random INTERFACE .)
TARGET_LINK_LIBRARIES(ynnpack_base_test_random INTERFACE ynnpack ynnpack_base)

ADD_LIBRARY(ynnpack_base_test_util STATIC util.cc)
TARGET_INCLUDE_DIRECTORIES(ynnpack_base_test_util PUBLIC .)
TARGET_LINK_LIBRARIES(ynnpack_base_test_util PUBLIC ynnpack_base gtest)

ADD_LIBRARY(ynnpack_base_test_fuzz INTERFACE)
TARGET_INCLUDE_DIRECTORIES(ynnpack_base_test_fuzz INTERFACE .)
TARGET_LINK_LIBRARIES(ynnpack_base_test_fuzz INTERFACE ynnpack_base gtest)

ADD_EXECUTABLE(ynnpack_random_test random_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_random_test ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random gtest gtest_main)
ADD_TEST(NAME ynnpack_random_test COMMAND ynnpack_random_test)

ADD_EXECUTABLE(ynnpack_tensor_test tensor_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_tensor_test ynnpack_base_test_fuzz ynnpack_base_test_tensor gmock gtest gtest_main)
ADD_TEST(NAME ynnpack_tensor_test COMMAND ynnpack_tensor_test)

ADD_EXECUTABLE(ynnpack_float_test float_test.cc)
TARGET_LINK_LIBRARIES(ynnpack_float_test ynnpack ynnpack_base ynnpack_base_test_fuzz ynnpack_base_test_random gtest gtest_main)
ADD_TEST(NAME ynnpack_float_test COMMAND ynnpack_float_test)
