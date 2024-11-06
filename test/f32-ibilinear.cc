// clang-format off
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-ibilinear
//   Generator: tools/generate-ibilinear-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, params_type, init_params) XNN_TEST_IBILINEAR_CHANNELS_EQ(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);\
XNN_TEST_IBILINEAR_CHANNELS_DIV(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                  \
XNN_TEST_IBILINEAR_CHANNELS_LT(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                   \
XNN_TEST_IBILINEAR_CHANNELS_GT(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                   \
XNN_TEST_IBILINEAR_PIXELS_DIV(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                    \
XNN_TEST_IBILINEAR_PIXELS_LT(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                     \
XNN_TEST_IBILINEAR_PIXELS_GT(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                     \
XNN_TEST_IBILINEAR_INPUT_OFFSET(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);                                                                                                                  \
XNN_TEST_IBILINEAR_OUTPUT_STRIDE(ukernel, arch_flags, channel_tile, pixel_tile, datatype, params_type, init_params);
#include "f32-ibilinear/f32-ibilinear.h"
#undef XNN_UKERNEL_WITH_PARAMS
