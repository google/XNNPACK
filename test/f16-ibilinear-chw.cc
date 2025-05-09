// clang-format off
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-ibilinear-chw
//   Generator: tools/generate-ibilinear-chw-test.py


#include <gtest/gtest.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, pixel_tile, channel_tile, datatype, params_type, init_params) XNN_TEST_IBILINEAR_CHW_PIXELS_DIV(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);\
XNN_TEST_IBILINEAR_CHW_PIXELS_LT(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                    \
XNN_TEST_IBILINEAR_CHW_PIXELS_GT(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                    \
XNN_TEST_IBILINEAR_CHW_CHANNELS_DIV(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                 \
XNN_TEST_IBILINEAR_CHW_CHANNELS_EQ(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                  \
XNN_TEST_IBILINEAR_CHW_CHANNELS_GT(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                  \
XNN_TEST_IBILINEAR_CHW_INPUT_OFFSET(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);                                                                                                                 \
XNN_TEST_IBILINEAR_CHW_INPUT_STRIDE(ukernel, arch_flags, pixel_tile, channel_tile, datatype, params_type, init_params);
#include "f16-ibilinear-chw/f16-ibilinear-chw.h"
#undef XNN_UKERNEL_WITH_PARAMS
