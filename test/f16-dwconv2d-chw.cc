// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-dwconv2d-chw
//   Generator: tools/generate-dwconv2d-chw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "dwconv2d-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params) XNN_TEST_DWCONV2D_OUTPUT_WIDTH_EQ(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);\
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_DIV(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                    \
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_LT(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                     \
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_GT(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                     \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_EQ(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                    \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_DIV(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                   \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_LT(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                    \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_GT(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);                                                                                                                                                                    \
XNN_TEST_DWCONV2D_OUTPUT_PADDING_TOP_EQ(ukernel, arch_flags, kernel_height, kernel_width, subsampling, padding, height_tile, width_tile, datatype, params_type, init_params);
#include "f16-dwconv2d-chw/f16-dwconv2d-chw.h"
#undef XNN_UKERNEL_WITH_PARAMS
