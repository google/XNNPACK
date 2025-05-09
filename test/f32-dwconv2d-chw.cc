// clang-format off
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-dwconv2d-chw
//   Generator: tools/generate-dwconv2d-chw-test.py


#include <gtest/gtest.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "dwconv2d-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params) XNN_TEST_DWCONV2D_OUTPUT_WIDTH_EQ(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);\
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_DIV(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                \
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_LT(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                 \
XNN_TEST_DWCONV2D_OUTPUT_WIDTH_GT(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                 \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_EQ(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_DIV(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                               \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_LT(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                \
XNN_TEST_DWCONV2D_OUTPUT_HEIGHT_GT(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);                                                                                                                                                                                \
XNN_TEST_DWCONV2D_OUTPUT_PADDING_TOP_EQ(arch_flags, ukernel, _kernel_height_, _kernel_width_, _subsampling_, _padding_, _height_tile_, _width_tile_, datatype, params_type, init_params);
#include "src/f32-dwconv2d-chw/f32-dwconv2d-chw.h"
#undef XNN_UKERNEL_WITH_PARAMS
