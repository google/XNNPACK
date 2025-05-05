// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: f32-vmulcaddc
//   Generator: tools/generate-vmulcaddc-test.py

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vmulcaddc.h"
#include "test/vmulcaddc-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile,   \
                                datatype, params_type, init_params)            \
  XNN_TEST_VMULCADDC_ROW_DIV(ukernel, arch_flags, row_tile, channel_tile,      \
                             datatype, params_type, init_params);              \
  XNN_TEST_VMULCADDC_ROW_LT(ukernel, arch_flags, row_tile, channel_tile,       \
                            datatype, params_type, init_params);               \
  XNN_TEST_VMULCADDC_ROW_GT(ukernel, arch_flags, row_tile, channel_tile,       \
                            datatype, params_type, init_params);               \
  XNN_TEST_VMULCADDC_CHANNEL_GT(ukernel, arch_flags, row_tile, channel_tile,   \
                                datatype, params_type, init_params);           \
  XNN_TEST_VMULCADDC_CHANNEL_EQ(ukernel, arch_flags, row_tile, channel_tile,   \
                                datatype, params_type, init_params);           \
  XNN_TEST_VMULCADDC_CHANNEL_DIV(ukernel, arch_flags, row_tile, channel_tile,  \
                                 datatype, params_type, init_params);          \
  XNN_TEST_VMULCADDC_CHANNEL_LT(ukernel, arch_flags, row_tile, channel_tile,   \
                                datatype, params_type, init_params);           \
  XNN_TEST_VMULCADDC_INPUT_STRIDE(ukernel, arch_flags, row_tile, channel_tile, \
                                  datatype, params_type, init_params);         \
  XNN_TEST_VMULCADDC_OUTPUT_STRIDE(ukernel, arch_flags, row_tile,              \
                                   channel_tile, datatype, params_type,        \
                                   init_params);                               \
  XNN_TEST_VMULCADDC_INPLACE(ukernel, arch_flags, row_tile, channel_tile,      \
                             datatype, params_type, init_params);              \
  XNN_TEST_VMULCADDC_QMAX(ukernel, arch_flags, row_tile, channel_tile,         \
                          datatype, params_type, init_params);                 \
  XNN_TEST_VMULCADDC_QMIN(ukernel, arch_flags, row_tile, channel_tile,         \
                          datatype, params_type, init_params);
#include "src/f32-vmulcaddc/f32-vmulcaddc.h"
#undef XNN_UKERNEL_WITH_PARAMS
