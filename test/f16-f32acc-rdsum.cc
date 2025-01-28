// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-f32acc-rdsum
//   Generator: tools/generate-rdsum-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rdsum-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params) XNN_TEST_RDSUM_CHANNELS_EQ(ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params);\
XNN_TEST_RDSUM_CHANNELS_DIV(ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params);                                                                                                                                                   \
XNN_TEST_RDSUM_CHANNELS_LT(ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params);                                                                                                                                                    \
XNN_TEST_RDSUM_CHANNELS_GT(ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params);                                                                                                                                                    \
XNN_TEST_RDSUM_OVERFLOW_ACCUMULATOR(ukernel, arch_flags, primary_tile, incremental_tile, channel_tile, datatype, output_type, params_type, init_params);
#include "f16-f32acc-rdsum/f16-f32acc-rdsum.h"
#undef XNN_UKERNEL_WITH_PARAMS
