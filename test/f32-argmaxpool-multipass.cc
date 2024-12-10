// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: f32-argmaxpool-multipass
//   Generator: tools/generate-argmaxpool-multipass-test.py


#include <gtest/gtest.h>
#include "xnnpack/argmaxpool.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "argmaxpool-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params) XNN_TEST_ARGMAXPOOL_CHANNELS_EQ_TWOPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);\
XNN_TEST_ARGMAXPOOL_CHANNELS_DIV_TWOPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                   \
XNN_TEST_ARGMAXPOOL_CHANNELS_LT_TWOPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                    \
XNN_TEST_ARGMAXPOOL_CHANNELS_GT_TWOPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                    \
XNN_TEST_ARGMAXPOOL_CHANNELS_EQ_MULTIPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                  \
XNN_TEST_ARGMAXPOOL_CHANNELS_DIV_MULTIPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                 \
XNN_TEST_ARGMAXPOOL_CHANNELS_LT_MULTIPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                  \
XNN_TEST_ARGMAXPOOL_CHANNELS_GT_MULTIPASS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);                                                                                                                                                  \
XNN_TEST_ARGMAXPOOL_FEW_OUTPUT_PIXELS(ukernel,arch_flags, primary_tile, incremental_tile, channel_tile, vector_tile, datatype, params_type, init_params);
#include "f32-argmaxpool/f32-argmaxpool-multipass.h"
#undef XNN_UKERNEL_WITH_PARAMS
