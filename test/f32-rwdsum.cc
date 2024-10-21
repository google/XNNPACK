// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-rwdsum
//   Generator: tools/generate-rwd-test.py


#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "reducewindow-d-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, init_params) \
XNN_TEST_RWD_CHANNEL_EQ_ROW_EQ(ukernel, ukernel, RWDMicrokernelTester::OpType::Sum, init_params);          \
XNN_TEST_RWD_CHANNEL_EQ_ROW_GT(ukernel, ukernel, RWDMicrokernelTester::OpType::Sum, init_params);          \
XNN_TEST_RWD_CHANNEL_GT_ROW_EQ(ukernel, ukernel, RWDMicrokernelTester::OpType::Sum, init_params);          \
XNN_TEST_RWD_CHANNEL_GT_ROW_GT(ukernel, ukernel, RWDMicrokernelTester::OpType::Sum, init_params);          
#include "src/f32-rwdsum/f32-rwdsum.h"
#undef XNN_UKERNEL_WITH_PARAMS
