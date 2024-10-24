// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-rwsum
//   Generator: tools/generate-rw-test.py


#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "reducewindow-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, init_params)   \
XNN_TEST_RW_BATCH_EQ(ukernel, batch_tile,  ukernel, ReduceWindowMicrokernelTester::OpType::Sum, init_params);\
XNN_TEST_RW_BATCH_GT(ukernel, batch_tile, ukernel, ReduceWindowMicrokernelTester::OpType::Sum, init_params); \
XNN_TEST_RW_BATCH_LT(ukernel, batch_tile, ukernel, ReduceWindowMicrokernelTester::OpType::Sum, init_params); \
XNN_TEST_RW_BATCH_DIV(ukernel, batch_tile, ukernel, ReduceWindowMicrokernelTester::OpType::Sum, init_params);
#include "f32-rwsum/f32-rwsum.h"
#undef XNN_UKERNEL_WITH_PARAMS
