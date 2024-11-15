// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-spmm-minmax
//   Generator: tools/generate-spmm-test.py

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/spmm.h"
#include "spmm-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params) XNN_TEST_SPMM_K_EQ(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);  \
XNN_TEST_SPMM_K_LT(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_K_GT(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_K_DIV(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                     \
XNN_TEST_SPMM_N_GT(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_N_DIV(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                     \
XNN_TEST_SPMM_M_LT(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_M_DIV(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                     \
XNN_TEST_SPMM_M_GT(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_OUTPUT_STRIDE(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                             \
XNN_TEST_SPMM_QMIN(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_QMAX(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                                      \
XNN_TEST_SPMM_HALF_SPARSE(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);                                                                                                               \
XNN_TEST_SPMM_ZERO_WEIGHTS(ukernel,arch_flags, mr, nr, pipelined, kblock, init_params);
#include "f16-spmm/f16-spmm-minmax.h"
#undef XNN_UKERNEL_WITH_PARAMS
