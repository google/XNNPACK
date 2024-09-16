// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: qu8-vcvt
//   Generator: tools/generate-vcvt-test.py


#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,                               \
                                datatype_in, datatype_out, params_type, init_params)                            \
XNN_TEST_CVT_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);        \
XNN_TEST_CVT_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);       \
XNN_TEST_CVT_BATCH_LT(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);        \
XNN_TEST_CVT_BATCH_GT(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);        \
                                                                                                                \
XNN_TEST_CVT_SCALE(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);           \
                                                                                                                \
XNN_TEST_CVT_INPUT_ZERO_POINT(ukernel, arch_flags, batch_tile, datatype_in, datatype_out, ukernel, init_params);
#include "src/qu8-vcvt/qu8-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS
