// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: f32-raddextexp
//   Generator: tools/generate-raddextexp-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/raddextexp.h"
#include "raddextexp-microkernel-tester.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) XNN_TEST_RADDEXTEXP_ELEMENT_EQ(ukernel,arch_flags, element_tile, init_params);  \
XNN_TEST_RADDEXTEXP_ELEMENT_DIV(ukernel,arch_flags, element_tile, init_params);                                                                                                        \
XNN_TEST_RADDEXTEXP_ELEMENT_LT(ukernel,arch_flags, element_tile, init_params);                                                                                                         \
XNN_TEST_RADDEXTEXP_ELEMENT_GT(ukernel,arch_flags, element_tile, init_params);
#include "src/f32-raddextexp/f32-raddextexp.h"
#undef XNN_UKERNEL_WITH_PARAMS
