// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: f32-vscaleextexp
//   Generator: tools/generate-vscaleextexp-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vscaleextexp.h"
#include "vscaleextexp-microkernel-tester.h"

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params)                 \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, init_params);                                    \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, init_params);                                   \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, init_params);                                    \
  XNN_TEST_VSCALEEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, init_params);
#include "f32-vscaleextexp/f32-vscaleextexp.h"
#undef XNN_UKERNEL_WITH_PARAMS
