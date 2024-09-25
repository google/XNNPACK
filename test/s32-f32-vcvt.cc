// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"
#include "vunary-microkernel-tester.h"

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,                                                              \
                                datatype_in, datatype_out, params_type, init_params)                                                           \
  TEST(ukernel, batch_eq) { TestBatchEq<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }                   \
  TEST(ukernel, batch_div) { TestBatchDiv<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }                 \
  TEST(ukernel, batch_lt) { TestBatchLT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }                   \
  TEST(ukernel, batch_gt) { TestBatchGT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }                   \
  TEST(ukernel, input_zero_point) { TestInputZeroPoint<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }
#include "s32-f32-vcvt/s32-f32-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS
