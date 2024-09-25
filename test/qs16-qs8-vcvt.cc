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
  TEST(ukernel, output_scale) { TestOutputScale<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }           \
  TEST(ukernel, output_zero_point) { TestOutputZeroPoint<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, ukernel, init_params); }
// TODO: Should there be testing of input scale/zero point? Or rename qs16 -> s16 in this kernel?
#include "qs16-qs8-vcvt/qs16-qs8-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS
