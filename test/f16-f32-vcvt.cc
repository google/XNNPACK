// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vcvt.h"
#include "test/vunary-microkernel-tester.h"

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile,         \
                                    vector_tile, datatype_in, datatype_out,  \
                                    params_type, init_params)                \
  TEST(ukernel, batch_eq) {                                                  \
    TestBatchEq<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,  \
                                                    ukernel, init_params);   \
  }                                                                          \
  TEST(ukernel, batch_div) {                                                 \
    TestBatchDiv<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, \
                                                     ukernel, init_params);  \
  }                                                                          \
  TEST(ukernel, batch_lt) {                                                  \
    TestBatchLT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,  \
                                                    ukernel, init_params);   \
  }                                                                          \
  TEST(ukernel, batch_gt) {                                                  \
    TestBatchGT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,  \
                                                    ukernel, init_params);   \
  }
#include "src/f16-f32-vcvt/f16-f32-vcvt.inc"
#undef XNN_UKERNEL
