// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vcvt.h"
#include "test/vunary-microkernel-tester.h"

#define XNN_QUANTIZED(T) xnnpack::quantized<T>
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
  }                                                                          \
  TEST(ukernel, output_scale) {                                              \
    TestOutputScale<Convert, datatype_in, datatype_out>(                     \
        arch_flags, batch_tile, ukernel, init_params);                       \
  }                                                                          \
  TEST(ukernel, output_zero_point) {                                         \
    TestOutputZeroPoint<Convert, datatype_in, datatype_out>(                 \
        arch_flags, batch_tile, ukernel, init_params);                       \
  }                                                                          \
  TEST(ukernel, output_saturation) {                                         \
    TestOutputSaturation<Convert, datatype_in, datatype_out>(                \
        arch_flags, batch_tile, ukernel, init_params);                       \
  }                                                                          \
  TEST(ukernel, output_overflow) {                                           \
    TestOutputOverflow<Convert, datatype_in, datatype_out>(                  \
        arch_flags, batch_tile, ukernel, init_params);                       \
  }
#include "src/f32-qu8-vcvt/f32-qu8-vcvt.inc"
#undef XNN_UKERNEL
#undef XNN_QUANTIZED
