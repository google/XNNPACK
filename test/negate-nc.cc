// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "unary-operator-tester.h"

namespace xnnpack {

class NegateOperatorTester : public UnaryOperatorTester {
 public:
  NegateOperatorTester() : UnaryOperatorTester() {
    range_f32_ = {-1.0f, 1.0f};
    range_f16_ = {-1.0f, 1.0f};
  }

 protected:
  // Computes the expected result for some input `x`. Subclasses should override
  // this function with their own reference function.
  float RefFunc(float x) const override { return -x; }

  CREATE_OP_OVERRIDES_F32(negate);
  CREATE_OP_OVERRIDES_F16(negate);
};

CREATE_UNARY_FLOAT_TESTS(F32, NegateOperatorTester);
CREATE_UNARY_FLOAT_TESTS(RunF32, NegateOperatorTester);
#ifndef XNN_EXCLUDE_F16_TESTS
CREATE_UNARY_FLOAT_TESTS(F16, NegateOperatorTester);
#endif  // XNN_EXCLUDE_F16_TESTS

};  // namespace xnnpack
