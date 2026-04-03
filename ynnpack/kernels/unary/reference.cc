// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/unary/reference.h"

#include "ynnpack/include/ynnpack.h"

namespace ynn {

std::unique_ptr<unary_op_info> get_unary_op_info(ynn_unary_operator op,
                                                 const unary_params& params) {
  switch (op) {
    case ynn_unary_abs:
      return std::make_unique<abs>(params);
    case ynn_unary_round:
      return std::make_unique<round>(params);
    case ynn_unary_ceil:
      return std::make_unique<ceil>(params);
    case ynn_unary_convert:
      return std::make_unique<convert>(params);
    case ynn_unary_exp:
      return std::make_unique<exp>(params);
    case ynn_unary_expm1:
      return std::make_unique<expm1>(params);
    case ynn_unary_erf:
      return std::make_unique<erf>(params);
    case ynn_unary_floor:
      return std::make_unique<floor>(params);
    case ynn_unary_log:
      return std::make_unique<log>(params);
    case ynn_unary_log1p:
      return std::make_unique<log1p>(params);
    case ynn_unary_negate:
      return std::make_unique<negate>(params);
    case ynn_unary_reciprocal_square_root:
      return std::make_unique<reciprocal_square_root>(params);
    case ynn_unary_square:
      return std::make_unique<square>(params);
    case ynn_unary_square_root:
      return std::make_unique<square_root>(params);
    case ynn_unary_tanh:
      return std::make_unique<tanh>(params);
    case ynn_unary_cube_root:
      return std::make_unique<cube_root>(params);
    case ynn_unary_sign:
      return std::make_unique<sign>(params);
    case ynn_unary_sine:
      return std::make_unique<sine>(params);
    case ynn_unary_cosine:
      return std::make_unique<cosine>(params);
    case ynn_unary_sigmoid:
      return std::make_unique<sigmoid>(params);
    case ynn_unary_hardswish:
      return std::make_unique<hardswish>(params);
    case ynn_unary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // namespace ynn
