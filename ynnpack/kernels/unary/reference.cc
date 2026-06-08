// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/unary/reference.h"

#include <memory>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {

std::unique_ptr<unary_op_info> get_unary_op_info(ynn_unary_operator op,
                                                 uint32_t flags,
                                                 const unary_params& params) {
  switch (op) {
    case ynn_unary_abs:
      return std::make_unique<abs>(flags, params);
    case ynn_unary_round:
      return std::make_unique<round>(flags, params);
    case ynn_unary_ceil:
      return std::make_unique<ceil>(flags, params);
    case ynn_unary_convert:
      return std::make_unique<convert>(flags, params);
    case ynn_unary_exp:
      return std::make_unique<exp>(flags, params);
    case ynn_unary_expm1:
      return std::make_unique<expm1>(flags, params);
    case ynn_unary_erf:
      return std::make_unique<erf>(flags, params);
    case ynn_unary_floor:
      return std::make_unique<floor>(flags, params);
    case ynn_unary_log:
      return std::make_unique<log>(flags, params);
    case ynn_unary_log1p:
      return std::make_unique<log1p>(flags, params);
    case ynn_unary_negate:
      return std::make_unique<negate>(flags, params);
    case ynn_unary_reciprocal_square_root:
      return std::make_unique<reciprocal_square_root>(flags, params);
    case ynn_unary_square:
      return std::make_unique<square>(flags, params);
    case ynn_unary_square_root:
      return std::make_unique<square_root>(flags, params);
    case ynn_unary_tanh:
      return std::make_unique<tanh>(flags, params);
    case ynn_unary_cube_root:
      return std::make_unique<cube_root>(flags, params);
    case ynn_unary_sign:
      return std::make_unique<sign>(flags, params);
    case ynn_unary_sine:
      return std::make_unique<sine>(flags, params);
    case ynn_unary_cosine:
      return std::make_unique<cosine>(flags, params);
    case ynn_unary_sigmoid:
      return std::make_unique<sigmoid>(flags, params);
    case ynn_unary_hardswish:
      return std::make_unique<hardswish>(flags, params);
    case ynn_unary_poly3:
      return std::make_unique<poly3>(flags, params);
    case ynn_unary_round_to_bf16:
      return std::make_unique<round_to_bf16>(flags, params);
    case ynn_unary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // namespace ynn
