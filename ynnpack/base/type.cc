// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/type.h"

#include <cassert>
#include <cstddef>

#include "ynnpack/base/base.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

bool type_is_integral(ynn_type t) {
  switch (t) {
    case ynn_type_int4:
    case ynn_type_uint4:
    case ynn_type_int8:
    case ynn_type_uint8:
    case ynn_type_int32:
      return true;
    case ynn_type_fp32:
    case ynn_type_fp16:
    case ynn_type_bf16:
    case ynn_type_opaque:
    case ynn_type_invalid:
      return false;
  }
  YNN_UNREACHABLE;
  return false;
}

size_t type_size_bytes(ynn_type t) {
  switch (t) {
    case ynn_type_int4:
    case ynn_type_uint4:
      return 1;
    case ynn_type_int8:
    case ynn_type_uint8:
      return 1;
    case ynn_type_fp16:
    case ynn_type_bf16:
      return 2;
    case ynn_type_int32:
    case ynn_type_fp32:
      return 4;
    case ynn_type_opaque:
      return 0;
    case ynn_type_invalid:
      break;
  }
  YNN_UNREACHABLE;
  return 0;
}

const char* to_string(ynn_type type) {
  switch (type) {
    case ynn_type_invalid:
      return "invalid";
    case ynn_type_opaque:
      return "opaque";
    case ynn_type_int4:
      return "int4";
    case ynn_type_uint4:
      return "uint4";
    case ynn_type_int8:
      return "int8";
    case ynn_type_uint8:
      return "uint8";
    case ynn_type_int32:
      return "int32";
    case ynn_type_fp32:
      return "fp32";
    case ynn_type_fp16:
      return "fp16";
    case ynn_type_bf16:
      return "bf16";
  }
  YNN_UNREACHABLE;
  return "unknown";
}

}  // namespace ynn
