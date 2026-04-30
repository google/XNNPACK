// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/type.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

bool type_is_integral(ynn_type t) {
  switch (t) {
    case ynn_type_int2:
    case ynn_type_uint2:
    case ynn_type_int4:
    case ynn_type_uint4:
    case ynn_type_int8:
    case ynn_type_uint8:
    case ynn_type_int32:
      return true;
    case ynn_type_fp64:
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

bool type_is_floating_point(ynn_type t) {
  switch (t) {
    case ynn_type_fp64:
    case ynn_type_fp32:
    case ynn_type_fp16:
    case ynn_type_bf16:
      return true;
    case ynn_type_int2:
    case ynn_type_uint2:
    case ynn_type_int4:
    case ynn_type_uint4:
    case ynn_type_int8:
    case ynn_type_uint8:
    case ynn_type_int32:
    case ynn_type_opaque:
    case ynn_type_invalid:
      return false;
  }
  YNN_UNREACHABLE;
  return false;
}

size_t type_size_bits(ynn_type t) {
  switch (t) {
    case ynn_type_int2:
    case ynn_type_uint2:
      return 2;
    case ynn_type_int4:
    case ynn_type_uint4:
      return 4;
    case ynn_type_int8:
    case ynn_type_uint8:
      return 8;
    case ynn_type_fp16:
    case ynn_type_bf16:
      return 16;
    case ynn_type_int32:
    case ynn_type_fp32:
      return 32;
    case ynn_type_fp64:
      return 64;
    case ynn_type_opaque:
      return 0;
    case ynn_type_invalid:
      break;
  }
  YNN_UNREACHABLE;
  return 0;
}

size_t type_size_bytes(ynn_type t) { return (type_size_bits(t) + 7) / 8; }

size_t type_mantissa_bits(ynn_type t) {
  switch (t) {
    case ynn_type_fp16:
      return 11;
    case ynn_type_bf16:
      return 8;
    case ynn_type_fp32:
      return 24;
    case ynn_type_fp64:
      return 53;
    default:
      // Treat all bits as mantissa for integers.
      return type_size_bits(t);
  }
}

size_t type_exponent_bits(ynn_type t) {
  switch (t) {
    case ynn_type_fp16:
      return 5;
    case ynn_type_bf16:
    case ynn_type_fp32:
      return 8;
    case ynn_type_fp64:
      return 11;
    default:
      // Integers don't have an exponent.
      return 0;
  }
}

bool is_convert_lossless(ynn_type from, ynn_type to) {
  return type_mantissa_bits(from) <= type_mantissa_bits(to) &&
         type_exponent_bits(from) <= type_exponent_bits(to);
}

const char* to_string(ynn_type type) {
  switch (type) {
    case ynn_type_invalid:
      return "invalid";
    case ynn_type_opaque:
      return "opaque";
    case ynn_type_int2:
      return "int2";
    case ynn_type_uint2:
      return "uint2";
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
    case ynn_type_fp64:
      return "fp64";
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

namespace {

template <typename T>
void convert_to_int(const float* src, size_t n, T* dst) {
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(dst, i, round_float_to_int<T>(src[i]));
  }
}

}  // namespace

void convert_n(const float* src, size_t n, ynn_type type, void* dst) {
  switch (type) {
    case ynn_type_fp64:
      std::copy_n(src, n, (double*)dst);
      return;
    case ynn_type_fp32:
      std::copy_n(src, n, (float*)dst);
      return;
    case ynn_type_fp16:
      std::copy_n(src, n, (half*)dst);
      return;
    case ynn_type_bf16:
      std::copy_n(src, n, (bfloat16*)dst);
      return;
    case ynn_type_int2:
      convert_to_int<int2x4>(src, n, (int2x4*)dst);
      return;
    case ynn_type_uint2:
      convert_to_int<uint2x4>(src, n, (uint2x4*)dst);
      return;
    case ynn_type_int4:
      convert_to_int<int4x2>(src, n, (int4x2*)dst);
      return;
    case ynn_type_uint4:
      convert_to_int<uint4x2>(src, n, (uint4x2*)dst);
      return;
    case ynn_type_int8:
      convert_to_int<int8_t>(src, n, (int8_t*)dst);
      return;
    case ynn_type_uint8:
      convert_to_int<uint8_t>(src, n, (uint8_t*)dst);
      return;
    case ynn_type_int32:
      convert_to_int<int32_t>(src, n, (int32_t*)dst);
      return;
    case ynn_type_opaque:
    case ynn_type_invalid:
      break;
  }
  YNN_UNREACHABLE;
}

}  // namespace ynn
