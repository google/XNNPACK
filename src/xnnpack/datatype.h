// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"

#ifdef __cplusplus
#include <type_traits>

extern "C" {
#endif

// Returns true if the datatype is floating point, or a quantized datatype
// (which is conceptually a real).
bool xnn_datatype_is_real(enum xnn_datatype t);

// Returns true if the datatype is an integer type, but not quantized.
bool xnn_datatype_is_integral(enum xnn_datatype t);

// Returns true if the datatype is a quantized real datatype.
bool xnn_datatype_is_quantized(enum xnn_datatype t);

// Returns the size of an element of the datatype.
size_t xnn_datatype_log2_size_bits(enum xnn_datatype t);
size_t xnn_datatype_log2_size_bytes(enum xnn_datatype t);
size_t xnn_datatype_size_bits(enum xnn_datatype t);
size_t xnn_datatype_size_bytes(enum xnn_datatype t);

// Returns true if the datatype can be addressed by a linear combination of
// indices and strides.
bool xnn_datatype_is_byte_addressable(enum xnn_datatype t);

#ifdef __cplusplus
}  // extern "C"

template <typename T>
xnn_datatype xnn_datatype_of() {
  if (std::is_same<T, uint8_t>::value) {
    return xnn_datatype_quint8;
  } else if (std::is_same<T, int8_t>::value) {
    return xnn_datatype_qint8;
  } else if (std::is_same<T, xnn_float16>::value) {
    return xnn_datatype_fp16;
  } else if (std::is_same<T, float>::value) {
    return xnn_datatype_fp32;
  } else if (std::is_same<T, int32_t>::value) {
    return xnn_datatype_int32;
  } else {
    return xnn_datatype_invalid;
  }
}
#endif