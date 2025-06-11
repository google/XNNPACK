// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

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
bool xnn_datatype_is_channelwise_quantized(enum xnn_datatype t);
bool xnn_datatype_is_blockwise_quantized(enum xnn_datatype t);

// Returns the size of an element of the datatype.
size_t xnn_datatype_log2_size_bits(enum xnn_datatype t);
size_t xnn_datatype_log2_size_bytes(enum xnn_datatype t);
size_t xnn_datatype_size_bits(enum xnn_datatype t);
size_t xnn_datatype_size_bytes(enum xnn_datatype t);

// Returns true if the datatype can be addressed by a linear combination of
// indices and strides.
bool xnn_datatype_is_byte_addressable(enum xnn_datatype t);

const char* xnn_datatype_to_string(enum xnn_datatype type);

#ifdef __cplusplus
}  // extern "C"

namespace xnnpack {

struct channelwise {};

// We need a type that distinguishes an intX_t from a quantized intX_t. We can't
// do arithmetic on these, because we don't know the quantization parameters.
template <typename T, typename Kind = void>
struct quantized {
  T value;
  using type = T;

  operator T() const { return value; }
  // Forward operator[] in case T is a sub-byte packed value.
  auto operator[](size_t i) const { return value[i]; }

  quantized() = default;
  quantized(T t) : value(t) {}
  quantized<T, Kind>& operator=(T t) {
    value = t;
    return *this;
  }
};

template <typename T>
struct is_quantized : std::false_type {};

template <typename T, typename Kind>
struct is_quantized<quantized<T, Kind>> : std::true_type {};

template <typename T>
struct unwrap_quantized {
  using type = T;
};

template <typename Kind>
struct unwrap_quantized<quantized<int8_t, Kind>> {
  using type = int8_t;
};

template <typename Kind>
struct unwrap_quantized<quantized<uint8_t, Kind>> {
  using type = uint8_t;
};

template <typename Kind>
struct unwrap_quantized<quantized<int32_t, Kind>> {
  using type = int32_t;
};

}  // namespace xnnpack

template <typename T>
xnn_datatype xnn_datatype_of() {
  if (std::is_same<T, xnnpack::quantized<uint8_t>>::value) {
    return xnn_datatype_quint8;
  } else if (std::is_same<T, xnnpack::quantized<int8_t>>::value) {
    return xnn_datatype_qint8;
  } else if (std::is_same<
                 T, xnnpack::quantized<int8_t, xnnpack::channelwise>>::value) {
    return xnn_datatype_qcint8;
  } else if (std::is_same<T, xnnpack::quantized<int32_t>>::value) {
    return xnn_datatype_qint32;
  } else if (std::is_same<T, xnn_float16>::value) {
    return xnn_datatype_fp16;
  } else if (std::is_same<T, xnn_bfloat16>::value) {
    return xnn_datatype_bf16;
  } else if (std::is_same<T, float>::value) {
    return xnn_datatype_fp32;
  } else if (std::is_same<T, int32_t>::value) {
    return xnn_datatype_int32;
  } else {
    return xnn_datatype_invalid;
  }
}
#endif
