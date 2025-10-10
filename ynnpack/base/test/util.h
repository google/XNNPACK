// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TEST_UTIL_H_
#define XNNPACK_YNNPACK_BASE_TEST_UTIL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"

namespace ynn {

// Returns a set of sizes that is likely to be relevant for SIMD operations.
std::vector<size_t> simd_sizes_up_to(size_t max_size, size_t alignment = 1);

template <typename T>
std::string str_join(const std::string&, const T& t) {
  using std::to_string;
  return to_string(t);
}

template <typename T, typename... Ts>
std::string str_join(const std::string& separator, const T& t,
                     const Ts&... ts) {
  using std::to_string;
  return to_string(t) + separator + str_join(separator, ts...);
}

template <typename T, size_t... Is>
std::string test_param_to_string_impl(const T& t, std::index_sequence<Is...>) {
  return str_join("_", std::get<Is>(t)...);
}

// Generate a string to name a test param object, assuming the object is a
// tuple.
template <typename T>
std::string test_param_to_string(const testing::TestParamInfo<T>& info) {
  constexpr size_t n = std::tuple_size<T>();
  std::string result =
      test_param_to_string_impl(info.param, std::make_index_sequence<n>());
  std::replace(result.begin(), result.end(), '-', '_');
  return result;
}

// This type enumerates a "tuple" of types. It supports mapping from up to 3
// compile-time types to a value of this enumeration (via `multi_type_of<>`),
// and then back to compile-time types, via. `SwitchTwoTypes` and
// `SwitchThreeTypes`.
enum class multi_type {
  fp32,
  fp16,
  bf16,
  int8,
  uint8,

  fp16_fp32,
  bf16_fp32,
  int8_int32,
  uint8_int32,

  fp16_fp16_fp32,
  bf16_bf16_fp32,
  int8_int8_int32,
  int8_int4_int32,
  uint8_int8_int32,
};

inline multi_type multi_type_of(float, float) { return multi_type::fp32; }
inline multi_type multi_type_of(bfloat16, bfloat16) { return multi_type::bf16; }
inline multi_type multi_type_of(half, half) { return multi_type::fp16; }
inline multi_type multi_type_of(int8_t, int8_t) { return multi_type::int8; }
inline multi_type multi_type_of(uint8_t, uint8_t) { return multi_type::uint8; }
inline multi_type multi_type_of(bfloat16, float) {
  return multi_type::bf16_fp32;
}
inline multi_type multi_type_of(half, float) { return multi_type::fp16_fp32; }
inline multi_type multi_type_of(int8_t, int32_t) {
  return multi_type::int8_int32;
}
inline multi_type multi_type_of(uint8_t, int32_t) {
  return multi_type::uint8_int32;
}

inline multi_type multi_type_of(float, float, float) {
  return multi_type::fp32;
}

inline multi_type multi_type_of(half, half, float) {
  return multi_type::fp16_fp16_fp32;
}
inline multi_type multi_type_of(bfloat16, bfloat16, float) {
  return multi_type::bf16_bf16_fp32;
}
inline multi_type multi_type_of(int8_t, int8_t, int32_t) {
  return multi_type::int8_int8_int32;
}
inline multi_type multi_type_of(uint8_t, int8_t, int32_t) {
  return multi_type::uint8_int8_int32;
}
inline multi_type multi_type_of(int8_t, int4x2, int32_t) {
  return multi_type::int8_int4_int32;
}

template <typename F>
constexpr decltype(auto) SwitchTwoTypes(multi_type type, F&& f) {
  switch (type) {
    case multi_type::fp32:
      return std::forward<F>(f)(float(), float());
    case multi_type::fp16:
      return std::forward<F>(f)(half(), half());
    case multi_type::bf16:
      return std::forward<F>(f)(bfloat16(), bfloat16());
    case multi_type::int8:
      return std::forward<F>(f)(int8_t(), int8_t());
    case multi_type::uint8:
      return std::forward<F>(f)(uint8_t(), uint8_t());
    case multi_type::fp16_fp32:
      return std::forward<F>(f)(half(), float());
    case multi_type::bf16_fp32:
      return std::forward<F>(f)(bfloat16(), float());
    case multi_type::int8_int32:
      return std::forward<F>(f)(int8_t(), int32_t());
    case multi_type::uint8_int32:
      return std::forward<F>(f)(uint8_t(), int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

template <typename F>
constexpr decltype(auto) SwitchThreeTypes(multi_type type, F&& f) {
  switch (type) {
    case multi_type::fp32:
      return std::forward<F>(f)(float(), float(), float());
    case multi_type::bf16:
      return std::forward<F>(f)(bfloat16(), bfloat16(), bfloat16());
    case multi_type::int8:
      return std::forward<F>(f)(int8_t(), int8_t(), int8_t());
    case multi_type::uint8:
      return std::forward<F>(f)(uint8_t(), uint8_t(), uint8_t());
    case multi_type::fp16_fp16_fp32:
      return std::forward<F>(f)(half(), half(), float());
    case multi_type::bf16_bf16_fp32:
      return std::forward<F>(f)(bfloat16(), bfloat16(), float());
    case multi_type::int8_int8_int32:
      return std::forward<F>(f)(int8_t(), int8_t(), int32_t());
    case multi_type::uint8_int8_int32:
      return std::forward<F>(f)(uint8_t(), int8_t(), int32_t());
    case multi_type::int8_int4_int32:
      return std::forward<F>(f)(int8_t(), int4x2(), int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

inline const char* to_string(multi_type type) {
  switch (type) {
    case multi_type::fp32:
      return "fp32";
    case multi_type::fp16:
      return "fp16";
    case multi_type::bf16:
      return "bf16";
    case multi_type::int8:
      return "int8";
    case multi_type::uint8:
      return "uint8";
    case multi_type::fp16_fp32:
      return "fp16_fp32";
    case multi_type::bf16_fp32:
      return "bf16_fp32";
    case multi_type::int8_int32:
      return "int8_int32";
    case multi_type::uint8_int32:
      return "uint8_int32";
    case multi_type::fp16_fp16_fp32:
      return "fp16_fp16_fp32";
    case multi_type::bf16_bf16_fp32:
      return "bf16_bf16_fp32";
    case multi_type::int8_int8_int32:
      return "int8_int8_int32";
    case multi_type::uint8_int8_int32:
      return "uint8_int8_int32";
    case multi_type::int8_int4_int32:
      return "int8_int4_int32";
  }
  YNN_UNREACHABLE;
  return nullptr;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_UTIL_H_
