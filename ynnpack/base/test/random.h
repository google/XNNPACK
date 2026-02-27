// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TEST_RANDOM_H_
#define XNNPACK_YNNPACK_BASE_TEST_RANDOM_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// This is a faster way of generating random numbers, by generating as many
// random values as possible for each call to rng(). Assumes that rng() returns
// entirely random bits.
template <typename T, typename Rng>
void fill_uniform_random_bits(T* data, size_t size, Rng& rng) {
  using RngT = decltype(rng());
  size_t size_bytes = size * sizeof(T);
  // Fill with as many RngT as we can.
  while (size_bytes >= sizeof(RngT)) {
    RngT bits = rng();
    memcpy(data, &bits, sizeof(RngT));
    data = offset_bytes(data, sizeof(RngT));
    size_bytes -= sizeof(RngT);
  }
  // Fill the remaining bytes.
  RngT bits = rng();
  memcpy(data, &bits, size_bytes);
}

// Generate a random shape of the given rank, where each dim is in [min_dim,
// max_dim].
template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t rank, size_t min_dim,
                                 size_t max_dim) {
  std::uniform_int_distribution<size_t> dim_dist(min_dim, max_dim);
  std::vector<size_t> shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    shape[i] = dim_dist(rng);
  }
  return shape;
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t rank) {
  return random_shape(rng, rank, 1, 9);
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t min_dim, size_t max_dim) {
  std::uniform_int_distribution<size_t> rank_dist(0, YNN_MAX_TENSOR_RANK - 1);
  return random_shape(rng, rank_dist(rng), min_dim, max_dim);
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng) {
  return random_shape(rng, 1, 9);
}

// Generate random quantization parameters for a given type.
template <typename T, typename Rng>
quantization_params random_quantization(T, Rng& rng, float min_scale = 0.25f,
                                        float max_scale = 8.0f) {
  return {};
}

template <typename T, typename Rng>
quantization_params random_quantization(quantized<T>, Rng& rng,
                                        float min_scale = 0.25f,
                                        float max_scale = 8.0f) {
  std::uniform_int_distribution<> T_dist{type_info<T>::min(),
                                         type_info<T>::max()};
  std::uniform_real_distribution<float> scale_dist{min_scale, max_scale};
  return {T_dist(rng), scale_dist(rng)};
}

// Bitcast a random number generator to type T.
template <typename T, typename Rng>
T random_bits(Rng& rng) {
  static_assert(Rng::min() == 0, "");
  static_assert(Rng::max() >= (1ull << (sizeof(T) * 8)) - 1, "");
  auto bits = rng();
  T result;
  static_assert(sizeof(result) <= sizeof(bits), "");
  memcpy(&result, &bits, sizeof(T));
  return result;
}

// Make a bitcasted random float, and then flush it to 0 if it is denormal (or
// Nan).
template <typename T, typename Rng>
T random_normal_float(Rng& rng) {
  T result = random_bits<T>(rng);
  if (std::abs(static_cast<float>(result)) >= type_info<T>::smallest_normal()) {
    return result;
  } else {
    return static_cast<T>(0.0f);
  }
}

template <typename T>
void replace_denormals_and_nans(T* data, size_t size) {
  if (type_info<T>::smallest_normal() <= 0) {
    // There are no denormals for this type.
    return;
  }
  for (size_t i = 0; i < size; ++i) {
    if (std::abs(static_cast<float>(data[i])) >=
        type_info<T>::smallest_normal()) {
      // This is a normal float, or infinity.
    } else {
      data[i] = static_cast<T>(0);
    }
  }
}

inline void replace_denormals_and_nans(int4x2*, size_t) {}

// Fill `[data, data + size)` with uniform random bits, excluding denormals and
// NaNs for floating point types.
template <typename T, typename Rng>
void fill_random(T* data, size_t size, Rng& rng,
                 const quantization_params& params = {}) {
  fill_uniform_random_bits(data, size, rng);
  replace_denormals_and_nans(data, size);
}

// Fill random values of a type T into the buffer [data, data + size). If [min,
// max] exceeds the range of T, the data is filled with random bits.
template <typename T, typename Rng>
void fill_random(T* data, size_t size, Rng& rng, double min, double max,
                 const quantization_params& params = {}) {
  if (min <= type_info<T>::min() && max >= type_info<T>::max()) {
    // [min, max] exceeds the range of T, just fill it with random bits.
    fill_random(data, size, rng);
  } else {
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<int> dist(
          std::max<int>(round_float_to_int<int>(min), type_info<T>::min()),
          std::min<int>(round_float_to_int<int>(max), type_info<T>::max()));
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(rng));
      }
    } else {
      // Floating point types
      std::uniform_real_distribution<float> dist(static_cast<float>(min),
                                                 static_cast<float>(max));
      for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(rng));
      }
    }
  }
}

// A specialization of the above for int4x2, which needs to set two values at
// once.
template <typename Rng>
void fill_random(int4x2* data, size_t size, Rng& rng, double min, double max,
                 const quantization_params& = {}) {
  if (min <= type_info<int4x2>::min() && max >= type_info<int4x2>::max()) {
    fill_random(data, size, rng);
  } else {
    std::uniform_int_distribution<int> dist(
        std::max<int>(round_float_to_int<int>(min), type_info<int4x2>::min()),
        std::min<int>(round_float_to_int<int>(max), type_info<int4x2>::max()));
    for (size_t i = 0; i < size; ++i) {
      data[i] = int4x2{static_cast<int8_t>(dist(rng)),
                       static_cast<int8_t>(dist(rng))};
    }
  }
}

// For quantized data, convert the min/max to the unquantized type, and generate
// that.
template <typename T, typename Rng>
void fill_random(quantized<T>* data, size_t size, Rng& rng, double min,
                 double max, const quantization_params& params) {
  float q_min = std::ceil(fake_quantize(
      static_cast<float>(min), 1.0f / params.scale, params.zero_point));
  float q_max = std::floor(fake_quantize(
      static_cast<float>(max), 1.0f / params.scale, params.zero_point));
  using UT = typename unwrap_quantized<T>::type;
  fill_random(reinterpret_cast<UT*>(data), size, rng, q_min, q_max);
}

template <typename T, typename Rng>
void fill_random(quantized<T>* data, size_t size, Rng& rng,
                 const quantization_params&) {
  fill_random(data, size, rng);
}

// Specialization for quantized<int32_t> to avoid values that can't losslessly
// convert to/from floats.
template <typename Rng>
void fill_random(quantized<int32_t>* data, size_t size, Rng& rng,
                 const quantization_params& params) {
  fill_random(data, size, rng, -(1 << 15), (1 << 15), params);
}

// Return a single random value of type T.
template <typename T, typename Rng, typename... Args>
T random_value(Rng& rng, Args&&... args) {
  T result;
  fill_random(&result, 1, rng, std::forward<Args>(args)...);
  return result;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_RANDOM_H_
