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
#include <limits>
#include <random>
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
  RngT* data_rng_t = reinterpret_cast<RngT*>(data);
  size_t size_bytes = size * sizeof(T);
  size_t i = 0;
  // Fill with as many RngT as we can.
  for (; i + sizeof(RngT) <= size_bytes; i += sizeof(RngT)) {
    *data_rng_t++ = rng();
  }
  // Fill the remaining bytes.
  char* data_char = reinterpret_cast<char*>(data_rng_t);
  for (; i < size_bytes; ++i) {
    *data_char++ = rng() & 0xff;
  }
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

// Make a generator of random values of a type T, suitable for use with
// std::generate/std::generate_n or similar.
template <typename T>
class TypeGenerator {
  std::uniform_real_distribution<float> dist_;
  bool reinterpret_ = false;

 public:
  TypeGenerator(double min, double max, const quantization_params& = {}) {
    if (min <= type_info<T>::min() && max >= type_info<T>::max()) {
      // The caller wants a full range of random value. Rather than generate
      // floats uniformly distributed across the range of floats, where a
      // negligible fraction of the range contains the "interesting" region near
      // 0, we generate random bits reinterpreted as a float instead. This
      // distribution more accurately reflects the numbers we want to test in
      // typical code.
      reinterpret_ = true;
    } else {
      reinterpret_ = false;
      min = std::max<double>(min, type_info<T>::min());
      max = std::min<double>(max, type_info<T>::max());
      dist_ = std::uniform_real_distribution<float>(min, max);
    }
  }
  explicit TypeGenerator(const quantization_params& = {})
      : TypeGenerator(type_info<T>::min(), type_info<T>::max()) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    if (reinterpret_) {
      static_assert(Rng::min() == 0, "");
      static_assert(Rng::max() >= (1ull << (sizeof(T) * 8)) - 1, "");
      auto bits = rng();
      T result;
      memcpy(&result, &bits, sizeof(T));
      if (std::abs(static_cast<float>(result)) >=
          type_info<T>::smallest_normal()) {
        return result;
      } else {
        // Flush denormals (and NaN) to 0.
        return static_cast<T>(0.0f);
      }
    } else {
      return dist_(rng);
    }
  }
};

// This specialization for integers doesn't include the lowest negative integer,
// because testing it is a headache due to undefined behavior when negating it.
template <>
class TypeGenerator<int> {
  std::uniform_int_distribution<int> dist_;

 public:
  TypeGenerator(float min, float max, const quantization_params& = {})
      : dist_(std::max<int>(round_float_to_int<int>(min),
                            -std::numeric_limits<int>::max()),
              std::min<int>(round_float_to_int<int>(max),
                            std::numeric_limits<int>::max())) {}
  explicit TypeGenerator(const quantization_params& = {})
      : dist_(-std::numeric_limits<int>::max(),
              std::numeric_limits<int>::max()) {}

  template <typename Rng>
  int operator()(Rng& rng) {
    return dist_(rng);
  }
};

template <typename T>
class TypeGenerator<quantized<T>> {
  std::uniform_int_distribution<int> dist_;

 public:
  TypeGenerator(float min, float max, const quantization_params params = {}) {
    min = std::ceil(fake_quantize(min, 1.0f / params.scale, params.zero_point));
    max =
        std::floor(fake_quantize(max, 1.0f / params.scale, params.zero_point));
    dist_ = std::uniform_int_distribution<int>(round_float_to_int<T>(min),
                                               round_float_to_int<T>(max));
  }
  explicit TypeGenerator(const quantization_params& params)
      : TypeGenerator(-1.0f, 1.0f, params) {}
  TypeGenerator()
      : TypeGenerator(std::numeric_limits<T>::min(),
                      std::numeric_limits<T>::max()) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    return static_cast<T>(dist_(rng));
  }
};

// Specialize quantized int32 to avoid huge numbers that cannot losslessly
// convert to/from float.
template <>
class TypeGenerator<quantized<int32_t>> {
  std::uniform_int_distribution<int32_t> dist_;

 public:
  TypeGenerator(float min, float max, const quantization_params params = {}) {
    min = std::ceil(fake_quantize(min, 1.0f / params.scale, params.zero_point));
    max =
        std::floor(fake_quantize(max, 1.0f / params.scale, params.zero_point));
    dist_ = std::uniform_int_distribution<int>(
        round_float_to_int<int32_t>(min), round_float_to_int<int32_t>(max));
  }
  explicit TypeGenerator(const quantization_params& params)
      : TypeGenerator(-1.0f, 1.0f, params) {}
  TypeGenerator() : TypeGenerator(-(1 << 15), (1 << 15)) {}

  template <typename Rng>
  int32_t operator()(Rng& rng) {
    return static_cast<int32_t>(dist_(rng));
  }
};

template <>
class TypeGenerator<int4x2> {
  std::uniform_int_distribution<int> dist_;

 public:
  TypeGenerator(float min, float max, const quantization_params& = {})
      : dist_(std::max<int>(round_float_to_int<int>(min),
                            type_info<int4x2>::min()),
              std::min<int>(round_float_to_int<int>(max),
                            type_info<int4x2>::max())) {}
  explicit TypeGenerator(const quantization_params& = {})
      : dist_(type_info<int4x2>::min(), type_info<int4x2>::max()) {}

  template <typename Rng>
  int4x2 operator()(Rng& rng) {
    return {static_cast<int8_t>(dist_(rng)), static_cast<int8_t>(dist_(rng))};
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_RANDOM_H_
