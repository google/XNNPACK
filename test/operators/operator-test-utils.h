// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_
#define _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_

#include <cstddef>
#include <random>

#include "test/replicable_random_device.h"

namespace xnnpack {

template <typename T, typename Buffer>
void randomize_int_buffer(xnn_datatype datatype,
                          xnnpack::ReplicableRandomDevice& rng, double min,
                          double max, Buffer& buf) {
  std::uniform_int_distribution<int> dist(static_cast<int>(min),
                                          static_cast<int>(max));
  const auto f = [&]() { return static_cast<T>(dist(rng)); };
  std::generate(reinterpret_cast<T*>(buf.begin()),
                reinterpret_cast<T*>(buf.end()), f);
}

template <typename T, typename Buffer>
void randomize_float_buffer(xnn_datatype datatype,
                            xnnpack::ReplicableRandomDevice& rng, double min,
                            double max, Buffer& buf) {
  std::uniform_real_distribution<float> dist(static_cast<float>(min),
                                             static_cast<float>(max));
  const auto f = [&]() { return dist(rng); };
  std::generate(reinterpret_cast<T*>(buf.begin()),
                reinterpret_cast<T*>(buf.end()), f);
}

// Given ann xnnpack::Buffer<char> type, initialize it with
// the given datatype using the given RNG and distribution.
template <typename Buffer>
void randomize_buffer(xnn_datatype datatype,
                      xnnpack::ReplicableRandomDevice& rng, double min,
                      double max, Buffer& buf) {
  switch (datatype) {
    case xnn_datatype_quint8:
      randomize_int_buffer<uint8_t>(datatype, rng, min, max, buf);
      break;
    case xnn_datatype_qint8:
      randomize_int_buffer<int8_t>(datatype, rng, min, max, buf);
      break;
    case xnn_datatype_int32:
      randomize_int_buffer<int32_t>(datatype, rng, min, max, buf);
      break;
    case xnn_datatype_fp16:
      randomize_float_buffer<xnn_float16>(datatype, rng, min, max, buf);
      break;
    case xnn_datatype_bf16:
      randomize_float_buffer<xnn_bfloat16>(datatype, rng, min, max, buf);
      break;
    case xnn_datatype_fp32:
      randomize_float_buffer<float>(datatype, rng, min, max, buf);
      break;
    default:
      assert(false);
      break;
  }
}

}  // namespace xnnpack

#endif  // _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_
