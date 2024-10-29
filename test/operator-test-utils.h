// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_
#define _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_

#include <cstddef>
#include <random>

#include "replicable_random_device.h"

namespace xnnpack {

// Given ann xnnpack::Buffer<char> type, initialize it with
// the given datatype using the given RNG and distribution.
template <typename Buffer>
void randomize_buffer(xnn_datatype datatype,
                      xnnpack::ReplicableRandomDevice& rng,
                      std::uniform_real_distribution<double>& dist,
                      Buffer& buf) {
  const auto f = [&]() { return dist(rng); };
  switch (datatype) {
    case xnn_datatype_quint8:
      std::generate(reinterpret_cast<uint8_t*>(buf.begin()),
                    reinterpret_cast<uint8_t*>(buf.end()), f);
      break;
    case xnn_datatype_qint8:
      std::generate(reinterpret_cast<int8_t*>(buf.begin()),
                    reinterpret_cast<int8_t*>(buf.end()), f);
      break;
    case xnn_datatype_int32:
      std::generate(reinterpret_cast<int32_t*>(buf.begin()),
                    reinterpret_cast<int32_t*>(buf.end()), f);
      break;
    case xnn_datatype_fp16:
      std::generate(reinterpret_cast<xnn_float16*>(buf.begin()),
                    reinterpret_cast<xnn_float16*>(buf.end()), f);
      break;
    case xnn_datatype_fp32:
      std::generate(reinterpret_cast<float*>(buf.begin()),
                    reinterpret_cast<float*>(buf.end()), f);
      break;
    default:
      assert(false);
      break;
  }
}

}  // namespace xnnpack


#endif  // _XNNPACK_TEST_OPERATOR_TEST_UTILS_H_
