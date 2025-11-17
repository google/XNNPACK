#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_SWITCH_ELEMENT_SIZE_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_SWITCH_ELEMENT_SIZE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "ynnpack/base/base.h"
#include "ynnpack/base/type.h"

namespace ynn {

using x128_t = std::array<uint8_t, 16>;
using x256_t = std::array<uint8_t, 32>;
using x512_t = std::array<uint8_t, 64>;
using x1024_t = std::array<uint8_t, 128>;
using x2048_t = std::array<uint8_t, 256>;

template <typename F>
constexpr decltype(auto) switch_element_size(size_t element_size_bits, F&& f) {
  switch (element_size_bits) {
    case 4:
      return std::forward<F>(f)(uint4x2());
    case 8:
      return std::forward<F>(f)(uint8_t());
    case 16:
      return std::forward<F>(f)(uint16_t());
    case 32:
      return std::forward<F>(f)(uint32_t());
    case 64:
      return std::forward<F>(f)(uint64_t());
    case 128:
      return std::forward<F>(f)(x128_t());
    case 256:
      return std::forward<F>(f)(x256_t());
    case 512:
      return std::forward<F>(f)(x512_t());
    case 1024:
      return std::forward<F>(f)(x1024_t());
    case 2048:
      return std::forward<F>(f)(x2048_t());
    default:
      YNN_UNREACHABLE;
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_SWITCH_ELEMENT_SIZE_H_
