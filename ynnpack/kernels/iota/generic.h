#ifndef XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ynnpack/base/simd/vec.h"

namespace ynn {

template <typename T, std::size_t... Is>
constexpr auto iota_impl(std::index_sequence<Is...>) {
  return std::array<T, sizeof...(Is)>{static_cast<T>(Is)...};
}

template <typename T, std::size_t N>
constexpr auto iota_impl() {
  return iota_impl<T>(std::make_index_sequence<N>{});
}

template <typename T, size_t N_>
static void iota_impl(size_t n, T begin, T stride, T* output) {
  constexpr std::integral_constant<size_t, N_> N = {};
  if (stride == 0) {
    auto tile = simd::broadcast<N>(static_cast<T>(begin));
    while (n >= N) {
      store(output, tile, N);
      output += N;
      n -= N;
    }
    store(output, tile, n);
  } else if constexpr (std::is_integral_v<T>) {
    static constexpr auto iota1_data = iota_impl<T, N>();

    // Make the first tile of iota.
    auto iota_i =
        simd::load(iota1_data.data(), N) * simd::broadcast<N>(stride) +
        simd::broadcast<N>(begin);
    auto step = simd::broadcast<N>(static_cast<int32_t>(N * stride));

    while (n >= N) {
      store(output, iota_i, N);
      output += N;
      iota_i += step;
      n -= N;
    }
    store(output, iota_i, n);
  } else {
    // Generating a numerically exact (iota * stride + begin) is difficult.
    // What we do here is compute cast<float>(iota) * stride + begin. This would
    // fail if the integer values overflow. We assume that `n` is less than
    // the max int value, because presumably this operation would be tiled
    // otherwise. This poses a different problem: the tiling operation
    // reassociates some of this arithmetic before we see it here (the `begin`
    // value has been adjusted to account for the beginning of the tile).
    assert(static_cast<size_t>(static_cast<int>(n)) == n);
    static constexpr auto iota1_data = iota_impl<int32_t, N>();

    // Make the first tile of iota.
    auto iota_i = simd::load(iota1_data.data(), N);
    auto step = simd::broadcast<N>(static_cast<int32_t>(N));
    auto stride_v = simd::broadcast<N>(stride);
    auto begin_v = simd::broadcast<N>(begin);

    while (n >= N) {
      store(output, simd::cast<T>(iota_i) * stride_v + begin_v, N);
      output += N;
      iota_i += step;
      n -= N;
    }
    store(output, simd::cast<T>(iota_i) * stride_v + begin_v, n);
  }
}

#define YNN_DEFINE_IOTA_KERNEL(arch, name, type, N)                          \
  void name(size_t n, const void* begin, const void* stride, void* output) { \
    iota_impl<type, N>(n, *static_cast<const type*>(begin),                  \
                       *static_cast<const type*>(stride),                    \
                       static_cast<type*>(output));                          \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_
