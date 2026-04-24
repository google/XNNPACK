#ifndef XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "ynnpack/base/simd/vec.h"

namespace ynn {

// For floating point types, we tile by this factor to make the result
// numerically consistent.
constexpr size_t consistent_tile_n = 16;

template <typename T, std::size_t... Is>
constexpr auto iota_impl(std::index_sequence<Is...>) {
  return std::array<T, sizeof...(Is)>{static_cast<T>(Is)...};
}

template <typename T, std::size_t N>
constexpr auto iota_impl() {
  return iota_impl<T>(std::make_index_sequence<N>{});
}

template <typename Vec>
static void iota_impl(size_t n, typename Vec::value_type begin,
                      typename Vec::value_type stride,
                      typename Vec::value_type* output) {
  using T = typename Vec::value_type;
  constexpr std::integral_constant<size_t, Vec::N> N = {};
  static constexpr auto iota1_data = iota_impl<T, N>();

  // Make the first tile of iota.
  auto tile = simd::load(iota1_data.data(), N);
  tile *= simd::broadcast<N>(static_cast<T>(stride));
  tile += simd::broadcast<N>(static_cast<T>(begin));

  // How much to add between each tile.
  auto step = simd::broadcast<N>(static_cast<T>(stride * N));

  if (stride == 0) {
    while (n >= N) {
      store(output, tile, N);
      output += N;
      n -= N;
    }
  } else {
    while (n >= N) {
      store(output, tile, N);
      tile += step;
      output += N;
      n -= N;
    }
  }
  store(output, tile, n);
}

#define YNN_DEFINE_IOTA_KERNEL(arch, name, type)                             \
  void name(size_t n, const void* begin, const void* stride, void* output) { \
    iota_impl<simd::vec<type, consistent_tile_n>>(                           \
        n, *static_cast<const type*>(begin),                                 \
        *static_cast<const type*>(stride), static_cast<type*>(output));      \
  }

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_IOTA_GENERIC_H_
