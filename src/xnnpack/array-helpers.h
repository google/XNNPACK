#pragma once

#include <array>
#include <cstddef>

namespace xnnpack {
namespace internal {
template <typename T, size_t N, typename F, size_t... Indx>
void ArrayApplyImpl(std::array<T, N>&& args, F&& f,
                    std::integer_sequence<size_t, Indx...> seq) {
  f(std::move(args[Indx])...);
}

template <typename T, size_t N, typename F,
          typename Indx = std::make_index_sequence<N> >
void ArrayApply(std::array<T, N>&& args, F&& f) {
  return ArrayApplyImpl(std::move(args), f, Indx{});
}

template <size_t... Is, typename V>
std::array<V, sizeof...(Is)> MakeArrayImpl(V value,
                                       std::integer_sequence<size_t, Is...>) {
  return {((void)Is, value)...};
}

template <size_t N, typename V>
std::array<V, N> MakeArray(V value) {
  return MakeArrayImpl(value, std::make_index_sequence<N>{});
}
}  // namespace internal
}  // namespace xnnpack
