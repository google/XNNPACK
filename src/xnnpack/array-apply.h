#pragma once

#include <cstddef>
#include <array>

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

}  // namespace internal
}  // namespace xnnpack
