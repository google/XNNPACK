#pragma once

#include <cstddef>
#include <array>

namespace xnnpack {
namespace internal {
template <typename T, size_t N, typename F, size_t... Indx>
void ArrayApplyImpl(const std::array<T, N>& args, F&& f,
               std::integer_sequence<size_t, Indx...> seq) {
  f(args[Indx]...);
}

template <typename T, size_t N, typename F,
          typename Indx = std::make_index_sequence<N> >
void ArrayApply(const std::array<T, N>& args, F&& f) {
  return ArrayApplyImpl(args, f, Indx{});
}

}  // namespace internal
}  // namespace xnnpack
