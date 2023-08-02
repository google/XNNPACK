// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

namespace xnnpack {
namespace internal {

template <typename F, typename R, typename... Args>
std::tuple<Args...> getArgsImpl(R (F::*)(Args...) const);

template <typename F>
decltype(getArgsImpl(&F::operator())) getArgsImpl(F);

template <typename F>
struct Args {
  using Type = decltype(getArgsImpl(std::declval<F>()));
};

template <typename F>
using ArgsT = typename Args<F>::Type;

template <typename... Args, typename F, size_t... Indx>
void TupleApplyImpl(std::tuple<Args...>&& args, F&& f,
                    std::integer_sequence<size_t, Indx...> seq) {
  f(std::move(std::get<Indx>(args))...);
}

template <typename... Args, typename F,
          typename Indx = std::make_index_sequence<sizeof...(Args)>>
void TupleApply(std::tuple<Args...>&& args, F&& f) {
  return TupleApplyImpl(std::move(args), f, Indx{});
}
}  // namespace internal
}  // namespace xnnpack
