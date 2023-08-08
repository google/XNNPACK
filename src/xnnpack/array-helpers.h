#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace xnnpack {
namespace internal {
template <typename T, size_t N, typename F, size_t... Indx>
void ArrayApplyImpl(std::array<T, N>&& args, F&& f,
                    std::integer_sequence<size_t, Indx...> seq) {
  f(std::move(args[Indx])...);
}

template <typename T, size_t N, typename F,
          typename Indx = std::make_index_sequence<N>>
void ArrayApply(std::array<T, N>&& args, F&& f) {
  return ArrayApplyImpl(std::move(args), f, Indx{});
}

template <size_t... Is, typename V>
constexpr std::array<V, sizeof...(Is)> MakeArrayImpl(
    V value, std::integer_sequence<size_t, Is...>) {
  return {((void)Is, value)...};
}

template <size_t N, typename V>
constexpr std::array<V, N> MakeArray(V value) {
  return MakeArrayImpl(value, std::make_index_sequence<N>{});
}

template <typename T, size_t max_size>
class ArrayPrefix {
 public:
  constexpr ArrayPrefix(size_t size, T t)
      : size_(size), array_(MakeArray<max_size>(t)) {
    assert(size_ <= max_size);
  }

  explicit constexpr ArrayPrefix(size_t size) : size_(size) {
    assert(size_ <= max_size);
  }

  template <typename Array,
            typename = decltype(std::declval<Array>().begin())>
  explicit constexpr ArrayPrefix(Array&& array, T placeholder)
      : ArrayPrefix(0, placeholder) {
    for (const auto& v : array) {
      push_back(v);
    }
  }

  constexpr ArrayPrefix(std::initializer_list<T> init)
      : ArrayPrefix(init.size(), T{}) {
    assert(size_ <= max_size);
    std::copy(init.begin(), init.end(), begin());
  }

  auto begin() { return array_.begin(); }
  auto begin() const { return array_.cbegin(); }
  auto end() {
    auto result = array_.begin();
    std::advance(result, size_);
    return result;
  }
  auto end() const {
    auto result = array_.cbegin();
    std::advance(result, size_);
    return result;
  }
  auto& operator[](size_t index) {
    assert(index < size_);
    return array_[index];
  }
  const auto& operator[](size_t index) const {
    assert(index < size_);
    return array_[index];
  }
  void push_back(const T& t) {
    assert(size_ + 1 <= max_size);
    array_[size_++] = t;
  }
  template <typename... Args>
  void emplace_back(Args&&... args) {
    if (size_ + 1 <= max_size) {
      new (&array_[size_++]) T(args...);
    }
  }
  size_t size() const { return size_; }

 private:
  size_t size_;
  std::array<T, max_size> array_;
};
}  // namespace internal
}  // namespace xnnpack
