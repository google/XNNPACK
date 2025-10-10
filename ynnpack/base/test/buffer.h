// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TEST_BUFFER_H_
#define XNNPACK_YNNPACK_BASE_TEST_BUFFER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace ynn {

struct Alignment {
  size_t bytes = 1;
};

// This is a container similar to std::vector, but:
// - It is move-only
// - It does not initialize its memory. This helps detect bugs with msan, and
//   speeds up tests/benchmarks (especially on slow emulators).
// - It supports allocating some extra bytes past the end, but does not consider
//   those bytes to be part of the container (size() and end() do not include
//   these bytes).
template <typename T>
class Buffer {
  static_assert(std::is_trivial<T>::value, "");

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size, Alignment alignment = {})
      : data_(
            reinterpret_cast<T*>(allocate(size * sizeof(T), alignment.bytes))),
        size_(size) {}
  Buffer(size_t size, T value, Alignment alignment = {})
      : Buffer(size, alignment) {
    std::fill(begin(), end(), value);
  }
  Buffer(std::initializer_list<T> init, Alignment alignment = {})
      : Buffer(init.size(), alignment) {
    std::copy(init.begin(), init.end(), begin());
  }
  Buffer(const Buffer& other) = delete;
  Buffer(Buffer&& other) : Buffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
  ~Buffer() { free(data_); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& other) {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    return *this;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T* data() { return data_; }
  const T* data() const { return data_; }
  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  const T* cbegin() const { return data_; }
  const T* cend() const { return data_ + size_; }
  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

  bool operator==(const Buffer& other) const {
    return size_ == other.size_ && std::equal(begin(), end(), other.begin());
  }
  bool operator!=(const Buffer& other) const {
    return size_ != other.size_ || !std::equal(begin(), end(), other.begin());
  }

 private:
  static void* allocate(size_t bytes, size_t alignment) {
    alignment = std::max(alignment, sizeof(void*));
    bytes = std::max(bytes, alignment);
#if defined(_WIN32)
    void* memory = nullptr;
    memory = _aligned_malloc(bytes, alignment);
    if (memory == 0) {
#if !defined(__GNUC__) && !defined(_MSC_VER) || defined(__EXCEPTIONS) || \
    defined(_CPPUNWIND)
      throw std::bad_alloc();
#endif
    }
#elif defined(__ANDROID__) || defined(__CYGWIN__)
    void* memory = memalign(alignment, bytes);
    if (memory == 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc();
#endif
    }
#else
    void* memory = nullptr;
    if (posix_memalign(&memory, alignment, bytes) != 0) {
#if !defined(__GNUC__) || defined(__EXCEPTIONS)
      throw std::bad_alloc();
#endif
    }
#endif
    return reinterpret_cast<T*>(memory);
  }

  static void free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    ::free(p);
#endif
  }

  T* data_;
  size_t size_;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_BUFFER_H_
