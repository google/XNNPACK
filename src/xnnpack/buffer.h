// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_TEST_BUFFER_H_
#define __XNNPACK_TEST_BUFFER_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/reference-utils.h"

#if XNN_COMPILER_HAS_FEATURE(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#define XNN_MSAN_POISON(x, size) __msan_poison(x, size)
#define XNN_MSAN_UNPOISON(x, size) __msan_unpoison(x, size)
#else
#define XNN_MSAN_POISON(x, size)
#define XNN_MSAN_UNPOISON(x, size)
#endif

#if XNN_COMPILER_HAS_FEATURE(address_sanitizer)
#include <sanitizer/asan_interface.h>
#define XNN_ASAN_POISON(x, size) __asan_poison_memory_region(x, size)
#define XNN_ASAN_UNPOISON(x, size) __asan_unpoison_memory_region(x, size)
#else
#define XNN_ASAN_POISON(x, size)
#define XNN_ASAN_UNPOISON(x, size)
#endif

namespace xnnpack {

template <typename T>
class NumericLimits {
 public:
  static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
  static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
  static constexpr T smallest_normal() { return std::numeric_limits<T>::min(); }

  // These are the identity values (values such that f(x, identity) = x) for a
  // min or max operation.
  static constexpr T min_identity() { return infinity(); }
  static constexpr T max_identity() { return -infinity(); }
};

template <>
class NumericLimits<xnn_float16> {
 public:
  static xnn_float16 epsilon() {
    return xnn_float16_from_bits(0x1400);  // 2^-10 = 0.0009765625
  }
  static xnn_float16 infinity() { return xnn_float16_from_bits(0x7c00); }
  static xnn_float16 min() { return xnn_float16_from_bits(0xfbff); }
  static xnn_float16 max() { return xnn_float16_from_bits(0x7bff); }
  static xnn_float16 smallest_normal() {
    return xnn_float16_from_bits(0x0400);  // 2^-14
  }
  static xnn_float16 min_identity() { return infinity(); }
  static xnn_float16 max_identity() { return -infinity(); }
};

template <>
class NumericLimits<xnn_bfloat16> {
 public:
  static xnn_bfloat16 epsilon() {
    return xnn_bfloat16_from_bits(0x3c00);  // 2^-7 = 0.0078125
  }
  static xnn_bfloat16 infinity() { return xnn_bfloat16_from_bits(0x7f80); }
  static xnn_bfloat16 min() { return xnn_bfloat16_from_bits(0xff7f); }
  static xnn_bfloat16 max() { return xnn_bfloat16_from_bits(0x7f7f); }
  static xnn_bfloat16 smallest_normal() {
    return xnn_bfloat16_from_bits(0x0080);  // 2^-126
  }
  static xnn_bfloat16 min_identity() { return infinity(); }
  static xnn_bfloat16 max_identity() { return -infinity(); }
};

template <typename T, typename Kind>
class NumericLimits<quantized<T, Kind>> {
 public:
  static quantized<T> min() { return {std::numeric_limits<T>::lowest()}; }
  static quantized<T> max() { return {std::numeric_limits<T>::max()}; }
  static quantized<T> smallest_normal() { return {0}; }
  static quantized<T> min_identity() { return max(); }
  static quantized<T> max_identity() { return min(); }
};

inline float epsilon(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_fp32:
      return NumericLimits<float>::epsilon();
    case xnn_datatype_fp16:
      return NumericLimits<xnn_float16>::epsilon();
    case xnn_datatype_bf16:
      return NumericLimits<xnn_bfloat16>::epsilon();
    default:
      return 1.0f;
  }
}

template <typename T>
T get_reduce_identity(xnn_reduce_operator op) {
  switch (op) {
    case xnn_reduce_sum:
    case xnn_reduce_mean:
      return 0;
    case xnn_reduce_max:
      return NumericLimits<T>::max_identity();
    case xnn_reduce_min:
      return NumericLimits<T>::min_identity();
    case xnn_reduce_invalid:
      break;
  }
  XNN_UNREACHABLE;
  return 0;
}

struct PaddingBytes {
  int value;
};

static constexpr PaddingBytes XnnExtraBytes = {XNN_EXTRA_BYTES};

// This is a container similar to std::vector, but:
// - It is move-only
// - It does not initialize its memory. This helps detect bugs with msan, and
//   speeds up tests/benchmarks (especially on slow emulators).
// - It supports allocating some extra bytes past the end, but does not consider
//   those bytes to be part of the container (size() and end() do not include
//   these bytes).
// - It allocates extra bytes before and after the buffer to detect memory
//   corruption. This is necessary because we have some code that only runs on
//   targets where we have no other reasonable way to detect out of bounds
//   memory writes (e.g. arm 32-bit assembly, which doesn't support asan).
template <typename T, size_t Alignment = alignof(T)>
class Buffer {
  static_assert(std::is_trivial<T>::value, "");
  T* data_;
  size_t size_;

  static void* allocate(size_t bytes) {
    size_t alignment = std::max(Alignment, sizeof(void*));
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

  // Some compilers can't handle static constexpr member variables.
  enum { guard_bytes = std::max<size_t>(64, Alignment) };
  enum { guard_signal = 0xB8AEBCB293DCA04F };

  static void fill_guard_bytes(uint8_t* x) {
    assert(guard_bytes % sizeof(guard_signal) == 0);
    const auto guard_signal_with_addr = guard_signal;
    for (size_t i = 0; i < guard_bytes; i += sizeof(guard_signal)) {
      memcpy(x + i, &guard_signal_with_addr, sizeof(guard_signal));
    }
  }

  static bool check_guard_bytes(const uint8_t* x) {
    assert(guard_bytes % sizeof(guard_signal) == 0);
    const auto guard_signal_with_addr = guard_signal;
    for (size_t i = 0; i < guard_bytes; i += sizeof(guard_signal)) {
      if (memcmp(x + i, &guard_signal_with_addr, sizeof(guard_signal)) != 0) {
        return false;
      }
    }
    return true;
  }

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size, PaddingBytes extra_bytes = {0})
      : data_(reinterpret_cast<T*>(
            allocate(size * sizeof(T) + guard_bytes +
                     std::max<size_t>(guard_bytes, extra_bytes.value)))),
        size_(size) {
    // Fill the region before the allocation with the guard bytes, and poison it
    // for sanitizers.
    uint8_t* before = reinterpret_cast<uint8_t*>(data_);
    fill_guard_bytes(before);
    XNN_MSAN_POISON(before, guard_bytes);
    XNN_ASAN_POISON(before, guard_bytes);

    // The actual allocation is after the guard bytes.
    data_ =
        reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(data_) + guard_bytes);

    // Fill the region after the allocation with the guard bytes.
    uint8_t* after = reinterpret_cast<uint8_t*>(data_ + size_);
    fill_guard_bytes(after);
    // Here, we only want to poison the memory for sanitizers after the extra
    // bytes. We want to allow reads past the end to not crash (in asan or
    // otherwise), but we don't want that memory to be considered initialized by
    // msan.
    XNN_ASAN_POISON(after + extra_bytes.value, guard_bytes - extra_bytes.value);
    XNN_MSAN_POISON(after, guard_bytes);
  }
  Buffer(size_t size, T value, PaddingBytes extra_bytes = {0})
      : Buffer(size, extra_bytes) {
    std::fill(begin(), end(), value);
  }
  Buffer(std::initializer_list<T> init) : Buffer(init.size()) {
    std::copy(init.begin(), init.end(), begin());
  }
  Buffer(const Buffer& other) = delete;
  Buffer(Buffer&& other) : Buffer() {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
  ~Buffer() {
    if (data_) {
      // Check that the guard bytes after the buffer have not been modified. We
      // need to unpoison the memory first so we can read it.
      const uint8_t* after = reinterpret_cast<uint8_t*>(data_ + size_);
      XNN_ASAN_UNPOISON(after, guard_bytes);
      XNN_MSAN_UNPOISON(after, guard_bytes);
      if (!check_guard_bytes(after)) {
        std::cerr << "guard bytes after allocation were corrupted" << std::endl;
        std::abort();
      }
      data_ =
          reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(data_) - guard_bytes);

      // Check that the guard bytes before the buffer have not been modified.
      const uint8_t* before = reinterpret_cast<uint8_t*>(data_);
      XNN_ASAN_UNPOISON(before, guard_bytes);
      XNN_MSAN_UNPOISON(before, guard_bytes);
      if (!check_guard_bytes(before)) {
        std::cerr << "guard bytes before allocation were corrupted"
                  << std::endl;
        std::abort();
      }

      free(data_);
    }
  }

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
};

// This is a faster way of generating random numbers, by generating as many
// random values as possible for each call to rng(). Assumes that rng() returns
// entirely random bits.
template <typename T, typename Rng>
void fill_uniform_random_bits(T* data, size_t size, Rng& rng) {
  using RngT = decltype(rng());
  RngT* data_rng_t = reinterpret_cast<RngT*>(data);
  size_t size_bytes = size * sizeof(T);
  size_t i = 0;
  // Fill with as many RngT as we can.
  for (; i + sizeof(RngT) <= size_bytes; i += sizeof(RngT)) {
    *data_rng_t++ = rng();
  }
  // Fill the remaining bytes.
  char* data_char = reinterpret_cast<char*>(data_rng_t);
  for (; i < size_bytes; ++i) {
    *data_char++ = rng() & 0xff;
  }
}

// Returns {x[i] for i in perm}
template <typename T>
std::vector<T> permute(const std::vector<size_t>& perm,
                       const std::vector<T>& x) {
  std::vector<T> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    result[i] = x[perm[i]];
  }
  return result;
}

// This stores a multi-dimensional array in a Buffer<T, Alignment> object
// (above). The sizes of dimensions are `extent`s, the distance between elements
// in a dimension in memory are `stride`s. The address of an element x is
// base() + sum(x[i]*stride(i) for i in rank)
// This buffer holds an std::shared_ptr to the underlying Buffer<T, Alignment>
// objects, i.e. copies are shallow.
template <typename T, size_t Alignment = alignof(T)>
class Tensor {
 public:
  using value_type = T;
  using iterator = typename xnnpack::Buffer<T>::iterator;
  using const_iterator = typename xnnpack::Buffer<T>::const_iterator;

  using index_type = std::vector<size_t>;

  Tensor() = default;
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) = default;
  // Constructs an array with strides in descending order, with no
  // padding/alignment between dimensions.
  explicit Tensor(index_type extents, PaddingBytes extra_bytes = {0})
      : extents_(std::move(extents)), strides_(extents_.size()) {
    size_t stride = 1;
    for (size_t i = rank(); i > 0; --i) {
      strides_[i - 1] = stride;
      stride *= extents_[i - 1];
    }
    data_ = std::make_shared<Buffer<T, Alignment>>(stride, extra_bytes);
    begin_ = data_->begin();
    end_ = data_->end();
  }
  template <size_t N>
  explicit Tensor(const std::array<size_t, N>& extents,
                  PaddingBytes extra_bytes = {0})
      : Tensor(index_type(extents.begin(), extents.end()), extra_bytes) {}
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) = default;

  // Returns true if every stride is the product of the following extents, i.e.
  // the buffer can be interpreted as a flat array without considering the
  // strides.
  bool is_contiguous() const {
    size_t stride = 1;
    for (size_t i = rank(); i > 0; --i) {
      if (extents_[i - 1] == 1) {
        // We don't care about the stride of extent 1 dimensions.
        continue;
      } else if (extents_[i - 1] == 0) {
        // Tensor is empty, it's contiguous.
        return true;
      }
      if (strides_[i - 1] != stride) {
        return false;
      }
      stride *= extents_[i - 1];
    }
    return true;
  }

  const index_type& extents() const { return extents_; }
  const index_type& shape() const { return extents_; }
  const index_type& strides() const { return strides_; }
  size_t extent(size_t dim) const { return extents_[dim]; }
  size_t stride(size_t dim) const { return strides_[dim]; }

  // This is a dangerous function, use carefully.
  void set_shape(index_type extents, index_type strides) {
    extents_ = std::move(extents);
    strides_ = std::move(strides);
  }

  XNN_INLINE size_t rank() const { return extents_.size(); }
  XNN_INLINE bool empty() const { return begin_ >= end_; }

  // Returns a pointer to the element {0,...}
  XNN_INLINE T* base() { return begin_; }
  XNN_INLINE const T* base() const { return begin_; }

  // Form a reference to an element at a particular index.
  T& operator()(const index_type& indices) {
    return *(begin_ + flat_offset(indices));
  }
  const T& operator()(const index_type& indices) const {
    return *(begin_ + flat_offset(indices));
  }

  template <typename... Args>
  T& operator()(Args... args) {
    assert(sizeof...(args) >= rank());
    return *(begin_ + flat_offset_variadic(sizeof...(args) - rank(), args...));
  }
  template <typename... Args>
  const T& operator()(Args... args) const {
    assert(sizeof...(args) >= rank());
    return *(begin_ + flat_offset_variadic(sizeof...(args) - rank(), args...));
  }

  // The following functions produce iterators or accessors to the "flat" array
  // in memory, and can only be used if `is_contiguous()` is true.
  T* data() {
    assert(is_contiguous());
    return begin_;
  }
  const T* data() const {
    assert(is_contiguous());
    return begin_;
  }
  size_t size() const {
    assert(is_contiguous());
    return end_ - begin_;
  }
  T* begin() { return data(); }
  T* end() { return end_; }
  const T* begin() const { return data(); }
  const T* end() const { return end_; }
  const T* cbegin() const { return data(); }
  const T* cend() const { return end_; }
  T& operator[](size_t index) { return data()[index]; }
  const T& operator[](size_t index) const { return data()[index]; }

  // The following manipulators only affect the strides or extents of the
  // Tensor, they do not affect the memory addressed by the Tensor. To realize
  // the effect of these operations, make a copy with `deep_copy`.

  // Reorder the dimensions in `dims`. Dimensions not in dims maintain their
  // relative ordering.
  Tensor<T, Alignment> transpose(std::vector<size_t> perm) const {
    // Sort idx to get the new locations
    std::vector<size_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());

    Tensor<T, Alignment> result(*this);
    for (size_t i = 0; i < sorted.size(); i++) {
      result.extents_[sorted[i]] = extent(perm[i]);
      result.strides_[sorted[i]] = stride(perm[i]);
    }
    return result;
  }

  // Reshape such that the shape has extent 1 dimensions at `new_axes`
  // positions.
  Tensor<T, Alignment> expand_dims(const std::vector<size_t>& new_axes) const {
    Tensor<T, Alignment> result(*this);
    size_t new_rank = rank() + new_axes.size();
    result.extents_.resize(new_rank);
    result.strides_.resize(new_rank);
    size_t i = 0;
    for (size_t j = 0; j < new_rank; ++j) {
      if (std::find(new_axes.begin(), new_axes.end(), j) != new_axes.end()) {
        result.extents_[j] = 1;
        result.strides_[j] = 0;
      } else {
        result.extents_[j] = extents_[i];
        result.strides_[j] = strides_[i];
        ++i;
      }
    }
    return result;
  }

  // Reshape the tensor to the given extents, assuming that the tensor is
  // contiguous, and produces a tensor with contiguous strides.
  Tensor<T, Alignment> reshape(const std::vector<size_t>& new_extents) const {
    assert(is_contiguous());
    Tensor<T, Alignment> result(*this);
    result.extents_ = new_extents;
    size_t stride = 1;
    result.strides_.resize(new_extents.size());
    for (size_t i = new_extents.size(); i > 0; --i) {
      result.strides_[i - 1] = stride;
      stride *= new_extents[i - 1];
    }
    assert(stride == size());
    return result;
  }

  // This uses the same rules for indexing as numpy, i.e. negative numbers are
  // offset are added to the extents.
  Tensor<T, Alignment> slice(const std::vector<int64_t>& begins,
                             const std::vector<int64_t>& ends) const {
    assert(rank() == begins.size());
    assert(rank() == ends.size());

    Tensor<T, Alignment> result(*this);
    std::vector<size_t> offsets(rank());
    std::vector<size_t> maxs(rank());
    for (size_t i = 0; i < rank(); ++i) {
      offsets[i] = begins[i] < 0 ? extents_[i] + begins[i] : begins[i];
      result.extents_[i] = std::max<int64_t>(
          0, (ends[i] <= 0 ? static_cast<int64_t>(extents_[i]) + ends[i]
                           : ends[i]) -
                 static_cast<int64_t>(offsets[i]));
      maxs[i] = doz(result.extents_[i], 1);
    }

    result.begin_ = begin_ + flat_offset(offsets);
    result.end_ = result.begin_ + result.flat_offset(maxs) + 1;

    return result;
  }

  // This is similar to the above, but only slices one dimension.
  Tensor<T, Alignment> slice(size_t dim, int64_t begin, int64_t end) const {
    assert(dim < rank());

    begin = begin < 0 ? extents_[dim] + begin : begin;
    end = end <= 0 ? extents_[dim] + end : end;

    Tensor<T, Alignment> result(*this);
    result.extents_[dim] = end - begin;
    result.begin_ = begin_ + strides_[dim] * begin;
    result.end_ = begin_ + strides_[dim] * end;

    return result;
  }

  Tensor<T, Alignment> slice(size_t dim, int64_t at) const {
    return slice(dim, at, at + 1);
  }

  // Slice the leading dimensions at the indices of `at`.
  Tensor<T, Alignment> slice_leading(std::vector<size_t> at) const {
    std::vector<int64_t> begins(rank());
    std::vector<int64_t> ends(rank());
    std::copy(at.begin(), at.end(), begins.begin());
    std::copy(at.begin(), at.end(), ends.begin());
    for (size_t i = 0; i < at.size(); ++i) {
      ends[i] += 1;
    }
    return slice(begins, ends);
  }

  // Split a dimension dim into dimensions of extent `split_extents`. The first
  // split extent of 0 will be replaced with
  // extent(dim) / product(non-zero split extents). The product of split_extents
  // must be equal to extent(dim).
  Tensor<T, Alignment> split(size_t dim,
                             std::vector<size_t> split_extents) const {
    assert(dim < rank());
    size_t splits_size = 1;
    for (size_t i : split_extents) {
      if (i != 0) {
        splits_size *= i;
      }
    }
    for (size_t& i : split_extents) {
      if (i == 0) {
        assert(extent(dim) % splits_size == 0);
        i = extent(dim) / splits_size;
        splits_size *= i;
      }
    }
    assert(splits_size == extent(dim) || stride(dim) == 0);
    std::vector<size_t> new_dims(split_extents.size() - 1);
    std::iota(new_dims.begin(), new_dims.end(), dim + 1);
    Tensor<T, Alignment> result = expand_dims(new_dims);
    for (size_t i = 0; i < split_extents.size(); ++i) {
      result.extents_[dim + i] = split_extents[i];
      splits_size /= split_extents[i];
      result.strides_[dim + i] = stride(dim) * splits_size;
    }
    return result;
  }

  // Fuse two dimensions into one, where the new dimension's extent is the
  // product of the extents of the two dimensions. The stride of the outer
  // dimension must match the product of the stride and extent of the inner
  // dimension.
  Tensor<T, Alignment> fuse(std::vector<size_t> dims) const {
    assert(!dims.empty());
    size_t a = dims.front();
    assert(a < rank());
    dims.erase(dims.begin());
    Tensor<T, Alignment> result(*this);
    for (size_t b : dims) {
      assert(b < rank());
      assert(stride(b) * extent(b) == stride(a));
      result.extents_[a] *= result.extent(b);
      result.strides_[a] = result.stride(b);
      result.extents_.erase(result.extents_.begin() + b);
      result.strides_.erase(result.strides_.begin() + b);
    }
    return result;
  }

  // Remove `pre` elements from the beginning of each dimension, and `post`
  // elements from the end of each dimension.
  Tensor<T, Alignment> crop_padding(const index_type& pre,
                                    const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    Tensor<T, Alignment> result(*this);
    result.begin_ = begin_ + flat_offset(pre);
    for (size_t i = 0; i < rank(); ++i) {
      result.extents_[i] -= pre[i] + post[i];
    }
    result.end_ = result.begin_ + result.flat_offset(result.extents_);
    return result;
  }

  // Add `pre` indices before, `post` indices after, of padding of `value` to
  // the tensor.
  Tensor<T, Alignment> pad(T value, const index_type& pre,
                           const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    std::vector<size_t> extents = extents_;
    for (size_t i = 0; i < rank(); ++i) {
      extents[i] += pre[i] + post[i];
    }

    Tensor<T, Alignment> result(extents);
    result.fill(value);
    result.crop_padding(pre, post).assign(*this);
    return result;
  }

  // Similar to the above, but repeats the edge value of the tensor instead of
  // padding with a constant value.
  Tensor<T, Alignment> pad(const index_type& pre,
                           const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    std::vector<size_t> extents = extents_;
    for (size_t i = 0; i < rank(); ++i) {
      extents[i] += pre[i] + post[i];
    }

    Tensor<T, Alignment> result(extents);
    result.crop_padding(pre, post).assign(*this);
    // Implementing "repeat edge" is tricky. For each dimension, we need to
    // slice the padding in that dimension, and copy from the edge of the valid
    // data. This starts by copying junk padding from the result, but each
    // dimension fills in more of this junk data (the regions in the "corners"
    // gets copied more than once).
    for (size_t dim = 0; dim < rank(); ++dim) {
      int64_t valid_begin = pre[dim];
      int64_t valid_end = valid_begin + extents_[dim];
      // Make a broadcasting set of strides in this dimension.
      std::vector<size_t> strides = result.strides();
      strides[dim] = 0;

      if (pre[dim] != 0) {
        // Copy the pre-padding.
        Tensor<T, Alignment> valid_pre =
            result.slice(dim, valid_begin, valid_begin + 1);
        valid_pre.set_shape(valid_pre.extents(), strides);
        Tensor<T, Alignment> padding = result.slice(dim, 0, valid_begin);
        padding.assign(valid_pre);
      }

      if (post[dim] != 0) {
        // Copy the post padding.
        Tensor<T, Alignment> valid_post =
            result.slice(dim, valid_end - 1, valid_end);
        valid_post.set_shape(valid_post.extents(), strides);
        result.slice(dim, valid_end, 0).assign(valid_post);
      }
    }
    return result;
  }

  // Copy the contents from other to this. The extents must match.
  void assign(const Tensor<T, Alignment>& other) {
    assert(rank() == other.rank());
    for (size_t i = 0; i < rank(); ++i) {
      assert(other.stride(i) == 0 || other.extent(i) == extent(i));
    }
    copy_impl(rank(), extents_.data(), other.strides_.data(), other.base(),
              strides_.data(), base());
  }

  // Make a copy of the buffer. The result will be contiguous, i.e. the strides
  // of this buffer are lost when copying.
  Tensor<T, Alignment> deep_copy(PaddingBytes extra_bytes = {0}) const {
    Tensor<T, Alignment> result(extents_, extra_bytes);
    result.assign(*this);
    return result;
  }

  // Fill each element of this tensor with the output of g() (compare to
  // std::generate).
  template <typename G>
  void generate(const G& g) {
    generate_impl(rank(), extents_.data(), strides_.data(), base(), g);
  }

  // Fill each element of this tensor with the given value (compare to
  // std::fill).
  void fill(T value) {
    generate([=]() { return value; });
  }

 private:
  static void copy_impl(size_t rank, const size_t* extents,
                        const size_t* src_strides, const T* src,
                        const size_t* dst_strides, T* dst) {
    if (rank == 0) {
      *dst = *src;
      return;
    } else {
      --rank;
      size_t extent = *extents++;
      size_t src_stride = *src_strides++;
      size_t dst_stride = *dst_strides++;
      if (rank == 0 && src_stride == 1 && dst_stride == 1) {
        std::copy_n(src, extent, dst);
      } else if (rank == 0 && src_stride == 0 && dst_stride == 1) {
        std::fill_n(dst, extent, *src);
      } else {
        for (size_t i = 0; i < extent; ++i) {
          copy_impl(rank, extents, src_strides, src, dst_strides, dst);
          src += src_stride;
          dst += dst_stride;
        }
      }
    }
  }

  template <typename G>
  static void generate_impl(size_t rank, const size_t* extents,
                            const size_t* strides, T* dst, const G& g) {
    if (rank == 0) {
      *dst = g();
      return;
    } else {
      --rank;
      size_t extent = *extents++;
      size_t stride = *strides++;
      if (rank == 0 && stride == 1) {
        std::generate_n(dst, extent, g);
      } else {
        for (size_t i = 0; i < extent; ++i) {
          generate_impl(rank, extents, strides, dst, g);
          dst += stride;
        }
      }
    }
  }

  // Compute the offset of an index from the pointer to element 0.
  // This allows broadcasting by according to numpy rules. The last index of
  // `indices` corresponds to the last dimension of this tensor. Missing
  // dimensions are treated as stride 0.
  size_t flat_offset(const index_type& indices) const {
    const size_t rank = this->rank();
    const size_t indices_rank = indices.size();
    assert(indices_rank >= rank);
    size_t result = 0;
    const size_t* strides = strides_.data();
    const size_t* indices_offset = indices.data() + indices_rank - rank;
    for (size_t i = 0; i < rank; ++i) {
      result += strides[i] * indices_offset[i];
    }
    return result;
  }

  XNN_INLINE size_t flat_offset_variadic(size_t /*dim0*/) const { return 0; }

  XNN_INLINE size_t flat_offset_variadic(size_t dim0, size_t idx0) const {
    if (dim0 > 0) {
      // We need to skip the leading dimensions that are broadcasts.
      return 0;
    } else {
      // We now have as many indices as there are dimensions.
      return flat_offset_no_broadcast(strides_.data(), idx0);
    }
  }

  template <typename... Args>
  XNN_INLINE size_t flat_offset_variadic(size_t dim0, size_t idx0,
                                         Args... idxs) const {
    if (dim0 > 0) {
      // We need to skip the leading dimensions that are broadcasts.
      return flat_offset_variadic(dim0 - 1, idxs...);
    } else {
      // We now have as many indices as there are dimensions.
      return flat_offset_no_broadcast(strides_.data(), idx0, idxs...);
    }
  }

  XNN_INLINE size_t flat_offset_no_broadcast(const size_t* strides) const {
    return 0;
  }
  XNN_INLINE size_t flat_offset_no_broadcast(const size_t* strides,
                                             size_t idx0) const {
    return *strides * idx0;
  }
  template <typename... Args>
  XNN_INLINE size_t flat_offset_no_broadcast(const size_t* strides, size_t idx0,
                                             Args... idxs) const {
    return *strides * idx0 + flat_offset_no_broadcast(strides + 1, idxs...);
  }

  index_type extents_;
  index_type strides_;
  std::shared_ptr<xnnpack::Buffer<T, Alignment>> data_;
  T* begin_ = nullptr;
  T* end_ = nullptr;
};

// Generate a random shape of the given rank, where each dim is in [min_dim,
// max_dim].
template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t rank, size_t min_dim,
                                 size_t max_dim) {
  std::uniform_int_distribution<size_t> dim_dist(min_dim, max_dim);
  std::vector<size_t> shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    shape[i] = dim_dist(rng);
  }
  return shape;
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t rank) {
  return random_shape(rng, rank, 1, 9);
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng, size_t min_dim, size_t max_dim) {
  std::uniform_int_distribution<size_t> rank_dist(0, XNN_MAX_TENSOR_DIMS - 1);
  return random_shape(rng, rank_dist(rng), min_dim, max_dim);
}

template <typename Rng>
std::vector<size_t> random_shape(Rng& rng) {
  return random_shape(rng, 1, 9);
}

// Like numpy.squeeze, remove dims of extent 1 from shape.
inline std::vector<size_t> squeeze(std::vector<size_t> shape) {
  shape.erase(std::remove_if(shape.begin(), shape.end(),
                             [](size_t x) { return x == 1; }),
              shape.end());
  return shape;
}

template <typename T>
void broadcast_extent_1(Tensor<T>& tensor) {
  std::vector<size_t> strides = tensor.strides();
  for (size_t i = 0; i < tensor.rank(); i++) {
    strides[i] = tensor.extent(i) == 1 ? 0 : strides[i];
  }
  tensor.set_shape(tensor.extents(), std::move(strides));
}

// Generate random quantization parameters for a given datatype.
template <typename Rng>
xnn_quantization_params random_quantization(xnn_datatype datatype, Rng& rng,
                                            float min_scale = 0.25f,
                                            float max_scale = 8.0f) {
  std::uniform_int_distribution<> u8_dist{std::numeric_limits<uint8_t>::min(),
                                          std::numeric_limits<uint8_t>::max()};
  std::uniform_real_distribution<float> scale_dist{min_scale, max_scale};
  switch (datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint4:
      // signed integer quantization assumes zero point is 0.
      return {0, scale_dist(rng)};
    case xnn_datatype_quint8:
      return {u8_dist(rng), scale_dist(rng)};
    default:
      return {0, 1.0f};
  }
}

// Convert a floating point value to a quantized type T.
template <typename T>
T quantize(float value, const xnn_quantization_params& params) {
  using Unwrapped = typename xnnpack::unwrap_quantized<T>::type;
  const float min = NumericLimits<Unwrapped>::min();
  const float max = NumericLimits<Unwrapped>::max();
  return static_cast<T>(std::lrintf(
      std::max(std::min(value / params.scale + params.zero_point, max), min)));
}

// Convert a floating point value to an "idealized" quantized value, still
// represented as a float.
inline float fake_quantize(float value, const xnn_quantization_params& params) {
  return std::round(value / params.scale + params.zero_point);
}

template <typename T>
float dequantize(T x, xnn_quantization_params params) {
  return (static_cast<float>(x) - params.zero_point) * params.scale;
}

// Make a generator of random values of a datatype T, suitable for use with
// std::generate/std::generate_n or similar.
template <typename T>
class DatatypeGenerator {
  std::uniform_real_distribution<float> dist_;
  bool reinterpret_ = false;

 public:
  DatatypeGenerator(float min, float max, const xnn_quantization_params& = {}) {
    if (min <= NumericLimits<T>::min() && max >= NumericLimits<T>::max()) {
      // The caller wants a full range of random value. Rather than generate
      // floats uniformly distributed across the range of floats, where a
      // negligible fraction of the range contains the "interesting" region near
      // 0, we generate random bits reinterpreted as a float instead. This
      // distribution more accurately reflects the numbers we want to test in
      // typical code.
      reinterpret_ = true;
    } else {
      reinterpret_ = false;
      dist_ = std::uniform_real_distribution<float>(min, max);
    }
  }
  DatatypeGenerator(const xnn_quantization_params& = {})
      : DatatypeGenerator(NumericLimits<T>::min(), NumericLimits<T>::max()) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    if (reinterpret_) {
      static_assert(Rng::min() == 0, "");
      static_assert(Rng::max() >= (1ull << (sizeof(T) * 8)) - 1, "");
      auto bits = rng();
      T result;
      memcpy(&result, &bits, sizeof(T));
      if (std::abs(static_cast<float>(result)) >
          NumericLimits<T>::smallest_normal()) {
        return result;
      } else {
        // Flush denormals (and NaN) to 0.
        return static_cast<T>(0.0f);
      }
    } else {
      return dist_(rng);
    }
  }
};

// This specialization for integers doesn't include the lowest negative integer,
// because testing it is a headache due to undefined behavior when negating it.
template <>
class DatatypeGenerator<int> {
  std::uniform_int_distribution<int> dist_;

 public:
  DatatypeGenerator(float min, float max, const xnn_quantization_params& = {})
      : dist_(std::max<int>(round_float_to_int<int>(min),
                            -std::numeric_limits<int>::max()),
              std::min<int>(round_float_to_int<int>(max),
                            std::numeric_limits<int>::max())) {}
  explicit DatatypeGenerator(const xnn_quantization_params& = {})
      : dist_(-std::numeric_limits<int>::max(),
              std::numeric_limits<int>::max()) {}

  template <typename Rng>
  int operator()(Rng& rng) {
    return dist_(rng);
  }
};

template <typename T>
class DatatypeGenerator<quantized<T>> {
  std::uniform_int_distribution<int> dist_;

 public:
  DatatypeGenerator(float min, float max,
                    const xnn_quantization_params& params) {
    min = std::ceil(fake_quantize(min, params));
    max = std::floor(fake_quantize(max, params));
    dist_ = std::uniform_int_distribution<int>(round_float_to_int<T>(min),
                                               round_float_to_int<T>(max));
  }
  explicit DatatypeGenerator(const xnn_quantization_params& params)
      : DatatypeGenerator(-1.0f, 1.0f, params) {}
  DatatypeGenerator()
      : dist_(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    return static_cast<T>(dist_(rng));
  }
};

namespace internal {

class IndexIterator {
  std::vector<size_t> extents_;
  std::vector<size_t> i_;

 public:
  static IndexIterator make_end(std::vector<size_t> extents) {
    IndexIterator result;
    result.extents_ = std::move(extents);
    if (result.extents_.empty()) {
      // This is a bit of a hack. For rank 0 shapes, we need a way of separating
      // "begin" from "end". We call "begin" {}, which is the rank 0 index.
      // "end" is {1}.
      result.i_ = {1};
    } else {
      // The "end" iterator is when we reach one past the end of the first
      // dimension, and the rest of the dimensions are 0.
      result.i_ = std::vector<size_t>(result.extents_.size(), 0);
      result.i_.front() = result.extents_.front();
    }
    return result;
  }
  static IndexIterator make_begin(std::vector<size_t> extents) {
    size_t size =
        std::accumulate(extents.begin(), extents.end(), static_cast<size_t>(1),
                        std::multiplies<size_t>());
    if (size == 0) {
      return make_end(extents);
    }
    IndexIterator result;
    result.extents_ = std::move(extents);
    result.i_ = std::vector<size_t>(result.extents_.size(), 0);
    return result;
  }

  IndexIterator& operator++() {
    if (i_.empty()) {
      i_.push_back(1);
    } else {
      i_.back() += 1;
      for (size_t d = i_.size() - 1; d > 0; --d) {
        if (i_[d] >= extents_[d]) {
          ++i_[d - 1];
          i_[d] = 0;
        } else {
          break;
        }
      }
    }
    return *this;
  }

  IndexIterator operator++(int) {
    IndexIterator result = *this;
    ++(*this);
    return result;
  }

  const std::vector<size_t>& operator*() const { return i_; }
  const std::vector<size_t>* operator->() const { return &i_; }

  bool operator!=(const IndexIterator& other) const { return i_ != other.i_; }
};

class IndexRange {
  IndexIterator begin_;
  IndexIterator end_;

 public:
  using value_type = std::vector<size_t>;

  IndexRange(IndexIterator begin, IndexIterator end)
      : begin_(std::move(begin)), end_(std::move(end)) {}

  const IndexIterator& begin() const { return begin_; }
  const IndexIterator& end() const { return end_; }
};

}  // namespace internal

// Enumerate the indices in a multidimensional range [0, extents)
// This is very inefficient, it is intended for use in tests, where a more
// efficient approach would result in calling ASSERT_* in another function.
inline internal::IndexRange EnumerateIndices(
    const std::vector<size_t>& extents) {
  return internal::IndexRange(internal::IndexIterator::make_begin(extents),
                              internal::IndexIterator::make_end(extents));
}

inline std::string index_to_string(const std::vector<size_t>& v) {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < v.size(); i++) {
    ss << v[i];
    if (i + 1 < v.size()) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

}  // namespace xnnpack

#endif  // __XNNPACK_TEST_BUFFER_H_
