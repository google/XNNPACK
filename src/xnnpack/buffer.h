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
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/datatype.h"
#include "xnnpack/math.h"

namespace xnnpack {

template <typename T>
class NumericLimits {
 public:
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
};

template <>
class NumericLimits<xnn_float16> {
 public:
  static xnn_float16 min() { return static_cast<xnn_float16>(-65504); }
  static xnn_float16 max() { return static_cast<xnn_float16>(65504); }
};

template <>
class NumericLimits<xnn_bfloat16> {
 public:
  static xnn_bfloat16 min() { return xnn_bfloat16_from_bits(0xff7f); }
  static xnn_bfloat16 max() { return xnn_bfloat16_from_bits(0x7f7f); }
};

template <typename T>
class NumericLimits<quantized<T>> {
 public:
  static quantized<T> min() {
    return {std::numeric_limits<T>::lowest()};
  }
  static quantized<T> max() {
    return {std::numeric_limits<T>::max()};
  }
};

struct PaddingBytes {
  size_t value;
};

// This is a container similar to std::vector, but it leaves the memory
// uninitialized, supports alignment.
// TODO: It would be good if this also managed padding in a way that allowed
// the client code to see the unpadded data, and the padding was hidden.
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

 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  Buffer() : data_(nullptr), size_(0) {}
  explicit Buffer(size_t size, PaddingBytes extra_bytes = {0})
      : data_(reinterpret_cast<T*>(
            allocate(size * sizeof(T) + extra_bytes.value))),
        size_(size) {}
  Buffer(size_t size, T value) : Buffer(size) {
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
    if (data_) free(data_);
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
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) = default;

  // Returns true if every stride is the product of the following extents, i.e.
  // the buffer can be interpreted as a flat array without considering the
  // strides.
  bool is_contiguous() const {
    size_t stride = 1;
    for (size_t i = rank(); i > 0; --i) {
      if (strides_[i - 1] != stride) {
        return false;
      }
      stride *= extents_[i - 1];
    }
    return true;
  }

  const index_type& extents() const { return extents_; }
  const index_type& strides() const { return strides_; }
  size_t extent(size_t dim) const { return extents_[dim]; }
  size_t stride(size_t dim) const { return strides_[dim]; }

  size_t rank() const { return extents_.size(); }
  bool empty() const { return begin_ >= end_; }

  // Returns a pointer to the element {0,...}
  T* base() { return begin_; }
  const T* base() const { return begin_; }

  // Form a reference to an element at a particular index.
  T& operator()(const index_type& indices) {
    return *(begin_ + flat_offset(indices));
  }
  const T& operator()(const index_type& indices) const {
    return *(begin_ + flat_offset(indices));
  }

  template <typename... Args>
  T& operator()(Args... args) {
    return operator()(index_type{static_cast<size_t>(args)...});
  }
  template <typename... Args>
  const T& operator()(Args... args) const {
    return operator()(index_type{static_cast<size_t>(args)...});
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
    return data_->size();
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

  // Reorder the dimensions to extents = {extent(i) for i in perm}, and similar
  // for strides.
  Tensor<T, Alignment> transpose(const std::vector<size_t>& perm) const {
    Tensor<T, Alignment> result(*this);
    result.extents_ = permute(perm, extents_);
    result.strides_ = permute(perm, strides_);
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
    for (size_t i = 0; i < rank(); ++i) {
      offsets[i] = begins[i] < 0 ? extents_[i] + begins[i] : begins[i];
      result.extents_[i] =
          (ends[i] <= 0 ? extents_[i] + ends[i] : ends[i]) - offsets[i];
    }

    result.begin_ = begin_ + flat_offset(offsets);
    result.end_ = result.begin_ + result.flat_offset(result.extents_);

    return result;
  }

  // Remove `pre` elements from the beginning of each dimension, and `post`
  // elements from the end of each dimension.
  Tensor<T, Alignment> crop_padding(const index_type& pre,
                                    const index_type& post) {
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

  // Copy the contents from other to this. The extents must match.
  void assign(const Tensor<T, Alignment>& other) {
    assert(extents_ == other.extents_);
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
  size_t flat_offset(const index_type& indices) const {
    assert(indices.size() == rank());
    size_t result = 0;
    for (size_t i = 0; i < rank(); ++i) {
      result += strides_[i] * indices[i];
    }
    return result;
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
std::vector<size_t> random_shape(Rng& rng, size_t rank, size_t min_dim = 1,
                                 size_t max_dim = 9) {
  std::uniform_int_distribution<size_t> dim_dist(min_dim, max_dim);
  std::vector<size_t> shape(rank);
  for (size_t i = 0; i < rank; ++i) {
    shape[i] = dim_dist(rng);
  }
  return shape;
}

// Generate random quantization parameters for a given datatype.
template <typename Rng>
xnn_quantization_params random_quantization(xnn_datatype datatype, Rng& rng) {
  std::uniform_int_distribution<> i8_dist{std::numeric_limits<int8_t>::min(),
                                          std::numeric_limits<int8_t>::max()};
  std::uniform_int_distribution<> u8_dist{std::numeric_limits<uint8_t>::min(),
                                          std::numeric_limits<uint8_t>::max()};
  std::uniform_real_distribution<float> scale_dist{0.1f, 10.0f};
  switch (datatype) {
    case xnn_datatype_qint8:
      return {i8_dist(rng), scale_dist(rng)};
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

// Make a generator of random values of a datatype T, suitable for use with
// std::generate/std::generate_n or similar.
template <typename T>
class DatatypeGenerator {
  std::uniform_real_distribution<float> dist_;

 public:
  DatatypeGenerator(float min, float max, const xnn_quantization_params& = {})
      : dist_(min, max) {}
  explicit DatatypeGenerator(const xnn_quantization_params& = {})
      : dist_(-1.0f, 1.0f) {}
  DatatypeGenerator() : dist_(-1.0f, 1.0f) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    return static_cast<T>(dist_(rng));
  }
};

template <typename T>
class DatatypeGenerator<quantized<T>> {
  std::uniform_int_distribution<int> dist_;

 public:
  DatatypeGenerator(float min, float max,
                    const xnn_quantization_params& params) {
    min = std::ceil(min / params.scale + params.zero_point);
    max = std::floor(max / params.scale + params.zero_point);
    dist_ = std::uniform_int_distribution<int>(static_cast<int>(min),
                                               static_cast<int>(max));
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

};  // namespace xnnpack

#endif  // __XNNPACK_TEST_BUFFER_H_
