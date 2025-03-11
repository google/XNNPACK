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

namespace xnnpack {

template <typename T>
class NumericLimits {
 public:
  static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }
  static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
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
};

template <typename T>
class NumericLimits<quantized<T>> {
 public:
  static quantized<T> min() { return {std::numeric_limits<T>::lowest()}; }
  static quantized<T> max() { return {std::numeric_limits<T>::max()}; }
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
      }
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

  // This is a dangerous function, use carefully.
  void set_shape(index_type extents, index_type strides) {
    extents_ = std::move(extents);
    strides_ = std::move(strides);
  }

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
  // This allows broadcasting by according to numpy rules. The last index of
  // `indices` corresponds to the last dimension of this tensor. Missing
  // dimensions are treated as stride 0.
  size_t flat_offset(const index_type& indices) const {
    size_t result = 0;
    assert(indices.size() >= rank());
    for (size_t i = 0; i < rank(); ++i) {
      result += strides_[i] * indices[i + indices.size() - rank()];
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
xnn_quantization_params random_quantization(xnn_datatype datatype, Rng& rng) {
  std::uniform_int_distribution<> i8_dist{std::numeric_limits<int8_t>::min(),
                                          std::numeric_limits<int8_t>::max()};
  std::uniform_int_distribution<> u8_dist{std::numeric_limits<uint8_t>::min(),
                                          std::numeric_limits<uint8_t>::max()};
  std::uniform_real_distribution<float> scale_dist{0.25f, 8.0f};
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

 public:
  DatatypeGenerator(float min, float max, const xnn_quantization_params& = {})
      : dist_(std::max<float>(min, NumericLimits<T>::min()),
              std::min<float>(max, NumericLimits<T>::max())) {}
  explicit DatatypeGenerator(const xnn_quantization_params& = {})
      : dist_(NumericLimits<T>::min(), NumericLimits<T>::max()) {}

  template <typename Rng>
  T operator()(Rng& rng) {
    while (true) {
      float result = dist_(rng);
      if (!std::isnan(result)) {
        return static_cast<T>(result);
      } else {
        // Don't allow generating NaN
      }
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
