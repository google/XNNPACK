// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_TEST_TENSOR_H_
#define XNNPACK_YNNPACK_BASE_TEST_TENSOR_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/buffer.h"

namespace ynn {

// This stores a multi-dimensional array in a Buffer<T> object
// (above). The sizes of dimensions are `extent`s, the distance between elements
// in a dimension in memory are `stride`s. The address of an element x is
// base() + sum(x[i]*stride(i) for i in rank)
// This buffer holds an std::shared_ptr to the underlying Buffer<T>
// objects, i.e. copies are shallow.
template <typename T>
class Tensor {
 public:
  using value_type = T;
  using iterator = typename Buffer<T>::iterator;
  using const_iterator = typename Buffer<T>::const_iterator;

  using index_type = std::vector<size_t>;

  Tensor() = default;
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) = default;
  // Constructs an array with strides in descending order, with no
  // padding/alignment between dimensions.
  explicit Tensor(index_type extents, Alignment alignment = {})
      : extents_(std::move(extents)), strides_(extents_.size()) {
    size_t stride = 1;
    for (size_t i = rank(); i > 0; --i) {
      strides_[i - 1] = stride;
      stride *= extents_[i - 1];
    }
    data_ = std::make_shared<Buffer<T>>(stride, alignment);
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

  YNN_ALWAYS_INLINE size_t rank() const { return extents_.size(); }
  YNN_ALWAYS_INLINE bool empty() const { return begin_ >= end_; }

  // Returns a pointer to the element {0,...}
  YNN_ALWAYS_INLINE T* base() { return begin_; }
  YNN_ALWAYS_INLINE const T* base() const { return begin_; }

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
  Tensor<T> transpose(std::vector<int32_t> perm) const {
    // Sort idx to get the new locations
    std::vector<int32_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());

    Tensor<T> result(*this);
    for (size_t i = 0; i < sorted.size(); i++) {
      result.extents_[sorted[i]] = extent(perm[i]);
      result.strides_[sorted[i]] = stride(perm[i]);
    }
    return result;
  }

  // Reshape such that the shape has extent 1 dimensions at `new_axes`
  // positions.
  Tensor<T> expand_dims(const std::vector<int32_t>& new_axes) const {
    Tensor<T> result(*this);
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
  Tensor<T> reshape(const std::vector<size_t>& new_extents) const {
    assert(is_contiguous());
    Tensor<T> result(*this);
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
  Tensor<T> slice(size_t dim, int64_t begin, int64_t end) const {
    assert(dim < rank());

    begin = begin < 0 ? extents_[dim] + begin : begin;
    end = end <= 0 ? extents_[dim] + end : end;

    begin = std::max<int64_t>(std::min<int64_t>(begin, extents_[dim]), 0);
    end = std::max<int64_t>(std::min<int64_t>(end, extents_[dim]), begin);

    Tensor<T> result(*this);
    result.extents_[dim] = end - begin;
    result.begin_ = begin_ + strides_[dim] * begin;
    result.end_ = begin_ + strides_[dim] * end;

    return result;
  }

  // This is similar to above, but slices all dimensions.
  Tensor<T> slice(const std::vector<int64_t>& begins,
                  const std::vector<int64_t>& ends) const {
    assert(rank() == begins.size());
    assert(rank() == ends.size());

    Tensor<T> result(*this);
    for (size_t i = 0; i < rank(); ++i) {
      result = result.slice(i, begins[i], ends[i]);
    }

    return result;
  }

  Tensor<T> slice(size_t dim, int64_t at) const {
    return slice(dim, at, at + 1);
  }

  // Slice the leading dimensions at the indices of `at`.
  Tensor<T> slice_leading(std::vector<size_t> at) const {
    Tensor<T> result(*this);
    for (size_t i = 0; i < at.size(); ++i) {
      result = result.slice(i, at[i], at[i] + 1);
    }
    return result;
  }

  Tensor<T> remove_dim(size_t dim) const {
    assert(dim < rank());
    Tensor<T> result(*this);
    result.extents_.erase(result.extents_.begin() + dim);
    result.strides_.erase(result.strides_.begin() + dim);
    return result;
  }

  Tensor<T> broadcast_like(const std::vector<size_t>& extents) {
    Tensor<T> result(*this);
    while (result.rank() < extents.size()) {
      result = result.expand_dims({0});
    }
    for (size_t i = 0; i < extents.size(); ++i) {
      if (result.extents_[i] == 1) {
        result.extents_[i] = extents[i];
        result.strides_[i] = 0;
      } else {
        assert(result.extents_[i] == extents[i]);
      }
    }
    return result;
  }

  // Split a dimension dim into dimensions of extent `split_extents`. The first
  // split extent of 0 will be replaced with
  // extent(dim) / product(non-zero split extents). The product of split_extents
  // must be equal to extent(dim).
  Tensor<T> split(size_t dim, std::vector<size_t> split_extents) const {
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
    std::vector<int32_t> new_dims(split_extents.size() - 1);
    std::iota(new_dims.begin(), new_dims.end(), dim + 1);
    Tensor<T> result = expand_dims(new_dims);
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
  Tensor<T> fuse(std::vector<size_t> dims) const {
    assert(!dims.empty());
    size_t a = dims.front();
    assert(a < rank());
    dims.erase(dims.begin());
    Tensor<T> result(*this);
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
  Tensor<T> crop_padding(const index_type& pre, const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    Tensor<T> result(*this);
    result.begin_ = begin_ + flat_offset(pre);
    index_type max(rank());
    for (size_t i = 0; i < rank(); ++i) {
      if (result.extents_[i] >= pre[i] + post[i]) {
        result.extents_[i] -= pre[i] + post[i];
        max[i] = result.extents_[i] - 1;
      } else {
        result.extents_[i] = 0;
      }
    }
    if (std::accumulate(result.extents_.begin(), result.extents_.end(),
                        static_cast<size_t>(1), std::multiplies<>()) == 0) {
      // The result is empty.
      result.end_ = result.begin_;
    } else {
      result.end_ = result.begin_ + result.flat_offset(max) + 1;
    }
    return result;
  }

  // Add `pre` indices before, `post` indices after, of padding of `value` to
  // the tensor.
  Tensor<T> pad(T value, const index_type& pre, const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    std::vector<size_t> extents = extents_;
    for (size_t i = 0; i < rank(); ++i) {
      extents[i] += pre[i] + post[i];
    }

    Tensor<T> result(extents);
    result.fill(value);
    result.crop_padding(pre, post).assign(*this);
    return result;
  }

  // Similar to the above, but repeats the edge value of the tensor instead of
  // padding with a constant value.
  Tensor<T> pad(const index_type& pre, const index_type& post) const {
    assert(rank() == pre.size());
    assert(rank() == post.size());

    std::vector<size_t> extents = extents_;
    for (size_t i = 0; i < rank(); ++i) {
      extents[i] += pre[i] + post[i];
    }

    Tensor<T> result(extents);
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
        Tensor<T> valid_pre = result.slice(dim, valid_begin, valid_begin + 1);
        valid_pre.set_shape(valid_pre.extents(), strides);
        Tensor<T> padding = result.slice(dim, 0, valid_begin);
        padding.assign(valid_pre);
      }

      if (post[dim] != 0) {
        // Copy the post padding.
        Tensor<T> valid_post = result.slice(dim, valid_end - 1, valid_end);
        valid_post.set_shape(valid_post.extents(), strides);
        result.slice(dim, valid_end, 0).assign(valid_post);
      }
    }
    return result;
  }

  // Copy the contents from other to this. The extents must match.
  void assign(const Tensor<T>& other) {
    assert(rank() == other.rank());
    for (size_t i = 0; i < rank(); ++i) {
      assert(other.stride(i) == 0 || other.extent(i) == extent(i));
    }
    copy_impl(rank(), extents_.data(), other.strides_.data(), other.base(),
              strides_.data(), base());
  }

  // Make a copy of the buffer. The result will be contiguous, i.e. the strides
  // of this buffer are lost when copying.
  Tensor<T> deep_copy(Alignment alignment = {}) const {
    Tensor<T> result(extents_, alignment);
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

  YNN_ALWAYS_INLINE size_t flat_offset_variadic(size_t /*dim0*/) const {
    return 0;
  }

  YNN_ALWAYS_INLINE size_t flat_offset_variadic(size_t dim0,
                                                size_t idx0) const {
    if (dim0 > 0) {
      // We need to skip the leading dimensions that are broadcasts.
      return 0;
    } else {
      // We now have as many indices as there are dimensions.
      return flat_offset_no_broadcast(strides_.data(), idx0);
    }
  }

  template <typename... Args>
  YNN_ALWAYS_INLINE size_t flat_offset_variadic(size_t dim0, size_t idx0,
                                                Args... idxs) const {
    if (dim0 > 0) {
      // We need to skip the leading dimensions that are broadcasts.
      return flat_offset_variadic(dim0 - 1, idxs...);
    } else {
      // We now have as many indices as there are dimensions.
      return flat_offset_no_broadcast(strides_.data(), idx0, idxs...);
    }
  }

  YNN_ALWAYS_INLINE size_t
  flat_offset_no_broadcast(const size_t* strides) const {
    return 0;
  }
  YNN_ALWAYS_INLINE size_t flat_offset_no_broadcast(const size_t* strides,
                                                    size_t idx0) const {
    return *strides * idx0;
  }
  template <typename... Args>
  YNN_ALWAYS_INLINE size_t flat_offset_no_broadcast(const size_t* strides,
                                                    size_t idx0,
                                                    Args... idxs) const {
    return *strides * idx0 + flat_offset_no_broadcast(strides + 1, idxs...);
  }

  index_type extents_;
  index_type strides_;
  std::shared_ptr<Buffer<T>> data_;
  T* begin_ = nullptr;
  T* end_ = nullptr;
};

template <typename T>
void broadcast_extent_1(Tensor<T>& tensor) {
  std::vector<size_t> strides = tensor.strides();
  for (size_t i = 0; i < tensor.rank(); i++) {
    strides[i] = tensor.extent(i) == 1 ? 0 : strides[i];
  }
  tensor.set_shape(tensor.extents(), std::move(strides));
}

// Like numpy.squeeze, remove dims of extent 1 from shape.
inline std::vector<size_t> squeeze(std::vector<size_t> shape) {
  shape.erase(std::remove_if(shape.begin(), shape.end(),
                             [](size_t x) { return x == 1; }),
              shape.end());
  return shape;
}

// Replace the dimension `dim` in x with two dimensions: a "spatial" dimension,
// and a "kernel" dimension. These dimensions "overlap" in memory, such that
// `result(di, i) = x(i * stride + di * dilation)`. The spatial dimension will
// be reduced in extent such that no padding is required.
template <typename T>
inline Tensor<T> make_stencil_dim(Tensor<T> x, int dim, int size,
                                  int stride = 1, int dilation = 1) {
  x = x.expand_dims({dim + 1});

  int dilated_kernel_size = (size - 1) * dilation + 1;

  std::vector<size_t> extents = x.extents();
  std::vector<size_t> strides = x.strides();
  extents[dim + 1] = size;
  strides[dim + 1] = strides[dim] * dilation;
  extents[dim] =
      std::max(0, static_cast<int>(extents[dim]) - (dilated_kernel_size - 1));
  extents[dim] = (extents[dim] + stride - 1) / stride;
  strides[dim] *= stride;
  std::swap(strides[dim], strides[dim + 1]);
  std::swap(extents[dim], extents[dim + 1]);
  x.set_shape(std::move(extents), std::move(strides));
  return x;
}

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

template <typename T, typename Scale, typename ZeroPoint>
Tensor<float> dequantize(const Tensor<T>& x, const Tensor<Scale>& scale,
                         const Tensor<ZeroPoint>& zero_point) {
  Tensor<float> result(x.extents());
  for (auto& i : EnumerateIndices(x.extents())) {
    result(i) = dequantize(x(i), scale(i), zero_point(i));
  }
  return result;
}

template <typename T, typename Scale, typename ZeroPoint>
Tensor<T> quantize(const Tensor<float>& x, const Tensor<Scale>& scale,
                   const Tensor<ZeroPoint>& zero_point) {
  Tensor<T> result(x.extents());
  for (auto& i : EnumerateIndices(x.extents())) {
    result(i) = quantize<T>(x(i), 1.0f / scale(i), zero_point(i));
  }
  return result;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_TEST_TENSOR_H_
