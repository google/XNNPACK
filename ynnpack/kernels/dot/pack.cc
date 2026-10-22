#include "ynnpack/kernels/dot/pack.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

namespace {

template <typename T>
void fill_interleaved(size_t factor, size_t m, size_t n, size_t stride_a,
                      const void* a, void* x) {
  if (n == 0) return;
  T* x_t = static_cast<T*>(x);
  const char* a_bytes = static_cast<const char*>(a);
  if (m == factor && stride_a == 0) {
    T val;
    std::memcpy(&val, a, sizeof(T));
    std::fill_n(x_t, n * factor, val);
  } else {
    for (size_t i = 0; i < m; ++i) {
      std::memcpy(&x_t[i], a_bytes + i * stride_a, sizeof(T));
    }
    std::fill_n(x_t + m, factor - m, T{0});
    for (size_t j = 1; j < n; ++j) {
      std::memcpy(x_t + j * factor, x_t, factor * sizeof(T));
    }
  }
}

interleave_kernel_fn get_broadcast_interleave_kernel(size_t elem_size_bits) {
  switch (elem_size_bits) {
    case 8:
      return fill_interleaved<uint8_t>;
    case 16:
      return fill_interleaved<uint16_t>;
    case 32:
      return fill_interleaved<uint32_t>;
    case 64:
      return fill_interleaved<uint64_t>;
    default:
      // The element size cannot be broadcasted.
      YNN_UNREACHABLE;
  }
}

}  // namespace

packer::packer(size_t stride_m, size_t stride_n, size_t elem_size_bits,
               size_t tile_m, size_t tile_n, size_t output_stride)
    : stride_m(stride_m),
      stride_n(stride_n),
      elem_size_bits(elem_size_bits),
      tile_m(tile_m),
      tile_n(tile_n),
      output_stride(output_stride) {
  // This operation is fusing 3 separate transposes (with padding as needed):
  //
  // 1. (optional) the caller might want to transpose the input prior to the
  // subsequent transposes (`transpose` = true). We assume this transpose is of
  // `tile_m` x elements at a time.
  // 2. A small transpose of `tile_m` x elements at a time
  // 3. A larger transpose of `tile_m * tile_m` x elements at a time
  //
  // If each transpose has an element size that is a multiple of the previous
  // transpose's element size, those transposes partially "cancel out",
  // resulting in a different transpose instead.
  if (stride_n == 0) {
    // The matrix is a broadcast across the columns. This is inefficient so
    // ideally would be an unused codepath, but we need it to support some use
    // cases.
    // TODO: b/517589771 - This could be avoided by rewriting dot (and reduce)
    // of broadcasts to be multiplication by the size of the reduction instead.
    input_stride = 0;
    interleave_fn = get_broadcast_interleave_kernel(elem_size_bits);
    return;
  }
  const size_t elem_size_bytes = elem_size_bits / 8;
  if (stride_m == elem_size_bytes && tile_n == 1 &&
      output_stride == elem_size_bytes * tile_m) {
    // This is a transpose of a row or column that remains contiguous.
    transpose_fn = [](size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                      const void* a, size_t stride_x, void* x) {
      assert(n == 1);
      const size_t copy_bytes = std::min(n_bytes_a, m * stride_x);
      std::memcpy(x, a, copy_bytes);
      if (m * stride_x > copy_bytes) {
        std::memset(static_cast<char*>(x) + copy_bytes, 0,
                    m * stride_x - copy_bytes);
      }
    };
  } else if (stride_m == elem_size_bytes &&
             (stride_n != elem_size_bytes || tile_n == 1)) {
    // We have all 3 transposes.
    // We're transposing columns of the input to rows of the output, but doing
    // `tile_m` of them at a time. In this case, (2) and (3) are equivalent to
    // "bitcasting" the input to have elements that are `tile_m` x larger (and
    // `tile_m` x fewer of them), and then transposing that. (3) is handled by a
    // loop over calls to this kernel below.
    // Since we're tiling this ourselves, we don't need the added overhead of a
    // tiled transpose.
    transpose_fn = get_transpose_kernel(elem_size_bits * tile_m);
    assert(transpose_fn);
  } else {
    // We only have (2) and (3).
    if (tile_m == 1) {
      // (2) is a no-op, we only need to do (3).
      transpose_blocks_fn = get_tiled_transpose(elem_size_bits * tile_n);
      assert(transpose_blocks_fn);
    } else {
      // We need to do (2).
      // We're interleaving rows of the input to produce rows of the output.
      interleave_fn = get_interleave_kernel(elem_size_bits, tile_m);
      assert(interleave_fn);
    }
  }
}

void packer::pack(size_t m, size_t n, const void* input,
                  size_t output_block_stride, void* output) {
  if (transpose_blocks_fn) {
    assert(tile_m == 1);
    const size_t n_bytes = n * elem_size_bits / 8;
    if (output_stride == tile_n * elem_size_bits / 8) {
      transpose_blocks_fn(ceil_div(n, tile_n), m, n_bytes, stride_m, input,
                          output_block_stride, output);
    } else {
      for (size_t i = 0; i < m; ++i) {
        transpose_blocks_fn(ceil_div(n, tile_n), 1, n_bytes, stride_m, input,
                            output_block_stride, output);
        input = offset_bytes(input, stride_m);
        output = offset_bytes(output, output_stride);
      }
    }
  } else if (transpose_fn) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      transpose_fn(ceil_div(m, tile_m), n_i, m * elem_size_bits / 8, stride_n,
                   input, output_stride, output);
      size_t elem_count = std::max<size_t>(8 / elem_size_bits, 1);
      input = offset_bytes(input, stride_n * (tile_n / elem_count));
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else if (interleave_fn) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      // In each row, we have a range that we produce via
      // interleaving, which handles padding of rows, but we also have
      // padding in the columns, which the interleave kernel does not
      // handle. Here we compute the size produced by interleaving
      // (including the padded rows), and then the size of the padding
      // in each row, which we set to 0.
      const size_t row_size = n_i * tile_m * elem_size_bits / 8;
      const size_t padding_size = (tile_n - n_i) * tile_m * elem_size_bits / 8;
      const void* input_i = input;
      void* output_i = output;
      for (size_t i = 0; i < m; i += tile_m) {
        const size_t m_i = std::min(m - i, tile_m);
        interleave_fn(tile_m, m_i, n_i, stride_m, input_i, output_i);
        memset(offset_bytes(output_i, row_size), 0, padding_size);
        input_i = offset_bytes(input_i, stride_m * tile_m);
        output_i = offset_bytes(output_i, output_stride);
      }
      if (stride_n != 0) {
        input = offset_bytes(input, tile_n * elem_size_bits / 8);
      }
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else {
    YNN_UNREACHABLE;
  }
}

}  // namespace ynn
