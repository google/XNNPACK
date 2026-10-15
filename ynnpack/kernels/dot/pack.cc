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
void fill_interleaved(size_t factor, size_t m, size_t n, size_t /*stride_a*/,
                      const void* a, void* x) {
  T val;
  std::memcpy(&val, a, sizeof(T));
  T* x_t = static_cast<T*>(x);
  if (m == factor) {
    std::fill_n(x_t, n * factor, val);
  } else {
    for (size_t j = 0; j < n; ++j) {
      T* col = x_t + j * factor;
      std::fill_n(col, m, val);
      std::fill_n(col + m, factor - m, T{0});
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
               size_t tile_m, size_t tile_n)
    : stride_m(stride_m),
      stride_n(stride_n),
      elem_size_bits(elem_size_bits),
      tile_m(tile_m),
      tile_n(tile_n) {
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
  if (stride_m == 0 && stride_n == 0) {
    // The matrix is a broadcast in both dimensions. This is inefficient so
    // ideally would be an unused codepath, but we need it to support some use
    // cases.
    // TODO: b/517589771 - This could be avoided by rewriting dot (and reduce)
    // of broadcasts to be multiplication by the size of the reduction instead.
    input_stride = 0;
    interleave_fn = get_broadcast_interleave_kernel(elem_size_bits);
    return;
  }
  const size_t elem_size_bytes = std::max<size_t>(elem_size_bits / 8, 1);
  if (stride_m == elem_size_bytes) {
    // We have all 3 transposes.
    // We're transposing columns of the input to rows of the output, but doing
    // `tile_m` of them at a time. In this case, (2) and (3) are equivalent to
    // "bitcasting" the input to have elements that are `tile_m` x larger (and
    // `tile_m` x fewer of them), and then transposing that. (3) is handled by a
    // loop over calls to this kernel below.
    // Since we're tiling this ourselves, we don't need the added overhead of a
    // tiled transpose.
    transpose_fn = get_transpose_kernel(elem_size_bits * tile_m);
  }
  // We might need to do (2).
  // We're interleaving rows of the input to produce rows of the output.
  interleave_fn = get_interleave_kernel(elem_size_bits, tile_m);
  if (tile_m == 1) {
      // (2) is a no-op, we only need to do (3).
    transpose_blocks_fn = get_tiled_transpose(elem_size_bits * tile_n);
  }
}

void packer::pack(size_t m, size_t n, const void* input, size_t output_stride,
                  size_t output_block_stride, void* output) {
  const size_t elem_size_bytes = std::max<size_t>(elem_size_bits / 8, 1);
  const bool is_transpose =
      transpose_fn && (stride_m == elem_size_bytes) &&
      (stride_n != elem_size_bytes || n == 1);
  if (is_transpose) {
    const size_t input_stride_tr = stride_n;
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      transpose_fn(ceil_div(m, tile_m), n_i, m * elem_size_bits / 8,
                   input_stride_tr, input, output_stride, output);
      size_t elem_count = std::max<size_t>(8 / elem_size_bits, 1);
      input = offset_bytes(input, input_stride_tr * (tile_n / elem_count));
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else if (transpose_blocks_fn &&
             (m == 1 || output_stride == tile_n * elem_size_bits / 8) &&
             (ceil_div(n, tile_n) <= 1 || output_block_stride != 0)) {
    assert(tile_m == 1);
    transpose_blocks_fn(ceil_div(n, tile_n), m, n * elem_size_bits / 8,
                        stride_m, input, output_block_stride, output);
  } else if (interleave_fn) {
    const size_t input_stride_row = stride_m;
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
        interleave_fn(tile_m, m_i, n_i, input_stride_row, input_i, output_i);
        memset(offset_bytes(output_i, row_size), 0, padding_size);
        input_i = offset_bytes(input_i, input_stride_row * tile_m);
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
