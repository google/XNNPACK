#include "ynnpack/kernels/dot/pack.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

packer::packer(bool transpose, size_t elem_size, size_t tile_m, size_t tile_n)
    : elem_size(elem_size), log2_tile_m(std::log2(tile_m)), tile_n(tile_n) {
  assert(tile_m == 1 << log2_tile_m);
  const size_t elem_size_bits = elem_size * 8;
  if (transpose) {
    // We're transposing columns of the input to rows of the output, but doing
    // tile_m of them at a time.
    this->transpose = get_transpose_kernel(elem_size_bits << log2_tile_m);
  } else {
    if (log2_tile_m == 0) {
      // This is just a transpose of entire blocks at once. Try to get a
      // kernel for that.
      transpose_blocks = get_transpose_kernel(elem_size_bits * tile_n);
    }
    if (!transpose_blocks) {
      // We're interleaving rows of the input to produce rows of the output.
      interleave = get_interleave_kernel(elem_size_bits, 1 << log2_tile_m);
    }
  }
}

void packer::pack(size_t m, size_t n, size_t input_stride, const void* input,
                  size_t output_stride, size_t output_block_stride,
                  void* output) {
  const size_t tile_m = 1 << log2_tile_m;
  if (transpose_blocks) {
    assert(tile_m == 1);
    assert(m == 1 || output_stride == elem_size * tile_n);
    transpose_blocks(ceil_div(n, tile_n), m, n * elem_size, input_stride, input,
                     output_block_stride, output);
  } else if (transpose) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      transpose(ceil_div(m, tile_m), n_i, m * elem_size, input_stride, input,
                output_stride, output);
      input = offset_bytes(input, input_stride * tile_n);
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else if (interleave) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      // In each row, we have a range that we produce via
      // interleaving, which handles padding of rows, but we also have
      // padding in the columns, which the interleave kernel does not
      // handle. Here we compute the size produced by interleaving
      // (including the padded rows), and then the size of the padding
      // in each row, which we set to 0.
      const size_t row_size = n_i * tile_m * elem_size;
      const size_t padding_size = (tile_n - n_i) * tile_m * elem_size;
      const void* input_i = input;
      void* output_i = output;
      for (size_t i = 0; i < m; i += tile_m) {
        const size_t m_i = std::min(m - i, tile_m);
        interleave(tile_m, m_i, n_i, input_stride, input_i, output_i);
        memset(offset_bytes(output_i, row_size), 0, padding_size);
        input_i = offset_bytes(input_i, input_stride * tile_m);
        output_i = offset_bytes(output_i, output_stride);
      }
      input = offset_bytes(input, elem_size * tile_n);
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else {
    YNN_UNREACHABLE;
  }
}

}  // namespace ynn
