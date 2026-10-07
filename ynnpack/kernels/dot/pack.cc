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

packer::packer(bool transpose, size_t elem_size_bits, size_t tile_m,
               size_t tile_n, bool allow_fused)
    : elem_size_bits(elem_size_bits), tile_m(tile_m), tile_n(tile_n) {
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
  if (transpose) {
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
      // Optional: if there is a kernel that can do the whole block in one
      // call, prefer it. `interleave_fn` stays populated as the fallback.
      if (allow_fused) {
        interleave_block_fn =
            get_interleave_block_kernel(elem_size_bits, tile_m);
      }
    }
  }
}

void packer::pack(size_t m, size_t n, size_t input_stride, const void* input,
                  size_t output_stride, size_t output_block_stride,
                  void* output) {
  if (transpose_blocks_fn) {
    assert(tile_m == 1);
    assert(m == 1 || output_stride == tile_n * elem_size_bits / 8);
    transpose_blocks_fn(ceil_div(n, tile_n), m, n * elem_size_bits / 8,
                        input_stride, input, output_block_stride, output);
  } else if (transpose_fn) {
    while (n > 0) {
      const size_t n_i = std::min(n, tile_n);
      transpose_fn(ceil_div(m, tile_m), n_i, m * elem_size_bits / 8,
                   input_stride, input, output_stride, output);
      size_t elem_count = std::max<size_t>(8 / elem_size_bits, 1);
      input = offset_bytes(input, input_stride * (tile_n / elem_count));
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else if (interleave_block_fn) {
    // Same as the `interleave_fn` case below, except the kernel owns the loop
    // over `m` and the column padding, so there is no per-row indirect call or
    // memset. Strip-mining `n` across `kChunkN` adjacent blocks and `m` in
    // 32-row panels keeps source cache lines and DTLB entries hot across
    // adjacent column blocks when `m` and `n` are large.
    constexpr size_t kChunkN = 4;
    const size_t chunk_m = tile_m * 16;
    const size_t elem_bytes = elem_size_bits / 8;
    const size_t blocks_n = ceil_div(n, tile_n);
    for (size_t bn0 = 0; bn0 < blocks_n; bn0 += kChunkN) {
      const size_t bn_end = std::min(bn0 + kChunkN, blocks_n);
      for (size_t m0 = 0; m0 < m; m0 += chunk_m) {
        const size_t m_i = std::min(m - m0, chunk_m);
        const size_t mo0 = m0 / tile_m;
        for (size_t bn = bn0; bn < bn_end; ++bn) {
          const size_t col0 = bn * tile_n;
          const size_t n_i = std::min(n - col0, tile_n);
          const void* in_tile =
              offset_bytes(input, m0 * input_stride + col0 * elem_bytes);
          void* out_tile = offset_bytes(
              output, bn * output_block_stride + mo0 * output_stride);
          interleave_block_fn(tile_m, m_i, n_i, tile_n, input_stride, in_tile,
                              output_stride, out_tile);
        }
      }
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
        interleave_fn(tile_m, m_i, n_i, input_stride, input_i, output_i);
        memset(offset_bytes(output_i, row_size), 0, padding_size);
        input_i = offset_bytes(input_i, input_stride * tile_m);
        output_i = offset_bytes(output_i, output_stride);
      }
      input = offset_bytes(input, tile_n * elem_size_bits / 8);
      output = offset_bytes(output, output_block_stride);
      n = sub_sat(n, tile_n);
    }
  } else {
    YNN_UNREACHABLE;
  }
}

}  // namespace ynn
