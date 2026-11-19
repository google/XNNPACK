#include "ynnpack/kernels/dot/pack.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/span.h"
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
               size_t tile_m, size_t tile_n, size_t output_stride,
               bool transpose)
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
  // Note that for sub-byte elements, a stride in bytes can never be equal to
  // the element size, so we can only detect that the input is transposed via
  // the `transpose` flag.
  const bool m_contiguous = stride_m * 8 == elem_size_bits;
  const bool n_contiguous = stride_n * 8 == elem_size_bits;
  if (m_contiguous && tile_n == 1 &&
      output_stride * 8 == elem_size_bits * tile_m) {
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
  } else if (transpose || (m_contiguous && (!n_contiguous || tile_n == 1))) {
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
      const size_t m_tiles = ceil_div(m, tile_m);
      transpose_fn(m_tiles, n_i, ceil_div(m * elem_size_bits, 8), stride_n,
                   input, output_stride, output);
      if (n_i < tile_n) {
        // The transpose only produces `n_i` columns of each row, set the
        // padding columns to 0. This should not be necessary, but the kernels
        // do process this data. We never care about the results in these
        // columns, but on some hardware, denormals/NaNs in this data causes the
        // dot kernel to be very slow.
        const size_t row_size = n_i * tile_m * elem_size_bits / 8;
        const size_t padding_size =
            (tile_n - n_i) * tile_m * elem_size_bits / 8;
        assert(m_tiles == 1 || row_size + padding_size <= output_stride);
        for (size_t i = 0; i < m_tiles; ++i) {
          memset(offset_bytes(output, i * output_stride + row_size), 0,
                 padding_size);
        }
      }
      input = offset_bytes(input, stride_n * tile_n);
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

const void* dot_packer::pack(size_t p_extent, span<const size_t> ks,
                             const void* src, span<const size_t> src_k_strides,
                             size_t& packed_stride_p,
                             size_t (&packed_k_strides)[3]) {
  const size_t k1 = !ks.empty() ? ks[0] : 1;
  const size_t k2 = ks.size() >= 2 ? ks[1] : 1;
  const size_t k3 = ks.size() >= 3 ? ks[2] : 1;
  const size_t block_stride = ceil_div(k1, tile_k) * ko_stride;
  const size_t p_blocks = ceil_div(p_extent, tile_p);

  packed_stride_p = block_stride / tile_p;
  packed_k_strides[0] = ko_stride / tile_k;
  packed_k_strides[1] = p_blocks * block_stride;
  packed_k_strides[2] = k2 * packed_k_strides[1];

  const size_t src_stride_k2 = src_k_strides.size() >= 2 ? src_k_strides[1] : 0;
  const size_t src_stride_k3 = src_k_strides.size() >= 3 ? src_k_strides[2] : 0;
  for (size_t i3 = 0; i3 < k3; ++i3) {
    for (size_t i2 = 0; i2 < k2; ++i2) {
      p.pack(k1, p_extent,
             offset_bytes(src, i3 * src_stride_k3 + i2 * src_stride_k2),
             block_stride,
             offset_bytes(buffer,
                          i3 * packed_k_strides[2] + i2 * packed_k_strides[1]));
    }
  }
  return buffer;
}

}  // namespace ynn
