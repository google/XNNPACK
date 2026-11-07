#ifndef XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_

#include <cassert>
#include <cstddef>
#include <cstring>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

// Packing is the following reshape + transpose operation:
//
// packed(mi, ni, mo, no) = input(mo * tile_m + mi, no * tile_n + ni)
//
// Where the input is padded with zeros where it is not aligned to a multiple of
// tile_m, tile_n.
class packer {
 public:
  // Prepare to run a packing operation for the given packing parameters.
  // `stride_m` and `stride_n` are the strides in bytes of the two input
  // dimensions.
  // `tile_m` must be a power of 2.
  // `transpose` indicates the input is transposed, i.e. the `m` dimension is
  // contiguous. This is inferred from the strides when possible, but it must be
  // specified for sub-byte types, where the strides are ambiguous.
  packer(size_t stride_m, size_t stride_n, size_t elem_size_bits, size_t tile_m,
         size_t tile_n, size_t output_stride, bool transpose = false);

  // Run the packing operation for input and output buffers. The input has an
  // un-transposed shape of `m` x `n`. The output will be rounded up to a
  // multiple of the tile size.
  void pack(size_t m, size_t n, const void* input, size_t output_block_stride,
            void* output);

 protected:
  size_t stride_m;
  size_t stride_n;
  size_t elem_size_bits;
  size_t tile_m;
  size_t tile_n;
  size_t output_stride;
  size_t input_stride = 0;
  interleave_kernel_fn interleave_fn = nullptr;
  transpose_kernel_fn transpose_fn = nullptr;
  ynn::transpose_fn transpose_blocks_fn = nullptr;
};

// Packs panels of a dot operand (either A or B) across (k3, k2, k1) into a
// pre-allocated buffer using `packer`, zero-padding `k1` up to `tile_k` and the
// non-reduction panel dimension `p` (`m` for A, `n` for B) up to `tile_p`.
struct dot_packer {
  packer p;
  size_t tile_k;
  size_t tile_p;
  size_t ko_stride;
  void* buffer;

  dot_packer(size_t stride_k1, size_t stride_p, size_t elem_size_bits,
             size_t tile_k, size_t tile_p, void* buffer, bool transpose = false)
      : p(stride_k1, stride_p, elem_size_bits, tile_k, tile_p,
          /*output_stride=*/tile_p * tile_k * elem_size_bits / 8, transpose),
        tile_k(tile_k),
        tile_p(tile_p),
        ko_stride(tile_p * tile_k * elem_size_bits / 8),
        buffer(buffer) {
    assert(tile_p * tile_k * elem_size_bits % 8 == 0);
  }

  static size_t packed_bytes(size_t p_extent, span<const size_t> ks,
                             size_t elem_size_bits, size_t tile_k,
                             size_t tile_p) {
    assert(tile_p * tile_k * elem_size_bits % 8 == 0);
    const size_t k1 = !ks.empty() ? ks[0] : 1;
    const size_t k2 = ks.size() >= 2 ? ks[1] : 1;
    const size_t k3 = ks.size() >= 3 ? ks[2] : 1;
    const size_t ko_stride = tile_p * tile_k * elem_size_bits / 8;
    return k3 * k2 * ceil_div(p_extent, tile_p) * ceil_div(k1, tile_k) *
           ko_stride;
  }

  const void* pack(size_t p_extent, span<const size_t> ks, const void* src,
                   span<const size_t> src_k_strides, size_t& packed_stride_p,
                   size_t (&packed_k_strides)[3]);
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
