#ifndef XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_

#include <cassert>
#include <cstddef>
#include <cstring>

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
  packer(size_t stride_m, size_t stride_n, size_t elem_size_bits, size_t tile_m,
         size_t tile_n);

  // Run the packing operation for input and output buffers. The input has an
  // un-transposed shape of `m` x `n`. The output will be rounded up to a
  // multiple of the tile size.
  void pack(size_t m, size_t n, const void* input, size_t output_stride,
            size_t output_block_stride, void* output);

 protected:
  size_t stride_m;
  size_t stride_n;
  size_t elem_size_bits;
  size_t tile_m;
  size_t tile_n;
  size_t input_stride = 0;
  interleave_kernel_fn interleave_fn = nullptr;
  transpose_kernel_fn transpose_fn = nullptr;
  ynn::transpose_fn transpose_blocks_fn = nullptr;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_PACK_H_
