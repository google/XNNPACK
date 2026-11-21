// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/slinky.h"

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "slinky/runtime/buffer.h"

namespace ynn {
namespace {

constexpr uintptr_t kBaseX = 0x10000;
constexpr uintptr_t kBaseA = 0x20000;
constexpr uintptr_t kBaseB = 0x30000;

slinky::raw_buffer make_raw_buffer(uintptr_t base, size_t elem_size,
                                   size_t rank, slinky::dim* dims) {
  slinky::raw_buffer buf;
  buf.base = reinterpret_cast<void*>(base);
  buf.elem_size = elem_size;
  buf.rank = rank;
  buf.dims = dims;
  return buf;
}

TEST(FuseAndSliceLeadingDims, LeadingPointDimWithScalarBroadcast) {
  // Pattern produced by elementwise ops after a keep_dims reduction along
  // dim 0, e.g., divide(1.0f, reduce_sum(x, axis=0, keep_dims=true)).
  slinky::dim x_storage[3] = {
      slinky::dim::broadcast(),
      slinky::dim(16, 143, 4),
      slinky::dim(3, 3, 512),
  };
  slinky::dim b_storage[3] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 255, 4),
      slinky::dim(0, 7, 512),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 3, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 0, nullptr);
  slinky::raw_buffer b = make_raw_buffer(kBaseB, 4, 3, b_storage);

  slinky::dim x_dims[2], a_dims[2], b_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a, b_dims, b));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 1);
  EXPECT_EQ(a_dims[0].stride(), 0);
  EXPECT_EQ(b_dims[0].stride(), 4);
  EXPECT_EQ(x.rank, 0);
  EXPECT_EQ(a.base, reinterpret_cast<void*>(kBaseA));
  EXPECT_EQ(b.base, reinterpret_cast<void*>(kBaseB + 16 * 4 + 3 * 512));
}

TEST(FuseAndSliceLeadingDims, MultipleLeadingPointDimsSkipped) {
  slinky::dim x_storage[3] = {
      slinky::dim::broadcast(),
      slinky::dim(5, 5, 512),
      slinky::dim(0, 63, 4),
  };
  slinky::dim a_storage[3] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 10, 512),
      slinky::dim(0, 63, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 3, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 3, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));

  EXPECT_EQ(x_dims[0].extent(), 64);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 1);
  EXPECT_EQ(a_dims[0].stride(), 4);
  EXPECT_EQ(x.rank, 0);
  EXPECT_EQ(a.base, reinterpret_cast<void*>(kBaseA + 5 * 512));
}

TEST(FuseAndSliceLeadingDims, LeadingPointDimNonContiguousNextNotReplaced) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 16),
  };
  slinky::dim a_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 16),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 2, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));

  EXPECT_EQ(x_dims[0].extent(), 1);
  EXPECT_EQ(x_dims[1].extent(), 128);
  EXPECT_EQ(x_dims[1].stride(), 16);
  EXPECT_EQ(a_dims[1].stride(), 16);
  EXPECT_EQ(x.rank, 0);
}

TEST(FuseAndSliceLeadingDims, MiddlePointDimDroppedAndFused) {
  slinky::dim x_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(2, 2, 1024),
      slinky::dim(0, 31, 512),
  };
  slinky::dim a_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 7, 1024),
      slinky::dim(0, 31, 512),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 3, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 3, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));

  EXPECT_EQ(x_dims[0].extent(), 128 * 32);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 1);
  EXPECT_EQ(a_dims[0].stride(), 4);
  EXPECT_EQ(x.rank, 0);
  EXPECT_EQ(a.base, reinterpret_cast<void*>(kBaseA + 2 * 1024));
}

TEST(FuseAndSliceLeadingDims, MiddlePointDimDroppedWithoutFusion) {
  slinky::dim x_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(3, 3, 512),
      slinky::dim(0, 31, 1024),
  };
  slinky::dim a_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 7, 512),
      slinky::dim(0, 31, 1024),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 3, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 3, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 32);
  EXPECT_EQ(x_dims[1].stride(), 1024);
  EXPECT_EQ(a_dims[1].stride(), 1024);
  EXPECT_EQ(x.rank, 0);
  EXPECT_EQ(a.base, reinterpret_cast<void*>(kBaseA + 3 * 512));
}

TEST(FuseAndSliceLeadingDims, ContiguousDimsFusedWithScalarInput) {
  slinky::dim x_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
  };
  slinky::dim b_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 0, nullptr);
  slinky::raw_buffer b = make_raw_buffer(kBaseB, 4, 2, b_storage);

  slinky::dim x_dims[2], a_dims[2], b_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a, b_dims, b));

  EXPECT_EQ(x_dims[0].extent(), 128 * 64);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 1);
  EXPECT_EQ(a_dims[0].stride(), 0);
  EXPECT_EQ(b_dims[0].stride(), 4);
  EXPECT_EQ(x.rank, 0);
}

TEST(FuseAndSliceLeadingDims, ContiguousDimsFusedWithExplicitBroadcastInput) {
  slinky::dim x_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
      slinky::dim(0, 3, 128 * 64 * 4),
  };
  slinky::dim a_storage[3] = {
      slinky::dim::broadcast(),
      slinky::dim::broadcast(),
      slinky::dim(0, 3, 4),
  };
  slinky::dim b_storage[3] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
      slinky::dim(0, 3, 128 * 64 * 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 3, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 3, a_storage);
  slinky::raw_buffer b = make_raw_buffer(kBaseB, 4, 3, b_storage);

  slinky::dim x_dims[2], a_dims[2], b_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a, b_dims, b));

  // Dims 0 and 1 fuse (a is broadcast across both); dim 2 stays separate
  // because a varies along dim 2.
  EXPECT_EQ(x_dims[0].extent(), 128 * 64);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(a_dims[0].stride(), 0);
  EXPECT_EQ(b_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 4);
  EXPECT_EQ(x_dims[1].stride(), 128 * 64 * 4);
  EXPECT_EQ(a_dims[1].stride(), 4);
  EXPECT_EQ(b_dims[1].stride(), 128 * 64 * 4);
  EXPECT_EQ(x.rank, 0);
}

TEST(FuseAndSliceLeadingDims, PartialBroadcastInnerNotFused) {
  // b is broadcast in dim 0 only: cannot fuse dim 0 and dim 1.
  slinky::dim x_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
  };
  slinky::dim b_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 63, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer b = make_raw_buffer(kBaseB, 4, 2, b_storage);

  slinky::dim x_dims[2], b_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, b_dims, b));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(b_dims[0].stride(), 0);
  EXPECT_EQ(x_dims[1].extent(), 64);
  EXPECT_EQ(b_dims[1].stride(), 4);
}

TEST(FuseAndSliceLeadingDims, PartialBroadcastOuterNotFused) {
  // b is broadcast in dim 1 only: cannot fuse dim 0 and dim 1.
  slinky::dim x_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
  };
  slinky::dim b_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim::broadcast(),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer b = make_raw_buffer(kBaseB, 4, 2, b_storage);

  slinky::dim x_dims[2], b_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, b_dims, b));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(b_dims[0].stride(), 4);
  EXPECT_EQ(x_dims[1].extent(), 64);
  EXPECT_EQ(b_dims[1].stride(), 0);
}

TEST(FuseAndSliceLeadingDims, UnaryLeadingPointDimReplacedWhenContiguous) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4),
  };
  slinky::dim a_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 2, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, /*x_elem_count=*/1, x,
                                             a_dims, /*a_elem_count=*/1, a));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(x_dims[0].stride(), 4);
  EXPECT_EQ(a_dims[0].stride(), 4);
  EXPECT_TRUE(is_contiguous(a_dims[0], a.elem_size));
  EXPECT_EQ(x_dims[1].extent(), 1);
}

TEST(FuseAndSliceLeadingDims, UnaryDoesNotReplaceWithBroadcastInput) {
  // Unary kernels require contiguous input along dim 0 (no broadcasting).
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 0, nullptr);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, /*x_elem_count=*/1, x,
                                             a_dims, /*a_elem_count=*/1, a));

  EXPECT_EQ(x_dims[0].extent(), 1);
  EXPECT_TRUE(is_contiguous(a_dims[0], a.elem_size));
  EXPECT_EQ(x_dims[1].extent(), 128);
  EXPECT_EQ(a_dims[1].stride(), 0);
}

TEST(FuseAndSliceLeadingDims, DisallowBroadcastDim0DoesNotFuse) {
  slinky::dim x_storage[2] = {
      slinky::dim(0, 127, 4),
      slinky::dim(0, 63, 512),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 0, nullptr);

  slinky::dim x_dims[1], a_dims[1];
  ASSERT_TRUE((fuse_and_slice_leading_dims<1, /*AllowBroadcastDim0=*/false>(
      x_dims, x, a_dims, a)));

  EXPECT_EQ(x_dims[0].extent(), 128);
  EXPECT_EQ(x.rank, 1);
}

TEST(FuseAndSliceLeadingDims, DisallowBroadcastDim0DoesNotReplacePointDim) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 0, nullptr);

  slinky::dim x_dims[1], a_dims[1];
  ASSERT_TRUE((fuse_and_slice_leading_dims<1, /*AllowBroadcastDim0=*/false>(
      x_dims, x, a_dims, a)));

  EXPECT_EQ(x_dims[0].extent(), 1);
  EXPECT_EQ(a_dims[0].stride(), 0);
  EXPECT_EQ(x.rank, 1);
}

TEST(FuseAndSliceLeadingDims, UnarySubByteDoesNotReplaceDim0) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 1),
  };
  slinky::dim a_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 1, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 2, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, /*x_elem_count=*/2, x,
                                             a_dims, /*a_elem_count=*/1, a));

  EXPECT_EQ(x_dims[0].extent(), 1);
  EXPECT_EQ(x_dims[1].extent(), 128);
  EXPECT_EQ(a_dims[1].stride(), 4);
}

TEST(FuseAndSliceLeadingDims, FoldedDimNotReplaced) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4, /*fold_factor=*/16),
  };
  slinky::dim a_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, 127, 4, /*fold_factor=*/16),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 2, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  ASSERT_TRUE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));

  EXPECT_EQ(x_dims[0].extent(), 1);
  EXPECT_EQ(x_dims[1].extent(), 128);
}

TEST(FuseAndSliceLeadingDims, EmptyDimReturnsFalse) {
  slinky::dim x_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, -1, 4),
  };
  slinky::dim a_storage[2] = {
      slinky::dim::broadcast(),
      slinky::dim(0, -1, 4),
  };
  slinky::raw_buffer x = make_raw_buffer(kBaseX, 4, 2, x_storage);
  slinky::raw_buffer a = make_raw_buffer(kBaseA, 4, 2, a_storage);

  slinky::dim x_dims[2], a_dims[2];
  EXPECT_FALSE(fuse_and_slice_leading_dims<2>(x_dims, x, a_dims, a));
}

}  // namespace
}  // namespace ynn
