// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "packq-microkernel-tester.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/packq.h"

namespace xnnpack {

void PackQMicrokernelTester::Test(xnn_x8_packq_f32qp8_ukernel_fn packq) const {
  // Allocate the input and output data.
  std::vector<float> input(m() * k() + XNN_EXTRA_BYTES / sizeof(float));
  const size_t packed_size =
      xnn_x8_packq_f32qp8_packed_size(m(), k(), mr(), kr(), sr());
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w(packed_size);
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> packed_w_ref(packed_size);

  // Populate the input and output data.
  std::iota(input.begin(), input.end(), 0);
  std::fill(packed_w.begin(), packed_w.end(), INT8_C(0x12));
  std::fill(packed_w_ref.begin(), packed_w_ref.end(), INT8_C(0x7B));

  // Compute reference results.
  xnn_x8_packq_f32qp8_ukernel__scalar_u1(
      m(), k(), mr(), kr(), sr(), /*m_idx_start=*/0, input.data(),
      /*lhs_stride=*/k() * sizeof(float), packed_w_ref.data());

  // Call optimized micro-kernel.
  packq(m(), k(), mr(), kr(), sr(), /*m_idx_start=*/0, input.data(),
        /*lhs_stride=*/k() * sizeof(float), packed_w.data());

  // Verify results.
  for (size_t i = 0; i < packed_size; i++) {
    if (packed_w_ref[i] != INT8_C(0x7B)) {  // Allow pad to differ
      ASSERT_EQ((int32_t)packed_w[i], (int32_t)packed_w_ref[i])
          << "at n " << i << " of " << packed_size << ", m=" << m()
          << ", k=" << k() << ", mr=" << mr() << ", kr=" << kr()
          << ", sr=" << sr();
    }
  }
}

};  // namespace xnnpack
