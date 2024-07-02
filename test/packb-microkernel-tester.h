// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"

// Reference bias packing function for f32.
static void f32_packb_reference(
    size_t groups,
    size_t channels,
    size_t kernel_tile,
    size_t channel_tile,
    size_t channel_subtile,
    size_t channel_round,
    const float* weights,
    const float* bias,
    float* out,
    size_t per_tile_extra_bytes,
    size_t per_subtile_extra_bytes) {
  assert(groups > 0);
  // Group loop.
  do {
    // Channel tile loop.
    size_t c = round_up_po2(channels, channel_round);
    size_t tiled_c = round_down_po2(c, channel_tile);

    size_t cr_block_start = 0;
    for (; cr_block_start < tiled_c; cr_block_start += channel_tile) {
      const size_t cr_block_size = min(channels - cr_block_start, channel_tile);
      if (bias != nullptr) {
        for (size_t i = 0; i < cr_block_size; i++) {
          *out++ = bias[cr_block_start + i];
        }
      } else {
        size_t i = cr_block_size;
        do {
          *out++ = 0.0f;
        } while (--i != 0);
      }
      out += channel_tile - cr_block_size;
      out += kernel_tile * channel_tile;
      out += per_tile_extra_bytes;
    }

    // Channel subtile loop.
    for (; cr_block_start < c; cr_block_start += channel_subtile) {
      const size_t cr_block_size = min(channels - cr_block_start, channel_subtile);
      if (bias != nullptr) {
        for (size_t i = 0; i < cr_block_size; i++) {
          *out++ = bias[cr_block_start + i];
        }
      } else {
        size_t i = cr_block_size;
        do {
          *out++ = 0.0f;
        } while (--i != 0);
      }
      out += channel_subtile - cr_block_size;
      out += kernel_tile * channel_subtile;
      out += per_subtile_extra_bytes;
    }
    if (bias != nullptr) {
      bias += channels;
    }
  } while (--groups > 0);
}

class PackBMicrokernelTester {
 public:

  PackBMicrokernelTester& groups(size_t groups) {
    this->groups_ = groups;
    return *this;
  }

  size_t groups() const {
    return this->groups_;
  }

  PackBMicrokernelTester& channel_tile(size_t channel_tile) {
    this->channel_tile_ = channel_tile;
    return *this;
  }

  size_t channel_tile() const {
    return this->channel_tile_;
  }

  PackBMicrokernelTester& channel_subtile(size_t channel_subtile) {
    this->channel_subtile_ = channel_subtile;
    return *this;
  }

  size_t channel_subtile() const {
    return this->channel_subtile_;
  }

  PackBMicrokernelTester& channel_round(size_t channel_round) {
    this->channel_round_ = channel_round;
    return *this;
  }

  size_t channel_round() const {
    return this->channel_round_;
  }

  PackBMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  size_t packed_channels() const {
    return round_up(channels(), channel_subtile());
  }

  PackBMicrokernelTester& kernel_tile(size_t kernel_tile) {
    this->kernel_tile_ = kernel_tile;
    return *this;
  }

  size_t kernel_tile() const {
    return this->kernel_tile_;
  }

  void Test(xnn_x32_packb_gemm_ukernel_fn packb) const {
    std::vector<uint32_t> weights(groups() * channels() * kernel_tile());
    std::vector<uint32_t> bias(groups() * channels());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(
      groups() * (packed_channels() * kernel_tile() + packed_channels()));
    std::vector<uint32_t> packed_w_ref(groups() * (packed_channels() * kernel_tile() + packed_channels()));

    std::fill(weights.begin(), weights.end(), 0xDEADBEEF);
    std::iota(bias.begin(), bias.end(), UINT32_C(0x80000000));
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), UINT32_C(0xDEADBEEF));

    // Compute reference results.
    f32_packb_reference(
      groups(), channels(), kernel_tile(), channel_tile(), channel_subtile(), channel_round(),
      reinterpret_cast<const float*>(weights.data()), reinterpret_cast<const float*>(bias.data()),
      reinterpret_cast<float*>(packed_w_ref.data()), /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0);

    // Call optimized micro-kernel.
    packb(
      groups(), channels(), bias.data(), packed_w.data(),
      /*channel_tile_stride=*/sizeof(float) * (kernel_tile() * channel_tile() + channel_tile()),
      /*channel_subtile_stride=*/sizeof(float) * (kernel_tile() * channel_subtile() + channel_subtile()),
      nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      if (packed_w_ref[i] !=  UINT32_C(0xDEADBEEF)) {  // Allow weights and padding to differ.
        EXPECT_EQ(packed_w[i], packed_w_ref[i]) << "at position " << i << " / " << packed_w.size() << ", channels "
                                                << channels() << ", kernel tile " << kernel_tile() << ", groups "
                                                << groups();
      } else {
        // These are weights, and should be unmodified.
        EXPECT_EQ(packed_w[i], 0x12345678) << "at position " << i << " / " << packed_w.size() << ", channels "
                                           << channels() << ", kernel tile " << kernel_tile() << ", groups "
                                           << groups();
      }
    }
  }

  void Test(xnn_x32_zerob_gemm_ukernel_fn zerob) const {
    std::vector<uint32_t> weights(groups() * channels() * kernel_tile());
    std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> packed_w(
      groups() * (packed_channels() * kernel_tile() + packed_channels()));
    std::vector<uint32_t> packed_w_ref(groups() * (packed_channels() * kernel_tile() + packed_channels()));

    std::fill(weights.begin(), weights.end(), 0xDEADBEEF);
    std::fill(packed_w.begin(), packed_w.end(), UINT32_C(0x12345678));
    std::fill(packed_w_ref.begin(), packed_w_ref.end(), UINT32_C(0xDEADBEEF));

    // Compute reference results.
    f32_packb_reference(
      groups(), channels(), kernel_tile(), channel_tile(), channel_subtile(), channel_round(),
      reinterpret_cast<const float*>(weights.data()), nullptr,
      reinterpret_cast<float*>(packed_w_ref.data()), /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0);

    // Call optimized micro-kernel.
    zerob(
      groups(), channels(), packed_w.data(),
      /*channel_tile_stride=*/sizeof(float) * (kernel_tile() * channel_tile() + channel_tile()),
      /*channel_subtile_stride=*/sizeof(float) * (kernel_tile() * channel_subtile() + channel_subtile()),
      nullptr);

    // Verify results.
    for (size_t i = 0; i < packed_w.size(); i++) {
      if (packed_w_ref[i] !=  UINT32_C(0xDEADBEEF)) {  // Allow weights and padding to differ.
        EXPECT_EQ(packed_w[i], packed_w_ref[i]) << "at position " << i << " / " << packed_w.size() << ", channels "
                                                << channels() << ", kernel tile " << kernel_tile();
        // Bias should be zero.
        EXPECT_EQ(packed_w[i], 0.0f) << "at position " << i << " / " << packed_w.size() << ", channels " << channels()
                                     << ", kernel tile " << kernel_tile();
      } else {
        // These are weights, and should be unmodified.
        EXPECT_EQ(packed_w[i], 0x12345678) << "at position " << i << " / " << packed_w.size() << ", channels "
                                           << channels() << ", kernel tile " << kernel_tile();
      }
    }
  }

 private:
  size_t groups_{1};
  size_t channels_{1};
  size_t channel_tile_{1};
  size_t channel_subtile_{1};
  size_t channel_round_{1};
  size_t kernel_tile_{1};
};
