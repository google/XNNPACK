// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/subgraph/stencil.h"

#include <cstddef>
#include <random>

#include <gtest/gtest.h>
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/operator-utils.h"
#include "test/replicable_random_device.h"

namespace xnnpack {

TEST(Stencil, extent) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<> output_dist{1, 20};

  for (int rep = 0; rep < 1000; ++rep) {
    StencilParams k = random_stencil_params(rng);
    for (int i = 100; i < 1000; ++i) {
      // Verify that a round trip from an output size, to an input size, and
      // back, matches xnnpack's understanding.
      size_t output_dimension = output_dist(rng);
      size_t input_dimension = k.input_extent(output_dimension);
      ASSERT_EQ(output_dimension, k.output_extent(input_dimension))
          << "input_dimension=" << input_dimension << ", k=" << k;
      ASSERT_EQ(output_dimension, xnn_compute_convolution_output_dimension(
                                      input_dimension + k.padding(), k.size,
                                      k.dilation, k.stride))
          << "input_dimension=" << input_dimension << ", k=" << k;
    }
  }
}

TEST(Stencil, make_stencil_dims_1d) {
  ReplicableRandomDevice rng;

  Tensor<int> buf({20});
  for (int rep = 0; rep < 100; ++rep) {
    StencilParams k = random_stencil_params(rng);
    Tensor<int> stencil = make_stencil_dim(buf, 0, k);
    ASSERT_EQ(stencil.rank(), 2);
    ASSERT_EQ(stencil.extent(1), k.size);
    for (size_t i = 0; i < stencil.extent(0); ++i) {
      for (size_t j = 0; j < stencil.extent(1); ++j) {
        ASSERT_EQ(&stencil(i, j), &buf(i * k.stride + j * k.dilation));
      }
    }
  }
}

TEST(Stencil, make_stencil_dims_2d) {
  ReplicableRandomDevice rng;

  Tensor<int> buf({20, 10});
  for (int rep = 0; rep < 10; ++rep) {
    StencilParams ky = random_stencil_params(rng);
    StencilParams kx = random_stencil_params(rng);
    Tensor<int> stencil = make_stencil_dim(buf, 1, kx);
    stencil = make_stencil_dim(stencil, 0, ky);
    ASSERT_EQ(stencil.rank(), 4);
    ASSERT_EQ(stencil.extent(1), ky.size);
    ASSERT_EQ(stencil.extent(3), kx.size);
    for (size_t y = 0; y < stencil.extent(0); ++y) {
      for (size_t x = 0; x < stencil.extent(2); ++x) {
        for (size_t dy = 0; dy < ky.size; ++dy) {
          for (size_t dx = 0; dx < kx.size; ++dx) {
            ASSERT_EQ(&stencil(y, dy, x, dx),
                      &buf(y * ky.stride + dy * ky.dilation,
                           x * kx.stride + dx * kx.dilation));
          }
        }
      }
    }
  }
}

TEST(Stencil, make_stencil_dims_2d_x) {
  ReplicableRandomDevice rng;

  Tensor<int> buf({20, 10});
  for (int rep = 0; rep < 10; ++rep) {
    StencilParams k = random_stencil_params(rng);
    Tensor<int> stencil = make_stencil_dim(buf, 1, k);
    ASSERT_EQ(stencil.rank(), 3);
    ASSERT_EQ(stencil.extent(2), k.size);
    for (size_t y = 0; y < stencil.extent(0); ++y) {
      for (size_t x = 0; x < stencil.extent(1); ++x) {
        for (size_t dx = 0; dx < k.size; ++dx) {
          ASSERT_EQ(&stencil(y, x, dx),
                    &buf(y, x * k.stride + dx * k.dilation));
        }
      }
    }
  }
}

TEST(Stencil, make_stencil_dims_2d_y) {
  ReplicableRandomDevice rng;

  Tensor<int> buf({20, 10});
  for (int rep = 0; rep < 10; ++rep) {
    StencilParams k = random_stencil_params(rng);
    Tensor<int> stencil = make_stencil_dim(buf, 0, k);
    ASSERT_EQ(stencil.rank(), 3);
    ASSERT_EQ(stencil.extent(1), k.size);
    for (size_t y = 0; y < stencil.extent(0); ++y) {
      for (size_t dy = 0; dy < k.size; ++dy) {
        for (size_t x = 0; x < stencil.extent(1); ++x) {
          ASSERT_EQ(&stencil(y, dy, x),
                    &buf(y * k.stride + dy * k.dilation, x));
        }
      }
    }
  }
}

}  // namespace xnnpack
