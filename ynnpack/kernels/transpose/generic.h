// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TRANSPOSE_GENERIC_H_
#define XNNPACK_YNNPACK_KERNELS_TRANSPOSE_GENERIC_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <tuple>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"

namespace ynn {

// When T is large or there are no SIMD instruction sets available, a simple
// memcpy implementation may be appropriate.
template <typename ElemSize>
static void transpose(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                      const void* a, size_t stride_x, void* x,
                      ElemSize elem_size) {
  // TODO(dsharlet): We could unroll this loop such that it reads and writes
  // at least one cache line at a time, and then add prefetching. Attempts
  // to do this yielded no improvement so far.
  while (m > 0) {
    const void* a_j = a;
    void* x_j = x;
    if (elem_size <= n_bytes_a) {
      // This column of input is fully in bounds.
      size_t j = n;
      while (j > 0) {
        memcpy(x_j, a_j, elem_size);
        x_j = offset_bytes(x_j, elem_size);
        a_j = offset_bytes(a_j, stride_a);
        --j;
      }
      n_bytes_a -= elem_size;
    } else if (n_bytes_a > 0) {
      // This column of input is partially in bounds.
      assert(n_bytes_a < elem_size);
      size_t j = n;
      while (j > 0) {
        memcpy(x_j, a_j, n_bytes_a);
        memset(offset_bytes(x_j, n_bytes_a), 0, elem_size - n_bytes_a);
        x_j = offset_bytes(x_j, elem_size);
        a_j = offset_bytes(a_j, stride_a);
        --j;
      }
      n_bytes_a = 0;
    } else {
      // This column of input is fully out of bounds. The entire output row is
      // 0.
      memset(x_j, 0, n * elem_size);
    }
    --m;
    a = offset_bytes(a, elem_size);
    x = offset_bytes(x, stride_x);
  }
}

// The following functions implement `transpose` and `interleave` assuming some
// helper functions exist to work with `Tile`:
// - `Tile` is an array of `M` vectors of size `N`.
// - `Tile load(Tile, from, stride, m)`: load a tile of `m` vectors from `from`,
// `stride` bytes apart in memory. `m` must not be greater than the number of
// vectors in the tile `M`, and if it is less, the extra vectors are padded with
// zero.
// - `Tile interleave(elem_size_bits, tile)`: Interleave elements of size
// `elem_size_bits` in each row of the tile into one row where the `i`th element
// comes from row `i % M`.
// - `store(tile, to, stride, m)`: Write `m` rows of `tile` to pointer `to`
// `stride` bytes apart in memory.

template <typename T, size_t ElemSize>
static std::array<T, 4> interleave(
    std::integral_constant<size_t, ElemSize> elem_size, std::array<T, 4> x) {
  std::integral_constant<size_t, ElemSize * 2> elem_size_x2;
  using t2x2 = std::array<T, 2>;

  // Transpose 2x2
  t2x2 x2[] = {
      interleave(elem_size, t2x2{{x[0], x[1]}}),
      interleave(elem_size, t2x2{{x[2], x[3]}}),
  };

  // Transpose 2x2 of 2x2
  t2x2 x4[] = {
      interleave(elem_size_x2, t2x2{{x2[0][0], x2[1][0]}}),
      interleave(elem_size_x2, t2x2{{x2[0][1], x2[1][1]}}),
  };

  return {x4[0][0], x4[0][1], x4[1][0], x4[1][1]};
}

template <typename T, size_t ElemSize>
static std::array<T, 8> interleave(
    std::integral_constant<size_t, ElemSize> elem_size, std::array<T, 8> x) {
  std::integral_constant<size_t, ElemSize * 4> elem_size_x4;
  using t2x2 = std::array<T, 2>;
  using t4x4 = std::array<T, 4>;

  // Transpose 4x4
  t4x4 x4[] = {
      interleave(elem_size, t4x4{{x[0], x[1], x[2], x[3]}}),
      interleave(elem_size, t4x4{{x[4], x[5], x[6], x[7]}}),
  };

  // Transpose 2x2 of 4x4
  auto x04 = interleave(elem_size_x4, t2x2{{x4[0][0], x4[1][0]}});
  auto x15 = interleave(elem_size_x4, t2x2{{x4[0][1], x4[1][1]}});
  auto x26 = interleave(elem_size_x4, t2x2{{x4[0][2], x4[1][2]}});
  auto x37 = interleave(elem_size_x4, t2x2{{x4[0][3], x4[1][3]}});

  return {x04[0], x04[1], x15[0], x15[1], x26[0], x26[1], x37[0], x37[1]};
}

template <typename T, size_t ElemSize>
static std::array<T, 16> interleave(
    std::integral_constant<size_t, ElemSize> elem_size, std::array<T, 16> x) {
  std::integral_constant<size_t, ElemSize * 4> elem_size_x4;
  using t4x4 = std::array<T, 4>;

  // Transpose 4x4
  t4x4 x4[] = {
      interleave(elem_size, t4x4{{x[0], x[1], x[2], x[3]}}),
      interleave(elem_size, t4x4{{x[4], x[5], x[6], x[7]}}),
      interleave(elem_size, t4x4{{x[8], x[9], x[10], x[11]}}),
      interleave(elem_size, t4x4{{x[12], x[13], x[14], x[15]}}),
  };

  // Transpose 4x4 of 4x4
  t4x4 x16[] = {
      interleave(elem_size_x4, t4x4{{x4[0][0], x4[1][0], x4[2][0], x4[3][0]}}),
      interleave(elem_size_x4, t4x4{{x4[0][1], x4[1][1], x4[2][1], x4[3][1]}}),
      interleave(elem_size_x4, t4x4{{x4[0][2], x4[1][2], x4[2][2], x4[3][2]}}),
      interleave(elem_size_x4, t4x4{{x4[0][3], x4[1][3], x4[2][3], x4[3][3]}}),
  };

  return {
      // clang-format off
      x16[0][0], x16[0][1], x16[0][2], x16[0][3],
      x16[1][0], x16[1][1], x16[1][2], x16[1][3],
      x16[2][0], x16[2][1], x16[2][2], x16[2][3],
      x16[3][0], x16[3][1], x16[3][2], x16[3][3],
      // clang-format on
  };
}

template <typename T, size_t ElemSize>
static std::array<T, 32> interleave(
    std::integral_constant<size_t, ElemSize> elem_size, std::array<T, 32> x) {
  std::integral_constant<size_t, ElemSize * 4> elem_size_x4;
  std::integral_constant<size_t, ElemSize * 16> elem_size_x16;
  using t4x4 = std::array<T, 4>;
  using t2x2 = std::array<T, 2>;

  // Transpose 4x4
  t4x4 x4[] = {
      interleave(elem_size, t4x4{{x[0], x[1], x[2], x[3]}}),
      interleave(elem_size, t4x4{{x[4], x[5], x[6], x[7]}}),
      interleave(elem_size, t4x4{{x[8], x[9], x[10], x[11]}}),
      interleave(elem_size, t4x4{{x[12], x[13], x[14], x[15]}}),
      interleave(elem_size, t4x4{{x[16], x[17], x[18], x[19]}}),
      interleave(elem_size, t4x4{{x[20], x[21], x[22], x[23]}}),
      interleave(elem_size, t4x4{{x[24], x[25], x[26], x[27]}}),
      interleave(elem_size, t4x4{{x[28], x[29], x[30], x[31]}}),
  };

  // Transpose 4x4 of 4x4
  t4x4 x16[] = {
      interleave(elem_size_x4, t4x4{{x4[0][0], x4[1][0], x4[2][0], x4[3][0]}}),
      interleave(elem_size_x4, t4x4{{x4[0][1], x4[1][1], x4[2][1], x4[3][1]}}),
      interleave(elem_size_x4, t4x4{{x4[0][2], x4[1][2], x4[2][2], x4[3][2]}}),
      interleave(elem_size_x4, t4x4{{x4[0][3], x4[1][3], x4[2][3], x4[3][3]}}),
      interleave(elem_size_x4, t4x4{{x4[4][0], x4[5][0], x4[6][0], x4[7][0]}}),
      interleave(elem_size_x4, t4x4{{x4[4][1], x4[5][1], x4[6][1], x4[7][1]}}),
      interleave(elem_size_x4, t4x4{{x4[4][2], x4[5][2], x4[6][2], x4[7][2]}}),
      interleave(elem_size_x4, t4x4{{x4[4][3], x4[5][3], x4[6][3], x4[7][3]}}),
  };

  // Transpose 2x2 of 16x16
  t2x2 x32[] = {
      interleave(elem_size_x16, t2x2{{x16[0][0], x16[4][0]}}),
      interleave(elem_size_x16, t2x2{{x16[0][1], x16[4][1]}}),
      interleave(elem_size_x16, t2x2{{x16[0][2], x16[4][2]}}),
      interleave(elem_size_x16, t2x2{{x16[0][3], x16[4][3]}}),
      interleave(elem_size_x16, t2x2{{x16[1][0], x16[5][0]}}),
      interleave(elem_size_x16, t2x2{{x16[1][1], x16[5][1]}}),
      interleave(elem_size_x16, t2x2{{x16[1][2], x16[5][2]}}),
      interleave(elem_size_x16, t2x2{{x16[1][3], x16[5][3]}}),
      interleave(elem_size_x16, t2x2{{x16[2][0], x16[6][0]}}),
      interleave(elem_size_x16, t2x2{{x16[2][1], x16[6][1]}}),
      interleave(elem_size_x16, t2x2{{x16[2][2], x16[6][2]}}),
      interleave(elem_size_x16, t2x2{{x16[2][3], x16[6][3]}}),
      interleave(elem_size_x16, t2x2{{x16[3][0], x16[7][0]}}),
      interleave(elem_size_x16, t2x2{{x16[3][1], x16[7][1]}}),
      interleave(elem_size_x16, t2x2{{x16[3][2], x16[7][2]}}),
      interleave(elem_size_x16, t2x2{{x16[3][3], x16[7][3]}}),
  };

  return {
      // clang-format off
      x32[0][0],  x32[0][1],
      x32[1][0],  x32[1][1],
      x32[2][0],  x32[2][1],
      x32[3][0],  x32[3][1],
      x32[4][0],  x32[4][1],
      x32[5][0],  x32[5][1],
      x32[6][0],  x32[6][1],
      x32[7][0],  x32[7][1],
      x32[8][0],  x32[8][1],
      x32[9][0],  x32[9][1],
      x32[10][0], x32[10][1],
      x32[11][0], x32[11][1],
      x32[12][0], x32[12][1],
      x32[13][0], x32[13][1],
      x32[14][0], x32[14][1],
      x32[15][0], x32[15][1],
      // clang-format on
  };
}

// These overloads of load/store work on partial tiles
template <typename Tile>
static Tile load(Tile, const void* a, size_t stride, size_t m, size_t n_bytes) {
  Tile result;
  memset(&result, 0, sizeof(Tile));
  for (size_t i = 0; i < m; ++i) {
    memcpy(&result[i], offset_bytes(a, i * stride), n_bytes);
  }
  return result;
}

template <typename Tile>
static void store(Tile tile, void* x, size_t stride, size_t m, size_t n_bytes) {
  for (size_t i = 0; i < m; ++i) {
    memcpy(offset_bytes(x, i * stride), &tile[i], n_bytes);
  }
}

template <typename Tile, typename ElemSize>
static void transpose(size_t m, size_t n, size_t n_bytes_a, size_t stride_a,
                      const void* a, size_t stride_x, void* x,
                      ElemSize elem_size_bits) {
  constexpr size_t M = std::tuple_size<Tile>::value;
  std::integral_constant<size_t, sizeof(std::declval<Tile>()[0])> N_bytes;
  constexpr size_t M_bytes = elem_size_bits * M / 8;
  constexpr size_t N = N_bytes * 8 / elem_size_bits;

  while (m >= M && n_bytes_a >= N_bytes) {
    // Handle a full set of M rows.
    size_t j = n;
    const void* a_j = a;
    void* x_j = x;
    while (j >= N) {
      // Handle a full set of M rows x N columns.
      Tile t = load(Tile{}, a_j, stride_a, M, N_bytes);
      t = interleave(elem_size_bits, t);
      store(t, x_j, stride_x, M, N_bytes);

      j -= N;
      a_j = offset_bytes(a_j, N * stride_a);
      x_j = offset_bytes(x_j, N_bytes);
    }
    if (j > 0) {
      // Handle a full set of M rows x partial set of j columns.
      Tile t = load(Tile{}, a_j, stride_a, j, N_bytes);
      t = interleave(elem_size_bits, t);
      store(t, x_j, stride_x, M, j * elem_size_bits / 8);
    }
    m -= M;
    n_bytes_a -= M_bytes;
    a = offset_bytes(a, M_bytes);
    x = offset_bytes(x, M * stride_x);
  }
  while (m > 0) {
    const size_t n_bytes =
        std::min({M_bytes, n_bytes_a, elem_size_bits * m / 8});
    size_t j = n;
    const void* a_j = a;
    void* x_j = x;
    while (j >= N) {
      // Handle a partial set of m rows x full set of N columns.
      Tile t = load(Tile{}, a_j, stride_a, M, n_bytes);
      t = interleave(elem_size_bits, t);
      store(t, x_j, stride_x, std::min(m, M), N_bytes);

      j -= N;
      a_j = offset_bytes(a_j, N * stride_a);
      x_j = offset_bytes(x_j, N_bytes);
    }
    if (j > 0) {
      // Handle a partial set of m rows x j columns.
      Tile t = load(Tile{}, a_j, stride_a, j, n_bytes);
      t = interleave(elem_size_bits, t);
      store(t, x_j, stride_x, std::min(m, M), j * elem_size_bits / 8);
    }
    m = sub_sat(m, M);
    n_bytes_a = sub_sat(n_bytes_a, M_bytes);
    a = offset_bytes(a, M_bytes);
    x = offset_bytes(x, M * stride_x);
  }
}

template <typename Tile, typename ElemSize>
static void interleave(size_t m, size_t n, size_t stride_a, const void* a,
                       void* x, ElemSize elem_size_bits) {
  constexpr size_t M = std::tuple_size<Tile>::value;
  std::integral_constant<size_t, sizeof(std::declval<Tile>()[0])> N_bytes;
  constexpr size_t N = N_bytes * 8 / elem_size_bits;

  while (n >= N) {
    Tile t = load(Tile{}, a, stride_a, m, N_bytes);
    t = interleave(elem_size_bits, t);
    store(t, x, N_bytes, M, N_bytes);

    n -= N;
    a = offset_bytes(a, N_bytes);
    x = offset_bytes(x, M * N_bytes);
  }
  if (n > 0) {
    size_t n_bytes = elem_size_bits * n / 8;
    Tile t = load(Tile{}, a, stride_a, m, n_bytes);
    t = interleave(elem_size_bits, t);
    memcpy(x, &t[0], M * n_bytes);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TRANSPOSE_GENERIC_H_
