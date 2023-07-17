// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <vector>

namespace xnnpack {
namespace internal {

// LEB128 adopted from llvm
template <typename Store>
static void StoreEncodedU32(uint32_t n, Store&& store) {
  do {
    uint8_t byte = n & 0x7f;
    n >>= 7;
    if (n != 0) {
      byte |= 0x80;
    }
    store(byte);
  } while (n != 0);
}

inline static uint32_t WidthEncodedU32(uint32_t n) {
  uint32_t cnt = 0;
  StoreEncodedU32(n, [&](auto _) { cnt++; });
  return cnt;
}

template <typename Store>
static void StoreEncodedS32(int32_t n, Store&& store) {
  auto more = true;
  do {
    uint8_t byte = n & 0x7f;
    n >>= 7;
    more = !((((n == 0) && ((byte & 0x40) == 0)) ||
              ((n == -1) && ((byte & 0x40) != 0))));
    if (more) {
      byte |= 0x80;
    }
    store(byte);
  } while (more);
}
}  // namespace internal
}  // namespace xnnpack
