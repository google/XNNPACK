#include "ynnpack/base/type.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace ynn {

using testing::ElementsAre;

TEST(copy_n, int4x2) {
  const int8_t src[6] = {0, 1, 2, 3, 4, 5};
  int4x2 dst[3];
  copy_n(src, 0, 6, dst, 0);
  EXPECT_THAT(dst, ElementsAre(int4x2(0, 1), int4x2(2, 3), int4x2(4, 5)));
}

TEST(copy_n, int2x4) {
  const int8_t src[12] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  int2x4 dst[3];
  copy_n(src, 0, 12, dst, 0);
  EXPECT_THAT(dst, ElementsAre(int2x4(0, 1, 2, 0), int2x4(1, 2, 0, 1),
                               int2x4(2, 0, 1, 2)));
}

TEST(generate_n, int4x2) {
  int i = 0;
  auto f = [&i]() { return i++; };
  int4x2 dst[3];
  generate_n(dst, 0, 6, f);
  EXPECT_THAT(dst, ElementsAre(int4x2(0, 1), int4x2(2, 3), int4x2(4, 5)));
}

TEST(generate_n, int2x4) {
  int i = 0;
  auto f = [&i]() { return (i++) % 3; };
  int2x4 dst[3];
  generate_n(dst, 0, 12, f);
  EXPECT_THAT(dst, ElementsAre(int2x4(0, 1, 2, 0), int2x4(1, 2, 0, 1),
                               int2x4(2, 0, 1, 2)));
}

}  // namespace ynn
