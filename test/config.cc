#include "src/xnnpack/config.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/experimental.h"

namespace {

using ::testing::Eq;

TEST(ConfigIdentifierTest, CreateAndExtractConfigIdentifier) {
  const int32_t expected_version = 3;
  const xnn_config_name expected_name = xnn_config_name_f32_gemm;
  const xnn_config_identifier identifier =
      xnn_create_config_identifier(expected_name, expected_version);
  EXPECT_THAT(xnn_get_config_name(&identifier), Eq(expected_name));
  EXPECT_THAT(xnn_get_config_version(&identifier), Eq(expected_version));
}

}  // namespace
