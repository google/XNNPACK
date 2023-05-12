// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <xnnpack/common.h>

#if XNN_PLATFORM_WEB

#include <xnnpack/memory.h>
#include <xnnpack/wasm-assembler.h>

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::NotNull;
using GetIntPtr = int (*)();
using AddPtr = int (*)(int, int);
namespace xnnpack {
namespace {
struct ValidCodeGenerator : WasmAssembler {
  explicit ValidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    AddFunc({i32}, "get5", {}, [this]() {
      i32_const(5);
      end();
    });
    AddFunc({i32}, "add", {i32, i32}, [this]() {
      local_get(0);
      local_get(1);
      i32_add();
      end();
    });
  }
};

struct InvalidCodeGenerator : WasmAssembler {
  explicit InvalidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    AddFunc({}, "invalid", {}, [this]() {
      i32_const(5);
      end();
    });
  }
};

}  // namespace

constexpr int32_t kExpectedGet5ReturnValue = 5;
constexpr int32_t kA = 55;
constexpr int32_t kB = 42;
constexpr int32_t kExpectedSum = kA + kB;


TEST(WasmAssebler, ValidCode) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  ValidCodeGenerator generator(&b);
  generator.Emit();
  ASSERT_THAT(generator.finalize(), NotNull());

  ASSERT_EQ(xnn_finalize_code_memory(&b), xnn_status_success);
  ASSERT_EQ(Error::kNoError, generator.error());
  auto get5 = (GetIntPtr)b.first_function_index;
  auto get7 = (AddPtr)(b.first_function_index + 1);
  EXPECT_EQ(get5(), kExpectedGet5ReturnValue);
  EXPECT_EQ(get7(kA, kB), kExpectedSum);
  ASSERT_EQ(xnn_release_code_memory(&b), xnn_status_success);
}

TEST(WasmAssebler, InvalidCode) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  InvalidCodeGenerator generator(&b);
  generator.Emit();
  EXPECT_THAT(generator.finalize(), NotNull());
  EXPECT_THAT(b.first_function_index, XNN_INVALID_FUNCTION_INDEX);
}
}  // namespace xnnpack
#endif
