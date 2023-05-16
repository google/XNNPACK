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
using Add5Ptr = int (*)(int);
using AddPtr = int (*)(int, int);
namespace xnnpack {
namespace {
struct ValidCodeGenerator : WasmAssembler {
  explicit ValidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, "get5", {}, no_locals, [this]() {
      i32_const(5);
      end();
    });
    AddFunc<2>({i32}, "add", {i32, i32}, no_locals,
               [this](const Local& a, const Local& b) {
                 local_get(a);
                 local_get(b);
                 i32_add();
                 end();
               });
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "add_with_local", {i32, i32}, single_local_int,
               [this](const Local& a, const Local& b) {
                 auto sum = MakeLocal(i32);
                 sum = I32Add(a, b);
                 local_get(sum);
                 end();
               });
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "add_twice", {i32, i32}, two_local_ints,
               [this](const Local& a, const Local& b) {
                 auto first = MakeLocal(i32);
                 auto second = MakeLocal(i32);
                 first = I32Add(a, b);
                 second = first;
                 second = I32Add(second, a);
                 second = I32Add(second, b);
                 local_get(second);
                 end();
               });
    AddFunc<2>({i32}, "add_twice_with_scopes", {i32, i32}, two_local_ints,
               [this](const Local& a, const Local& b) {
                 auto first = MakeLocal(i32);
                 {
                   auto sum = MakeLocal(i32);
                   sum = I32Add(a, b);
                   first = sum;
                 }
                 {
                   auto sum = MakeLocal(i32);
                   sum = I32Add(a, b);
                   first = I32Add(first, sum);
                 }
                 local_get(first);
                 end();
               });
    AddFunc<1>({i32}, "add5", {i32}, two_local_ints, [this](const Local& a) {
      auto five = MakeLocal(i32);
      auto sum = MakeLocal(i32);
      five = I32Const(5);
      sum = I32Add(five, a);
      local_get(sum);
      end();
    });
  }
};

struct InvalidCodeGenerator : WasmAssembler {
  ValTypesToInt no_locals;
  explicit InvalidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    AddFunc<0>({}, "invalid", {}, no_locals, [this]() {
      i32_const(5);
      end();
    });
  }
};

}  // namespace

constexpr int32_t kExpectedGet5ReturnValue = 5;
constexpr int32_t kA = 55;
constexpr int32_t kAPlusFive = kA + kExpectedGet5ReturnValue;
constexpr int32_t kB = 42;
constexpr int32_t kExpectedSum = kA + kB;
constexpr int32_t kExpectedSumTwice = 2 * kExpectedSum;

TEST(WasmAsseblerTest, ValidCode) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  ValidCodeGenerator generator(&b);
  generator.Emit();
  ASSERT_THAT(generator.finalize(), NotNull());

  ASSERT_EQ(xnn_finalize_code_memory(&b), xnn_status_success);
  ASSERT_EQ(Error::kNoError, generator.error());
  auto get5 = (GetIntPtr)b.first_function_index;
  auto add = (AddPtr)(b.first_function_index + 1);
  auto add_with_local = (AddPtr)(b.first_function_index + 2);
  auto add_twice = (AddPtr)(b.first_function_index + 3);
  auto add_twice_with_scopes = (AddPtr)(b.first_function_index + 4);
  auto add_five_ptr = (Add5Ptr)(b.first_function_index + 5);
  EXPECT_EQ(get5(), kExpectedGet5ReturnValue);
  EXPECT_EQ(add(kA, kB), kExpectedSum);
  EXPECT_EQ(add_with_local(kA, kB), kExpectedSum);
  EXPECT_EQ(add_twice(kA, kB), 2 * kExpectedSum);
  EXPECT_EQ(add_twice_with_scopes(kA, kB), 2 * kExpectedSum);
  EXPECT_EQ(add_five_ptr(kA), kAPlusFive);
  ASSERT_EQ(xnn_release_code_memory(&b), xnn_status_success);
}

TEST(WasmAsseblerTest, InvalidCode) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  InvalidCodeGenerator generator(&b);
  generator.Emit();
  EXPECT_THAT(generator.finalize(), NotNull());
  EXPECT_THAT(b.first_function_index, XNN_INVALID_FUNCTION_INDEX);
}

using ::xnnpack::internal::At;
using ::xnnpack::internal::LocalsManager;

TEST(WasmAssembler, LocalsManager) {
  ValType i32(1);
  ValType f32(20);
  ValType f16(10);
  constexpr uint32_t params = 4;
  ValTypesToInt local_declaration = {{i32, 2}, {f32, 0}, {f16, 3}};

  LocalsManager manager;
  manager.ResetLocalsManager(params, local_declaration);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 6);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 7);
  manager.DestructLocal(f16);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 7);
  // assert fails
  // manager.MakeLocal(f32);
  EXPECT_EQ(manager.GetNewLocalIndex(i32), 4);
}

}  // namespace xnnpack
#endif
