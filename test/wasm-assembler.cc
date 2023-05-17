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

constexpr int32_t kExpectedGet5ReturnValue = 5;
constexpr int32_t kA = 55;
constexpr int32_t kAPlusFive = kA + kExpectedGet5ReturnValue;
constexpr int32_t kB = 42;
constexpr int32_t kExpectedSum = kA + kB;
constexpr int32_t kExpectedSumTwice = 2 * kExpectedSum;

struct Get5Generator : WasmAssembler {
  explicit Get5Generator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, "get5", {}, no_locals, [this]() {
      i32_const(5);
      end();
    });
  }
};

struct Get5TestSuite {
  using Generator = Get5Generator;
  using Func = GetIntPtr;
  static void ExpectFuncCorrect(Func get5) {
    EXPECT_EQ(get5(), kExpectedGet5ReturnValue);
  }
};

struct AddGenerator : WasmAssembler {
  explicit AddGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<2>({i32}, "add", {i32, i32}, no_locals,
               [this](const Local& a, const Local& b) {
                 local_get(a);
                 local_get(b);
                 i32_add();
                 end();
               });
  }
};

template <typename G, uint32_t kSum>
struct AddTestSuiteTmpl {
  using Generator = G;
  using Func = AddPtr;
  static void ExpectFuncCorrect(Func add) { EXPECT_EQ(add(kA, kB), kSum); }
};

struct AddTestSuite : AddTestSuiteTmpl<AddGenerator, kExpectedSum> {};

struct AddWithLocalGenerator : WasmAssembler {
  explicit AddWithLocalGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "add_with_local", {i32, i32}, single_local_int,
               [this](const Local& a, const Local& b) {
                 auto sum = MakeLocal(i32);
                 sum = I32Add(a, b);
                 local_get(sum);
                 end();
               });
  }
};

struct AddWithLocalTestSuite
    : AddTestSuiteTmpl<AddWithLocalGenerator, kExpectedSum> {};

struct AddTwiceGenerator : WasmAssembler {
  explicit AddTwiceGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
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
  }
};

struct AddTwiceTestSuite
    : AddTestSuiteTmpl<AddTwiceGenerator, kExpectedSumTwice> {};

struct AddTwiceWithScopesGenerator : WasmAssembler {
  explicit AddTwiceWithScopesGenerator(xnn_code_buffer* buf)
      : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
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
  }
};

struct AddTwiceWithScopesTestSuite
    : AddTestSuiteTmpl<AddTwiceWithScopesGenerator, kExpectedSumTwice> {};

struct Add5CodeGenerator : WasmAssembler {
  explicit Add5CodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
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

struct Add5TestSuite {
  using Generator = Add5CodeGenerator;
  using Func = Add5Ptr;
  static void ExpectFuncCorrect(Func add_five) {
    EXPECT_EQ(add_five(kA), kAPlusFive);
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

template <typename TestSuite>
class WasmAssemblerTest : public testing::Test {};

}  // namespace

TYPED_TEST_SUITE_P(WasmAssemblerTest);

TYPED_TEST_P(WasmAssemblerTest, ValidCode) {
  using TestSuite = TypeParam;
  using Generator = typename TestSuite::Generator;
  using Func = typename TestSuite::Func;

  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  Generator generator(&b);
  generator.Emit();
  ASSERT_THAT(generator.finalize(), NotNull());

  ASSERT_EQ(xnn_finalize_code_memory(&b), xnn_status_success);
  ASSERT_EQ(Error::kNoError, generator.error());
  auto func = (Func)b.first_function_index;
  TestSuite::ExpectFuncCorrect(func);

  ASSERT_EQ(xnn_release_code_memory(&b), xnn_status_success);
}

REGISTER_TYPED_TEST_SUITE_P(WasmAssemblerTest, ValidCode);

using WasmAssemblerTestSuits =
    testing::Types<Get5TestSuite, AddTestSuite, AddWithLocalTestSuite,
                   AddTwiceTestSuite, AddTwiceWithScopesTestSuite,
                   Add5TestSuite>;
INSTANTIATE_TYPED_TEST_SUITE_P(WasmAssemblerTestSuits, WasmAssemblerTest,
                               WasmAssemblerTestSuits);

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
