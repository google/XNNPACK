// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <xnnpack/common.h>

#if XNN_PLATFORM_WEB

#include <xnnpack/memory.h>
#include <xnnpack/wasm-assembler.h>

#include <array>
#include <cstdint>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::NotNull;
using GetIntPtr = int (*)();
using Add5Ptr = int (*)(int);
using AddPtr = int (*)(int, int);
using MaxPtr = AddPtr;
using SumUntil = Add5Ptr;
using DoWhile = Add5Ptr;
using SumUntilManyLocals = int (*)();
using SumArray = int (*)(const int*, int);
using MemCpy = void (*)(int*, const int*, int);

namespace xnnpack {
namespace {

constexpr int32_t kExpectedGet5ReturnValue = 5;
constexpr int32_t kPositive = 5;
constexpr int32_t kNegative = -5;
constexpr int32_t kA = 55;
constexpr int32_t kAPlusFive = kA + kExpectedGet5ReturnValue;
constexpr int32_t kB = 42;
constexpr int32_t kExpectedSum = kA + kB;
constexpr int32_t kExpectedSumTwice = 2 * kExpectedSum;
constexpr uint32_t kLargeNumberOfLocals = 300;
constexpr uint32_t kLargeNumberOfFunctions = 300;
constexpr size_t kArraySize = 5;
constexpr std::array<int, kArraySize> kArray = {1, 2, 3, 45, 6};
const int kExpectedArraySum = std::accumulate(kArray.begin(), kArray.end(), 0);

struct Get5Generator : WasmAssembler {
  explicit Get5Generator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, "get5", {}, no_locals, [this]() { i32_const(5); });
  }
};

template <typename G, typename F>
struct GeneratorTestSuite {
  using Generator = G;
  using Func = F;
};

struct Get5TestSuite : GeneratorTestSuite<Get5Generator, GetIntPtr> {
  static void ExpectFuncCorrect(GetIntPtr get5) {
    EXPECT_EQ(get5(), kExpectedGet5ReturnValue);
  }
};

struct AddGenerator : WasmAssembler {
  explicit AddGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<2>({i32}, "add", {i32, i32}, no_locals, [this](Local a, Local b) {
      local_get(a);
      local_get(b);
      i32_add();
    });
  }
};

template <typename G, uint32_t kSum>
struct AddTestSuiteTmpl : GeneratorTestSuite<G, AddPtr> {
  static void ExpectFuncCorrect(AddPtr add) { EXPECT_EQ(add(kA, kB), kSum); }
};

struct AddTestSuite : AddTestSuiteTmpl<AddGenerator, kExpectedSum> {};

struct AddWithLocalGenerator : WasmAssembler {
  explicit AddWithLocalGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "add_with_local", {i32, i32}, single_local_int,
               [this](Local a, Local b) {
                 auto sum = MakeLocal(i32);
                 sum = I32Add(a, b);
                 local_get(sum);
               });
  }
};

struct AddWithLocalTestSuite
    : AddTestSuiteTmpl<AddWithLocalGenerator, kExpectedSum> {};

struct AddTwiceGenerator : WasmAssembler {
  explicit AddTwiceGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "add_twice", {i32, i32}, two_local_ints,
               [this](Local a, Local b) {
                 auto first = MakeLocal(i32);
                 auto second = MakeLocal(i32);
                 first = I32Add(a, b);
                 second = first;
                 second = I32Add(second, a);
                 second = I32Add(second, b);
                 local_get(second);
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
               [this](Local a, Local b) {
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
               });
  }
};

struct AddTwiceWithScopesTestSuite
    : AddTestSuiteTmpl<AddTwiceWithScopesGenerator, kExpectedSumTwice> {};

struct Add5CodeGenerator : WasmAssembler {
  explicit Add5CodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals = {};
    AddFunc<1>({i32}, "add5", {i32}, no_locals, [this](Local a) {
      a = I32Add(I32Const(5), a);
      local_get(a);
    });
  }
};

struct Add5TestSuite : GeneratorTestSuite<Add5CodeGenerator, Add5Ptr> {
  static void ExpectFuncCorrect(Add5Ptr add_five) {
    EXPECT_EQ(add_five(kA), kAPlusFive);
  }
};

struct MaxCodeGenerator : WasmAssembler {
  explicit MaxCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "max", {i32, i32}, single_local_int,
               [this](Local a, Local b) {
                 auto result = MakeLocal(i32);
                 IfElse([&] { I32LtS(a, b); }, [&] { result = b; },
                        [&] { result = a; });
                 local_get(result);
               });
  }
};

template <typename G>
struct GenericMaxTestSuite : GeneratorTestSuite<G, MaxPtr> {
  static void ExpectFuncCorrect(MaxPtr max) {
    EXPECT_EQ(max(2, 3), 3);
    EXPECT_EQ(max(3, 2), 3);
  }
};

struct MaxTestSuite : GenericMaxTestSuite<MaxCodeGenerator> {};

struct MaxIncompleteIfCodeGenerator : WasmAssembler {
  explicit MaxIncompleteIfCodeGenerator(xnn_code_buffer* buf)
      : WasmAssembler(buf) {
    ValTypesToInt no_locals = {};
    AddFunc<2>({i32}, "max_incomplete_if", {i32, i32}, no_locals,
               [this](Local a, Local b) {
                 If([&] { I32LtS(a, b); }, [&] { a = b; });
                 local_get(a);
               });
  }
};

struct MaxIncompleteIfTestSuite
    : GenericMaxTestSuite<MaxIncompleteIfCodeGenerator> {};

struct SumUntilCodeGenerator : WasmAssembler {
  explicit SumUntilCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<1>({i32}, "SumUntil", {i32}, two_local_ints, [&](Local n) {
      auto i = MakeLocal(i32);
      auto result = MakeLocal(i32);
      While([&] { I32LtS(i, n); },
            [&] {
              result = I32Add(result, i);
              i = I32Add(i, I32Const(1));
            });
      local_get(result);
    });
  }
};

static int ReferenceSumUntil(int n) {
  int i = 0;
  int result = 0;
  while (i < n) {
    result += i;
    i++;
  }
  return result;
}

struct SumUntilTestSuite : GeneratorTestSuite<SumUntilCodeGenerator, SumUntil> {
  static void ExpectFuncCorrect(SumUntil sum_until) {
    static constexpr int kN = 5;
    static constexpr int kNoIters = 0;
    EXPECT_EQ(sum_until(kN), ReferenceSumUntil(kN));
    EXPECT_EQ(sum_until(kNoIters), ReferenceSumUntil(kNoIters));
  }
};

struct DoWhileCodeGenerator : WasmAssembler {
  explicit DoWhileCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<1>({i32}, "DoWhile", {i32}, two_local_ints, [&](Local n) {
      auto i = MakeLocal(i32);
      auto result = MakeLocal(i32);
      result = I32Const(kPositive);
      DoWhile(
          [&] {
            result = I32Add(result, result);
            i = I32Add(i, I32Const(1));
          },
          [&] { I32LtS(i, n); });
      local_get(result);
    });
  }
};

struct DoWhileTestSuite : GeneratorTestSuite<DoWhileCodeGenerator, DoWhile> {
  static int ReferenceDoWhile(int n) {
    int result = kPositive;
    int i = 0;
    do {
      result += result;
      i++;
    } while (i < n);
    return result;
  }

  static void ExpectFuncCorrect(DoWhile do_while) {
    EXPECT_EQ(do_while(kPositive), ReferenceDoWhile(kPositive));
    EXPECT_EQ(do_while(kNegative), ReferenceDoWhile(kNegative));
  }
};

struct SumArrayMemory : WasmAssembler {
  explicit SumArrayMemory(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "SumArray", {i32, i32}, two_local_ints,
               [&](Local array, Local n) {
                 auto i = MakeLocal(i32);
                 auto result = MakeLocal(i32);
                 While([&] { I32LtS(i, n); },
                       [&] {
                         result = I32Add(result, I32Load(array, i));
                         i = I32Add(i, I32Const(1));
                       });
                 local_get(result);
               });
  }
};

struct SumArrayTestSuite : GeneratorTestSuite<SumArrayMemory, SumArray> {
  static void ExpectFuncCorrect(SumArray sum_array) {
    EXPECT_EQ(sum_array(kArray.data(), kArraySize), kExpectedArraySum);
  }
};

struct MemCpyGenerator : WasmAssembler {
  explicit MemCpyGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<3>({}, "mymemcpy", {i32, i32, i32}, two_local_ints,
               [&](Local dst, Local src, Local n) {
                 auto i = MakeLocal(i32);
                 auto value = MakeLocal(i32);

                 While([&] { I32LtS(i, n); },
                       [&] {
                         value = I32Load(src, i);
                         I32Store(dst, i, value);
                         i = I32Add(i, I32Const(1));
                       });
               });
  }
};

struct MemCpyTestSuite : GeneratorTestSuite<MemCpyGenerator, MemCpy> {
  static void ExpectFuncCorrect(MemCpy mem_cpy) {
    std::array<int, kArraySize> dst;
    mem_cpy(dst.data(), kArray.data(), kArraySize);
    EXPECT_THAT(dst, testing::ElementsAreArray(kArray));
  }
};

struct AddDelayedInitLocalsGenerator : WasmAssembler {
  explicit AddDelayedInitLocalsGenerator(xnn_code_buffer* b)
      : WasmAssembler(b) {
    ValTypesToInt three_ints = {{i32, 3}};
    AddFunc<2>({i32}, "add_delayed_init", {i32, i32}, three_ints,
               [this](Local a, Local b) {
                 Local c;
                 c = MakeLocal(i32);
                 Local d;
                 d = MakeLocal(i32);
                 Local sum;
                 c = I32Add(c, a);
                 d = I32Add(c, b);
                 sum = MakeLocal(i32);
                 sum = I32Add(a, sum);
                 sum = I32Add(b, sum);
                 local_get(sum);
               });
  }
};

struct AddDelayedInitTestSuite
    : AddTestSuiteTmpl<AddDelayedInitLocalsGenerator, kExpectedSum> {};

struct ManyFunctionsGenerator : WasmAssembler {
  explicit ManyFunctionsGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    for (uint32_t func_index = 0; func_index < kLargeNumberOfFunctions;
         func_index++) {
      AddGet5(func_index);
    }
  }

 private:
  void AddGet5(uint32_t index) {
    function_names[index] = "get5_" + std::to_string(index);
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, function_names[index].c_str(), {}, no_locals,
               [this]() { i32_const(5); });
  }
  static std::array<std::string, kLargeNumberOfFunctions> function_names;
};

std::array<std::string, kLargeNumberOfFunctions>
    ManyFunctionsGenerator::function_names = {};

struct ManyFunctionsGeneratorTestSuite
    : GeneratorTestSuite<ManyFunctionsGenerator, GetIntPtr> {
  static void ExpectFuncCorrect(GetIntPtr get_int) {
    EXPECT_EQ(get_int(), kExpectedGet5ReturnValue);
  }
};

struct ManyLocalsGenerator : WasmAssembler {
  explicit ManyLocalsGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt many_ints = {{i32, kLargeNumberOfFunctions + 1}};
    AddFunc<0>({i32}, "sum_until_with_many_locals", {}, many_ints, [this]() {
      std::array<Local, kLargeNumberOfFunctions> locals;
      for (int i = 0; i < kLargeNumberOfFunctions; i++) {
        locals[i] = MakeLocal(i32);
        locals[i] = I32Const(i);
      }
      auto sum = MakeLocal(i32);

      for (int i = 0; i < kLargeNumberOfFunctions; i++) {
        sum = I32Add(sum, locals[i]);
      }
      local_get(sum);
    });
  }
};

struct ManyLocalsGeneratorTestSuite
    : GeneratorTestSuite<ManyLocalsGenerator, SumUntilManyLocals> {
  static void ExpectFuncCorrect(SumUntilManyLocals sum_until) {
    EXPECT_EQ(sum_until(), ReferenceSumUntil(kLargeNumberOfFunctions));
  }
};

struct InvalidCodeGenerator : WasmAssembler {
  ValTypesToInt no_locals;
  explicit InvalidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    AddFunc<0>({}, "invalid", {}, no_locals, [this]() { i32_const(5); });
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
  static constexpr size_t kBufferSize = 131072;

  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, kBufferSize);

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

using WasmAssemblerTestSuits = testing::Types<
    Get5TestSuite, AddTestSuite, AddWithLocalTestSuite, AddTwiceTestSuite,
    AddTwiceWithScopesTestSuite, Add5TestSuite, MaxTestSuite,
    MaxIncompleteIfTestSuite, SumUntilTestSuite, DoWhileTestSuite,
    SumArrayTestSuite, MemCpyTestSuite, AddDelayedInitTestSuite,
    ManyFunctionsGeneratorTestSuite, ManyLocalsGeneratorTestSuite>;
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
