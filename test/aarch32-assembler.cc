#include <xnnpack/aarch32-assembler.h>

#include <ios>

#include <gtest/gtest.h>

#define CHECK_INSTR(expected, expr) \
  EXPECT_PRED_FORMAT2(AssertInstructionEqual, expected, expr)

#define EXPECT_INSTR_EQ(expected, call) \
  a.reset();                            \
  CHECK_INSTR(expected, call);

#define EXPECT_ERROR(expected, call) \
  a.reset();                         \
  call;                              \
  EXPECT_EQ(expected, a.error());

namespace xnnpack {
namespace aarch32 {

// Formats an int value as "0x%x".
std::string FormatHexUInt32(uint32_t value) {
  std::stringstream ss;
  ss << "0x" << std::hex << value;
  return ss.str();
}

testing::AssertionResult AssertInstructionEqual(const char* expected_expr,
                                                const char* actual_expr,
                                                uint32_t expected,
                                                Assembler& a) {
  uint32_t actual = *a.start();
  if (expected == actual) {
    return testing::AssertionSuccess();
  }

  return testing::AssertionFailure()
         << "Encoding of '" << actual_expr << "' does not match "
         << expected_expr << std::endl
         << " expected: " << FormatHexUInt32(expected) << std::endl
         << "   actual: " << FormatHexUInt32(actual);
}

TEST(AArch32Assembler, InstructionEncoding) {
  Assembler a;

  EXPECT_INSTR_EQ(0xE0810002, a.add(r0, r1, r2));

  EXPECT_INSTR_EQ(0xE3500002, a.cmp(r0, 2));

  // Offset addressing mode.
  EXPECT_INSTR_EQ(0xE59D7060, a.ldr(r7, mem[sp, 96]));
  // Post-indexed addressing mode.
  EXPECT_INSTR_EQ(0xE490B000, a.ldr(r11, mem[r0], 0));
  EXPECT_INSTR_EQ(0xE490B060, a.ldr(r11, mem[r0], 96));
  // Offsets out of bounds.
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, 4096)));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, -4096)));

  EXPECT_INSTR_EQ(0x31A0C003, a.movlo(r12, r3));
  EXPECT_INSTR_EQ(0x91A0A00C, a.movls(r10, r12));
  EXPECT_INSTR_EQ(0xE1A0A00C, a.mov(r10, r12));

  EXPECT_INSTR_EQ(0xE92D0FF0, a.push({r4, r5, r6, r7, r8, r9, r10, r11}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({r1}));

  EXPECT_INSTR_EQ(0xF5D3F000, a.pld(MemOperand(r3, 0)));
  EXPECT_INSTR_EQ(0xF5D3F040, a.pld(MemOperand(r3, 64)));

  EXPECT_INSTR_EQ(0xE0487002, a.sub(r7, r8, r2));
  EXPECT_INSTR_EQ(0xE2525010, a.subs(r5, r2, 16));
}

TEST(AArch32Assembler, CoreRegisterList) {
  EXPECT_EQ(0x3, CoreRegisterList({r0, r1}));
  EXPECT_EQ(0xFC00, CoreRegisterList({r10, r11, r12, r13, r14, r15}));

  EXPECT_FALSE(CoreRegisterList({}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r0}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r1}).has_more_than_one_register());
  EXPECT_TRUE(CoreRegisterList({r0, r1}).has_more_than_one_register());
}

TEST(AArch32Assembler, MemOperand) {
  EXPECT_EQ(MemOperand(r0, 4, AddressingMode::kOffset), (mem[r0, 4]));
}
}  // namespace aarch32
}  // namespace xnnpack

#undef CHECK_INSTR
#undef EXPECT_INSTR_EQ
#undef EXPECT_ERROR
