#include <xnnpack/aarch32-assembler.h>

#include <ios>

#include <gtest/gtest.h>

// clang-format off
#define EXPECT_INSTR(expected, actual)                                                                        \
  EXPECT_EQ(expected, actual) << "expected = 0x" << std::hex << std::setw(8) << std::setfill('0') << expected \
                              << std::endl << "  actual = 0x" << actual;
// clang-format on

#define CHECK_ENCODING(expected, call) \
  a.reset();                           \
  call;                                \
  EXPECT_INSTR(expected, *a.start())

#define EXPECT_ERROR(expected, call) \
  a.reset();                         \
  call;                              \
  EXPECT_EQ(expected, a.error());

namespace xnnpack {
namespace aarch32 {
TEST(AArch32Assembler, InstructionEncoding) {
  Assembler a;

  CHECK_ENCODING(0xE0810002, a.add(r0, r1, r2));
  CHECK_ENCODING(0xE28A9080, a.add(r9, r10, 128));

  CHECK_ENCODING(0xE3500002, a.cmp(r0, 2));

  // Offset addressing mode.
  CHECK_ENCODING(0xE59D7060, a.ldr(r7, mem[sp, 96]));
  // Post-indexed addressing mode.
  CHECK_ENCODING(0xE490B000, a.ldr(r11, mem[r0], 0));
  CHECK_ENCODING(0xE490B060, a.ldr(r11, mem[r0], 96));
  // Offsets out of bounds.
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, 4096)));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, -4096)));

  CHECK_ENCODING(0x31A0C003, a.movlo(r12, r3));
  CHECK_ENCODING(0x91A0A00C, a.movls(r10, r12));
  CHECK_ENCODING(0xE1A0A00C, a.mov(r10, r12));

  CHECK_ENCODING(0xE92D0FF0, a.push({r4, r5, r6, r7, r8, r9, r10, r11}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({r1}));

  CHECK_ENCODING(0xF5D3F000, a.pld(MemOperand(r3, 0)));
  CHECK_ENCODING(0xF5D3F040, a.pld(MemOperand(r3, 64)));

  CHECK_ENCODING(0xE0487002, a.sub(r7, r8, r2));
  CHECK_ENCODING(0xE2525010, a.subs(r5, r2, 16));

  CHECK_ENCODING(0xE315000F, a.tst(r5, 15));

  CHECK_ENCODING(0xF42C178F, a.vld1_32({d1}, mem[r12]));
  CHECK_ENCODING(0xF42C178D, a.vld1_32({d1}, mem[r12]++));

  CHECK_ENCODING(0xF4A54CAF, a.vld1r_32({d4, d5}, mem[r5]));
  CHECK_ENCODING(0xF4A54CAD, a.vld1r_32({d4, d5}, mem[r5]++));

  CHECK_ENCODING(0xECF90B08, a.vldm(r9, {d16, d19}, true));
  CHECK_ENCODING(0xEC998B08, a.vldm(r9, {d8, d11}, false));
  CHECK_ENCODING(0xEC998B08, a.vldm(r9, {d8, d11}));
  CHECK_ENCODING(0xECB30A01, a.vldm(r3, {s0}, true));
  CHECK_ENCODING(0xEC930A01, a.vldm(r3, {s0}));

  CHECK_ENCODING(0xED99FB0E, a.vldr(d15, mem[r9, 56]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, MemOperand(r9, 56, AddressingMode::kPostIndexed)));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, mem[r9, 256]));

  CHECK_ENCODING(0xF24ECFC4, a.vmax_f32(q14, q15, q2));

  CHECK_ENCODING(0xF220EFC6, a.vmin_f32(q7, q8, q3));

  CHECK_ENCODING(0xF3E80140, a.vmla_f32(q8, q4, d0[0]));
  CHECK_ENCODING(0xF3EC0160, a.vmla_f32(q8, q6, d0[1]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vmla_f32(q8, q4, d0[2]));

  CHECK_ENCODING(0xEEB0EA4F, a.vmov(s28, s30));
  CHECK_ENCODING(0xF26101B1, a.vmov(d16, d17));
  CHECK_ENCODING(0xEC420B1F, a.vmov(d15, r0, r2));
  CHECK_ENCODING(0xF26041F0, a.vmov(q10, q8));

  CHECK_ENCODING(0xED2D4A08, a.vpush({s8, s15}));
  CHECK_ENCODING(0xED2DAA04, a.vpush({s20, s23}));
  CHECK_ENCODING(0xED2D8B10, a.vpush({d8, d15}));
  CHECK_ENCODING(0xED6D4B08, a.vpush({d20, d23}));

  CHECK_ENCODING(0xF44B0280, a.vst1_32({d16, d19}, mem[r11], r0));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vst1_32({d0, d4}, mem[r11], r0));
  EXPECT_ERROR(Error::kInvalidOperand, a.vst1_32({d16, d19}, mem[r11], sp));
  EXPECT_ERROR(Error::kInvalidOperand, a.vst1_32({d16, d19}, mem[r11], pc));
  CHECK_ENCODING(0xF404168F, a.vst1_32({d1, d3}, mem[r4]));
  CHECK_ENCODING(0xF44B0A8D, a.vst1_32({d16, d17}, mem[r11]++));
  CHECK_ENCODING(0xF4CB080F, a.vst1_32({d16[0]}, mem[r11]));
  // The surrounding braces are optional, but makes it look closer to native assembly.
  CHECK_ENCODING(0xF4CB080F, a.vst1_32(d16[0], mem[r11]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vst1_32(d16[2], mem[r11]));
  CHECK_ENCODING(0xF4C6C80D, a.vst1_32({d28[0]}, mem[r6]++));
}

TEST(AArch32Assembler, Label) {
  Assembler a;

  Label l1;
  a.add(r0, r0, r0);

  // Branch to unbound label.
  auto b1 = a.offset();
  a.beq(l1);

  a.add(r1, r1, r1);

  auto b2 = a.offset();
  a.bne(l1);

  a.add(r2, r2, r2);

  a.bind(l1);

  // Check that b1 and b2 are both patched after binding l1.
  EXPECT_INSTR(0x0A000002, *b1);
  EXPECT_INSTR(0x1A000000, *b2);

  a.add(r0, r1, r2);

  // Branch to bound label.
  auto b3 = a.offset();
  a.bhi(l1);
  auto b4 = a.offset();
  a.bhs(l1);
  auto b5 = a.offset();
  a.blo(l1);

  EXPECT_INSTR(0x8AFFFFFD, *b3);
  EXPECT_INSTR(0x2AFFFFFC, *b4);
  EXPECT_INSTR(0x3AFFFFFB, *b5);

  // Binding a bound label is an error.
  a.bind(l1);
  EXPECT_ERROR(Error::kLabelAlreadyBound, a.bind(l1));

  // Check for bind failure due to too many users of label.
  Label lfail;
  a.reset();
  // Arbitrary high number of users that we probably won't support.
  for (int i = 0; i < 1000; i++) {
    a.beq(lfail);
  }
  EXPECT_EQ(Error::kLabelHasTooManyUsers, a.error());
}

TEST(AArch32Assembler, CoreRegisterList) {
  EXPECT_EQ(0x3, CoreRegisterList({r0, r1}));
  EXPECT_EQ(0xFC00, CoreRegisterList({r10, r11, r12, r13, r14, r15}));

  EXPECT_FALSE(CoreRegisterList({}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r0}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r1}).has_more_than_one_register());
  EXPECT_TRUE(CoreRegisterList({r0, r1}).has_more_than_one_register());
}

TEST(AArch32Assembler, ConsecutiveRegisterList) {
  SRegisterList s1 = SRegisterList(s0, s9);
  EXPECT_EQ(s1.start, s0);
  EXPECT_EQ(s1.length, 10);

  DRegisterList d1 = DRegisterList(d4, d5);
  EXPECT_EQ(d1.start, d4);
  EXPECT_EQ(d1.length, 2);
}

TEST(AArch32Assembler, MemOperand) {
  EXPECT_EQ(MemOperand(r0, 4, AddressingMode::kOffset), (mem[r0, 4]));
}

TEST(AArch32Assembler, DRegisterLane) {
  EXPECT_EQ((DRegisterLane{2, 0}), d2[0]);
  EXPECT_EQ((DRegisterLane{2, 1}), d2[1]);
}
}  // namespace aarch32
}  // namespace xnnpack
