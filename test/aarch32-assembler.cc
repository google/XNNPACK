// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <ios>
#include <random>

#include <gtest/gtest.h>

#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/common.h>
#include "assembler-helpers.h"


namespace xnnpack {
namespace aarch32 {
TEST(AArch32Assembler, InstructionEncoding) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  CHECK_ENCODING(0xE086600B, a.add(r6, r11));
  CHECK_ENCODING(0xE0810002, a.add(r0, r1, r2));
  CHECK_ENCODING(0xE28A9080, a.add(r9, r10, 128));
  CHECK_ENCODING(0xE29D5008, a.adds(r5, r13, 8));

  CHECK_ENCODING(0xE2025007, a.and_(r5, r2, 7));

  CHECK_ENCODING(0xE12FFF38, a.blx(r8));

  CHECK_ENCODING(0xE3CC2003, a.bic(r2, r12, 3));

  CHECK_ENCODING(0xE12FFF1E, a.bx(lr));

  CHECK_ENCODING(0xE3500002, a.cmp(r0, 2));
  CHECK_ENCODING(0xE1530007, a.cmp(r3, r7));

  // Offset addressing mode.
  CHECK_ENCODING(0xE59D7060, a.ldr(r7, mem[sp, 96]));
  // Post-indexed addressing mode.
  CHECK_ENCODING(0xE490B000, a.ldr(r11, mem[r0], 0));
  CHECK_ENCODING(0xE490B060, a.ldr(r11, mem[r0], 96));
  // Offsets out of bounds.
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, 4096)));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(r7, MemOperand(sp, -4096)));

  CHECK_ENCODING(0xE1CD66D8, a.ldrd(r6, r7, mem[sp, 104]));
  CHECK_ENCODING(0xE0CD66D8, a.ldrd(r6, r7, MemOperand(sp, 104, AddressingMode::kPostIndexed)));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldrd(r6, r8, mem[sp, 104]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldrd(r6, r7, mem[sp, 4096]));

  CHECK_ENCODING(0x01A0C007, a.moveq(r12, r7));
  CHECK_ENCODING(0x31A0C003, a.movlo(r12, r3));
  CHECK_ENCODING(0x91A0A00C, a.movls(r10, r12));
  CHECK_ENCODING(0xE1A0A00C, a.mov(r10, r12));

  CHECK_ENCODING(0xE30D3EAD, a.mov(r3, 0xDEAD));
  CHECK_ENCODING(0xE34D3EAD, a.movt(r3, 0xDEAD));

  CHECK_ENCODING(0xE320F000, a.nop());

  CHECK_ENCODING(0xE8BD0FF0, a.pop({r4, r5, r6, r7, r8, r9, r10, r11}));
  EXPECT_ERROR(Error::kInvalidOperand, a.pop({}));
  EXPECT_ERROR(Error::kInvalidOperand, a.pop({r1}));

  CHECK_ENCODING(0xE92D0FF0, a.push({r4, r5, r6, r7, r8, r9, r10, r11}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({r1}));

  CHECK_ENCODING(0xF5D3F000, a.pld(MemOperand(r3, 0)));
  CHECK_ENCODING(0xF5D3F040, a.pld(MemOperand(r3, 64)));

  CHECK_ENCODING(0xE58D5068, a.str(r5, mem[sp, 104]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(r5, MemOperand(sp, 4096)));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(r5, MemOperand(sp, -4096)));

  CHECK_ENCODING(0xE0487002, a.sub(r7, r8, r2));
  CHECK_ENCODING(0xE2425010, a.sub(r5, r2, 16));
  CHECK_ENCODING(0xE2525010, a.subs(r5, r2, 16));

  CHECK_ENCODING(0xE315000F, a.tst(r5, 15));

  CHECK_ENCODING(0xF3B9676E, a.vabs_f32(q3, q15));

  CHECK_ENCODING(0xF24E2DC2, a.vadd_f32(q9, q15, q1));

  CHECK_ENCODING(0xEEB44AC8, a.vcmpe_f32(s8, s16));

  CHECK_ENCODING(0xF3FBE646, a.vcvt_f32_s32(q15, q3));
  CHECK_ENCODING(0xF3FB6748, a.vcvt_s32_f32(q11, q4));

  CHECK_ENCODING(0xF3FB6148, a.vcvtn_s32_f32(q11, q4));

  CHECK_ENCODING(0xF3FF8C4F, a.vdup_8(q12, d15[7]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vdup_8(q12, d15[8]));
  CHECK_ENCODING(0xF3FE8C4F, a.vdup_16(q12, d15[3]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vdup_16(q12, d15[4]));
  CHECK_ENCODING(0xF3FC8C4F, a.vdup_32(q12, d15[1]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vdup_32(q12, d15[2]));

  CHECK_ENCODING(0xF2BE04C6, a.vext_8(q0, q15, q3, 4));
  EXPECT_ERROR(Error::kInvalidOperand, a.vext_8(q0, q15, q3, 16));

  CHECK_ENCODING(0xF423070F, a.vld1_8({d0}, mem[r3]));
  CHECK_ENCODING(0xF423070D, a.vld1_8({d0}, mem[r3]++));
  CHECK_ENCODING(0xF4230A0F, a.vld1_8({d0-d1}, mem[r3]));
  CHECK_ENCODING(0xF423060F, a.vld1_8({d0-d2}, mem[r3]));
  CHECK_ENCODING(0xF423020F, a.vld1_8({d0-d3}, mem[r3]));
  CHECK_ENCODING(0xF42A4705, a.vld1_8({d4}, mem[r10], r5));
  CHECK_ENCODING(0xF4294A0D, a.vld1_8({q2}, mem[r9]++));

  CHECK_ENCODING(0xF42C178F, a.vld1_32({d1}, mem[r12]));
  CHECK_ENCODING(0xF42C178D, a.vld1_32({d1}, mem[r12]++));
  CHECK_ENCODING(0xF42C1A8D, a.vld1_32({d1-d2}, mem[r12]++));
  CHECK_ENCODING(0xF42C168D, a.vld1_32({d1-d3}, mem[r12]++));
  CHECK_ENCODING(0xF42C128D, a.vld1_32({d1-d4}, mem[r12]++));

  CHECK_ENCODING(0xF4A8780F, a.vld1_32({d7[0]}, mem[r8]));
  CHECK_ENCODING(0xF4A3488D, a.vld1_32({d4[1]}, mem[r3]++));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vld1_32({d0[2]}, mem[r3]));

  CHECK_ENCODING(0xF4294A8D, a.vld1_32({q2}, mem[r9]++));

  CHECK_ENCODING(0xF4A54C8F, a.vld1r_32({d4}, mem[r5]));
  CHECK_ENCODING(0xF4A54CAF, a.vld1r_32({d4, d5}, mem[r5]));
  CHECK_ENCODING(0xF4A54CAD, a.vld1r_32({d4, d5}, mem[r5]++));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld1r_32({d4, d5}, mem[r5, 4]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld1r_32({d4, d6}, mem[r5]));

  CHECK_ENCODING(0xF4A54D8F, a.vld2r_32({d4, d5}, mem[r5]));
  CHECK_ENCODING(0xF4A54DAF, a.vld2r_32({d4, d6}, mem[r5]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld2r_32({d4, d5}, mem[r5, 4]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld2r_32({d4, d7}, mem[r5]));

  CHECK_ENCODING(0xF4A54E8F, a.vld3r_32({d4, d5, d6}, mem[r5]));
  CHECK_ENCODING(0xF4A54EAF, a.vld3r_32({d4, d6, d8}, mem[r5]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld3r_32({d4, d5, d6}, mem[r5, 4]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld3r_32({d4, d5, d7}, mem[r5]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld3r_32({d4, d6, d7}, mem[r5]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vld3r_32({d4, d6, d9}, mem[r5]));

  CHECK_ENCODING(0xECD90B08, a.vldm(mem[r9], {d16-d19}));
  CHECK_ENCODING(0xECF90B08, a.vldm(mem[r9]++, {d16-d19}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vldm(mem[r9], {d8-d0}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vldm(mem[r9], {d0-d16}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vldm(mem[r9], DRegisterList(d31, 2)));

  CHECK_ENCODING(0xEC930A01, a.vldm(mem[r3], {s0}));
  CHECK_ENCODING(0xECB30A01, a.vldm(mem[r3]++, {s0}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vldm(mem[r3], {s4-s0}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vldm(mem[r3], SRegisterList(s31, 2)));

  CHECK_ENCODING(0xEDD97A0E, a.vldr(s15, mem[r9, 56]));
  CHECK_ENCODING(0xEDD97AFF, a.vldr(s15, mem[r9, 1020]));
  CHECK_ENCODING(0xED597AFF, a.vldr(s15, mem[r9, -1020]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(s15, MemOperand(r9, 56, AddressingMode::kPostIndexed)));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(s15, mem[r9, 1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(s15, mem[r9, -1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(s15, mem[r9, 1018]));

  CHECK_ENCODING(0xED99FB0E, a.vldr(d15, mem[r9, 56]));
  CHECK_ENCODING(0xED99FBFF, a.vldr(d15, mem[r9, 1020]));
  CHECK_ENCODING(0xED19FBFF, a.vldr(d15, mem[r9, -1020]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, MemOperand(r9, 56, AddressingMode::kPostIndexed)));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, mem[r9, 1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, mem[r9, -1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vldr(d15, mem[r9, 1018]));

  CHECK_ENCODING(0xF20E26C6, a.vmax_s8(q1, q15, q3));
  CHECK_ENCODING(0xF24ECFC4, a.vmax_f32(q14, q15, q2));

  CHECK_ENCODING(0xF20E26D6, a.vmin_s8(q1, q15, q3));
  CHECK_ENCODING(0xF220EFC6, a.vmin_f32(q7, q8, q3));

  CHECK_ENCODING(0xEE04AA01, a.vmla_f32(s20, s8, s2));

  CHECK_ENCODING(0xF3E80140, a.vmla_f32(q8, q4, d0[0]));
  CHECK_ENCODING(0xF3EC0160, a.vmla_f32(q8, q6, d0[1]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vmla_f32(q8, q4, d0[2]));

  CHECK_ENCODING(0xF2D9E246, a.vmlal_s16(q15, d9, d6[0]));
  CHECK_ENCODING(0xF2D8424A, a.vmlal_s16(q10, d8, d2[1]));
  CHECK_ENCODING(0xF2D88264, a.vmlal_s16(q12, d8, d4[2]));
  CHECK_ENCODING(0xF2D8626A, a.vmlal_s16(q11, d8, d2[3]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vmlal_s16(q15, d9, d6[4]));

  CHECK_ENCODING(0xF2C0E050, a.vmov(q15, 0));
  EXPECT_ERROR(Error::kInvalidOperand, a.vmov(q15, 1));

  CHECK_ENCODING(0xEE1E4A10, a.vmov(r4, s28));
  CHECK_ENCODING(0xEE0E4A10, a.vmov(s28, r4));
  CHECK_ENCODING(0xEEB0EA4F, a.vmov(s28, s30));
  CHECK_ENCODING(0xF2245114, a.vmov(d5, d4));
  CHECK_ENCODING(0xF26101B1, a.vmov(d16, d17));
  CHECK_ENCODING(0xEC420B1F, a.vmov(d15, r0, r2));
  CHECK_ENCODING(0xEC454B18, a.vmov(d8, r4, r5));
  CHECK_ENCODING(0xEC554B18, a.vmov(r4, r5, d8));
  CHECK_ENCODING(0xF26041F0, a.vmov(q10, q8));

  CHECK_ENCODING(0xEEB08A49, a.vmov_f32(s16, s18));
  CHECK_ENCODING(0x5EB08A44, a.vmovpl_f32(s16, s8));
  CHECK_ENCODING(0x4EB08A64, a.vmovmi_f32(s16, s9));

  CHECK_ENCODING(0xEEB0AB48, a.vmov_f64(d10, d8));

  CHECK_ENCODING(0xF2880A10, a.vmovl_s8(q0, d0));

  CHECK_ENCODING(0xEEF1FA10, a.vmrs(APSR_nzcv, FPSCR));

  CHECK_ENCODING(0xF34E2DD2, a.vmul_f32(q9, q15, q1));
  CHECK_ENCODING(0xF3EE29C7, a.vmul_f32(q9, q15, d7[0]));
  CHECK_ENCODING(0xF3EE29E7, a.vmul_f32(q9, q15, d7[1]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vmul_f32(q9, q15, d7[2]));

  CHECK_ENCODING(0xF3F927EE, a.vneg_f32(q9, q15));

  CHECK_ENCODING(0xECBD8B10, a.vpop({d8-d15}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpop({d0-d16}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpop({d4-d0}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpop(DRegisterList(d31, 2)));

  CHECK_ENCODING(0xED2D8B10, a.vpush({d8-d15}));
  CHECK_ENCODING(0xED6D4B08, a.vpush({d20-d23}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpush({d8-d7}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpush({d0-d16}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpush(DRegisterList(d31, 2)));

  CHECK_ENCODING(0xED2D4A08, a.vpush({s8-s15}));
  CHECK_ENCODING(0xED2DAA04, a.vpush({s20-s23}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpush({s8-s2}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vpush(SRegisterList(s31, 2)));

  CHECK_ENCODING(0xF25E00D2, a.vqadd_s16(q8, q15, q1));

  CHECK_ENCODING(0xF3A82CCE, a.vqdmulh_s32(q1, q12, d14[0]));
  CHECK_ENCODING(0xF3A82CEE, a.vqdmulh_s32(q1, q12, d14[1]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vqdmulh_s32(q1, q12, d14[2]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vqdmulh_s32(q1, q12, d16[0]));

  CHECK_ENCODING(0xF3B232A6, a.vqmovn_s16(d3, q11));
  CHECK_ENCODING(0xF3F602A0, a.vqmovn_s32(d16, q8));

  CHECK_ENCODING(0xF22C247E, a.vqshl_s32(q1, q15, q6));

  CHECK_ENCODING(0xF264C560, a.vrshl_s32(q14, q8, q2));

  CHECK_ENCODING(0xFE666D41, a.vsdot_s8(q11, q3, d1[0]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vsdot_s8(q11, q3, d1[2]));

  CHECK_ENCODING(0xF40B070F, a.vst1_8({d0}, mem[r11]));
  CHECK_ENCODING(0xF40B070D, a.vst1_8({d0}, mem[r11]++));
  CHECK_ENCODING(0xF40B0707, a.vst1_8({d0}, mem[r11], r7));
  CHECK_ENCODING(0xF48B000F, a.vst1_8({d0[0]}, mem[r11]));
  CHECK_ENCODING(0xF48B00EF, a.vst1_8({d0[7]}, mem[r11]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vst1_8(d0[8], mem[r11]));

  CHECK_ENCODING(0xF40B074F, a.vst1_16({d0}, mem[r11]));
  CHECK_ENCODING(0xF40B074D, a.vst1_16({d0}, mem[r11]++));
  CHECK_ENCODING(0xF40B0747, a.vst1_16({d0}, mem[r11], r7));
  CHECK_ENCODING(0xF48B040F, a.vst1_16({d0[0]}, mem[r11]));
  CHECK_ENCODING(0xF48B04CF, a.vst1_16({d0[3]}, mem[r11]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vst1_16(d0[4], mem[r11]));

  CHECK_ENCODING(0xF44B0280, a.vst1_32({d16-d19}, mem[r11], r0));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vst1_32({d0-d4}, mem[r11], r0));
  EXPECT_ERROR(Error::kInvalidOperand, a.vst1_32({d16-d19}, mem[r11], sp));
  EXPECT_ERROR(Error::kInvalidOperand, a.vst1_32({d16-d19}, mem[r11], pc));
  CHECK_ENCODING(0xF404168F, a.vst1_32({d1-d3}, mem[r4]));
  CHECK_ENCODING(0xF44B0A8D, a.vst1_32({d16-d17}, mem[r11]++));
  CHECK_ENCODING(0xF4CB080F, a.vst1_32({d16[0]}, mem[r11]));
  // The surrounding braces are optional, but makes it look closer to native assembly.
  CHECK_ENCODING(0xF4CB080F, a.vst1_32(d16[0], mem[r11]));
  CHECK_ENCODING(0xF4CB088F, a.vst1_32(d16[1], mem[r11]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.vst1_32(d16[2], mem[r11]));
  CHECK_ENCODING(0xF4C6C80D, a.vst1_32({d28[0]}, mem[r6]++));

  CHECK_ENCODING(0xEC868B04, a.vstm(mem[r6], {d8-d9}));
  CHECK_ENCODING(0xECA7EB02, a.vstm(mem[r7]++, {d14}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vstm(mem[r6], {d8-d28}));
  EXPECT_ERROR(Error::kInvalidRegisterListLength, a.vstm(mem[r6], DRegisterList(d31, 2)));

  CHECK_ENCODING(0xED868A00, a.vstr(s16, mem[r6]));
  CHECK_ENCODING(0xED868A02, a.vstr(s16, mem[r6, 8]));
  CHECK_ENCODING(0xED868AFF, a.vstr(s16, mem[r6, 1020]));
  CHECK_ENCODING(0xED068AFF, a.vstr(s16, mem[r6, -1020]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vstr(s16, MemOperand(r6, 8, AddressingMode::kPostIndexed)));
  EXPECT_ERROR(Error::kInvalidOperand, a.vstr(s16, mem[r6, 1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vstr(s16, mem[r6, -1024]));
  EXPECT_ERROR(Error::kInvalidOperand, a.vstr(s16, mem[r6, 1018]));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch32Assembler, Label) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  Label l1;
  a.add(r0, r0, r0);

  // Branch to unbound label.
  auto b1 = a.offset<uint32_t*>();
  a.beq(l1);

  a.add(r1, r1, r1);

  auto b2 = a.offset<uint32_t*>();
  a.bne(l1);

  a.add(r2, r2, r2);

  a.bind(l1);

  // Check that b1 and b2 are both patched after binding l1.
  EXPECT_INSTR(0x0A000002, *b1);
  EXPECT_INSTR(0x1A000000, *b2);

  a.add(r0, r1, r2);

  // Branch to bound label.
  auto b3 = a.offset<uint32_t*>();
  a.bhi(l1);
  auto b4 = a.offset<uint32_t*>();
  a.bhs(l1);
  auto b5 = a.offset<uint32_t*>();
  a.blo(l1);
  auto b6 = a.offset<uint32_t*>();
  a.b(l1);

  EXPECT_INSTR(0x8AFFFFFD, *b3);
  EXPECT_INSTR(0x2AFFFFFC, *b4);
  EXPECT_INSTR(0x3AFFFFFB, *b5);
  EXPECT_INSTR(0xEAFFFFFA, *b6);

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

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch32Assembler, Align) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  a.add(r0, r1, r2);
  a.align(4);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x3);
  EXPECT_EQ(4, a.code_size_in_bytes());

  a.align(8);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x7);
  EXPECT_EQ(8, a.code_size_in_bytes());

  a.add(r0, r1, r2);
  a.align(8);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x7);
  EXPECT_EQ(16, a.code_size_in_bytes());

  a.add(r0, r1, r2);
  EXPECT_EQ(20, a.code_size_in_bytes());

  a.align(16);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0xF);
  EXPECT_EQ(32, a.code_size_in_bytes());

  a.add(r0, r1, r2);
  a.add(r0, r1, r2);
  EXPECT_EQ(40, a.code_size_in_bytes());

  a.align(16);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0xF);
  EXPECT_EQ(48, a.code_size_in_bytes());

  // Not power-of-two.
  EXPECT_ERROR(Error::kInvalidOperand, a.align(6));
  // Is power-of-two but is not a multiple of instruction size.
  EXPECT_ERROR(Error::kInvalidOperand, a.align(2));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
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
  SRegisterList s_list_1 = SRegisterList(s0, s9);
  EXPECT_EQ(s_list_1.start, s0);
  EXPECT_EQ(s_list_1.length, 10);

  SRegisterList s_list_2 = {s4 - s11};
  EXPECT_EQ(s_list_2.start, s4);
  EXPECT_EQ(s_list_2.length, 8);

  DRegisterList d_list_1 = DRegisterList(d4, d5);
  EXPECT_EQ(d_list_1.start, d4);
  EXPECT_EQ(d_list_1.length, 2);

  DRegisterList d_list_2 = {d4 - d11};
  EXPECT_EQ(d_list_2.start, d4);
  EXPECT_EQ(d_list_2.length, 8);

  QRegisterList q_list_1 = {q3-q3};
  EXPECT_EQ(q_list_1.start, q3);
  EXPECT_EQ(q_list_1.length, 1);

  DRegisterList d_from_q_1 = static_cast<DRegisterList>(q_list_1);
  EXPECT_EQ(d_from_q_1.start, d6);
  EXPECT_EQ(d_from_q_1.length, 2);

  QRegisterList q_list_2 = {q4-q9};
  EXPECT_EQ(q_list_2.start, q4);
  EXPECT_EQ(q_list_2.length, 6);

  DRegisterList d_from_q_2 = static_cast<DRegisterList>(q_list_2);
  EXPECT_EQ(d_from_q_2.start, d8);
  EXPECT_EQ(d_from_q_2.length, 12);
}

TEST(AArch32Assembler, MemOperand) {
  EXPECT_EQ(MemOperand(r0, 4, AddressingMode::kOffset), (mem[r0, 4]));
}

TEST(AArch32Assembler, DRegisterLane) {
  EXPECT_EQ((DRegisterLane{2, 0}), d2[0]);
  EXPECT_EQ((DRegisterLane{2, 1}), d2[1]);
}

TEST(AArch32Assembler, QRegister) {
  EXPECT_EQ(q0.low(), d0);
  EXPECT_EQ(q0.high(), d1);
  EXPECT_EQ(q1.low(), d2);
  EXPECT_EQ(q1.high(), d3);
  EXPECT_EQ(q15.low(), d30);
  EXPECT_EQ(q15.high(), d31);
}

TEST(AArch32Assembler, CodeBufferOverflow) {
  xnn_code_buffer b;
  // Requested memory is rounded to page size.
  xnn_allocate_code_memory(&b, 4);
  Assembler a(&b);
  for (int i = 0; i < b.capacity; i += 1 << kInstructionSizeInBytesLog2) {
    a.add(r0, r0, 2);
  }
  EXPECT_EQ(Error::kNoError, a.error());

  a.bx(lr);
  EXPECT_EQ(Error::kOutOfMemory, a.error());

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch32Assembler, BoundOverflow) {
  xnn_code_buffer b;
  // Requested memory is rounded to page size.
  xnn_allocate_code_memory(&b, 4);
  Assembler a(&b);
  Label l1;
  for (int i = 0; i < b.capacity; i += 1 << kInstructionSizeInBytesLog2) {
    a.add(r0, r0, 2);
  }
  EXPECT_EQ(Error::kNoError, a.error());

  // This is out of bounds, not written.
  a.bhi(l1);
  EXPECT_EQ(Error::kOutOfMemory, a.error());

  a.bind(l1);
  EXPECT_EQ(false, l1.bound);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
TEST(AArch32Assembler, JitAllocCodeBuffer) {
  typedef uint32_t (*Func)(uint32_t);

  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  Assembler a(&b);
  a.add(r0, r0, 2);
  a.bx(lr);

  Func fn = reinterpret_cast<Func>(a.finalize());
  xnn_finalize_code_memory(&b);

  ASSERT_EQ(3, fn(1));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
JitF32HardswishFn GenerateF32Hardswish(MacroAssembler& a, std::vector<QRegister> accs, std::vector<QRegister> tmps) {
  const QRegister sixth = q0;
  const QRegister three = q1;
  const QRegister six = q2;
  const QRegister zero = q3;

  // No callee-saved GPR registers used.
  a.vpush({d8-d15});  // callee-saved NEON registers
  // Load params.
  a.vld3r_32({sixth.low(), three.low(), six.low()}, mem[r2]);
  a.vmov(three.high(), three.low());
  a.vmov(six.high(), six.low());
  a.vmov(zero, 0);
  // Load inputs.
  for (size_t i = 0; i < accs.size(); i++) {
    a.vld1_32({accs[i].low(), accs[i].high()}, mem[r0]++);
  }
  a.f32_hardswish(
      sixth, three, six, zero,
      accs.data(),
      accs.size(),
      tmps.data(),
      tmps.size());
  // Write results of hardswish.
  for (size_t i = 0; i < accs.size(); i++) {
    a.vst1_32({accs[i].low(), accs[i].high()}, mem[r1]++);
  }
  a.vpop({d8-d15});
  a.bx(lr);

  return reinterpret_cast<JitF32HardswishFn>(a.finalize());
}

class F32HardswishTest : public testing::TestWithParam<std::vector<QRegister>> {};

TEST_P(F32HardswishTest, F32Hardswish) {
  xnn_code_buffer buffer;
  xnn_allocate_code_memory(&buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  MacroAssembler assembler(&buffer);

  const std::vector<QRegister> accs = GetParam();
  const std::vector<QRegister> tmps = {q8, q9, q10, q11};

  std::random_device random_device;
  std::mt19937 rng(random_device());
  std::uniform_real_distribution<float> f32dist(-6.0f, 6.0f);

  xnn_f32_hswish_params params;
  xnn_init_f32_hswish_scalar_params(&params);

  std::vector<float> input(4 * accs.size());
  std::vector<float> output(4 * accs.size());
  std::vector<float> expected_output(output);

  std::generate(input.begin(), input.end(), [&]{ return f32dist(rng); });
  std::fill(output.begin(), output.end(), std::nanf(""));
  std::fill(expected_output.begin(), expected_output.end(), std::nanf(""));

  // Call generated function.
  JitF32HardswishFn jit_f32_hardswish_fn = GenerateF32Hardswish(assembler, accs, tmps);
  EXPECT_EQ(Error::kNoError, assembler.error());
  xnn_finalize_code_memory(&buffer);
  jit_f32_hardswish_fn(input.data(), output.data(), &params);

  // Compute reference results.
  std::transform(input.begin(), input.end(), expected_output.begin(), hardswish);

  // Verify results.
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], expected_output[i], std::max(5.0e-6, std::abs(expected_output[i]) * 1.0e-5))
        << "at " << i << " / " << output.size() << ", x[" << i << "] = " << input[i];
  }
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));
}

INSTANTIATE_TEST_SUITE_P(
  AArch32Assembler,
  F32HardswishTest,
  testing::Values(
    std::vector<QRegister>({q4, q5, q6, q7}), std::vector<QRegister>({q4, q5, q6, q7, q12, q13, q14, q15})));

typedef void (*MovFn)(uint32_t*);

TEST(MovTest, Mov) {
  xnn_code_buffer buffer;
  xnn_allocate_code_memory(&buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  MacroAssembler assm(&buffer);

  const uint32_t expected = 0x01234567;
  assm.Mov(r1, expected);
  assm.str(r1, mem[r0]);
  assm.bx(lr);

  MovFn mov_fn = reinterpret_cast<MovFn>(assm.finalize());
  xnn_finalize_code_memory(&buffer);

  uint32_t out = 0;
  mov_fn(&out);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));

  EXPECT_EQ(expected, out);
}

#endif
}  // namespace aarch32
}  // namespace xnnpack
