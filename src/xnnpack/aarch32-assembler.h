#include <array>
#include <cstdint>
#include <initializer_list>

namespace xnnpack {
namespace aarch32 {

struct CoreRegister {
  uint8_t code;
};

constexpr CoreRegister r0{0};
constexpr CoreRegister r1{1};
constexpr CoreRegister r2{2};
constexpr CoreRegister r3{3};
constexpr CoreRegister r4{4};
constexpr CoreRegister r5{5};
constexpr CoreRegister r6{6};
constexpr CoreRegister r7{7};
constexpr CoreRegister r8{8};
constexpr CoreRegister r9{9};
constexpr CoreRegister r10{10};
constexpr CoreRegister r11{11};
constexpr CoreRegister r12{12};
constexpr CoreRegister r13{13};
constexpr CoreRegister r14{14};
constexpr CoreRegister r15{15};
constexpr CoreRegister sp = r13;
constexpr CoreRegister lr = r14;
constexpr CoreRegister pc = r15;

static inline bool operator==(const CoreRegister lhs, const CoreRegister rhs) {
  return lhs.code == rhs.code;
}

struct CoreRegisterList {
  CoreRegisterList(std::initializer_list<CoreRegister> rs) {
    for (auto r : rs) {
      list |= 1 << r.code;
    }
  }

  bool has_more_than_one_register() { return (list & (list - 1)) != 0; }

  // Bit i is set if CoreRegister is in the list.
  uint16_t list = 0;
};

static inline bool operator==(int i, CoreRegisterList registers) {
  return i == registers.list;
}

struct SRegister {
  uint8_t code;
  uint8_t d() const { return code & 0x1; }
  uint8_t vd() const { return (code & 0x1e) >> 1; }
};

static inline bool operator==(const SRegister lhs, const SRegister rhs) {
  return lhs.code == rhs.code;
}

constexpr SRegister s0{0};
constexpr SRegister s1{1};
constexpr SRegister s2{2};
constexpr SRegister s3{3};
constexpr SRegister s4{4};
constexpr SRegister s5{5};
constexpr SRegister s6{6};
constexpr SRegister s7{7};
constexpr SRegister s8{8};
constexpr SRegister s9{9};
constexpr SRegister s10{10};
constexpr SRegister s11{11};
constexpr SRegister s12{12};
constexpr SRegister s13{13};
constexpr SRegister s14{14};
constexpr SRegister s15{15};
constexpr SRegister s16{16};
constexpr SRegister s17{17};
constexpr SRegister s18{18};
constexpr SRegister s19{19};
constexpr SRegister s20{20};
constexpr SRegister s21{21};
constexpr SRegister s22{22};
constexpr SRegister s23{23};
constexpr SRegister s24{24};
constexpr SRegister s25{25};
constexpr SRegister s26{26};
constexpr SRegister s27{27};
constexpr SRegister s28{28};
constexpr SRegister s29{29};
constexpr SRegister s30{30};
constexpr SRegister s31{31};

struct DRegister {
  uint8_t code;
  uint8_t d() const { return (code & 0x10) >> 4; }
  uint8_t vd() const { return code & 0xf; }
};

static inline bool operator==(const DRegister lhs, const DRegister rhs) {
  return lhs.code == rhs.code;
}

constexpr DRegister d0{0};
constexpr DRegister d1{1};
constexpr DRegister d2{2};
constexpr DRegister d3{3};
constexpr DRegister d4{4};
constexpr DRegister d5{5};
constexpr DRegister d6{6};
constexpr DRegister d7{7};
constexpr DRegister d8{8};
constexpr DRegister d9{9};
constexpr DRegister d10{10};
constexpr DRegister d11{11};
constexpr DRegister d12{12};
constexpr DRegister d13{13};
constexpr DRegister d14{14};
constexpr DRegister d15{15};
constexpr DRegister d16{16};
constexpr DRegister d17{17};
constexpr DRegister d18{18};
constexpr DRegister d19{19};
constexpr DRegister d20{20};
constexpr DRegister d21{21};
constexpr DRegister d22{22};
constexpr DRegister d23{23};
constexpr DRegister d24{24};
constexpr DRegister d25{25};
constexpr DRegister d26{26};
constexpr DRegister d27{27};
constexpr DRegister d28{28};
constexpr DRegister d29{29};
constexpr DRegister d30{30};
constexpr DRegister d31{31};

struct QRegister {
  uint8_t code;
};

constexpr QRegister q0{0};
constexpr QRegister q1{1};
constexpr QRegister q2{2};
constexpr QRegister q3{3};
constexpr QRegister q4{4};
constexpr QRegister q5{5};
constexpr QRegister q6{6};
constexpr QRegister q7{7};
constexpr QRegister q8{8};
constexpr QRegister q9{9};
constexpr QRegister q10{10};
constexpr QRegister q11{11};
constexpr QRegister q12{12};
constexpr QRegister q13{13};
constexpr QRegister q14{14};
constexpr QRegister q15{15};

// SIMD register lists are used in a more restrictive way, compared to core
// registers, only consecutive registers are used as an operand to instruction.
template <typename RegType>
struct ConsecutiveRegisterList {
  // End must be >= start.
  ConsecutiveRegisterList(RegType s, RegType end)
      : start(s), length(end.code - s.code + 1) {}
  ConsecutiveRegisterList(RegType start)
      : ConsecutiveRegisterList(start, start) {}

  RegType start;
  int length;
};

using SRegisterList = ConsecutiveRegisterList<SRegister>;
using DRegisterList = ConsecutiveRegisterList<DRegister>;

constexpr size_t max_label_users = 10;
// Label is a target of a branch. You call Assembler::bind to bind a label to an
// actual location in the instruction stream.
//
// ```
// Label l;
// b(kAl, l1); // branch to an unbound label is fine, it will be patched later.
// a.bind(l); // binds label to this location in the instruction stream.
// b(kAl, l1); // branch to an already bound label.
// ```
struct Label {
  // Location of label within Assembler buffer.
  uint32_t* offset = nullptr;
  // A label can only be bound once, binding it again leads to an error.
  bool bound = (offset != nullptr);
  // All users of this label, recorded by their offset in the Assembler buffer.
  std::array<uint32_t*, max_label_users> users = {0};
  size_t num_users = 0;

  // Records a user (e.g. branch instruction) of this label.
  // Returns true if success, false if number of users exceeds maximum.
  bool add_use(uint32_t* offset) {
    if (num_users >= max_label_users) {
      return false;
    }
    users[num_users++] = offset;
    return true;
  }
};

// A8.5 Addressing modes for memory access.
enum class AddressingMode {
  // [<Rn>, <offset>], offset applied to address in Rn.
  kOffset,
  // Pre-indexed not used, so not implemented.
  // [<Rn>], <offset>, address from Rn, offset applied, written back to Rn.
  kPostIndexed,
};

// Memory operands, operands for memory access instructions. See
// "MemOperandHelper mem" for a nicer syntax that is closer to assembly.
class MemOperand {
 public:
  MemOperand(CoreRegister rn, int32_t offset)
      : mode_(AddressingMode::kOffset), rn_(rn), offset_(offset) {}

  MemOperand(CoreRegister rn, int32_t offset, AddressingMode mode)
      : mode_(mode), rn_(rn), offset_(offset) {}

  CoreRegister base() const { return rn_; }
  int32_t offset() const { return offset_; }
  AddressingMode mode() const { return mode_; }

  // These are bits used for encoding, named based on the encoding description.
  int32_t u() { return offset_ >= 0; }
  int32_t p() { return mode_ != AddressingMode::kPostIndexed; }
  // Note, kPostIndexed will write back, but doesn't need to set bit w.
  int32_t w() { return 0; }

 private:
  AddressingMode mode_;
  CoreRegister rn_;
  int32_t offset_;
};

static inline bool operator==(const MemOperand lhs, const MemOperand rhs) {
  return lhs.mode() == rhs.mode() && lhs.base() == rhs.base() &&
         lhs.offset() == rhs.offset();
}

static inline MemOperand operator,(CoreRegister r, int32_t offset) {
  return MemOperand(r, offset);
}

// Helper struct for some syntax sugar to look like native assembly, see mem.
struct MemOperandHelper {
  const MemOperand operator[](MemOperand op) const { return op; }
  const MemOperand operator[](CoreRegister r) const { return MemOperand(r, 0); }
};

// Use "mem" (and its overload of array subscript operator) to get some syntax
// that looks closer to native assembly when accessing memory. For example:
// - ldr(r0, mem[rn, offset]); // offset
// - ldr(r0, mem[rn], offset); // post-indexed
constexpr MemOperandHelper mem;

// Conditional execution, only support AL (always) for now.
enum Condition : uint32_t {
  kEQ = 0x00000000,
  kNE = 0x10000000,
  kCS = 0x20000000,
  kCC = 0x30000000,
  kMI = 0x40000000,
  kPL = 0x50000000,
  kVS = 0x60000000,
  kVC = 0x70000000,
  kHI = 0x80000000,
  kLS = 0x90000000,
  kGE = 0xa0000000,
  kLT = 0xB0000000,
  kGT = 0xC0000000,
  kLE = 0xD0000000,
  kAL = 0xE0000000,
  kHS = kCS,
  kLO = kCC,
};

enum class Error {
  kNoError,
  kOutOfMemory,
  kInvalidOperand,
  kLabelAlreadyBound,
  kLabelOffsetOutOfBounds,
  kLabelHasTooManyUsers,
};

// A simple AAarch32 assembler.
// Right now it allocates its own memory (using `new`) to write code into (for
// testing), but will be updated to be more customizable.
class Assembler {
 public:
  explicit Assembler();
  ~Assembler();

  Assembler& add(CoreRegister Rd, CoreRegister Rn, CoreRegister Rm);
  Assembler& beq(Label& l) { return b(kEQ, l); }
  Assembler& bne(Label& l) { return b(kNE, l); }
  Assembler& bhi(Label& l) { return b(kHI, l); }
  Assembler& bhs(Label& l) { return b(kHS, l); }
  Assembler& blo(Label& l) { return b(kLO, l); }
  // Cmp supports a subset of uint32_t offsets, see "A5.2.4 Modified immediate
  // constants in ARM instructions", for simplicity we start with uint8_t, which
  // is fully representation using a "rotation" of 0.
  Assembler& cmp(CoreRegister Rn, uint8_t imm);
  Assembler& ldr(CoreRegister Rt, MemOperand operand, int32_t offset);
  Assembler& ldr(CoreRegister Rt, MemOperand operand);
  Assembler& mov(CoreRegister Rd, CoreRegister Rm);
  Assembler& movlo(CoreRegister Rd, CoreRegister Rm);
  Assembler& movls(CoreRegister Rd, CoreRegister Rm);
  Assembler& pld(MemOperand operand);
  Assembler& push(CoreRegisterList regs);
  Assembler& sub(CoreRegister Rd, CoreRegister Rn, CoreRegister Rm);
  // Only support uint8_t immediates for now, it simplifies encoding.
  Assembler& subs(CoreRegister Rd, CoreRegister Rn, uint8_t imm);

  // SIMD instructions.
  // VLDM <Rn>{!}, <list>. {!} is indicated by setting `wb` argument.
  Assembler& vldm(CoreRegister rn, SRegisterList regs, bool wb);
  Assembler& vldm(CoreRegister rn, DRegisterList regs, bool wb);
  Assembler& vldm(CoreRegister rn, SRegisterList regs) {
    return vldm(rn, regs, false);
  }
  Assembler& vldm(CoreRegister rn, DRegisterList regs) {
    return vldm(rn, regs, false);
  }
  Assembler& vpush(SRegisterList regs);
  Assembler& vpush(DRegisterList regs);

  // Binds Label l to the current location in the code buffer.
  Assembler& bind(Label& l);

  // Reset the assembler state (no memory is freed).
  void reset();

  // Get a pointer to the start of code buffer.
  const uint32_t* const start() { return buffer_; }
  const uint32_t* const offset() { return cursor_; }
  const Error error() { return error_; }

 private:
  // Emits a 32-bit value to the code buffer.
  Assembler& emit32(uint32_t value);
  Assembler& mov(Condition c, CoreRegister Rd, CoreRegister Rm);
  Assembler& b(Condition c, Label& l);

  // Pointer to start of code buffer.
  uint32_t* buffer_;
  // Pointer to current place in code buffer.
  uint32_t* cursor_;
  // Pointer to out-of-bounds of code buffer.
  uint32_t* top_;
  // Errors encountered while assembling code.
  Error error_;
};
}  // namespace aarch32
}  // namespace xnnpack
