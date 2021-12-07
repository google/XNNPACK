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
  kAL = 0xE0000000,
};

enum class Error {
  kNoError,
  kOutOfMemory,
  kInvalidOperand,
};

// A simple AAarch32 assembler.
// Right now it allocates its own memory (using `new`) to write code into (for
// testing), but will be updated to be more customizable.
class Assembler {
 public:
  explicit Assembler();
  ~Assembler();

  Assembler& add(CoreRegister Rd, CoreRegister Rn, CoreRegister Rm);
  // Cmp supports a subset of uint32_t offsets, see "A5.2.4 Modified immediate
  // constants in ARM instructions", for simplicity we start with uint8_t, which
  // is fully representation using a "rotation" of 0.
  Assembler& cmp(CoreRegister Rn, uint8_t imm);
  Assembler& ldr(CoreRegister Rt, MemOperand operand, int32_t offset);
  Assembler& ldr(CoreRegister Rt, MemOperand operand);
  Assembler& push(CoreRegisterList registers);

  // Reset the assembler state (no memory is freed).
  void reset();

  // Get a pointer to the start of code buffer.
  const uint32_t* const start() { return buffer_; }
  const Error error() { return error_; }

 private:
  // Emits a 32-bit value to the code buffer.
  Assembler& emit32(uint32_t value);

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
