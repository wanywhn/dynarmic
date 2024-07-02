/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/block_of_code.h"


#include <sys/mman.h>


#include <array>
#include <cstring>

#include <mcl/assert.hpp>

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"

#include "dynarmic/backend/loongarch64/stack_layout.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

namespace {

class CustomXbyakAllocator : public Xbyak_loongarch64::Allocator {
public:
    static constexpr size_t DYNARMIC_PAGE_SIZE = 4096;

    // Can't subclass Xbyak_loongarch64::MmapAllocator because it is not a pure interface
    // and doesn't expose its construtor
    uint32_t * alloc(size_t size) override {
        // Waste a page to store the size
        size += DYNARMIC_PAGE_SIZE;

#    if defined(MAP_ANONYMOUS)
        int mode = MAP_PRIVATE | MAP_ANONYMOUS;
#    elif defined(MAP_ANON)
        int mode = MAP_PRIVATE | MAP_ANON;
#    else
#        error "not supported"
#    endif
#    ifdef MAP_JIT
        mode |= MAP_JIT;
#    endif

        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, mode, -1, 0);
        if (p == MAP_FAILED) {
            throw Xbyak_loongarch64::Error(Xbyak_loongarch64::ERR_CANT_ALLOC);
        }
        std::memcpy(p, &size, sizeof(size_t));
        return (uint32_t *)((uint8_t *)p + DYNARMIC_PAGE_SIZE);
    }

    void free(uint32_t *p) override {
        size_t size;
        std::memcpy(&size, (uint8_t *)p - DYNARMIC_PAGE_SIZE, sizeof(size_t));
        munmap(p - DYNARMIC_PAGE_SIZE, size);
    }

#    ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
    bool useProtect() const override { return false; }
#    endif
};

// This is threadsafe as Xbyak_loongarch64::Allocator does not contain any state; it is a pure interface.
CustomXbyakAllocator s_allocator;

#ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
void ProtectMemory(const void* base, size_t size, bool is_executable) {
#    ifdef _WIN32
    DWORD oldProtect = 0;
    VirtualProtect(const_cast<void*>(base), size, is_executable ? PAGE_EXECUTE_READ : PAGE_READWRITE, &oldProtect);
#    else
    static const size_t pageSize = sysconf(_SC_PAGESIZE);
    const size_t iaddr = reinterpret_cast<size_t>(base);
    const size_t roundAddr = iaddr & ~(pageSize - static_cast<size_t>(1));
    const int mode = is_executable ? (PROT_READ | PROT_EXEC) : (PROT_READ | PROT_WRITE);
    mprotect(reinterpret_cast<void*>(roundAddr), size + (iaddr - roundAddr), mode);
#    endif
}
#endif


}  // anonymous namespace

BlockOfCode::BlockOfCode(size_t total_code_size, JitStateInfo jsi)
        : Xbyak_loongarch64::CodeGenerator(total_code_size, nullptr, &s_allocator), jsi(jsi) {
}

void BlockOfCode::PreludeComplete() {
    prelude_complete = true;
//    code_begin = (CodePtr) getCurr();
    ClearCache();
    DisableWriting();
}

void BlockOfCode::EnableWriting() {
#ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
#    ifdef _WIN32
    ProtectMemory(getCode(), committed_size, false);
#    else
    ProtectMemory(getCode(), maxSize_, false);
#    endif
#endif
}

void BlockOfCode::DisableWriting() {
#ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
#    ifdef _WIN32
    ProtectMemory(getCode(), committed_size, true);
#    else
    ProtectMemory(getCode(), maxSize_, true);
#    endif
#endif
}

void BlockOfCode::ClearCache() {
    ASSERT(prelude_complete);
    SetCodePtr(code_begin);
}

size_t BlockOfCode::SpaceRemaining() const {
    ASSERT(prelude_complete);
    uint32_t * current_ptr = getCurr<uint32_t *>();
    if (current_ptr >= &top_[maxSize_])
        return 0;
    return &top_[maxSize_] - current_ptr;
}

void BlockOfCode::EnsureMemoryCommitted([[maybe_unused]] size_t codesize) {
#ifdef _WIN32
    if (committed_size < size_ + codesize) {
        committed_size = std::min<size_t>(maxSize_, committed_size + codesize);
#    ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
        VirtualAlloc(top_, committed_size, MEM_COMMIT, PAGE_READWRITE);
#    else
        VirtualAlloc(top_, committed_size, MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#    endif
    }
#endif
}

HaltReason BlockOfCode::RunCode(void* jit_state, CodePtr code_ptr) const {
    return run_code(jit_state, code_ptr);
}

HaltReason BlockOfCode::StepCode(void* jit_state, CodePtr code_ptr) const {
    return step_code(jit_state, code_ptr);
}

void BlockOfCode::SwitchMxcsrOnEntry() {
        movfcsr2gr(Wscratch0, fcsr0);
        st_w(Wscratch0, sp, offsetof(StackLayout, save_host_fpcr));
        ld_w(Wscratch0, Xstate, jsi.offsetof_guest_FCSR);
        movgr2fcsr(fcsr0, Wscratch0);
}

void BlockOfCode::SwitchMxcsrOnExit() {
    movfcsr2gr(Wscratch0, Wscratch0);
    st_w(Wscratch0, Xstate, jsi.offsetof_guest_FCSR);

    ld_w(Wscratch0, sp, offsetof(StackLayout, save_host_fpcr));
    movgr2fcsr(Wscratch0, Wscratch0);
    // FIXME r15?

//    stmxcsr(dword[r15 + jsi.offsetof_guest_FCSR]);
}

void BlockOfCode::EnterStandardASIMD() {
    movfcsr2gr(Wscratch0, Wscratch0);
    st_w(Wscratch0, Xstate, jsi.offsetof_guest_FCSR);

    ld_w(Wscratch0, Xstate, jsi.offsetof_asimd_MXCSR);
    movgr2fcsr(Wscratch0, Wscratch0);
//    stmxcsr(dword[r15 + jsi.offsetof_guest_FCSR]);
//    ldmxcsr(dword[r15 + jsi.offsetof_asimd_MXCSR]);
}

void BlockOfCode::LeaveStandardASIMD() {
    movfcsr2gr(Wscratch0, Wscratch0);
    st_w(Wscratch0, Xstate, jsi.offsetof_asimd_MXCSR);

    ld_w(Wscratch0, Xstate, jsi.offsetof_guest_FCSR);
    movgr2fcsr(Wscratch0, Wscratch0);
//    stmxcsr(dword[r15 + jsi.offsetof_asimd_MXCSR]);
//    ldmxcsr(dword[r15 + jsi.offsetof_guest_FCSR]);
}


    void BlockOfCode::LoadRequiredFlagsForCondFromRax(IR::Cond cond) {
    // sahf restores SF, ZF, CF
    // add al, 0x7F restores OF

    switch (cond) {
    case IR::Cond::EQ:  // z
    case IR::Cond::NE:  // !z
    case IR::Cond::CS:  // c
    case IR::Cond::CC:  // !c
    case IR::Cond::MI:  // n
    case IR::Cond::PL:  // !n
    // FIXME
//        sahf();
        break;
    case IR::Cond::VS:  // v
    case IR::Cond::VC:  // !v
//        cmp(al, 0x81);
        break;
    case IR::Cond::HI:  // c & !z
    case IR::Cond::LS:  // !c | z
//        sahf();
//        cmc();
        break;
    case IR::Cond::GE:  // n == v
    case IR::Cond::LT:  // n != v
    case IR::Cond::GT:  // !z & (n == v)
    case IR::Cond::LE:  // z | (n != v)
//        cmp(al, 0x81);
//        sahf();
        break;
    case IR::Cond::AL:
    case IR::Cond::NV:
        break;
    default:
        ASSERT_MSG(false, "Unknown cond {}", static_cast<size_t>(cond));
        break;
    }
}

CodePtr BlockOfCode::GetCodeBegin() const {
    return code_begin;
}

size_t BlockOfCode::GetTotalCodeSize() const {
    return maxSize_;
}

void* BlockOfCode::AllocateFromCodeSpace(size_t alloc_size) {
    if (size_ + alloc_size >= maxSize_) {
        throw Xbyak_loongarch64::Error(Xbyak_loongarch64::ERR_CODE_IS_TOO_BIG);
    }

    EnsureMemoryCommitted(alloc_size);

    void* ret = getCurr<void*>();
    size_ += alloc_size;
    memset(ret, 0, alloc_size);
    return ret;
}

void BlockOfCode::SetCodePtr(CodePtr code_ptr) {
    // The "size" defines where top_, the insertion point, is.
    size_t required_size = reinterpret_cast<const u8*>(code_ptr) - getCode();
    setSize(required_size);
}

    void BlockOfCode::B(const void *a) {
        auto fn = (std::uint64_t (*)() )a;
        JumpFunction(fn);
    }

    void BlockOfCode::BL(const void *a) {
    auto fn = (std::uint64_t (*)() )a;
    CallFunction(fn);
    }

    void BlockOfCode::LDLableData_d(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::Label &label) {
        pcaddi(rd, label);
        ld_d(rd, rd, 0);
    }

    void BlockOfCode::LDLableData_w(const Xbyak_loongarch64::XReg &rd, const Xbyak_loongarch64::Label &label) {
        pcaddi(rd, label);
        ld_w(rd, rd, 0);
    }

}  // namespace Dynarmic::Backend::LoongArch64
