/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mutex>

#include "dynarmic/backend/loongarch64/abi.h"
//#include "dynarmic/backend/loongarch64/hostloc.h"
#include "dynarmic/common/spin_lock.h"

namespace Dynarmic {
using Backend::LoongArch64::Wscratch0;
using Backend::LoongArch64::Wscratch1;
    using Backend::LoongArch64::Wscratch2;

    void EmitSpinLockLock(Xbyak_loongarch64::CodeGenerator& code, Xbyak_loongarch64::XReg ptr) {
    Xbyak_loongarch64::Label start, loop;
    code.b(start);
    code.L(loop);
    code.nop();
    code.L(start);
    code.addi_d(Wscratch1, code.zero, 1);
    code.ll_w(Wscratch0, ptr, 0);
    code.sc_w(Wscratch1, ptr, 0);
//    code.amswap_w(Wscratch1, Wscratch1, ptr);
    code.beqz(Wscratch1, loop);
}

void EmitSpinLockUnlock(Xbyak_loongarch64::CodeGenerator& code, Xbyak_loongarch64::XReg ptr) {
        code.st_w(code.zero, ptr, 0);
}

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

struct SpinLockImpl {
    SpinLockImpl();

    Xbyak_loongarch64::CodeGenerator code;

    void (*lock)(volatile int*);
    void (*unlock)(volatile int*);
};

std::once_flag flag;
SpinLockImpl impl;

SpinLockImpl::SpinLockImpl(): code(4096, nullptr, &s_allocator) {

    lock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockLock(code, code.a0);
    code.jirl(code.zero, code.ra, 0);

    unlock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockUnlock(code, code.a0);
    code.jirl(code.zero, code.ra, 0);
}

}  // namespace

void SpinLock::Lock() {
    impl.lock(&storage);
}

void SpinLock::Unlock() {
    impl.unlock(&storage);
}

}  // namespace Dynarmic
