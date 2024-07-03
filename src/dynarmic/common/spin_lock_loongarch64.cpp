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
    code.amswap_d(Wscratch1, Wscratch1, ptr);
    code.bnez(Wscratch1, loop);
}

void EmitSpinLockUnlock(Xbyak_loongarch64::CodeGenerator& code, Xbyak_loongarch64::XReg ptr) {
        code.amswap_db_d(Wscratch1, Wscratch0, ptr);
}

namespace {

struct SpinLockImpl {
    void Initialize();

    Xbyak_loongarch64::CodeGenerator code;

    void (*lock)(volatile int*);
    void (*unlock)(volatile int*);
};

std::once_flag flag;
SpinLockImpl impl;

void SpinLockImpl::Initialize() {

    lock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockLock(code, code.a0);
    code.jirl(code.zero, code.ra, 0);

    unlock = code.getCurr<void (*)(volatile int*)>();
    EmitSpinLockUnlock(code, code.a0);
    code.jirl(code.zero, code.ra, 0);
}

}  // namespace

void SpinLock::Lock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.lock(&storage);
}

void SpinLock::Unlock() {
    std::call_once(flag, &SpinLockImpl::Initialize, impl);
    impl.unlock(&storage);
}

}  // namespace Dynarmic
