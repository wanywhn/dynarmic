/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/fpsr_manager.h"

#include "dynarmic/backend/loongarch64/abi.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

FpsrManager::FpsrManager(BlockOfCode& code, size_t state_fpsr_offset)
        : code{code}, state_fpsr_offset{state_fpsr_offset} {}

void FpsrManager::Spill() {
    if (!fpsr_loaded)
        return;

    code.ld_d(Wscratch0, Xstate, state_fpsr_offset);
//    code.MRS(Xscratch1, Xbyak_loongarch64::SystemReg::FPSR);
    code.or_(Wscratch0, Wscratch0, Wscratch1);
    code.st_d(Wscratch0, Xstate, state_fpsr_offset);

    fpsr_loaded = false;
}

void FpsrManager::Load() {
    if (fpsr_loaded)
        return;

//    code.MSR(Xbyak_loongarch64::SystemReg::FPSR, code.zero);

    fpsr_loaded = true;
}

}  // namespace Dynarmic::Backend::LoongArch64
