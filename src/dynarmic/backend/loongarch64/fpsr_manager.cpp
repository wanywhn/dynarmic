/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/fpsr_manager.h"

#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

#include "dynarmic/backend/loongarch64/abi.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

FpsrManager::FpsrManager(Xbyak_loongarch64::CodeGenerator& code, size_t state_fpsr_offset)
        : code{code}, state_fpsr_offset{state_fpsr_offset} {}

void FpsrManager::Spill() {
    if (!fpsr_loaded)
        return;

    code.pcaddi(Wscratch0, Xstate, state_fpsr_offset);
    code.MRS(Xscratch1, Xbyak_loongarch64::SystemReg::FPSR);
    code.ORR(Wscratch0, Wscratch0, Wscratch1);
    code.STR(Wscratch0, Xstate, state_fpsr_offset);

    fpsr_loaded = false;
}

void FpsrManager::Load() {
    if (fpsr_loaded)
        return;

    code.MSR(Xbyak_loongarch64::SystemReg::FPSR, XZR);

    fpsr_loaded = true;
}

}  // namespace Dynarmic::Backend::LoongArch64
