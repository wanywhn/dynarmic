/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <xbyak_loongarch64/xbyak_loongarch64.h>

namespace Dynarmic {

void EmitSpinLockLock(Xbyak_loongarch64::CodeGenerator& code,Xbyak_loongarch64::XReg ptr);
void EmitSpinLockUnlock(Xbyak_loongarch64::CodeGenerator& code,Xbyak_loongarch64::XReg ptr);

}  // namespace Dynarmic

