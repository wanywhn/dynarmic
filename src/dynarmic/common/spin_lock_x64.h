/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once
#ifdef  _WIN32
#include <xbyak/xbyak.h>

namespace Dynarmic {

void EmitSpinLockLock(Xbyak::CodeGenerator& code, Xbyak::Reg64 ptr, Xbyak::Reg32 tmp);
void EmitSpinLockUnlock(Xbyak::CodeGenerator& code, Xbyak::Reg64 ptr, Xbyak::Reg32 tmp);

}  // namespace Dynarmic
#else
#include <xbyak_loongarch64/xbyak_loongarch64.h>

namespace Dynarmic {

void EmitSpinLockLock(Xbyak_loongarch64::CodeGenerator& code,Xbyak_loongarch64::XReg ptr,Xbyak_loongarch64::WReg tmp);
void EmitSpinLockUnlock(Xbyak_loongarch64::CodeGenerator& code,Xbyak_loongarch64::XReg ptr,Xbyak_loongarch64::WReg tmp);

}  // namespace Dynarmic
#endif
