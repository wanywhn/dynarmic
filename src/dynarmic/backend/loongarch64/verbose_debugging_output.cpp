/* This file is part of the dynarmic project.
 * Copyright (c) 2023 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/verbose_debugging_output.h"

#include <fmt/format.h>

#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/ir/type.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

void EmitVerboseDebuggingOutput(BlockOfCode& code, EmitContext& ctx) {
    code.sub_imm(code.sp, code.sp, sizeof(RegisterData), code.t0);
    for (unsigned int i = 2; i < 32; i++) {
        if (i == 21) {
            continue;  // Platform register
        }
        code.st_d(Xbyak_loongarch64::XReg{i}, code.sp, offsetof(RegisterData, x) + i * sizeof(u64));
    }
    for (unsigned int i = 0; i < 32; i++) {
        code.vst(Xbyak_loongarch64::VReg{i}, code.sp, offsetof(RegisterData, q) + i * sizeof(Vector));
    }

    code.add_imm(Xscratch0, code.sp, sizeof(RegisterData) + offsetof(StackLayout, spill), Xscratch0);
    code.st_d(Xscratch0, code.sp, offsetof(RegisterData, spill));
    code.movfcsr2gr(Xscratch0, code.fcsr0);
    code.st_d(Xscratch0, code.sp, offsetof(RegisterData, fpsr));

    ctx.reg_alloc.EmitVerboseDebuggingOutput();

    code.ld_d(Xscratch0, code.sp, offsetof(RegisterData, fpsr));
    code.movgr2fcsr(Xscratch0, code.fcsr0);
    for (unsigned int i = 0; i < 32; i++) {
        code.vld(Xbyak_loongarch64::VReg{i}, code.sp, offsetof(RegisterData, q) + i * sizeof(Vector));
    }
    for (unsigned int i = 0; i < 30; i++) {
        if (i == 18) {
            continue;  // Platform register
        }
        code.ld_d(Xbyak_loongarch64::XReg{i}, code.sp, offsetof(RegisterData, x) + i * sizeof(u64));
    }
    code.add_imm(code.sp, code.sp, sizeof(RegisterData), Xscratch0);
}

void PrintVerboseDebuggingOutputLine(RegisterData& reg_data, HostLocType reg_type, size_t reg_index, size_t inst_index, IR::Type inst_type) {
    fmt::print("dynarmic debug: %{:05} = ", inst_index);

    Vector value = [&]() -> Vector {
        switch (reg_type) {
        case HostLocType::X:
            return {reg_data.x[reg_index], 0};
        case HostLocType::Q:
            return reg_data.q[reg_index];
        case HostLocType::Nzcv:
            return {reg_data.nzcv, 0};
        case HostLocType::Spill:
            return (*reg_data.spill)[reg_index];
        }
        fmt::print("invalid reg_type! ");
        return {0, 0};
    }();

    switch (inst_type) {
    case IR::Type::U1:
    case IR::Type::U8:
        fmt::print("{:02x}", value[0] & 0xff);
        break;
    case IR::Type::U16:
        fmt::print("{:04x}", value[0] & 0xffff);
        break;
    case IR::Type::U32:
    case IR::Type::NZCVFlags:
        fmt::print("{:08x}", value[0] & 0xffffffff);
        break;
    case IR::Type::U64:
        fmt::print("{:016x}", value[0]);
        break;
    case IR::Type::U128:
        fmt::print("{:016x}{:016x}", value[1], value[0]);
        break;
    case IR::Type::A32Reg:
    case IR::Type::A32ExtReg:
    case IR::Type::A64Reg:
    case IR::Type::A64Vec:
    case IR::Type::CoprocInfo:
    case IR::Type::Cond:
    case IR::Type::Void:
    case IR::Type::Table:
    case IR::Type::AccType:
    case IR::Type::Opaque:
    default:
        fmt::print("invalid inst_type!");
        break;
    }

    fmt::print("\n");
}

}  // namespace Dynarmic::Backend::LoongArch64
