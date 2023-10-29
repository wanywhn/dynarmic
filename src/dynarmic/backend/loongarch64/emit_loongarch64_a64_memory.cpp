/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a64_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_loongarch64_memory.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/ir/acc_type.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

    using namespace Xbyak_loongarch64::util;

    template<>
    void EmitIR<IR::Opcode::A64ClearExclusive>(BlockOfCode &code, EmitContext &, IR::Inst *) {
        code.st_d(code.zero, Xstate, offsetof(A64JitState, exclusive_state));
    }

    template<>
    void EmitIR<IR::Opcode::A64ReadMemory8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReadMemory<8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ReadMemory16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReadMemory<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ReadMemory32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReadMemory<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ReadMemory64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReadMemory<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ReadMemory128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitReadMemory<128>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveReadMemory8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveReadMemory<8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveReadMemory16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveReadMemory<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveReadMemory32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveReadMemory<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveReadMemory64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveReadMemory<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveReadMemory128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveReadMemory<128>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64WriteMemory8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitWriteMemory<8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64WriteMemory16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitWriteMemory<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64WriteMemory32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitWriteMemory<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64WriteMemory64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitWriteMemory<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64WriteMemory128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitWriteMemory<128>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveWriteMemory8>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveWriteMemory<8>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveWriteMemory16>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveWriteMemory<16>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveWriteMemory32>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveWriteMemory<32>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveWriteMemory64>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveWriteMemory<64>(code, ctx, inst);
    }

    template<>
    void EmitIR<IR::Opcode::A64ExclusiveWriteMemory128>(BlockOfCode &code, EmitContext &ctx, IR::Inst *inst) {
        EmitExclusiveWriteMemory<128>(code, ctx, inst);
    }

}  // namespace Dynarmic::Backend::LoongArch64
