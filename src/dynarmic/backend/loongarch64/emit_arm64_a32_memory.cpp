/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/loongarch64/a32_jitstate.h"
#include "dynarmic/backend/loongarch64/abi.h"
#include "dynarmic/backend/loongarch64/emit_arm64.h"
#include "dynarmic/backend/loongarch64/emit_arm64_memory.h"
#include "dynarmic/backend/loongarch64/emit_context.h"
#include "dynarmic/backend/loongarch64/reg_alloc.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "xbyak_loongarch64.h"
#include "xbyak_loongarch64_util.h"

namespace Dynarmic::Backend::LoongArch64 {

using namespace Xbyak_loongarch64::util;

template<>
void EmitIR<IR::Opcode::A32ClearExclusive>(BlockOfCode& code, EmitContext&, IR::Inst*) {
    code.st_d(code.zero, Xstate, offsetof(A32JitState, exclusive_state));
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitReadMemory<8>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitReadMemory<16>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitReadMemory<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ReadMemory64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitReadMemory<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveReadMemory<8>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveReadMemory<16>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveReadMemory<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveReadMemory64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveReadMemory<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitWriteMemory<8>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitWriteMemory<16>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitWriteMemory<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32WriteMemory64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitWriteMemory<64>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory8>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveWriteMemory<8>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory16>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveWriteMemory<16>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory32>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveWriteMemory<32>(code, ctx, inst);
}

template<>
void EmitIR<IR::Opcode::A32ExclusiveWriteMemory64>(BlockOfCode& code, EmitContext& ctx, IR::Inst* inst) {
    EmitExclusiveWriteMemory<64>(code, ctx, inst);
}

}  // namespace Dynarmic::Backend::LoongArch64
