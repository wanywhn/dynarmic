/* This file is part of the dynarmic project.
 * Copyright (c) 2023 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>

#include <mcl/stdint.hpp>

#include "dynarmic/backend/loongarch64/stack_layout.h"
#include "xbyak_loongarch64.h"
#include "block_of_code.h"

namespace oaknut {
struct PointerCodeGeneratorPolicy;
template<typename>
class BasicCodeGenerator;
using CodeGenerator = BasicCodeGenerator<PointerCodeGeneratorPolicy>;
struct Label;
}  // namespace oaknut

namespace Dynarmic::IR {
enum class Type;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::LoongArch64 {

struct EmitContext;

using Vector = std::array<u64, 2>;

enum class HostLocType {
    X,
    Q,
    Nzcv,
    Spill,
};

struct alignas(16) RegisterData {
    std::array<u64, 30> x;
    std::array<Vector, 32> q;
    u32 nzcv;
    decltype(StackLayout::spill)* spill;
    u32 fpsr;
};

void EmitVerboseDebuggingOutput(BlockOfCode& code, EmitContext& ctx);
void PrintVerboseDebuggingOutputLine(RegisterData& reg_data, HostLocType reg_type, size_t reg_index, size_t inst_index, IR::Type inst_type);

}  // namespace Dynarmic::Backend::LoongArch64
