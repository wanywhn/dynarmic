/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <mcl/stdint.hpp>

namespace oaknut {
struct PointerCodeGeneratorPolicy;
template<typename>
class BasicCodeGenerator;
using CodeGenerator = BasicCodeGenerator<PointerCodeGeneratorPolicy>;
struct Label;
}  // namespace oaknut

namespace Dynarmic::IR {
enum class AccType;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::LoongArch64 {

struct EmitContext;
enum class LinkTarget;

template<size_t bitsize>
void EmitReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template<size_t bitsize>
void EmitExclusiveReadMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template<size_t bitsize>
void EmitWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);
template<size_t bitsize>
void EmitExclusiveWriteMemory(Xbyak_loongarch64::CodeGenerator& code, EmitContext& ctx, IR::Inst* inst);

}  // namespace Dynarmic::Backend::LoongArch64
