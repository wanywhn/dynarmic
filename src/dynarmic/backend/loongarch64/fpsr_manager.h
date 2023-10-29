/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <mcl/stdint.hpp>

#include "xbyak_loongarch64.h"
#include "block_of_code.h"

namespace oaknut {
    struct PointerCodeGeneratorPolicy;

    template<typename>
    class BasicCodeGenerator;

    using CodeGenerator = BasicCodeGenerator<PointerCodeGeneratorPolicy>;
}  // namespace oaknut

namespace Dynarmic::Backend::LoongArch64 {

    class FpsrManager {
    public:
        explicit FpsrManager(BlockOfCode &code, size_t state_fpsr_offset);

        void Spill();

        void Load();

        void Overwrite() { fpsr_loaded = false; }

    private:
        BlockOfCode &code;
        size_t state_fpsr_offset;
        bool fpsr_loaded = false;
    };

}  // namespace Dynarmic::Backend::LoongArch64
