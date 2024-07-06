/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <initializer_list>
#include <stdexcept>
#include <type_traits>

#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/common/always_false.h"
#include "xbyak_loongarch64.h"

namespace Dynarmic::Backend::LoongArch64 {

constexpr Xbyak_loongarch64::XReg Xstate{31};
constexpr Xbyak_loongarch64::XReg Xhalt{30};
constexpr Xbyak_loongarch64::XReg Xticks{29};
constexpr Xbyak_loongarch64::XReg Xfastmem{28};
constexpr Xbyak_loongarch64::XReg Xpagetable{27};

constexpr size_t ABI_PARAM_COUNT = 6;


constexpr Xbyak_loongarch64::XReg Xscratch0{18}, Xscratch1{19}, Xscratch2{20};
constexpr Xbyak_loongarch64::WReg Wscratch0{18}, Wscratch1{19}, Wscratch2{20};

constexpr Xbyak_loongarch64::VReg Fscratch0{21}, Fscratch1{22}, Fscratch2{23};
constexpr Xbyak_loongarch64::VReg Vscratch0{21}, Vscratch1{22}, Vscratch2{23};

// TODO no need in LoongArch ,LoongArch use instruction to distinguish
template<size_t bitsize>
constexpr auto Rscratch0() {
    if constexpr (bitsize == 32) {
        return Wscratch0;
    } else if constexpr (bitsize == 64) {
        return Xscratch0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<bitsize>>);
    }
}

template<size_t bitsize>
constexpr auto Rscratch1() {
    if constexpr (bitsize == 32) {
        return Wscratch1;
    } else if constexpr (bitsize == 64) {
        return Xscratch1;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<bitsize>>);
    }
}

constexpr std::initializer_list<int> GPR_ORDER{23, 24, 25, 26, 12, 13, 14, 15, 16, 17, 18, 19, 20, 4, 5, 6, 7, 8, 9, 10, 11};
constexpr std::initializer_list<int> FPR_ORDER{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31};

using RegisterList = u64;

constexpr RegisterList ToRegList(Xbyak_loongarch64::Reg reg) {
     if (reg.isXVReg() || reg.isVReg()) {
     return RegisterList{1} << (reg.getIdx() + 32);
     }

    if (reg.getIdx() == 0) {
        throw std::out_of_range("ZR not allowed in reg list");
    }

    // if (reg.getIdx() == -1) {
    // return RegisterList{1} << 31;
    // }

    return RegisterList{1} << reg.getIdx();
}

// constexpr RegisterList ABI_CALLEE_SAVE = 0x0000ff00'7ff80000;
constexpr RegisterList ABI_CALLEE_SAVE = 0xff000000'ff800000;
// constexpr RegisterList ABI_CALLER_SAVE = 0xffffffff'4000ffff;
constexpr RegisterList ABI_CALLER_SAVE = 0xffffffff'001FFFF2;
//constexpr RegisterList ABI_CALLER_SAVE = 0x000000ff'001FFFF0;


void ABI_PushRegisters(Xbyak_loongarch64::CodeGenerator& code, RegisterList rl, size_t stack_space);
void ABI_PopRegisters(Xbyak_loongarch64::CodeGenerator& code, RegisterList rl, size_t stack_space);

}  // namespace Dynarmic::Backend::LoongArch64
