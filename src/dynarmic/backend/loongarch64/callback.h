/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <functional>
#include <vector>

#include <mcl/stdint.hpp>
#include "xbyak_loongarch64.h"

namespace Dynarmic::Backend::LoongArch64 {

using RegList = std::vector<Xbyak_loongarch64::XReg>;

class BlockOfCode;

}  // namespace Dynarmic::Backend::LoongArch64
