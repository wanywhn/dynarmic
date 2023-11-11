/*******************************************************************************
 * Copyright 2019-2020 LOONGSON LIMITED
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <xbyak_loongarch64/xbyak_loongarch64.h>
using namespace Xbyak_loongarch64;
class Generator : public CodeGenerator {
public:
  Generator() {
    add_d(v0, a0, a1);
    jirl(zero, ra, 0);
  }
};

TEST(testDump, dumpCode) {
  Generator gen;
  gen.ready();

  std::cout << "size:" << gen.getSize() << std::endl;
  gen.dump();
  const uint8_t expect[] = {0x84, 0x94, 0x10 ,0x00, 0x20, 0x00, 0x00, 0x4C};
  const uint8_t *code = gen.getCode();
  EXPECT_TRUE(0 == std::memcmp(expect, code, gen.getSize()));
}
int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
