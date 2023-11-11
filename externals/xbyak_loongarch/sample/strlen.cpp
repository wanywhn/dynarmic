/*******************************************************************************
 * Copyright 2019-2022 LOONGSON LIMITED
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
#include <xbyak_loongarch64/xbyak_loongarch64.h>
using namespace Xbyak_loongarch64;
class Generator : public CodeGenerator {
public:
  Generator() {
    Label label_process, label_result;

    add_d(a1, a0, zero);
    L(label_process);
    ld_bu(t0, a0, 0);
    addi_d(a0, a0, 1);

    bnez(t0, label_process);
    beqz(t0, label_result);

    L(label_result);
    addi_d(a0, a0, -1);
    sub_d(v0, a0, a1);
    jirl(zero, ra, 0);
  }
};
int main() {
  Generator gen;
  gen.ready();
  auto f = gen.getCode<int (*)(char *)>();
  char str0[] = "loongarch is the best";
  gen.dump();
  printf("(%s) length is %d\n", str0, f(str0));
}
int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
