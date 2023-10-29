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
    sub_d(v0, a0, a1);
    jirl(zero, ra, 0);
  }
};
int main() {
  Generator gen;
  gen.ready();
  auto f = gen.getCode<int (*)(int, int)>();
  int a = 6;
  int b = 8;
  gen.dump();
  printf("%d - %d = %d\n", a, b, f(a, b));
}
