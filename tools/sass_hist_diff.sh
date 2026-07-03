#!/usr/bin/env bash
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# sass_hist_diff.sh — 跨轮 SASS「指令类别直方图」对比（不要裸 diff SASS）
#
# 为什么不用裸 diff：SASS 每行带 /* 0x.. */ 编码列、物理寄存器号、指令地址，
# ptxas 还会重排——任何小改动都让逐行 diff 几乎全变（实测两版 5千行 SASS 裸 diff
# 7900+ 行）。判「这轮改动落在哪类指令上」要按助记符做直方图 Δ。
#
# 用法:
#   tools/sass_hist_diff.sh <A> <B>
#   A/B 可以是 SASS 文本文件，或一个 profiles/v<N>/ 目录（自动取其中的 kernel.sass）。
#   SASS 由 profile_iter_nvidia.sh 存成 <output-dir>/kernel.sass（extract_nvidia_asm.py 抽取）。
# 例:
#   tools/sass_hist_diff.sh profiles/v0/kernel.sass profiles/v1/kernel.sass
#   tools/sass_hist_diff.sh profiles/v0 profiles/v1        # 目录形式，自动找 kernel.sass
set -euo pipefail

resolve() {  # 入参 -> sass 文件路径（目录则取 atrex 约定的 kernel.sass）
  if [ -d "$1" ]; then echo "$1/kernel.sass"
  else echo "$1"; fi
}
A=$(resolve "${1:?usage: sass_hist_diff.sh <A> <B>}")
B=$(resolve "${2:?usage: sass_hist_diff.sh <A> <B>}")
for f in "$A" "$B"; do [ -f "$f" ] || { echo "缺文件: $f（该轮没存 kernel.sass？）" >&2; exit 1; }; done

# 提取助记符（类别级：去谓词 @P0/@!UPT，去 /*addr*/，取到第一个 . 或 ; 之前）
mnem() {
  grep -E '/\*[0-9a-f]+\*/' "$1" \
  | sed -E 's#.*/\*[0-9a-f]+\*/[[:space:]]*##; s/^@!?[A-Za-z0-9_]+[[:space:]]+//' \
  | awk '{print $1}' | sed -E 's/[.;,].*//' | grep -E '^[A-Z]'
}

# 把「@!UPT UIADD3 URZ」这种调度填充单独归一成 FILLER（裸数 NOP 会严重漏数）
fillerA=$(grep -c '@!UPT UIADD3 URZ' "$A" || true)
fillerB=$(grep -c '@!UPT UIADD3 URZ' "$B" || true)

mnem "$A" | sort | uniq -c | awk '{print $2, $1}' > /tmp/.hA.$$
mnem "$B" | sort | uniq -c | awk '{print $2, $1}' > /tmp/.hB.$$

echo "A = $A"
echo "B = $B"
echo
printf "%-14s %8s %8s %8s\n" "指令类" "A" "B" "Δ(B-A)"
printf "%-14s %8s %8s %8s\n" "------" "----" "----" "----"
awk -v fa="$fillerA" -v fb="$fillerB" '
  FNR==NR { a[$1]=$2; next }
  { b[$1]=$2 }
  END {
    a["FILLER@!UPT"]=fa; b["FILLER@!UPT"]=fb
    for (k in a) keys[k]=1; for (k in b) keys[k]=1
    n=0; for (k in keys) order[n++]=k
    # 按 |Δ| 降序
    for (i=0;i<n;i++) for (j=i+1;j<n;j++){
      di=(b[order[i]]-a[order[i]]); dj=(b[order[j]]-a[order[j]])
      if ((di<0?-di:di) < (dj<0?-dj:dj)) { t=order[i]; order[i]=order[j]; order[j]=t }
    }
    for (i=0;i<n;i++){ k=order[i]; d=b[k]-a[k]; if(d!=0) printf "%-14s %8d %8d %+8d\n", k, a[k]+0, b[k]+0, d }
  }
' /tmp/.hA.$$ /tmp/.hB.$$
rm -f /tmp/.hA.$$ /tmp/.hB.$$
echo
echo "注: FILLER@!UPT = ptxas 在 QMMA 间插的调度填充槽（吞吐 stall 标志）；只数字面 NOP 会漏。"
