#!/bin/bash
# jobs4.sh <prefix> [maxb]: prints one run line per build of a step (tag elf extra-args), for xargs.
p=$1; MAXB=${2:-15}
for v in i8 i4 si8 si4; do
  case $v in i8) B="";; i4) B="-p top.memory.params.ext_ram_base 589477888";; si8) B="-p top.memory.params.ext_ram_base 544289280";; si4) B="-p top.memory.params.ext_ram_base 541405696";; esac
  L="b8 b4 b2 b1 ttft32"; [ ${v:0:1} = s ] && L="b$MAXB $L"
  for t in $L; do echo "${p}${v}_$t ${p}${v}_$t${B:+ $B}"; done
done
