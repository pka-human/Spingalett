#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# build.sh <cmake --build arguments>: the build, whose errors also become annotations of the job (the compilers'
# error lines, and the end of the output), which the Actions page shows to anyone, without its logs.
log=$(mktemp)
cmake --build "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    grep -E 'error|Error' "$log" | head -n 40 | while IFS= read -r line; do echo "::error::$line"; done
    printf '::error title=end of the output::'
    tail -n 60 "$log" | sed 's/%/%25/g' | awk '{ printf "%s%%0A", $0 }' | head -c 60000
    echo
fi
exit "$status"
