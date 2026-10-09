#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# ctest.sh <ctest arguments>: ctest with --output-on-failure, whose failures also become annotations of the job
# (the failed tests, and the end of the output), which the Actions page shows to anyone, without its logs.
log=$(mktemp)
ctest "$@" --output-on-failure 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    grep -E '\*\*\*|Failed|FAILED|failed' "$log" | head -n 20 | while IFS= read -r line; do echo "::error::$line"; done
    printf '::error title=end of the output::'
    tail -n 100 "$log" | sed 's/%/%25/g' | awk '{ printf "%s%%0A", $0 }' | head -c 60000
    echo
fi
exit "$status"
