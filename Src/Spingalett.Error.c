/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdio.h>

static _Thread_local int last_error_code = SPINGALETT_OK;
static _Thread_local char last_error_message[SPINGALETT_ERRMSG_MAX];

int spingalett_last_error_code(void) {
    return last_error_code;
}

const char *spingalett_last_error_message(void) {
    return last_error_message;
}

void spingalett_clear_error(void) {
    last_error_code = SPINGALETT_OK;
    last_error_message[0] = '\0';
}

void set_error(int code, const char *msg) {
    last_error_code = code;
    (void)snprintf(last_error_message, SPINGALETT_ERRMSG_MAX, "%s", msg ? msg : "unknown error");
}
