/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * A reader of JSON documents (RFC 8259), internal to the library: tokenizer.json, config.json and the
 * index of sharded safetensors checkpoints. A document is parsed into one array of values in document
 * order, each container followed by its elements (an object's as key, value, key, value, ...) and each
 * value recording the index just past its subtree, so that siblings are reached without recursion and
 * nothing is allocated per value. Strings point into the text and are decoded (escapes, surrogate
 * pairs) when copied. The parser checks the whole grammar, UTF-8 included, and rejects documents
 * nested deeper than SPG_JSON_MAX_DEPTH or larger than 4 GB.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SPG_JSON_MAX_DEPTH 256u

typedef enum {
    SPG_JSON_NULL,
    SPG_JSON_BOOL,
    SPG_JSON_NUMBER,
    SPG_JSON_STRING,
    SPG_JSON_ARRAY,
    SPG_JSON_OBJECT
} SpgJsonType;

typedef struct {
    uint32_t start;         /* string: its first byte after the quote; others: their first character */
    uint32_t length;        /* string: its bytes up to the closing quote, escapes as written; number: its
                               characters; container: its bytes up to the closing bracket */
    uint32_t next;          /* the index of the value after this one and its elements */
    uint32_t count;         /* container: elements (an object's members); string: bytes once decoded;
                               bool: its value */
    uint8_t type;           /* SpgJsonType */
    bool escaped;           /* string: holds escapes, so that its bytes differ from the text's */
} SpgJsonValue;

typedef struct {
    const char *text;
    SpgJsonValue *values;   /* values[SPG_JSON_ROOT] is the document's value; values[0] stands for no value
                               (a null), so that the accessors given the 0 that a missing member gives answer
                               as for a missing value */
    uint32_t count;
    char error[160];
} SpgJson;

#define SPG_JSON_ROOT 1u

/* Parses size bytes of text (no NUL needed), which must outlive the document; false with
   json->error set (and nothing to free) when the text is not one JSON value. */
bool spingalett_json_parse(SpgJson *json, const char *text, size_t size);
void spingalett_json_free(SpgJson *json);

/* The value of an object's member by key (the first of equal keys), 0 when v is no object or has no
   such member. */
uint32_t spingalett_json_member(const SpgJson *json, uint32_t v, const char *key);
/* The first element of a container (an object's first key; its value follows it), 0 when it has none.
   The next is values[e].next for an array's elements, values[e + 1].next for an object's keys. */
uint32_t spingalett_json_first(const SpgJson *json, uint32_t v);
/* Element i of an array (linear in i), 0 when there is none. */
uint32_t spingalett_json_element(const SpgJson *json, uint32_t v, uint32_t i);

/* Whether v is a string equal to s. */
bool spingalett_json_is(const SpgJson *json, uint32_t v, const char *s);
/* String v decoded into out, which holds values[v].count + 1 bytes (a NUL is added); false when v is
   no string. */
bool spingalett_json_copy(const SpgJson *json, uint32_t v, char *out);
/* String v decoded into a malloc'ed buffer (NUL-terminated; *length, when given, its bytes), NULL when
   v is no string or memory runs out. */
char *spingalett_json_strdup(const SpgJson *json, uint32_t v, size_t *length);
/* Number v; false when v is no number. */
bool spingalett_json_number(const SpgJson *json, uint32_t v, double *out);
/* Number v when it is an integer within [min, max]; false otherwise. */
bool spingalett_json_integer(const SpgJson *json, uint32_t v, int64_t min, int64_t max, int64_t *out);
bool spingalett_json_bool(const SpgJson *json, uint32_t v, bool *out);
