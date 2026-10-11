/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Tokenizers of language models (Spingalett.h): the byte-pair encodings that Hugging Face's
 * tokenizer.json files and GGUF files describe, encoding and decoding as Hugging Face's tokenizers
 * library does, id for id.
 *
 * Encoding runs the library's steps. The added tokens are found first, leftmost and longest first (the
 * library's Aho-Corasick automaton), with their lstrip, rstrip and single_word rules: those whose
 * "normalized" is false in the text as given, the others in the normalized text. The rest of the text
 * is normalized (Prepend, Replace, NFC, Strip), cut into pieces by the pre-tokenizers (ByteLevel,
 * Split, Metaspace, Digits) and each piece encoded by the BPE model: its characters (or, with byte
 * fallback, the <0xXX> tokens of characters the vocabulary lacks) merged pair by pair, always the pair
 * of lowest rank in the merges and the leftmost of equals, through a heap of the candidate pairs, so a
 * piece of n characters takes n log n steps. A post-processor's template adds the tokens around a text
 * (add_special).
 *
 * The library runs the Split pre-tokenizers' regular expressions on Oniguruma. Those of tokenizer.json
 * files in use are few: GPT-2's (ByteLevel's own), tiktoken's cl100k (LLaMA 3; Qwen2's with single
 * digits) and o200k (with its variant of Mistral's tekken). They are recognized by their text and
 * matched by hand, with Oniguruma's semantics: the first alternative that matches wins, quantifiers
 * are greedy and give back characters only as far as the rest of the alternative needs; \s, \p{L},
 * \p{N} and the letters' categories come from the Unicode tables of Spingalett.Unicode.h. Every
 * character matches one of their alternatives, so the matches tile a piece.
 *
 * Decoding gives each id its token (the added tokens' contents included, special ones left out when
 * asked) and runs the decoders over the list (ByteLevel, ByteFallback, Replace, Fuse, Strip, Metaspace),
 * bytes that are no UTF-8 becoming U+FFFD as Rust's from_utf8_lossy makes them.
 *
 * Every text is UTF-8: input that is not is refused, and strings of the files are checked as they are
 * read (the JSON reader checks its strings; a GGUF file's are checked here).
 */

#include "Spingalett.Private.h"
#include "Spingalett.Json.h"
#include "Spingalett.Tokenizer.h"
#include "Spingalett.Unicode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NONE SPINGALETT_NO_TOKEN
#define MAX_IDS (1u << 24)          /* ids an embedding's float inputs hold exactly */

/* ------------------------------------------------------------------------- UTF-8 and Unicode */

/* The code point at s[*i] (n bytes in all), *i moved past it; a malformed sequence reads as U+FFFD of
   one byte. */
static uint32_t next_cp(const char *str, size_t n, size_t *i) {
    const unsigned char *s = (const unsigned char *)str;
    size_t k = *i;
    unsigned c = s[k];
    if (c < 0x80) { *i = k + 1; return c; }
    uint32_t cp;
    size_t len;
    if (c >= 0xC2 && c < 0xE0) { cp = c & 0x1Fu; len = 2; }
    else if (c >= 0xE0 && c < 0xF0) { cp = c & 0x0Fu; len = 3; }
    else if (c >= 0xF0 && c < 0xF5) { cp = c & 0x07u; len = 4; }
    else { *i = k + 1; return 0xFFFD; }
    if (k + len > n) { *i = k + 1; return 0xFFFD; }
    for (size_t j = 1; j < len; j++) {
        if ((s[k + j] & 0xC0) != 0x80) { *i = k + 1; return 0xFFFD; }
        cp = (cp << 6) | (s[k + j] & 0x3Fu);
    }
    if ((len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp < 0xE000))) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF))) {
        *i = k + 1;
        return 0xFFFD;
    }
    *i = k + len;
    return cp;
}

/* The length of the well-formed UTF-8 sequence at s (n bytes left), 0 when it is malformed. */
static size_t utf8_valid_at(const char *s, size_t n) {
    size_t i = 0;
    uint32_t cp = next_cp(s, n, &i);
    if (cp == 0xFFFD && !(i == 3 && (unsigned char)s[0] == 0xEF && (unsigned char)s[1] == 0xBF && (unsigned char)s[2] == 0xBD))
        return 0;
    return i;
}

static bool utf8_valid(const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        if ((unsigned char)s[i] < 0x80) { i++; continue; }
        size_t k = utf8_valid_at(s + i, n - i);
        if (!k) return false;
        i += k;
    }
    return true;
}

static size_t utf8_put(char *out, uint32_t cp) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18)); out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

enum { CAT_NONE, CAT_LU, CAT_LL, CAT_LT, CAT_LM, CAT_LO, CAT_M, CAT_N };

static int category(uint32_t cp) {
    if (cp < 0x80) return cp >= 'A' && cp <= 'Z' ? CAT_LU : cp >= 'a' && cp <= 'z' ? CAT_LL : cp >= '0' && cp <= '9' ? CAT_N : CAT_NONE;
    size_t lo = 0, hi = sizeof unicode_categories / sizeof unicode_categories[0];
    while (hi - lo > 1) {                   /* the last run starting at or before cp */
        size_t mid = (lo + hi) / 2;
        if ((unicode_categories[mid] >> 3) <= cp) lo = mid;
        else hi = mid;
    }
    return (int)(unicode_categories[lo] & 7u);
}

static bool is_letter(uint32_t cp) { int c = category(cp); return c >= CAT_LU && c <= CAT_LO; }
static bool is_number(uint32_t cp) { return category(cp) == CAT_N; }

/* \s of Oniguruma's Unicode syntax, and the White_Space property Rust's char::is_whitespace() tests. */
static bool is_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/* ------------------------------------------------------------------------- NFC */

static unsigned combining_class(uint32_t cp) {
    if (cp < 0x300) return 0;
    size_t lo = 0, hi = sizeof unicode_classes / sizeof unicode_classes[0];
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        uint32_t c = unicode_classes[mid] >> 8;
        if (c == cp) return unicode_classes[mid] & 0xFFu;
        if (c < cp) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

#define HANGUL_S 0xAC00u
#define HANGUL_L 0x1100u
#define HANGUL_V 0x1161u
#define HANGUL_T 0x11A7u
#define HANGUL_LC 19u
#define HANGUL_VC 21u
#define HANGUL_TC 28u
#define HANGUL_N (HANGUL_VC * HANGUL_TC)
#define HANGUL_SC (HANGUL_LC * HANGUL_N)

static bool decomposition(uint32_t cp, uint32_t *a, uint32_t *b) {
    size_t lo = 0, hi = sizeof unicode_decompositions / sizeof unicode_decompositions[0];
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        uint32_t c = unicode_decompositions[mid][0];
        if (c == cp) { *a = unicode_decompositions[mid][1]; *b = unicode_decompositions[mid][2]; return true; }
        if (c < cp) lo = mid + 1;
        else hi = mid;
    }
    return false;
}

static uint32_t composite(uint32_t a, uint32_t b) {
    if (a >= HANGUL_L && a < HANGUL_L + HANGUL_LC && b >= HANGUL_V && b < HANGUL_V + HANGUL_VC)
        return HANGUL_S + ((a - HANGUL_L) * HANGUL_VC + (b - HANGUL_V)) * HANGUL_TC;
    if (a >= HANGUL_S && a < HANGUL_S + HANGUL_SC && (a - HANGUL_S) % HANGUL_TC == 0 && b > HANGUL_T && b < HANGUL_T + HANGUL_TC)
        return a + (b - HANGUL_T);
    size_t lo = 0, hi = sizeof unicode_composites / sizeof unicode_composites[0];
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        const uint32_t *c = unicode_composites[mid];
        if (c[0] == a && c[1] == b) return c[2];
        if (c[0] < a || (c[0] == a && c[1] < b)) lo = mid + 1;
        else hi = mid;
    }
    return 0;
}

/* Code points of the canonical decomposition of cp appended at out (room for 8: canonical decompositions
   are at most four code points long); returns how many. */
static size_t decompose(uint32_t cp, uint32_t *out, int depth) {
    if (cp >= HANGUL_S && cp < HANGUL_S + HANGUL_SC) {
        uint32_t s = cp - HANGUL_S, t = s % HANGUL_TC;
        out[0] = HANGUL_L + s / HANGUL_N;
        out[1] = HANGUL_V + (s % HANGUL_N) / HANGUL_TC;
        if (!t) return 2;
        out[2] = HANGUL_T + t;
        return 3;
    }
    uint32_t a, b;
    if (cp < 0xC0 || depth > 4 || !decomposition(cp, &a, &b)) { out[0] = cp; return 1; }
    size_t n = decompose(a, out, depth + 1);
    if (b && n < 8) out[n++] = b;
    return n;
}

/* NFC of s (n bytes of UTF-8) appended to *out; text below U+0300 is its own NFC. */
typedef struct { char *buf; size_t len, cap; } Bytes;

static bool bytes_reserve(Bytes *b, size_t more) {
    if (more > SIZE_MAX / 2 - b->len) return false;
    if (b->len + more <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + more) cap *= 2;
    char *p = (char *)realloc(b->buf, cap);
    if (!p) return false;
    b->buf = p;
    b->cap = cap;
    return true;
}

static bool bytes_add(Bytes *b, const char *s, size_t n) {
    if (!bytes_reserve(b, n)) return false;
    if (n) memcpy(b->buf + b->len, s, n);
    b->len += n;
    return true;
}

static bool nfc(const char *s, size_t n, Bytes *out) {
    bool simple = true;
    for (size_t i = 0; i < n && simple; i++) simple = (unsigned char)s[i] < 0xCC;
    if (simple) return bytes_add(out, s, n);
    uint32_t *cps = (uint32_t *)malloc((8 * n + 8) * sizeof(uint32_t));
    if (!cps) return false;
    size_t m = 0;
    for (size_t i = 0; i < n;) m += decompose(next_cp(s, n, &i), cps + m, 0);
    /* canonical order: each run of non-starters sorted by class, stably */
    for (size_t i = 1; i < m; i++) {
        unsigned c = combining_class(cps[i]);
        if (!c) continue;
        for (size_t j = i; j > 0; j--) {
            unsigned p = combining_class(cps[j - 1]);
            if (p <= c) break;
            uint32_t t = cps[j]; cps[j] = cps[j - 1]; cps[j - 1] = t;
        }
    }
    /* composition (the algorithm of UAX #15): a character joins the last starter unless something of
       its class or higher stands between them */
    size_t starter = 0, kept = m ? 1 : 0;
    unsigned last = m ? combining_class(cps[0]) : 0;
    if (last) last = 256;
    for (size_t i = 1; i < m; i++) {
        uint32_t ch = cps[i];
        unsigned c = combining_class(ch);
        uint32_t comp = composite(cps[starter], ch);
        if (comp && (last < c || last == 0)) {
            cps[starter] = comp;
            continue;
        }
        if (c == 0) starter = kept;
        last = c;
        cps[kept++] = ch;
    }
    bool ok = bytes_reserve(out, 4 * kept);
    for (size_t i = 0; ok && i < kept; i++) out->len += utf8_put(out->buf + out->len, cps[i]);
    free(cps);
    return ok;
}

/* ------------------------------------------------------------------------- GPT-2's bytes as characters */

/* The characters of GPT-2's byte-level alphabet: the printable bytes of Latin-1 themselves, the others
   256 and up in their order. */
static bool byte_printable(unsigned b) { return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE; }

/* ------------------------------------------------------------------------- the tokenizer */

typedef struct {
    uint32_t id;
    uint32_t off, len;              /* its content in the pool */
    bool special, normalized, lstrip, rstrip, single_word;
} Added;

enum {
    /* normalizers */
    STEP_PREPEND, STEP_REPLACE, STEP_NFC, STEP_STRIP,
    /* pre-tokenizers */
    STEP_BYTE_LEVEL, STEP_SPLIT, STEP_METASPACE, STEP_DIGITS,
    /* decoders (STEP_REPLACE, STEP_STRIP, STEP_BYTE_LEVEL and STEP_METASPACE too) */
    STEP_BYTE_FALLBACK, STEP_FUSE
};

enum { PATTERN_STRING, PATTERN_GPT2, PATTERN_CL100K, PATTERN_O200K };
enum { SPLIT_REMOVED, SPLIT_ISOLATED, SPLIT_MERGED_PREVIOUS, SPLIT_MERGED_NEXT, SPLIT_CONTIGUOUS };
enum { PREPEND_ALWAYS, PREPEND_FIRST, PREPEND_NEVER };

typedef struct {
    int kind;
    char *a, *b;                    /* Prepend's, Replace's pattern and content, Split's string, Metaspace's
                                       replacement, Strip's content */
    uint32_t alen, blen;
    int pattern, behavior;          /* Split */
    int digits;                     /* the patterns' \p{N}{1,digits} */
    bool contractions;              /* o200k: the contractions after a word */
    bool slash;                     /* o200k: [\r\n/]* after punctuation, else [\r\n]* */
    bool invert;                    /* Split */
    bool prefix_space, use_regex;   /* ByteLevel */
    int prepend;                    /* Metaspace */
    bool split;                     /* Metaspace */
    bool individual;                /* Digits */
    bool left, right;               /* Strip (normalizer) */
    uint32_t start, stop;           /* Strip (decoder) */
} Step;

typedef struct {
    uint64_t key;                   /* (left << 32) | right, UINT64_MAX for an empty slot */
    uint32_t rank, id;
} Pair;

struct SpingalettTokenizer {
    char *pool;                     /* every string, each followed by a NUL */
    size_t pool_len, pool_cap;
    uint32_t size;                  /* ids */
    uint32_t *model_off;            /* [size]: the model's token of each id (UINT32_MAX: none) */
    uint32_t *model_len;
    int32_t *added_of;              /* [size]: the added token of each id, -1 for none */
    uint32_t *vocab;                /* the model's tokens by hash: id + 1, 0 for an empty slot */
    uint32_t vocab_mask;
    Pair *pairs;                    /* merges by hash */
    uint32_t pair_mask;
    Added *added;
    uint32_t added_count;
    uint32_t *added_order;          /* added tokens by first byte, longest first */
    uint32_t added_first[257];      /* their ranges in added_order by first byte */
    Step *norm, *pre, *dec;
    uint32_t norm_count, pre_count, dec_count;
    bool has_decoder;
    uint32_t unk;
    bool byte_fallback, fuse_unk, ignore_merges;
    bool split_marks;               /* pieces merged word by word at U+2581 (bpe()) */
    char *prefix, *suffix;          /* continuing_subword_prefix, end_of_word_suffix (NULL: none) */
    uint32_t prefix_len, suffix_len;
    uint32_t byte_tokens[256];      /* the <0xXX> tokens */
    char byte_chars[256][2];        /* GPT-2's alphabet: each byte's character as UTF-8 */
    uint8_t byte_char_len[256];
    int16_t char_bytes[324];        /* and each character's byte (-1 for none) */
    uint32_t before[4], after[4];   /* the template of a single text */
    uint32_t before_count, after_count;
    uint32_t bos, eos, pad;
};

static const char *tok_str(const SpingalettTokenizer *t, uint32_t off) { return t->pool + off; }

static uint64_t hash_bytes(const char *s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 0x100000001b3ull;
    return h ^ (h >> 29);
}

static uint64_t hash_pair(uint64_t key) {
    key ^= key >> 31;
    key *= 0x9E3779B97F4A7C15ull;
    return key ^ (key >> 29);
}

/* Appends a string and its NUL to the pool; UINT32_MAX when memory runs out or the pool would pass 4 GB. */
static uint32_t pool_add(SpingalettTokenizer *t, const char *s, size_t n) {
    if (n >= UINT32_MAX - 1 || t->pool_len + n + 1 >= UINT32_MAX) return UINT32_MAX;
    if (t->pool_len + n + 1 > t->pool_cap) {
        size_t cap = t->pool_cap ? t->pool_cap : 4096;
        while (cap < t->pool_len + n + 1) cap *= 2;
        char *p = (char *)realloc(t->pool, cap);
        if (!p) return UINT32_MAX;
        t->pool = p;
        t->pool_cap = cap;
    }
    uint32_t off = (uint32_t)t->pool_len;
    if (n) memcpy(t->pool + off, s, n);
    t->pool[off + n] = '\0';
    t->pool_len += n + 1;
    return off;
}

/* The model's token s, NONE when the vocabulary lacks it. */
static uint32_t vocab_get(const SpingalettTokenizer *t, const char *s, size_t n) {
    if (!t->vocab) return NONE;
    for (uint64_t i = hash_bytes(s, n) & t->vocab_mask;; i = (i + 1) & t->vocab_mask) {
        uint32_t slot = t->vocab[i];
        if (!slot) return NONE;
        uint32_t id = slot - 1;
        if (t->model_len[id] == n && (n == 0 || memcmp(t->pool + t->model_off[id], s, n) == 0)) return id;
    }
}

static const Pair *pair_get(const SpingalettTokenizer *t, uint32_t a, uint32_t b) {
    if (!t->pairs) return NULL;
    const uint64_t key = (uint64_t)a << 32 | b;
    for (uint64_t i = hash_pair(key) & t->pair_mask;; i = (i + 1) & t->pair_mask) {
        const Pair *p = &t->pairs[i];
        if (p->key == key) return p;
        if (p->key == UINT64_MAX) return NULL;
    }
}

static bool tfail(char *error, size_t cap, const char *fmt, const char *arg) {
    if (!error[0]) snprintf(error, cap, fmt, arg ? arg : "");
    return false;
}

/* Sets the model's token of id (count ids in all) and indexes it. */
static bool set_model_token(SpingalettTokenizer *t, uint32_t id, const char *s, size_t n) {
    uint32_t off = pool_add(t, s, n);
    if (off == UINT32_MAX) return false;
    t->model_off[id] = off;
    t->model_len[id] = (uint32_t)n;
    return true;
}

/* The tables of ids, the vocabulary's hash, once the model's tokens are set (ids without one have
   model_off UINT32_MAX). Of equal tokens the lowest id is found. */
static bool index_vocab(SpingalettTokenizer *t) {
    uint64_t slots = 16;
    while (slots < 2ull * t->size + 2) slots *= 2;
    t->vocab = (uint32_t *)calloc(slots, sizeof(uint32_t));
    if (!t->vocab) return false;
    t->vocab_mask = (uint32_t)(slots - 1);
    for (uint32_t id = 0; id < t->size; id++) {
        if (t->model_off[id] == UINT32_MAX) continue;
        const char *s = t->pool + t->model_off[id];
        uint32_t n = t->model_len[id];
        if (vocab_get(t, s, n) != NONE) continue;
        uint64_t i = hash_bytes(s, n) & t->vocab_mask;
        while (t->vocab[i]) i = (i + 1) & t->vocab_mask;
        t->vocab[i] = id + 1;
    }
    char name[8];
    for (unsigned b = 0; b < 256; b++) {
        snprintf(name, sizeof name, "<0x%02X>", b);
        t->byte_tokens[b] = vocab_get(t, name, 6);
    }
    return true;
}

static bool alloc_ids(SpingalettTokenizer *t, uint32_t size) {
    t->size = size;
    t->model_off = (uint32_t *)malloc((size_t)size * sizeof(uint32_t));
    t->model_len = (uint32_t *)calloc(size, sizeof(uint32_t));
    t->added_of = (int32_t *)malloc((size_t)size * sizeof(int32_t));
    if (!t->model_off || !t->model_len || !t->added_of) return false;
    memset(t->model_off, 0xFF, (size_t)size * sizeof(uint32_t));
    memset(t->added_of, 0xFF, (size_t)size * sizeof(int32_t));
    return true;
}

/* The merges' table for count merges, then each merge added by merge_add(). */
static bool pairs_alloc(SpingalettTokenizer *t, uint32_t count) {
    uint64_t slots = 16;
    while (slots < 2ull * count + 2) slots *= 2;
    t->pairs = (Pair *)malloc(slots * sizeof(Pair));
    if (!t->pairs) return false;
    for (uint64_t i = 0; i < slots; i++) t->pairs[i].key = UINT64_MAX;
    t->pair_mask = (uint32_t)(slots - 1);
    return true;
}

/* Merge a + b of the given rank: a later merge of the same pair replaces an earlier one, as in the
   library's map. */
static void merge_put(SpingalettTokenizer *t, uint32_t a, uint32_t b, uint32_t rank, uint32_t id) {
    const uint64_t key = (uint64_t)a << 32 | b;
    uint64_t i = hash_pair(key) & t->pair_mask;
    while (t->pairs[i].key != UINT64_MAX && t->pairs[i].key != key) i = (i + 1) & t->pair_mask;
    t->pairs[i].key = key;
    t->pairs[i].rank = rank;
    t->pairs[i].id = id;
}

/* The merge of tokens a and b (bytes): the ids of both and of their concatenation (b without the
   continuing prefix). */
static bool merge_add(SpingalettTokenizer *t, const char *a, size_t an, const char *b, size_t bn, uint32_t rank,
                      char *error, size_t cap) {
    uint32_t ia = vocab_get(t, a, an), ib = vocab_get(t, b, bn);
    size_t skip = t->prefix && bn >= t->prefix_len && memcmp(b, t->prefix, t->prefix_len) == 0 ? t->prefix_len : 0;
    char small[256];
    size_t n = an + bn - skip;
    char *joined = n <= sizeof small ? small : (char *)malloc(n);
    if (!joined) return tfail(error, cap, "out of memory", NULL);
    memcpy(joined, a, an);
    memcpy(joined + an, b + skip, bn - skip);
    uint32_t id = vocab_get(t, joined, n);
    if (joined != small) free(joined);
    if (ia == NONE || ib == NONE || id == NONE) {
        char what[96];
        snprintf(what, sizeof what, "%.*s %.*s", (int)(an < 40 ? an : 40), a, (int)(bn < 40 ? bn : 40), b);
        return tfail(error, cap, "tokenizer: the merge '%s' names tokens the vocabulary lacks", what);
    }
    merge_put(t, ia, ib, rank, id);
    return true;
}

/* Indexes the added tokens by first byte, longest first, and fills added_of. */
static bool index_added(SpingalettTokenizer *t) {
    t->added_order = (uint32_t *)malloc(((size_t)t->added_count + 1) * sizeof(uint32_t));
    if (!t->added_order) return false;
    memset(t->added_first, 0, sizeof t->added_first);
    for (uint32_t k = 0; k < t->added_count; k++) {
        const Added *a = &t->added[k];
        if (a->len) t->added_first[(unsigned char)t->pool[a->off] + 1]++;
        if (a->id < t->size) t->added_of[a->id] = (int32_t)k;
    }
    for (unsigned b = 0; b < 256; b++) t->added_first[b + 1] += t->added_first[b];
    uint32_t fill[256];
    for (unsigned b = 0; b < 256; b++) fill[b] = t->added_first[b];
    for (uint32_t k = 0; k < t->added_count; k++)
        if (t->added[k].len) t->added_order[fill[(unsigned char)t->pool[t->added[k].off]]++] = k;
    for (unsigned b = 0; b < 256; b++)      /* longest first, then in the file's order */
        for (uint32_t i = t->added_first[b] + 1; i < t->added_first[b + 1]; i++)
            for (uint32_t j = i; j > t->added_first[b] && t->added[t->added_order[j - 1]].len < t->added[t->added_order[j]].len; j--) {
                uint32_t x = t->added_order[j]; t->added_order[j] = t->added_order[j - 1]; t->added_order[j - 1] = x;
            }
    return true;
}

static void free_steps(Step *s, uint32_t n) {
    for (uint32_t k = 0; k < n; k++) { free(s[k].a); free(s[k].b); }
    free(s);
}

void spingalett_tokenizer_free(SpingalettTokenizer *t) {
    if (!t) return;
    free(t->pool);
    free(t->model_off);
    free(t->model_len);
    free(t->added_of);
    free(t->vocab);
    free(t->pairs);
    free(t->added);
    free(t->added_order);
    free_steps(t->norm, t->norm_count);
    free_steps(t->pre, t->pre_count);
    free_steps(t->dec, t->dec_count);
    free(t->prefix);
    free(t->suffix);
    free(t);
}

static SpingalettTokenizer *tokenizer_new(void) {
    SpingalettTokenizer *t = (SpingalettTokenizer *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->unk = t->bos = t->eos = t->pad = NONE;
    memset(t->char_bytes, 0xFF, sizeof t->char_bytes);
    for (unsigned b = 0, extra = 0; b < 256; b++) {
        uint32_t cp = byte_printable(b) ? b : 256 + extra++;
        t->byte_char_len[b] = (uint8_t)utf8_put(t->byte_chars[b], cp);
        t->char_bytes[cp] = (int16_t)b;
    }
    return t;
}

static Step *add_step(Step **steps, uint32_t *count, int kind) {
    Step *s = (Step *)realloc(*steps, ((size_t)*count + 1) * sizeof(Step));
    if (!s) return NULL;
    *steps = s;
    Step *st = &s[(*count)++];
    memset(st, 0, sizeof *st);
    st->kind = kind;
    return st;
}

static char *dup_bytes(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    if (n) memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* ------------------------------------------------------------------------- the patterns */

/* The code point at i, or 0 past the end (callers test the end first where it matters). */
static uint32_t cp_at(const char *s, size_t n, size_t i, size_t *len) {
    if (i >= n) { *len = 0; return 0; }
    size_t k = i;
    uint32_t cp = next_cp(s, n, &k);
    *len = k - i;
    return cp;
}

/* (?i:'s|'t|'re|'ve|'m|'ll|'d) at i (case-sensitive for GPT-2's): its length, 0 when none matches.
   Oniguruma's case folding makes s match ſ (U+017F) as well. */
static size_t contraction(const char *s, size_t n, size_t i, bool fold) {
    if (i >= n || s[i] != '\'') return 0;
    size_t l1, l2;
    uint32_t c1 = cp_at(s, n, i + 1, &l1);
    if (!l1) return 0;
    uint32_t c2 = cp_at(s, n, i + 1 + l1, &l2);
    if (fold) {
        if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
        if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
        if (c1 == 0x17F) c1 = 's';
    }
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 1 + l1;
    if (l2 && ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l'))) return 1 + l1 + l2;
    return 0;
}

/* The end of a run of characters of a class from i. */
typedef bool (*CharClass)(uint32_t);

static size_t run_end(const char *s, size_t n, size_t i, CharClass in, size_t max_chars) {
    size_t chars = 0;
    while (i < n && chars < max_chars) {
        size_t len;
        uint32_t c = cp_at(s, n, i, &len);
        if (!in(c)) break;
        i += len;
        chars++;
    }
    return i;
}

static bool is_other(uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); }
static bool is_newline(uint32_t c) { return c == '\r' || c == '\n'; }
static bool is_newline_or_slash(uint32_t c) { return c == '\r' || c == '\n' || c == '/'; }
/* [^\r\n\p{L}\p{N}]: the character a word may start with. */
static bool is_word_lead(uint32_t c) { return c != '\r' && c != '\n' && !is_letter(c) && !is_number(c); }
/* o200k's [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and [\p{Ll}\p{Lm}\p{Lo}\p{M}] */
static bool is_upper_class(uint32_t c) { int k = category(c); return k == CAT_LU || k == CAT_LT || k == CAT_LM || k == CAT_LO || k == CAT_M; }
static bool is_lower_class(uint32_t c) { int k = category(c); return k == CAT_LL || k == CAT_LM || k == CAT_LO || k == CAT_M; }

/* \s+(?!\S) and \s+ from i: the whitespace run, less its last character when something follows it and
   the run has more than one (lookahead); 0 when the lookahead form fails. */
static size_t spaces_not_before_text(const char *s, size_t n, size_t i) {
    size_t j = run_end(s, n, i, is_space, SIZE_MAX);
    if (j == i) return 0;
    if (j == n) return j - i;
    /* give back the last character */
    size_t last = j - 1;
    while (last > i && ((unsigned char)s[last] & 0xC0) == 0x80) last--;
    return last > i ? last - i : 0;
}

/* \s*[\r\n]+ from i: up to the run's last \r or \n; 0 when the run has none. */
static size_t spaces_to_newline(const char *s, size_t n, size_t i) {
    size_t j = i, end = 0;
    while (j < n) {
        size_t len;
        uint32_t c = cp_at(s, n, j, &len);
        if (!is_space(c)) break;
        j += len;
        if (is_newline(c)) end = j;
    }
    return end ? end - i : 0;
}

/* GPT-2's 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+ at i. */
static size_t match_gpt2(const char *s, size_t n, size_t i) {
    size_t m = contraction(s, n, i, false);
    if (m) return m;
    static const CharClass classes[3] = {is_letter, is_number, is_other};
    size_t l0, l1;
    uint32_t c0 = cp_at(s, n, i, &l0);
    for (int k = 0; k < 3; k++) {
        if (c0 == ' ' && i + 1 < n && classes[k](cp_at(s, n, i + 1, &l1))) return run_end(s, n, i + 1, classes[k], SIZE_MAX) - i;
        if (classes[k](c0)) return run_end(s, n, i, classes[k], SIZE_MAX) - i;
    }
    m = spaces_not_before_text(s, n, i);
    if (m) return m;
    m = run_end(s, n, i, is_space, SIZE_MAX) - i;
    return m ? m : l0;
}

/* cl100k's (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,d}| ?[^\s\p{L}\p{N}]+[\r\n]*|
   \s*[\r\n]+|\s+(?!\S)|\s+ at i. */
static size_t match_cl100k(const char *s, size_t n, size_t i, int digits) {
    size_t m = contraction(s, n, i, true);
    if (m) return m;
    size_t l0, l1;
    uint32_t c0 = cp_at(s, n, i, &l0);
    if (is_word_lead(c0) && i + l0 < n && is_letter(cp_at(s, n, i + l0, &l1)))
        return run_end(s, n, i + l0, is_letter, SIZE_MAX) - i;
    if (is_letter(c0)) return run_end(s, n, i, is_letter, SIZE_MAX) - i;
    if (is_number(c0)) return run_end(s, n, i, is_number, (size_t)digits) - i;
    size_t j = 0;
    if (c0 == ' ' && i + 1 < n && is_other(cp_at(s, n, i + 1, &l1))) j = run_end(s, n, i + 1, is_other, SIZE_MAX);
    else if (is_other(c0)) j = run_end(s, n, i, is_other, SIZE_MAX);
    if (j) return run_end(s, n, j, is_newline, SIZE_MAX) - i;
    m = spaces_to_newline(s, n, i);
    if (m) return m;
    m = spaces_not_before_text(s, n, i);
    if (m) return m;
    m = run_end(s, n, i, is_space, SIZE_MAX) - i;
    return m ? m : l0;
}

/* o200k's [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+ from j: its end, 0 when it fails. The
   upper run is greedy; when no lower character follows it, it gives back characters until one it gave
   back is of the lower class too (Lm, Lo, M), which then ends the match. */
static size_t o200k_lower_word(const char *s, size_t n, size_t j) {
    size_t k = run_end(s, n, j, is_upper_class, SIZE_MAX), len;
    if (k < n && is_lower_class(cp_at(s, n, k, &len))) return run_end(s, n, k, is_lower_class, SIZE_MAX);
    /* the run's last character of both classes */
    size_t end = 0;
    for (size_t p = j; p < k;) {
        uint32_t c = cp_at(s, n, p, &len);
        p += len;
        if (is_lower_class(c)) end = p;
    }
    return end;
}

/* o200k's (and tekken's) [^\r\n\p{L}\p{N}]?U*L+C?|[^\r\n\p{L}\p{N}]?U+L*C?|\p{N}{1,d}| ?[^\s\p{L}\p{N}]+[\r\n/]*|
   \s*[\r\n]+|\s+(?!\S)|\s+ at i, U and L the upper and lower classes, C the optional contractions. */
static size_t match_o200k(const char *s, size_t n, size_t i, int digits, bool contractions, bool slash) {
    size_t l0, l1;
    uint32_t c0 = cp_at(s, n, i, &l0);
    const bool lead = is_word_lead(c0);
    for (int alt = 0; alt < 2; alt++)
        for (int with_lead = lead ? 1 : 0; with_lead >= 0; with_lead--) {
            size_t j = with_lead ? i + l0 : i, e;
            if (alt == 0) {
                e = o200k_lower_word(s, n, j);
            } else {
                e = run_end(s, n, j, is_upper_class, SIZE_MAX);
                if (e == j) e = 0;
                else e = run_end(s, n, e, is_lower_class, SIZE_MAX);
            }
            if (!e) continue;
            if (contractions) e += contraction(s, n, e, true);
            return e - i;
        }
    if (is_number(c0)) return run_end(s, n, i, is_number, (size_t)digits) - i;
    size_t j = 0;
    if (c0 == ' ' && i + 1 < n && is_other(cp_at(s, n, i + 1, &l1))) j = run_end(s, n, i + 1, is_other, SIZE_MAX);
    else if (is_other(c0)) j = run_end(s, n, i, is_other, SIZE_MAX);
    if (j) return run_end(s, n, j, slash ? is_newline_or_slash : is_newline, SIZE_MAX) - i;
    size_t m = spaces_to_newline(s, n, i);
    if (m) return m;
    m = spaces_not_before_text(s, n, i);
    if (m) return m;
    m = run_end(s, n, i, is_space, SIZE_MAX) - i;
    return m ? m : l0;
}

/* Recognizes a Split's regular expression; false for one there is no matcher of. */
static bool recognize_pattern(const char *re, Step *st) {
    static const char gpt2[] = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    if (!strcmp(re, gpt2)) { st->pattern = PATTERN_GPT2; return true; }
    static const char *const digit_forms[2] = {"\\p{N}{1,3}", "\\p{N}"};
    char text[1024];
    for (int d = 0; d < 2; d++) {
        snprintf(text, sizeof text, "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|%s| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
                 "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", digit_forms[d]);
        if (!strcmp(re, text)) { st->pattern = PATTERN_CL100K; st->digits = d ? 1 : 3; return true; }
        for (int c = 0; c < 2; c++)
            for (int sl = 0; sl < 2; sl++) {
                const char *con = c ? "(?i:'s|'t|'re|'ve|'m|'ll|'d)?" : "";
                snprintf(text, sizeof text,
                         "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+%s|"
                         "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*%s|"
                         "%s| ?[^\\s\\p{L}\\p{N}]+[\\r\\n%s]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                         con, con, digit_forms[d], sl ? "/" : "");
                if (!strcmp(re, text)) {
                    st->pattern = PATTERN_O200K;
                    st->digits = d ? 1 : 3;
                    st->contractions = c;
                    st->slash = sl;
                    return true;
                }
            }
    }
    return false;
}

/* ------------------------------------------------------------------------- pieces of a text */

typedef struct {
    uint32_t off, len;              /* text: its bytes in the buffer */
    uint32_t id;                    /* an added token, NONE for text */
    bool first;                     /* text that starts the input (Metaspace's "first") */
} Piece;

typedef struct {
    Bytes text;
    Piece *items;
    size_t count, cap;
    bool failed;
} Pieces;

static void pieces_clear(Pieces *p) { p->text.len = 0; p->count = 0; }

static Piece *new_piece(Pieces *p) {
    if (p->count == p->cap) {
        size_t cap = p->cap ? 2 * p->cap : 64;
        Piece *items = (Piece *)realloc(p->items, cap * sizeof(Piece));
        if (!items) { p->failed = true; return NULL; }
        p->items = items;
        p->cap = cap;
    }
    return &p->items[p->count++];
}

static void push_text(Pieces *p, const char *s, size_t n, bool first) {
    if (!n || p->failed) return;            /* empty pieces vanish, as in the library */
    if (p->text.len + n >= UINT32_MAX || !bytes_add(&p->text, s, n)) { p->failed = true; return; }
    Piece *pc = new_piece(p);
    if (!pc) return;
    pc->off = (uint32_t)(p->text.len - n);
    pc->len = (uint32_t)n;
    pc->id = NONE;
    pc->first = first;
}

/* A text piece made in the buffer by the caller (its last n bytes). */
static void push_made(Pieces *p, size_t n, bool first) {
    if (p->failed) return;
    if (!n) return;
    if (p->text.len >= UINT32_MAX) { p->failed = true; return; }
    Piece *pc = new_piece(p);
    if (!pc) return;
    pc->off = (uint32_t)(p->text.len - n);
    pc->len = (uint32_t)n;
    pc->id = NONE;
    pc->first = first;
}

static void push_added(Pieces *p, uint32_t id) {
    if (p->failed) return;
    Piece *pc = new_piece(p);
    if (!pc) return;
    pc->off = pc->len = 0;
    pc->id = id;
    pc->first = false;
}

/* ------------------------------------------------------------------------- added tokens */

static bool ends_with_word(const char *s, size_t n) {
    if (!n) return false;
    size_t k = n - 1;
    while (k > 0 && ((unsigned char)s[k] & 0xC0) == 0x80) k--;
    size_t i = k;
    uint32_t c = next_cp(s, n, &i);
    return is_letter(c) || is_number(c);
}

static bool starts_with_word(const char *s, size_t n) {
    if (!n) return false;
    size_t i = 0;
    uint32_t c = next_cp(s, n, &i);
    return is_letter(c) || is_number(c);
}

/* Splits text s at the added tokens whose "normalized" equals `normalized`, as the library's
   find_matches() does; special tokens are found only with `special`, but are matched (and skipped)
   either way, as the library's automaton matches them. */
static void split_added(const SpingalettTokenizer *t, Pieces *out, const char *s, size_t n, bool first, bool normalized,
                        bool special) {
    size_t start_offset = 0, i = 0;
    while (i < n) {
        const unsigned b = (unsigned char)s[i];
        const Added *hit = NULL;
        for (uint32_t k = t->added_first[b]; k < t->added_first[b + 1] && !hit; k++) {
            const Added *a = &t->added[t->added_order[k]];
            if (a->normalized == normalized && a->len <= n - i && memcmp(s + i, t->pool + a->off, a->len) == 0) hit = a;
        }
        if (!hit) { i++; continue; }
        size_t start = i, stop = i + hit->len;
        i = stop;
        if (hit->special && !special) continue;
        if (hit->single_word && ((start > 0 && ends_with_word(s, start)) || (stop < n && starts_with_word(s + stop, n - stop))))
            continue;
        if (hit->lstrip) {
            size_t ns = start;
            while (ns > 0) {
                size_t k = ns - 1;
                while (k > 0 && ((unsigned char)s[k] & 0xC0) == 0x80) k--;
                size_t q = k;
                if (!is_space(next_cp(s, n, &q))) break;
                ns = k;
            }
            start = ns > start_offset ? ns : start_offset;
        }
        if (hit->rstrip) stop = run_end(s, n, stop, is_space, SIZE_MAX);
        if (start_offset < start) push_text(out, s + start_offset, start - start_offset, first && start_offset == 0);
        push_added(out, hit->id);
        start_offset = stop;
        if (i < stop) i = stop;
    }
    if (start_offset < n) push_text(out, s + start_offset, n - start_offset, first && start_offset == 0);
}

/* ------------------------------------------------------------------------- normalizers */

static void replace_all(Bytes *out, const char *s, size_t n, const char *pat, size_t pn, const char *with, size_t wn,
                        bool *failed) {
    size_t i = 0;
    if (!pn) { if (!bytes_add(out, s, n)) *failed = true; return; }
    while (i < n) {
        if (i + pn <= n && memcmp(s + i, pat, pn) == 0) {
            if (!bytes_add(out, with, wn)) { *failed = true; return; }
            i += pn;
        } else {
            if (!bytes_add(out, s + i, 1)) { *failed = true; return; }
            i++;
        }
    }
}

/* The normalizers on s into out (cleared first); false when memory runs out. */
static bool normalize(const SpingalettTokenizer *t, const char *s, size_t n, Bytes *out, Bytes *tmp) {
    out->len = 0;
    if (!bytes_add(out, s, n)) return false;
    for (uint32_t k = 0; k < t->norm_count; k++) {
        const Step *st = &t->norm[k];
        tmp->len = 0;
        bool failed = false;
        switch (st->kind) {
            case STEP_PREPEND:
                if (out->len && (!bytes_add(tmp, st->a, st->alen) || !bytes_add(tmp, out->buf, out->len))) failed = true;
                break;
            case STEP_REPLACE:
                replace_all(tmp, out->buf, out->len, st->a, st->alen, st->b, st->blen, &failed);
                break;
            case STEP_NFC:
                failed = !nfc(out->buf, out->len, tmp);
                break;
            default: {                      /* STEP_STRIP */
                size_t a = 0, b = out->len;
                if (st->left) a = run_end(out->buf, out->len, 0, is_space, SIZE_MAX);
                if (st->right)
                    while (b > a) {
                        size_t q = b - 1;
                        while (q > a && ((unsigned char)out->buf[q] & 0xC0) == 0x80) q--;
                        size_t r = q;
                        if (!is_space(next_cp(out->buf, out->len, &r))) break;
                        b = q;
                    }
                failed = !bytes_add(tmp, out->buf + a, b - a);
                break;
            }
        }
        if (failed) return false;
        Bytes x = *out; *out = *tmp; *tmp = x;
    }
    return true;
}

/* ------------------------------------------------------------------------- pre-tokenizers */

/* Spans of s: matches and the text between them, as the library's find_matches() lists them. */
typedef struct { size_t start, end; bool match; } Span;
typedef struct { Span *items; size_t count, cap; } Spans;

static bool add_span(Spans *sp, size_t start, size_t end, bool match) {
    if (sp->count == sp->cap) {
        size_t cap = sp->cap ? 2 * sp->cap : 64;
        Span *p = (Span *)realloc(sp->items, cap * sizeof(Span));
        if (!p) return false;
        sp->items = p;
        sp->cap = cap;
    }
    sp->items[sp->count++] = (Span){start, end, match};
    return true;
}

static bool find_matches(const Step *st, const char *s, size_t n, Spans *sp) {
    sp->count = 0;
    if (!n) return add_span(sp, 0, 0, false);
    if (st->pattern == PATTERN_STRING) {
        size_t prev = 0;
        if (st->alen)
            for (size_t i = 0; i + st->alen <= n;) {
                if (memcmp(s + i, st->a, st->alen) == 0) {
                    if (prev != i && !add_span(sp, prev, i, false)) return false;
                    if (!add_span(sp, i, i + st->alen, true)) return false;
                    i += st->alen;
                    prev = i;
                } else {
                    i++;
                }
            }
        return prev == n || add_span(sp, prev, n, false);
    }
    for (size_t i = 0; i < n;) {
        size_t m = st->pattern == PATTERN_GPT2 ? match_gpt2(s, n, i)
                 : st->pattern == PATTERN_CL100K ? match_cl100k(s, n, i, st->digits)
                 : match_o200k(s, n, i, st->digits, st->contractions, st->slash);
        if (!add_span(sp, i, i + m, true)) return false;
        i += m;
    }
    return true;
}

/* The pieces of s by the spans and a behavior, as the library's NormalizedString::split(). */
static void split_by(Pieces *out, const char *s, Spans *sp, int behavior, bool invert, bool first) {
    if (invert)
        for (size_t k = 0; k < sp->count; k++) sp->items[k].match = !sp->items[k].match;
    size_t count = 0;
    bool previous = false;
    if (behavior == SPLIT_MERGED_NEXT) {
        /* from the end: a match joins the piece after it */
        for (size_t k = sp->count; k-- > 0;) {
            Span x = sp->items[k];
            if (x.match && !previous && count) sp->items[sp->count - count].start = x.start;
            else sp->items[sp->count - ++count] = (Span){x.start, x.end, false};
            previous = x.match;
        }
        for (size_t k = sp->count - count; k < sp->count; k++)
            push_text(out, s + sp->items[k].start, sp->items[k].end - sp->items[k].start, first && sp->items[k].start == 0);
        return;
    }
    for (size_t k = 0; k < sp->count; k++) {
        Span x = sp->items[k];
        bool remove = false;
        if (behavior == SPLIT_REMOVED) {
            remove = x.match;
        } else if (behavior == SPLIT_CONTIGUOUS || behavior == SPLIT_MERGED_PREVIOUS) {
            bool merge = behavior == SPLIT_CONTIGUOUS ? x.match == previous : x.match && !previous;
            previous = x.match;
            if (merge && count) { sp->items[count - 1].end = x.end; continue; }
        }
        sp->items[count++] = (Span){x.start, x.end, remove};
    }
    for (size_t k = 0; k < count; k++)
        if (!sp->items[k].match)
            push_text(out, s + sp->items[k].start, sp->items[k].end - sp->items[k].start, first && sp->items[k].start == 0);
}

/* Bytes s as characters of GPT-2's alphabet appended to b. */
static bool byte_level_map(const SpingalettTokenizer *t, Bytes *b, const char *s, size_t n) {
    if (!bytes_reserve(b, 2 * n)) return false;
    for (size_t i = 0; i < n; i++) {
        const unsigned c = (unsigned char)s[i];
        memcpy(b->buf + b->len, t->byte_chars[c], 2);
        b->len += t->byte_char_len[c];
    }
    return true;
}

/* One pre-tokenizer over the pieces of in, into out. */
static bool pre_tokenize(const SpingalettTokenizer *t, const Step *st, const Pieces *in, Pieces *out, Spans *sp,
                         Bytes *tmp) {
    pieces_clear(out);
    for (size_t k = 0; k < in->count && !out->failed; k++) {
        const Piece *pc = &in->items[k];
        if (pc->id != NONE) { push_added(out, pc->id); continue; }
        const char *s = in->text.buf + pc->off;
        size_t n = pc->len;
        switch (st->kind) {
            case STEP_BYTE_LEVEL: {
                tmp->len = 0;
                if (st->prefix_space && s[0] != ' ' && !bytes_add(tmp, " ", 1)) return false;
                if (!bytes_add(tmp, s, n)) return false;
                if (st->use_regex) {
                    Step g = {.pattern = PATTERN_GPT2};
                    if (!find_matches(&g, tmp->buf, tmp->len, sp)) return false;
                } else {
                    sp->count = 0;
                    if (!add_span(sp, 0, tmp->len, true)) return false;
                }
                for (size_t m = 0; m < sp->count; m++) {
                    size_t before = out->text.len;
                    if (!byte_level_map(t, &out->text, tmp->buf + sp->items[m].start, sp->items[m].end - sp->items[m].start))
                        return false;
                    push_made(out, out->text.len - before, pc->first && sp->items[m].start == 0);
                }
                break;
            }
            case STEP_SPLIT:
                if (!find_matches(st, s, n, sp)) return false;
                split_by(out, s, sp, st->behavior, st->invert, pc->first);
                break;
            case STEP_METASPACE: {
                tmp->len = 0;
                bool failed = false;
                replace_all(tmp, s, n, " ", 1, st->a, st->alen, &failed);
                if (failed) return false;
                const bool starts = tmp->len >= st->alen && memcmp(tmp->buf, st->a, st->alen) == 0;
                if (!starts && (st->prepend == PREPEND_ALWAYS || (st->prepend == PREPEND_FIRST && pc->first))) {
                    if (!bytes_reserve(tmp, st->alen)) return false;
                    memmove(tmp->buf + st->alen, tmp->buf, tmp->len);
                    memcpy(tmp->buf, st->a, st->alen);
                    tmp->len += st->alen;
                }
                if (st->split) {
                    Step d = {.pattern = PATTERN_STRING, .a = st->a, .alen = st->alen};
                    if (!find_matches(&d, tmp->buf, tmp->len, sp)) return false;
                    split_by(out, tmp->buf, sp, SPLIT_MERGED_NEXT, false, pc->first);
                } else {
                    push_text(out, tmp->buf, tmp->len, pc->first);
                }
                break;
            }
            default: {                      /* STEP_DIGITS: each digit, or each run of them, a piece */
                sp->count = 0;
                size_t last = 0;
                for (size_t i = 0; i < n;) {
                    size_t j = i;
                    uint32_t c = next_cp(s, n, &j);
                    if (is_number(c)) {
                        if (last < i && !add_span(sp, last, i, false)) return false;
                        if (!add_span(sp, i, j, true)) return false;
                        last = j;
                    }
                    i = j;
                }
                if (last < n && !add_span(sp, last, n, false)) return false;
                split_by(out, s, sp, st->individual ? SPLIT_ISOLATED : SPLIT_CONTIGUOUS, false, pc->first);
                break;
            }
        }
    }
    return !out->failed;
}

/* ------------------------------------------------------------------------- the BPE model */

typedef struct { uint32_t *items; size_t count, cap; } Ids;

static bool ids_add(Ids *ids, uint32_t id) {
    if (ids->count == ids->cap) {
        size_t cap = ids->cap ? 2 * ids->cap : 256;
        uint32_t *p = (uint32_t *)realloc(ids->items, cap * sizeof(uint32_t));
        if (!p) return false;
        ids->items = p;
        ids->cap = cap;
    }
    ids->items[ids->count++] = id;
    return true;
}

typedef struct { uint32_t id; int32_t prev, next; uint32_t len; } Symbol;
typedef struct { uint32_t rank, pos, id; } Candidate;

typedef struct {
    Symbol *symbols;
    size_t symbol_cap;
    Candidate *heap;
    size_t heap_count, heap_cap;
    Bytes scratch;
} Work;

static bool heap_less(const Candidate *a, const Candidate *b) {
    return a->rank != b->rank ? a->rank < b->rank : a->pos < b->pos;
}

static bool heap_push(Work *w, Candidate c) {
    if (w->heap_count == w->heap_cap) {
        size_t cap = w->heap_cap ? 2 * w->heap_cap : 64;
        Candidate *h = (Candidate *)realloc(w->heap, cap * sizeof(Candidate));
        if (!h) return false;
        w->heap = h;
        w->heap_cap = cap;
    }
    size_t i = w->heap_count++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!heap_less(&c, &w->heap[parent])) break;
        w->heap[i] = w->heap[parent];
        i = parent;
    }
    w->heap[i] = c;
    return true;
}

static Candidate heap_pop(Work *w) {
    Candidate top = w->heap[0], last = w->heap[--w->heap_count];
    size_t i = 0, n = w->heap_count;
    for (;;) {
        size_t c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && heap_less(&w->heap[c + 1], &w->heap[c])) c++;
        if (!heap_less(&w->heap[c], &last)) break;
        w->heap[i] = w->heap[c];
        i = c;
    }
    if (n) w->heap[i] = last;
    return top;
}

static bool add_symbol(Work *w, size_t *count, uint32_t id, uint32_t len) {
    if (*count == w->symbol_cap) {
        size_t cap = w->symbol_cap ? 2 * w->symbol_cap : 64;
        Symbol *s = (Symbol *)realloc(w->symbols, cap * sizeof(Symbol));
        if (!s) return false;
        w->symbols = s;
        w->symbol_cap = cap;
    }
    w->symbols[*count] = (Symbol){id, (int32_t)*count - 1, (int32_t)*count + 1, len};
    (*count)++;
    return true;
}

/* The BPE model's tokens of a piece s, appended to out: the library's merge_word() and merge_all(). */
static bool bpe_word(const SpingalettTokenizer *t, const char *s, size_t n, Work *w, Ids *out) {
    if (!n) return true;
    if (n >= INT32_MAX / 2) return false;
    if (t->ignore_merges) {
        uint32_t id = vocab_get(t, s, n);
        if (id != NONE) return ids_add(out, id);
    }
    size_t count = 0;
    uint32_t unk_id = NONE, unk_len = 0;
    for (size_t i = 0; i < n;) {
        size_t j = i;
        (void)next_cp(s, n, &j);
        const bool is_first = i == 0, is_last = j == n;
        const char *piece = s + i;
        size_t len = j - i;
        if ((!is_first && t->prefix) || (is_last && t->suffix)) {
            w->scratch.len = 0;
            if ((!is_first && t->prefix && !bytes_add(&w->scratch, t->prefix, t->prefix_len)) ||
                !bytes_add(&w->scratch, s + i, j - i) || (is_last && t->suffix && !bytes_add(&w->scratch, t->suffix, t->suffix_len)))
                return false;
            piece = w->scratch.buf;
            len = w->scratch.len;
        }
        const uint32_t byte_len = (uint32_t)(j - i);
        uint32_t id = vocab_get(t, piece, len);
        i = j;
        if (id != NONE) {
            if (unk_id != NONE) { if (!add_symbol(w, &count, unk_id, unk_len)) return false; unk_id = NONE; }
            if (!add_symbol(w, &count, id, byte_len)) return false;
            continue;
        }
        if (t->byte_fallback) {
            bool all = true;
            for (size_t k = 0; k < len && all; k++) all = t->byte_tokens[(unsigned char)piece[k]] != NONE;
            if (all) {
                for (size_t k = 0; k < len; k++)
                    if (!add_symbol(w, &count, t->byte_tokens[(unsigned char)piece[k]], 1)) return false;
                continue;
            }
        }
        if (t->unk != NONE) {
            if (unk_id != NONE && t->fuse_unk) { unk_len += byte_len; continue; }
            if (unk_id != NONE && !add_symbol(w, &count, unk_id, unk_len)) return false;
            unk_id = t->unk;
            unk_len = byte_len;
        }
    }
    if (unk_id != NONE && !add_symbol(w, &count, unk_id, unk_len)) return false;
    if (!count) return true;
    Symbol *sy = w->symbols;
    sy[count - 1].next = -1;
    /* every adjacent pair with a merge, then the lowest rank first, the leftmost of equals */
    w->heap_count = 0;
    for (size_t k = 0; k + 1 < count; k++) {
        const Pair *p = pair_get(t, sy[k].id, sy[k + 1].id);
        if (p && !heap_push(w, (Candidate){p->rank, (uint32_t)k, p->id})) return false;
    }
    while (w->heap_count) {
        Candidate top = heap_pop(w);
        sy = w->symbols;
        Symbol *cur = &sy[top.pos];
        if (cur->len == 0 || cur->next == -1) continue;
        const int32_t next_pos = cur->next;
        Symbol right = sy[next_pos];
        const Pair *p = pair_get(t, cur->id, right.id);
        if (!p || p->id != top.id) continue;            /* an expired candidate */
        cur->id = top.id;
        cur->len += right.len;
        cur->next = right.next;
        sy[next_pos].len = 0;
        if (right.next >= 0) sy[right.next].prev = (int32_t)top.pos;
        if (cur->prev >= 0) {
            const Pair *q = pair_get(t, sy[cur->prev].id, cur->id);
            if (q && !heap_push(w, (Candidate){q->rank, (uint32_t)cur->prev, q->id})) return false;
        }
        if (cur->next >= 0) {
            const Pair *q = pair_get(t, cur->id, sy[cur->next].id);
            if (q && !heap_push(w, (Candidate){q->rank, top.pos, q->id})) return false;
        }
    }
    for (size_t k = 0; k < count; k++)
        if (sy[k].len && !ids_add(out, sy[k].id)) return false;
    return true;
}

#define SPACE_MARK "\xE2\x96\x81"           /* U+2581, SentencePiece's space */

/* A piece by the BPE model. Where no token of the vocabulary holds U+2581 after another character
   (split_marks: SentencePiece's vocabularies, whose words begin with it), no merge can join the
   characters before such a U+2581 to it, so the piece is merged word by word: the same tokens, as the
   merges of one word never depend on another's, without a heap of the whole text (a SentencePiece text
   is one piece). */
static bool bpe(const SpingalettTokenizer *t, const char *s, size_t n, Work *w, Ids *out) {
    if (!t->split_marks) return bpe_word(t, s, n, w, out);
    size_t start = 0;
    for (size_t i = 1; i + 3 <= n; i++)
        if (s[i] == SPACE_MARK[0] && !memcmp(s + i, SPACE_MARK, 3) && (i < 3 || memcmp(s + i - 3, SPACE_MARK, 3) != 0)) {
            if (!bpe_word(t, s + start, i - start, w, out)) return false;
            start = i;
            i += 2;
        }
    return bpe_word(t, s + start, n - start, w, out);
}

/* Whether pieces may be merged word by word (bpe()): no prefix or suffix of subwords, no lookup of whole
   pieces, and no token holding U+2581 after another character. */
static bool marks_split(const SpingalettTokenizer *t) {
    if (t->prefix || t->suffix || t->ignore_merges) return false;
    for (uint32_t id = 0; id < t->size; id++) {
        if (t->model_off[id] == UINT32_MAX) continue;
        const char *s = t->pool + t->model_off[id];
        for (uint32_t k = 1; k + 3 <= t->model_len[id]; k++)
            if (!memcmp(s + k, SPACE_MARK, 3) && (k < 3 || memcmp(s + k - 3, SPACE_MARK, 3) != 0)) return false;
    }
    return true;
}

/* ------------------------------------------------------------------------- encoding */

int64_t spingalett_tokenize_args(SpingalettTokenizeArgs args) {
    const SpingalettTokenizer *t = args.tokenizer;
    if (!t || !args.text || (args.capacity && !args.tokens)) {
        set_error(SPINGALETT_ERR_INVALID, "tokenize: tokenizer or text is NULL, or tokens with a capacity");
        return -1;
    }
    const size_t n = args.length ? args.length : strlen(args.text);
    if (n >= UINT32_MAX / 4) {
        set_error(SPINGALETT_ERR_INVALID, "tokenize: the text exceeds 1 GB");
        return -1;
    }
    if (!utf8_valid(args.text, n)) {
        set_error(SPINGALETT_ERR_INVALID, "tokenize: the text is not UTF-8");
        return -1;
    }
    Pieces a = {0}, b = {0};
    Bytes norm = {0}, tmp = {0};
    Spans sp = {0};
    Work w = {0};
    Ids ids = {0};
    bool ok = true;
    if (args.add_special)
        for (uint32_t k = 0; k < t->before_count && ok; k++) ok = ids_add(&ids, t->before[k]);
    /* the added tokens of the text as given, then those of the normalized text */
    split_added(t, &a, args.text, n, true, false, args.special);
    for (size_t k = 0; k < a.count && !b.failed && ok; k++) {
        const Piece *pc = &a.items[k];
        if (pc->id != NONE) { push_added(&b, pc->id); continue; }
        ok = normalize(t, a.text.buf + pc->off, pc->len, &norm, &tmp);
        if (ok) split_added(t, &b, norm.buf, norm.len, pc->first, true, args.special);
    }
    ok = ok && !a.failed && !b.failed;
    Pieces *cur = &b, *next = &a;
    for (uint32_t k = 0; k < t->pre_count && ok; k++) {
        ok = pre_tokenize(t, &t->pre[k], cur, next, &sp, &tmp);
        Pieces *x = cur; cur = next; next = x;
    }
    for (size_t k = 0; k < cur->count && ok; k++) {
        const Piece *pc = &cur->items[k];
        ok = pc->id != NONE ? ids_add(&ids, pc->id) : bpe(t, cur->text.buf + pc->off, pc->len, &w, &ids);
    }
    if (args.add_special)
        for (uint32_t k = 0; k < t->after_count && ok; k++) ok = ids_add(&ids, t->after[k]);
    int64_t result = -1;
    if (ok) {
        size_t copy = ids.count < args.capacity ? ids.count : args.capacity;
        if (copy) memcpy(args.tokens, ids.items, copy * sizeof(uint32_t));
        result = (int64_t)ids.count;
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "tokenize: out of memory");
    }
    free(a.text.buf); free(a.items); free(b.text.buf); free(b.items);
    free(norm.buf); free(tmp.buf); free(sp.items);
    free(w.symbols); free(w.heap); free(w.scratch.buf);
    free(ids.items);
    return result;
}

/* ------------------------------------------------------------------------- decoding */

/* A list of strings in one buffer. */
typedef struct { Bytes text; size_t *off, *len; size_t count, cap; bool failed; } Strs;

static void strs_clear(Strs *s) { s->text.len = 0; s->count = 0; }

static void strs_end(Strs *s, size_t start) {
    if (s->failed) return;
    if (s->count == s->cap) {
        size_t cap = s->cap ? 2 * s->cap : 64;
        size_t *o = (size_t *)realloc(s->off, cap * sizeof(size_t));
        if (o) s->off = o;
        size_t *l = o ? (size_t *)realloc(s->len, cap * sizeof(size_t)) : NULL;
        if (!o || !l) { s->failed = true; return; }
        s->len = l;
        s->cap = cap;
    }
    s->off[s->count] = start;
    s->len[s->count] = s->text.len - start;
    s->count++;
}

static void strs_add(Strs *s, const char *p, size_t n) {
    size_t start = s->text.len;
    if (!bytes_add(&s->text, p, n)) { s->failed = true; return; }
    strs_end(s, start);
}

/* Bytes as UTF-8, each maximal malformed part replaced by U+FFFD, as Rust's from_utf8_lossy(). */
static void lossy_utf8(Bytes *out, const char *s, size_t n, bool *failed) {
    for (size_t i = 0; i < n;) {
        size_t k = (unsigned char)s[i] < 0x80 ? 1 : utf8_valid_at(s + i, n - i);
        if (k) {
            if (!bytes_add(out, s + i, k)) { *failed = true; return; }
            i += k;
            continue;
        }
        /* the maximal prefix of a well-formed sequence counts as one malformed part */
        unsigned c = (unsigned char)s[i];
        size_t need = c >= 0xC2 && c < 0xE0 ? 2 : c >= 0xE0 && c < 0xF0 ? 3 : c >= 0xF0 && c < 0xF5 ? 4 : 1, j = 1;
        for (; j < need && i + j < n; j++) {
            unsigned d = (unsigned char)s[i + j];
            unsigned lo = 0x80, hi = 0xBF;
            if (j == 1) {
                if (c == 0xE0) lo = 0xA0;
                else if (c == 0xED) hi = 0x9F;
                else if (c == 0xF0) lo = 0x90;
                else if (c == 0xF4) hi = 0x8F;
            }
            if (d < lo || d > hi) break;
        }
        if (!bytes_add(out, "\xEF\xBF\xBD", 3)) { *failed = true; return; }
        i += j;
    }
}

static bool is_byte_token(const char *s, size_t n, unsigned *byte) {
    if (n != 6 || memcmp(s, "<0x", 3) != 0 || s[5] != '>') return false;
    int h = s[3] >= '0' && s[3] <= '9' ? s[3] - '0' : s[3] >= 'a' && s[3] <= 'f' ? s[3] - 'a' + 10 : s[3] >= 'A' && s[3] <= 'F' ? s[3] - 'A' + 10 : -1;
    int l = s[4] >= '0' && s[4] <= '9' ? s[4] - '0' : s[4] >= 'a' && s[4] <= 'f' ? s[4] - 'a' + 10 : s[4] >= 'A' && s[4] <= 'F' ? s[4] - 'A' + 10 : -1;
    if (h < 0 || l < 0) return false;
    *byte = (unsigned)(h * 16 + l);
    return true;
}

/* One decoder over the strings of in, into out. */
static bool decode_step(const SpingalettTokenizer *t, const Step *st, const Strs *in, Strs *out) {
    strs_clear(out);
    bool failed = false;
    switch (st->kind) {
        case STEP_BYTE_LEVEL: {
            /* every token's characters as bytes (a token with others as its bytes), then one string */
            Bytes raw = {0};
            for (size_t k = 0; k < in->count && !failed; k++) {
                const char *s = in->text.buf + in->off[k];
                size_t n = in->len[k], start = raw.len;
                bool all = true;
                for (size_t i = 0; i < n && all && !failed;) {
                    uint32_t cp = next_cp(s, n, &i);
                    int b = cp < 324 ? t->char_bytes[cp] : -1;
                    char c = (char)b;
                    if (b < 0) all = false;
                    else if (!bytes_add(&raw, &c, 1)) failed = true;
                }
                if (!all) { raw.len = start; if (!bytes_add(&raw, s, n)) failed = true; }
            }
            size_t start = out->text.len;
            if (!failed) lossy_utf8(&out->text, raw.buf, raw.len, &failed);
            free(raw.buf);
            if (!failed) strs_end(out, start);
            break;
        }
        case STEP_REPLACE:
            for (size_t k = 0; k < in->count && !failed; k++) {
                size_t start = out->text.len;
                replace_all(&out->text, in->text.buf + in->off[k], in->len[k], st->a, st->alen, st->b, st->blen, &failed);
                strs_end(out, start);
            }
            break;
        case STEP_BYTE_FALLBACK: {
            Bytes run = {0};
            size_t run_tokens = 0;
            for (size_t k = 0; k <= in->count && !failed; k++) {
                unsigned byte;
                const char *s = k < in->count ? in->text.buf + in->off[k] : NULL;
                if (s && is_byte_token(s, in->len[k], &byte)) {
                    unsigned char c = (unsigned char)byte;
                    if (!bytes_add(&run, (const char *)&c, 1)) failed = true;
                    run_tokens++;
                    continue;
                }
                if (run_tokens) {
                    if (utf8_valid(run.buf, run.len)) strs_add(out, run.buf, run.len);
                    else for (size_t r = 0; r < run_tokens; r++) strs_add(out, "\xEF\xBF\xBD", 3);
                    run.len = 0;
                    run_tokens = 0;
                }
                if (s) strs_add(out, s, in->len[k]);
            }
            free(run.buf);
            break;
        }
        case STEP_FUSE: {
            size_t start = out->text.len;
            for (size_t k = 0; k < in->count && !failed; k++)
                if (!bytes_add(&out->text, in->text.buf + in->off[k], in->len[k])) failed = true;
            strs_end(out, start);
            break;
        }
        case STEP_STRIP:
            for (size_t k = 0; k < in->count; k++) {
                const char *s = in->text.buf + in->off[k];
                size_t n = in->len[k], a = 0, b = n;
                for (uint32_t r = 0; r < st->start && a + st->alen <= n && memcmp(s + a, st->a, st->alen) == 0; r++) a += st->alen;
                for (uint32_t r = 0; r < st->stop && b >= a + st->alen && memcmp(s + b - st->alen, st->a, st->alen) == 0; r++) b -= st->alen;
                strs_add(out, s + a, b - a);
            }
            break;
        default:                            /* STEP_METASPACE */
            for (size_t k = 0; k < in->count && !failed; k++) {
                const char *s = in->text.buf + in->off[k];
                size_t n = in->len[k], start = out->text.len;
                for (size_t i = 0; i < n && !failed;) {
                    if (st->alen && i + st->alen <= n && memcmp(s + i, st->a, st->alen) == 0) {
                        if (!(k == 0 && st->prepend != PREPEND_NEVER) && !bytes_add(&out->text, " ", 1)) failed = true;
                        i += st->alen;
                    } else {
                        if (!bytes_add(&out->text, s + i, 1)) failed = true;
                        i++;
                    }
                }
                strs_end(out, start);
            }
            break;
    }
    return !failed && !out->failed;
}

int64_t spingalett_detokenize_args(SpingalettDetokenizeArgs args) {
    const SpingalettTokenizer *t = args.tokenizer;
    if (!t || (args.count && !args.tokens) || (args.capacity && !args.text)) {
        set_error(SPINGALETT_ERR_INVALID, "detokenize: tokenizer or tokens is NULL, or text with a capacity");
        return -1;
    }
    Strs a = {0}, b = {0};
    for (size_t k = 0; k < args.count && !a.failed; k++) {
        const uint32_t id = args.tokens[k];
        if (id >= t->size || (t->added_of[id] < 0 && t->model_off[id] == UINT32_MAX)) {
            char msg[96];
            snprintf(msg, sizeof msg, "detokenize: %u is the id of no token", id);
            set_error(SPINGALETT_ERR_INVALID, msg);
            free(a.text.buf); free(a.off); free(a.len);
            return -1;
        }
        if (t->added_of[id] >= 0) {
            const Added *ad = &t->added[t->added_of[id]];
            if (args.skip_special && ad->special) continue;
            strs_add(&a, t->pool + ad->off, ad->len);
        } else {
            strs_add(&a, t->pool + t->model_off[id], t->model_len[id]);
        }
    }
    bool ok = !a.failed;
    Strs *cur = &a, *next = &b;
    for (uint32_t k = 0; k < t->dec_count && ok; k++) {
        ok = decode_step(t, &t->dec[k], cur, next);
        Strs *x = cur; cur = next; next = x;
    }
    /* the strings joined (with spaces when there is no decoder, as the library joins them) */
    Bytes text = {0};
    for (size_t k = 0; k < cur->count && ok; k++) {
        if (!t->has_decoder && k && !bytes_add(&text, " ", 1)) ok = false;
        if (ok && !bytes_add(&text, cur->text.buf + cur->off[k], cur->len[k])) ok = false;
    }
    int64_t result = -1;
    if (ok) {
        if (args.capacity) {
            size_t copy = text.len < args.capacity - 1 ? text.len : args.capacity - 1;
            if (copy) memcpy(args.text, text.buf, copy);
            args.text[copy] = '\0';
        }
        result = (int64_t)text.len;
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "detokenize: out of memory");
    }
    free(text.buf);
    free(a.text.buf); free(a.off); free(a.len);
    free(b.text.buf); free(b.off); free(b.len);
    return result;
}

/* ------------------------------------------------------------------------- lookups */

SpingalettTokenizerInfo spingalett_tokenizer_info(const SpingalettTokenizer *t) {
    SpingalettTokenizerInfo info;
    memset(&info, 0, sizeof info);
    info.bos = info.eos = info.pad = info.unk = NONE;
    if (!t) {
        set_error(SPINGALETT_ERR_INVALID, "tokenizer_info: tokenizer is NULL");
        return info;
    }
    info.vocabulary = t->size;
    info.added = t->added_count;
    info.bos = t->bos;
    info.eos = t->eos;
    info.pad = t->pad;
    info.unk = t->unk;
    return info;
}

const char *spingalett_tokenizer_token(const SpingalettTokenizer *t, uint32_t id) {
    if (!t || id >= t->size) return NULL;
    if (t->added_of[id] >= 0) return tok_str(t, t->added[t->added_of[id]].off);
    return t->model_off[id] == UINT32_MAX ? NULL : tok_str(t, t->model_off[id]);
}

uint32_t spingalett_tokenizer_id(const SpingalettTokenizer *t, const char *token) {
    if (!t || !token) return NONE;
    size_t n = strlen(token);
    for (uint32_t k = 0; k < t->added_count; k++)
        if (t->added[k].len == n && memcmp(t->pool + t->added[k].off, token, n) == 0) return t->added[k].id;
    return vocab_get(t, token, n);
}

/* The id of a token by its content (added tokens first), NONE when none. */
static uint32_t find_token(const SpingalettTokenizer *t, const char *s, size_t n) {
    for (uint32_t k = 0; k < t->added_count; k++)
        if (t->added[k].len == n && memcmp(t->pool + t->added[k].off, s, n) == 0) return t->added[k].id;
    return vocab_get(t, s, n);
}

/* The special added token of a content, NONE when there is none. */
static uint32_t find_special(const SpingalettTokenizer *t, const char *s) {
    const size_t n = strlen(s);
    for (uint32_t k = 0; k < t->added_count; k++)
        if (t->added[k].special && t->added[k].len == n && memcmp(t->pool + t->added[k].off, s, n) == 0) return t->added[k].id;
    return NONE;
}

/* BOS, EOS and padding by the names special tokens usually have, where nothing else named them. */
static void conventional_tokens(SpingalettTokenizer *t) {
    static const char *const bos[] = {"<s>", "<|begin_of_text|>", "<|startoftext|>", "<bos>"};
    static const char *const eos[] = {"</s>", "<|endoftext|>", "<|end_of_text|>", "<eos>"};
    static const char *const pad[] = {"<pad>", "<|pad|>", "[PAD]"};
    for (size_t k = 0; t->bos == NONE && k < sizeof bos / sizeof *bos; k++) t->bos = find_special(t, bos[k]);
    for (size_t k = 0; t->eos == NONE && k < sizeof eos / sizeof *eos; k++) t->eos = find_special(t, eos[k]);
    for (size_t k = 0; t->pad == NONE && k < sizeof pad / sizeof *pad; k++) t->pad = find_special(t, pad[k]);
}

/* ------------------------------------------------------------------------- tokenizer.json */

typedef struct {
    SpgJson *j;
    SpingalettTokenizer *t;
    char error[200];
} Loader;

static bool lfail(Loader *l, const char *fmt, const char *arg) { return tfail(l->error, sizeof l->error, fmt, arg); }

static char *member_string(Loader *l, uint32_t obj, const char *key, uint32_t *len) {
    uint32_t v = spingalett_json_member(l->j, obj, key);
    if (!v || l->j->values[v].type != SPG_JSON_STRING) return NULL;
    size_t n = 0;
    char *s = spingalett_json_strdup(l->j, v, &n);
    if (len) *len = (uint32_t)n;
    return s;
}

static bool member_bool(Loader *l, uint32_t obj, const char *key, bool fallback) {
    bool b;
    return spingalett_json_bool(l->j, spingalett_json_member(l->j, obj, key), &b) ? b : fallback;
}

/* A "pattern": {"String": "..."} into st->a, or {"Regex": "..."} recognized. */
static bool read_pattern(Loader *l, uint32_t obj, Step *st, bool regex_allowed) {
    uint32_t p = spingalett_json_member(l->j, obj, "pattern");
    uint32_t s = spingalett_json_member(l->j, p, "String"), r = spingalett_json_member(l->j, p, "Regex");
    if (s) {
        size_t n = 0;
        st->a = spingalett_json_strdup(l->j, s, &n);
        st->alen = (uint32_t)n;
        st->pattern = PATTERN_STRING;
        return st->a ? true : lfail(l, "tokenizer: a pattern cannot be read", NULL);
    }
    if (r && regex_allowed) {
        char *re = spingalett_json_strdup(l->j, r, NULL);
        if (!re) return lfail(l, "out of memory", NULL);
        bool known = recognize_pattern(re, st);
        if (!known) {
            char shown[80];
            snprintf(shown, sizeof shown, "%.70s", re);
            lfail(l, "tokenizer: no matcher of the pattern %s (GPT-2's, cl100k's and o200k's are known)", shown);
        }
        free(re);
        return known;
    }
    return lfail(l, "tokenizer: a %s pattern is not supported", r ? "regular expression" : "missing");
}

static bool read_normalizer(Loader *l, uint32_t v) {
    if (!v || l->j->values[v].type == SPG_JSON_NULL) return true;
    uint32_t type = spingalett_json_member(l->j, v, "type");
    SpingalettTokenizer *t = l->t;
    if (spingalett_json_is(l->j, type, "Sequence")) {
        uint32_t list = spingalett_json_member(l->j, v, "normalizers");
        for (uint32_t e = spingalett_json_first(l->j, list); e && e < l->j->values[list].next; e = l->j->values[e].next)
            if (!read_normalizer(l, e)) return false;
        return true;
    }
    Step *st = NULL;
    if (spingalett_json_is(l->j, type, "Prepend")) {
        if (!(st = add_step(&t->norm, &t->norm_count, STEP_PREPEND))) return lfail(l, "out of memory", NULL);
        if (!(st->a = member_string(l, v, "prepend", &st->alen))) return lfail(l, "tokenizer: Prepend has no string", NULL);
    } else if (spingalett_json_is(l->j, type, "Replace")) {
        if (!(st = add_step(&t->norm, &t->norm_count, STEP_REPLACE))) return lfail(l, "out of memory", NULL);
        if (!read_pattern(l, v, st, false)) return false;
        if (!(st->b = member_string(l, v, "content", &st->blen))) return lfail(l, "tokenizer: Replace has no content", NULL);
    } else if (spingalett_json_is(l->j, type, "NFC")) {
        if (!add_step(&t->norm, &t->norm_count, STEP_NFC)) return lfail(l, "out of memory", NULL);
    } else if (spingalett_json_is(l->j, type, "Strip")) {
        if (!(st = add_step(&t->norm, &t->norm_count, STEP_STRIP))) return lfail(l, "out of memory", NULL);
        st->left = member_bool(l, v, "strip_left", true);
        st->right = member_bool(l, v, "strip_right", true);
    } else {
        char name[64];
        if (!type || l->j->values[type].count >= sizeof name || !spingalett_json_copy(l->j, type, name)) strcpy(name, "?");
        return lfail(l, "tokenizer: the normalizer %s is not supported", name);
    }
    return true;
}

static int prepend_scheme(Loader *l, uint32_t v) {
    uint32_t s = spingalett_json_member(l->j, v, "prepend_scheme");
    if (spingalett_json_is(l->j, s, "first")) return PREPEND_FIRST;
    if (spingalett_json_is(l->j, s, "never")) return PREPEND_NEVER;
    if (s) return PREPEND_ALWAYS;
    return member_bool(l, v, "add_prefix_space", true) ? PREPEND_ALWAYS : PREPEND_NEVER;
}

static bool read_pre_tokenizer(Loader *l, uint32_t v) {
    if (!v || l->j->values[v].type == SPG_JSON_NULL) return true;
    uint32_t type = spingalett_json_member(l->j, v, "type");
    SpingalettTokenizer *t = l->t;
    if (spingalett_json_is(l->j, type, "Sequence")) {
        uint32_t list = spingalett_json_member(l->j, v, "pretokenizers");
        for (uint32_t e = spingalett_json_first(l->j, list); e && e < l->j->values[list].next; e = l->j->values[e].next)
            if (!read_pre_tokenizer(l, e)) return false;
        return true;
    }
    Step *st;
    if (spingalett_json_is(l->j, type, "ByteLevel")) {
        if (!(st = add_step(&t->pre, &t->pre_count, STEP_BYTE_LEVEL))) return lfail(l, "out of memory", NULL);
        st->prefix_space = member_bool(l, v, "add_prefix_space", true);
        st->use_regex = member_bool(l, v, "use_regex", true);
    } else if (spingalett_json_is(l->j, type, "Split")) {
        if (!(st = add_step(&t->pre, &t->pre_count, STEP_SPLIT))) return lfail(l, "out of memory", NULL);
        if (!read_pattern(l, v, st, true)) return false;
        uint32_t b = spingalett_json_member(l->j, v, "behavior");
        static const char *const names[] = {"Removed", "Isolated", "MergedWithPrevious", "MergedWithNext", "Contiguous"};
        st->behavior = -1;
        for (int k = 0; k < 5; k++) if (spingalett_json_is(l->j, b, names[k])) st->behavior = k;
        if (st->behavior < 0) return lfail(l, "tokenizer: a Split has no behavior it knows", NULL);
        st->invert = member_bool(l, v, "invert", false);
    } else if (spingalett_json_is(l->j, type, "Metaspace")) {
        if (!(st = add_step(&t->pre, &t->pre_count, STEP_METASPACE))) return lfail(l, "out of memory", NULL);
        if (!(st->a = member_string(l, v, "replacement", &st->alen))) return lfail(l, "tokenizer: Metaspace has no replacement", NULL);
        st->prepend = prepend_scheme(l, v);
        st->split = member_bool(l, v, "split", true);
    } else if (spingalett_json_is(l->j, type, "Digits")) {
        if (!(st = add_step(&t->pre, &t->pre_count, STEP_DIGITS))) return lfail(l, "out of memory", NULL);
        st->individual = member_bool(l, v, "individual_digits", false);
    } else {
        char name[64];
        if (!type || l->j->values[type].count >= sizeof name || !spingalett_json_copy(l->j, type, name)) strcpy(name, "?");
        return lfail(l, "tokenizer: the pre-tokenizer %s is not supported", name);
    }
    return true;
}

static bool read_decoder(Loader *l, uint32_t v) {
    if (!v || l->j->values[v].type == SPG_JSON_NULL) return true;
    uint32_t type = spingalett_json_member(l->j, v, "type");
    SpingalettTokenizer *t = l->t;
    t->has_decoder = true;
    if (spingalett_json_is(l->j, type, "Sequence")) {
        uint32_t list = spingalett_json_member(l->j, v, "decoders");
        for (uint32_t e = spingalett_json_first(l->j, list); e && e < l->j->values[list].next; e = l->j->values[e].next)
            if (!read_decoder(l, e)) return false;
        return true;
    }
    Step *st;
    if (spingalett_json_is(l->j, type, "ByteLevel")) {
        if (!add_step(&t->dec, &t->dec_count, STEP_BYTE_LEVEL)) return lfail(l, "out of memory", NULL);
    } else if (spingalett_json_is(l->j, type, "Replace")) {
        if (!(st = add_step(&t->dec, &t->dec_count, STEP_REPLACE))) return lfail(l, "out of memory", NULL);
        if (!read_pattern(l, v, st, false)) return false;
        if (!(st->b = member_string(l, v, "content", &st->blen))) return lfail(l, "tokenizer: Replace has no content", NULL);
    } else if (spingalett_json_is(l->j, type, "ByteFallback")) {
        if (!add_step(&t->dec, &t->dec_count, STEP_BYTE_FALLBACK)) return lfail(l, "out of memory", NULL);
    } else if (spingalett_json_is(l->j, type, "Fuse")) {
        if (!add_step(&t->dec, &t->dec_count, STEP_FUSE)) return lfail(l, "out of memory", NULL);
    } else if (spingalett_json_is(l->j, type, "Strip")) {
        if (!(st = add_step(&t->dec, &t->dec_count, STEP_STRIP))) return lfail(l, "out of memory", NULL);
        if (!(st->a = member_string(l, v, "content", &st->alen)) || !st->alen) return lfail(l, "tokenizer: Strip has no content", NULL);
        int64_t x = 0;
        if (spingalett_json_integer(l->j, spingalett_json_member(l->j, v, "start"), 0, 1 << 20, &x)) st->start = (uint32_t)x;
        if (spingalett_json_integer(l->j, spingalett_json_member(l->j, v, "stop"), 0, 1 << 20, &x)) st->stop = (uint32_t)x;
    } else if (spingalett_json_is(l->j, type, "Metaspace")) {
        if (!(st = add_step(&t->dec, &t->dec_count, STEP_METASPACE))) return lfail(l, "out of memory", NULL);
        if (!(st->a = member_string(l, v, "replacement", &st->alen))) return lfail(l, "tokenizer: Metaspace has no replacement", NULL);
        st->prepend = prepend_scheme(l, v);
    } else {
        char name[64];
        if (!type || l->j->values[type].count >= sizeof name || !spingalett_json_copy(l->j, type, name)) strcpy(name, "?");
        return lfail(l, "tokenizer: the decoder %s is not supported", name);
    }
    return true;
}

/* The id of a token named in the post-processor or a config, NONE when the tokenizer lacks it. */
static uint32_t named_token(Loader *l, uint32_t v) {
    if (!v) return NONE;
    if (l->j->values[v].type == SPG_JSON_OBJECT) v = spingalett_json_member(l->j, v, "content");
    if (!v || l->j->values[v].type != SPG_JSON_STRING) return NONE;
    char *s = spingalett_json_strdup(l->j, v, NULL);
    if (!s) return NONE;
    uint32_t id = find_token(l->t, s, strlen(s));
    free(s);
    return id;
}

/* The template of a single text: TemplateProcessing's special tokens before and after $A (a Sequence's
   too), or BertProcessing's and RobertaProcessing's cls and sep. */
static bool read_post_processor(Loader *l, uint32_t v) {
    if (!v || l->j->values[v].type == SPG_JSON_NULL) return true;
    SpingalettTokenizer *t = l->t;
    uint32_t type = spingalett_json_member(l->j, v, "type");
    if (spingalett_json_is(l->j, type, "Sequence")) {
        uint32_t list = spingalett_json_member(l->j, v, "processors");
        for (uint32_t e = spingalett_json_first(l->j, list); e && e < l->j->values[list].next; e = l->j->values[e].next)
            if (!read_post_processor(l, e)) return false;
        return true;
    }
    if (spingalett_json_is(l->j, type, "BertProcessing") || spingalett_json_is(l->j, type, "RobertaProcessing")) {
        int64_t cls, sep;
        if (!spingalett_json_integer(l->j, spingalett_json_element(l->j, spingalett_json_member(l->j, v, "cls"), 1), 0, UINT32_MAX - 1, &cls) ||
            !spingalett_json_integer(l->j, spingalett_json_element(l->j, spingalett_json_member(l->j, v, "sep"), 1), 0, UINT32_MAX - 1, &sep))
            return lfail(l, "tokenizer: the post-processor's cls or sep cannot be read", NULL);
        t->before[0] = (uint32_t)cls; t->before_count = 1;
        t->after[0] = (uint32_t)sep; t->after_count = 1;
        return true;
    }
    if (!spingalett_json_is(l->j, type, "TemplateProcessing")) return true;     /* ByteLevel: offsets only */
    uint32_t single = spingalett_json_member(l->j, v, "single"), specials = spingalett_json_member(l->j, v, "special_tokens");
    bool seen_sequence = false;
    t->before_count = t->after_count = 0;
    for (uint32_t e = spingalett_json_first(l->j, single); e && e < l->j->values[single].next; e = l->j->values[e].next) {
        uint32_t sp = spingalett_json_member(l->j, e, "SpecialToken");
        if (!sp) { seen_sequence = true; continue; }
        uint32_t id_name = spingalett_json_member(l->j, sp, "id");
        char *name = spingalett_json_strdup(l->j, id_name, NULL);
        if (!name) return lfail(l, "tokenizer: the template cannot be read", NULL);
        uint32_t entry = spingalett_json_member(l->j, specials, name);
        free(name);
        uint32_t ids = spingalett_json_member(l->j, entry, "ids");
        for (uint32_t i = spingalett_json_first(l->j, ids); i && i < l->j->values[ids].next; i = l->j->values[i].next) {
            int64_t id;
            if (!spingalett_json_integer(l->j, i, 0, UINT32_MAX - 1, &id)) return lfail(l, "tokenizer: the template's ids cannot be read", NULL);
            uint32_t *list = seen_sequence ? t->after : t->before, *count = seen_sequence ? &t->after_count : &t->before_count;
            if (*count == 4) return lfail(l, "tokenizer: the template adds more than four tokens on a side", NULL);
            list[(*count)++] = (uint32_t)id;
        }
    }
    return true;
}

static bool read_added(Loader *l, uint32_t list, uint32_t *max_id) {
    SpingalettTokenizer *t = l->t;
    uint32_t count = list && l->j->values[list].type == SPG_JSON_ARRAY ? l->j->values[list].count : 0;
    t->added = (Added *)calloc(count + 1u, sizeof(Added));
    if (!t->added) return lfail(l, "out of memory", NULL);
    for (uint32_t e = spingalett_json_first(l->j, list); e && e < l->j->values[list].next; e = l->j->values[e].next) {
        int64_t id;
        uint32_t c = spingalett_json_member(l->j, e, "content");
        if (!spingalett_json_integer(l->j, spingalett_json_member(l->j, e, "id"), 0, MAX_IDS - 1, &id) ||
            !c || l->j->values[c].type != SPG_JSON_STRING)
            return lfail(l, "tokenizer: an added token has no id or content", NULL);
        Added *a = &t->added[t->added_count++];
        char *s = spingalett_json_strdup(l->j, c, NULL);
        if (!s) return lfail(l, "out of memory", NULL);
        a->off = pool_add(t, s, l->j->values[c].count);
        a->len = l->j->values[c].count;
        free(s);
        if (a->off == UINT32_MAX) return lfail(l, "out of memory", NULL);
        a->id = (uint32_t)id;
        a->special = member_bool(l, e, "special", false);
        a->normalized = member_bool(l, e, "normalized", !a->special);
        a->lstrip = member_bool(l, e, "lstrip", false);
        a->rstrip = member_bool(l, e, "rstrip", false);
        a->single_word = member_bool(l, e, "single_word", false);
        if ((uint32_t)id + 1 > *max_id) *max_id = (uint32_t)id + 1;
    }
    return true;
}

/* The BPE model: its vocabulary (ids up to size, which the added tokens may raise), options and merges. */
static bool read_model(Loader *l, uint32_t model, uint32_t added) {
    SpgJson *j = l->j;
    SpingalettTokenizer *t = l->t;
    uint32_t type = spingalett_json_member(j, model, "type");
    uint32_t vocab = spingalett_json_member(j, model, "vocab"), merges = spingalett_json_member(j, model, "merges");
    if (type && !spingalett_json_is(j, type, "BPE")) {
        char name[64];
        if (j->values[type].count >= sizeof name || !spingalett_json_copy(j, type, name)) strcpy(name, "?");
        return lfail(l, "tokenizer: the model %s is not supported (BPE is)", name);
    }
    if (!vocab || j->values[vocab].type != SPG_JSON_OBJECT || !merges || j->values[merges].type != SPG_JSON_ARRAY)
        return lfail(l, "tokenizer: the model has no vocabulary or merges", NULL);
    /* the largest id first, for the tables */
    uint32_t size = 0;
    for (uint32_t k = spingalett_json_first(j, vocab); k && k < j->values[vocab].next; k = j->values[k + 1].next) {
        int64_t id;
        if (!spingalett_json_integer(j, k + 1, 0, MAX_IDS - 1, &id))
            return lfail(l, "tokenizer: a token's id cannot be read (ids are below 2^24)", NULL);
        if ((uint32_t)id + 1 > size) size = (uint32_t)id + 1;
    }
    if (!read_added(l, added, &size)) return false;
    if (!size || !alloc_ids(t, size)) return lfail(l, size ? "out of memory" : "tokenizer: the vocabulary is empty", NULL);
    char small[512];
    for (uint32_t k = spingalett_json_first(j, vocab); k && k < j->values[vocab].next; k = j->values[k + 1].next) {
        int64_t id;
        spingalett_json_integer(j, k + 1, 0, MAX_IDS - 1, &id);
        const uint32_t n = j->values[k].count;
        char *s = n < sizeof small ? small : (char *)malloc((size_t)n + 1);
        if (!s) return lfail(l, "out of memory", NULL);
        spingalett_json_copy(j, k, s);
        bool ok = set_model_token(t, (uint32_t)id, s, n);
        if (s != small) free(s);
        if (!ok) return lfail(l, "out of memory", NULL);
    }
    if (!index_vocab(t) || !index_added(t)) return lfail(l, "out of memory", NULL);
    t->byte_fallback = member_bool(l, model, "byte_fallback", false);
    t->fuse_unk = member_bool(l, model, "fuse_unk", false);
    t->ignore_merges = member_bool(l, model, "ignore_merges", false);
    t->prefix = member_string(l, model, "continuing_subword_prefix", &t->prefix_len);
    t->suffix = member_string(l, model, "end_of_word_suffix", &t->suffix_len);
    if (t->prefix && !t->prefix_len) { free(t->prefix); t->prefix = NULL; }
    if (t->suffix && !t->suffix_len) { free(t->suffix); t->suffix = NULL; }
    uint32_t unk = spingalett_json_member(j, model, "unk_token");
    if (unk && j->values[unk].type == SPG_JSON_STRING) {
        char *s = spingalett_json_strdup(j, unk, NULL);
        if (!s) return lfail(l, "out of memory", NULL);
        t->unk = vocab_get(t, s, strlen(s));
        free(s);
    }
    /* merges: "a b" strings or ["a", "b"] pairs */
    if (!pairs_alloc(t, j->values[merges].count)) return lfail(l, "out of memory", NULL);
    uint32_t rank = 0;
    for (uint32_t e = spingalett_json_first(j, merges); e && e < j->values[merges].next; e = j->values[e].next, rank++) {
        const SpgJsonValue *m = &j->values[e];
        char *a = NULL, *b = NULL;
        size_t an = 0, bn = 0;
        char *whole = NULL;
        if (m->type == SPG_JSON_STRING) {
            whole = spingalett_json_strdup(j, e, &an);
            char *space = whole ? memchr(whole, ' ', an) : NULL;
            if (!space || memchr(space + 1, ' ', an - (size_t)(space + 1 - whole))) { free(whole); return lfail(l, "tokenizer: a merge is not two tokens", NULL); }
            a = whole;
            b = space + 1;
            bn = an - (size_t)(b - whole);
            an = (size_t)(space - whole);
        } else if (m->type == SPG_JSON_ARRAY && m->count == 2) {
            a = spingalett_json_strdup(j, e + 1, &an);
            b = spingalett_json_strdup(j, j->values[e + 1].next, &bn);
            if (!a || !b) { free(a); free(b); return lfail(l, "tokenizer: a merge is not two tokens", NULL); }
        } else {
            return lfail(l, "tokenizer: a merge is not two tokens", NULL);
        }
        bool ok = merge_add(t, a, an, b, bn, rank, l->error, sizeof l->error);
        if (whole) free(whole);
        else { free(a); free(b); }
        if (!ok) return false;
    }
    t->split_marks = marks_split(t);
    return true;
}

/* bos_token, eos_token, pad_token of a tokenizer_config.json (strings or {"content": ...}). */
static void read_config(SpingalettTokenizer *t, const char *text, size_t size) {
    SpgJson j;
    if (!spingalett_json_parse(&j, text, size)) return;
    Loader l = {&j, t, ""};
    uint32_t id;
    if ((id = named_token(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "bos_token"))) != NONE) t->bos = id;
    if ((id = named_token(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "eos_token"))) != NONE) t->eos = id;
    if ((id = named_token(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "pad_token"))) != NONE) t->pad = id;
    spingalett_json_free(&j);
}

static SpingalettTokenizer *load_json(const char *text, size_t size, const char *config, size_t config_size) {
    SpgJson j;
    if (!spingalett_json_parse(&j, text, size)) {
        char msg[220];
        snprintf(msg, sizeof msg, "tokenizer: %s", j.error);
        set_error(SPINGALETT_ERR_INVALID, msg);
        return NULL;
    }
    SpingalettTokenizer *t = tokenizer_new();
    Loader l = {&j, t, ""};
    bool ok = t != NULL;
    if (!ok) lfail(&l, "out of memory", NULL);
    if (ok && j.values[SPG_JSON_ROOT].type != SPG_JSON_OBJECT) ok = lfail(&l, "tokenizer: not a tokenizer.json file", NULL);
    ok = ok && read_model(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "model"),
                          spingalett_json_member(&j, SPG_JSON_ROOT, "added_tokens"));
    ok = ok && read_normalizer(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "normalizer"));
    ok = ok && read_pre_tokenizer(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "pre_tokenizer"));
    ok = ok && read_decoder(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "decoder"));
    ok = ok && read_post_processor(&l, spingalett_json_member(&j, SPG_JSON_ROOT, "post_processor"));
    for (uint32_t k = 0; ok && k < t->before_count + t->after_count; k++) {
        uint32_t id = k < t->before_count ? t->before[k] : t->after[k - t->before_count];
        if (id >= t->size) ok = lfail(&l, "tokenizer: the template names an id beyond the vocabulary", NULL);
    }
    spingalett_json_free(&j);
    if (!ok) {
        set_error(strcmp(l.error, "out of memory") ? SPINGALETT_ERR_INVALID : SPINGALETT_ERR_ALLOC, l.error);
        spingalett_tokenizer_free(t);
        return NULL;
    }
    if (t->before_count) t->bos = t->before[0];
    if (t->after_count) t->eos = t->after[t->after_count - 1];
    if (config) read_config(t, config, config_size);
    conventional_tokens(t);
    return t;
}

SpingalettTokenizer *spingalett_tokenizer_from_memory(const void *data, size_t size) {
    if (!data) {
        set_error(SPINGALETT_ERR_INVALID, "tokenizer: data is NULL");
        return NULL;
    }
    if (size >= 4 && !memcmp(data, "GGUF", 4)) return spingalett_gguf_tokenizer((const uint8_t *)data, size);
    return load_json((const char *)data, size, NULL, 0);
}

SpingalettTokenizer *spingalett_tokenizer_load(const char *path) {
    if (!path) {
        set_error(SPINGALETT_ERR_INVALID, "tokenizer: path is NULL");
        return NULL;
    }
    /* a directory: its tokenizer.json and tokenizer_config.json */
    size_t n = strlen(path);
    char *joined = (char *)malloc(n + 32);
    if (!joined) { set_error(SPINGALETT_ERR_ALLOC, "tokenizer: out of memory"); return NULL; }
    snprintf(joined, n + 32, "%s/tokenizer.json", path);
    FILE *probe = fopen(joined, "rb");
    SpgFileView file, config = {0};
    SpingalettTokenizer *t = NULL;
    if (probe) {
        fclose(probe);
        if (spingalett_file_open(&file, joined)) {
            snprintf(joined, n + 32, "%s/tokenizer_config.json", path);
            probe = fopen(joined, "rb");
            if (probe) {
                fclose(probe);
                if (!spingalett_file_open(&config, joined)) memset(&config, 0, sizeof config);
            }
            t = load_json((const char *)file.data, file.size, (const char *)config.data, config.size);
            spingalett_file_close(&config);
            spingalett_file_close(&file);
        }
    } else if (spingalett_file_open(&file, path)) {
        t = spingalett_tokenizer_from_memory(file.data, file.size);
        spingalett_file_close(&file);
    }
    free(joined);
    return t;
}

/* ------------------------------------------------------------------------- GGUF vocabularies */

/* Merges of a SentencePiece vocabulary from its scores, as Hugging Face's converter derives them: every
   token that splits into two tokens is the merge of the two, ranked by its score (the highest first;
   equal scores share a rank, so the leftmost pair goes first, as in llama.cpp's tokenizer). */
static bool merges_from_scores(SpingalettTokenizer *t, const float *scores) {
    float *sorted = (float *)malloc((size_t)t->size * sizeof(float));
    if (!sorted) return false;
    uint32_t distinct = 0;
    for (uint32_t id = 0; id < t->size; id++) sorted[id] = scores[id];
    /* ranks: the distinct scores in descending order (a heap sort, no recursion) */
    for (uint32_t n = t->size, i = n / 2; i-- > 0;) {
        for (uint32_t r = i;;) {
            uint32_t c = 2 * r + 1;
            if (c >= n) break;
            if (c + 1 < n && sorted[c + 1] < sorted[c]) c++;
            if (!(sorted[c] < sorted[r])) break;
            float x = sorted[c]; sorted[c] = sorted[r]; sorted[r] = x;
            r = c;
        }
    }
    for (uint32_t n = t->size; n > 1;) {
        float x = sorted[0]; sorted[0] = sorted[--n]; sorted[n] = x;
        for (uint32_t r = 0;;) {
            uint32_t c = 2 * r + 1;
            if (c >= n) break;
            if (c + 1 < n && sorted[c + 1] < sorted[c]) c++;
            if (!(sorted[c] < sorted[r])) break;
            float y = sorted[c]; sorted[c] = sorted[r]; sorted[r] = y;
            r = c;
        }
    }
    for (uint32_t i = 0; i < t->size; i++)
        if (!distinct || sorted[i] != sorted[distinct - 1]) sorted[distinct++] = sorted[i];
    uint64_t pairs = 0;
    for (uint32_t id = 0; id < t->size; id++) pairs += t->model_len[id];
    bool ok = pairs_alloc(t, (uint32_t)(pairs < UINT32_MAX / 4 ? pairs : UINT32_MAX / 4));
    for (uint32_t id = 0; ok && id < t->size; id++) {
        if (t->model_off[id] == UINT32_MAX || t->added_of[id] >= 0) continue;
        const char *s = t->pool + t->model_off[id];
        uint32_t n = t->model_len[id];
        uint32_t lo = 0, hi = distinct;     /* the rank of its score */
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (sorted[mid] > scores[id]) lo = mid + 1;
            else hi = mid;
        }
        for (uint32_t k = 1; k < n; k++) {
            if (((unsigned char)s[k] & 0xC0) == 0x80) continue;
            uint32_t a = vocab_get(t, s, k), b = vocab_get(t, s + k, n - k);
            if (a == NONE || b == NONE) continue;
            const Pair *p = pair_get(t, a, b);
            if (!p || p->rank > lo) merge_put(t, a, b, lo, id);
        }
    }
    free(sorted);
    return ok;
}

SpingalettTokenizer *spingalett_tokenizer_build(const SpgTokenizerSpec *spec) {
    char error[200] = "";
    const bool spm = spec->model && !strcmp(spec->model, "llama");
    if (!spec->model || (!spm && strcmp(spec->model, "gpt2"))) {
        tfail(error, sizeof error, "GGUF: the tokenizer model %s is not supported (gpt2 and llama are)", spec->model ? spec->model : "(none)");
        set_error(SPINGALETT_ERR_INVALID, error);
        return NULL;
    }
    if (!spec->count || spec->count > MAX_IDS || (spm && !spec->scores) || (!spm && !spec->merges)) {
        set_error(SPINGALETT_ERR_INVALID, spm ? "GGUF: the vocabulary has no tokens or scores" : "GGUF: the vocabulary has no tokens or merges");
        return NULL;
    }
    SpingalettTokenizer *t = tokenizer_new();
    bool ok = t && alloc_ids(t, spec->count);
    uint32_t added = 0;
    for (uint32_t id = 0; ok && id < spec->count; id++) {
        const SpgString *s = &spec->tokens[id];
        ok = set_model_token(t, id, s->s, s->n);
        int type = spec->types ? spec->types[id] : 1;
        if (type == 2 || type == 3 || type == 4) added++;
    }
    ok = ok && index_vocab(t);
    if (ok) {
        t->added = (Added *)calloc(added + 1u, sizeof(Added));
        ok = t->added != NULL;
    }
    /* unknown and control tokens are special added tokens, user-defined ones added tokens found in text */
    for (uint32_t id = 0; ok && id < spec->count && spec->types; id++) {
        int type = spec->types[id];
        if (type != 2 && type != 3 && type != 4) continue;
        Added *a = &t->added[t->added_count++];
        a->id = id;
        a->off = t->model_off[id];
        a->len = t->model_len[id];
        a->special = type != 4;
    }
    ok = ok && index_added(t);
    t->unk = spec->unk < spec->count ? spec->unk : NONE;
    t->bos = spec->bos < spec->count ? spec->bos : NONE;
    t->eos = spec->eos < spec->count ? spec->eos : NONE;
    t->pad = spec->pad < spec->count ? spec->pad : NONE;
    Step *st;
    if (ok && spm) {
        /* SentencePiece: a space before the text, spaces as U+2581, byte fallback; decoded back */
        t->byte_fallback = true;
        if (spec->add_space_prefix) {
            ok = (st = add_step(&t->norm, &t->norm_count, STEP_PREPEND)) && (st->a = dup_bytes("\xE2\x96\x81", 3));
            if (ok) st->alen = 3;
        }
        ok = ok && (st = add_step(&t->norm, &t->norm_count, STEP_REPLACE)) && (st->a = dup_bytes(" ", 1)) && (st->b = dup_bytes("\xE2\x96\x81", 3));
        if (ok) { st->alen = 1; st->blen = 3; }
        ok = ok && (st = add_step(&t->dec, &t->dec_count, STEP_REPLACE)) && (st->a = dup_bytes("\xE2\x96\x81", 3)) && (st->b = dup_bytes(" ", 1));
        if (ok) { st->alen = 3; st->blen = 1; }
        ok = ok && add_step(&t->dec, &t->dec_count, STEP_BYTE_FALLBACK) && add_step(&t->dec, &t->dec_count, STEP_FUSE);
        if (ok && spec->add_space_prefix) {
            ok = (st = add_step(&t->dec, &t->dec_count, STEP_STRIP)) && (st->a = dup_bytes(" ", 1));
            if (ok) { st->alen = 1; st->start = 1; }
        }
        ok = ok && merges_from_scores(t, spec->scores);
    } else if (ok) {
        /* byte-level BPE: the pre-tokenizer of the tokenizer.json llama.cpp's converter recognized (its name) */
        const char *pre = spec->pre ? spec->pre : "default";
        int digits = 0;
        bool digits_apart = false;          /* SmolLM's and StarCoder's: each digit a piece, then GPT-2's */
        if (!strcmp(pre, "llama3") || !strcmp(pre, "llama-bpe") || !strcmp(pre, "llama-v3") || !strcmp(pre, "smaug-bpe") ||
            !strcmp(pre, "dbrx") || !strcmp(pre, "falcon3") || !strcmp(pre, "pixtral")) {
            digits = 3;
            /* LLaMA 3's tokenizer.json looks whole pieces up first (ignore_merges), as llama.cpp does for these */
            t->ignore_merges = strcmp(pre, "smaug-bpe") && strcmp(pre, "dbrx");
        } else if (!strcmp(pre, "qwen2") || !strcmp(pre, "deepseek-r1-qwen") || !strcmp(pre, "megrez"))
            digits = 1;
        else if (!strcmp(pre, "smollm") || !strcmp(pre, "starcoder"))
            digits_apart = true;
        else if (strcmp(pre, "default") && strcmp(pre, "gpt-2") && strcmp(pre, "gpt2"))
            spingalett_log(LOG_WARNING, "GGUF: the pre-tokenizer '%s' is unknown; GPT-2's is used", pre);
        if (digits == 1) ok = add_step(&t->norm, &t->norm_count, STEP_NFC) != NULL;
        if (ok && digits) {
            ok = (st = add_step(&t->pre, &t->pre_count, STEP_SPLIT)) != NULL;
            if (ok) { st->pattern = PATTERN_CL100K; st->digits = digits; st->behavior = SPLIT_ISOLATED; }
        }
        if (ok && digits_apart) {
            ok = (st = add_step(&t->pre, &t->pre_count, STEP_DIGITS)) != NULL;
            if (ok) st->individual = true;
        }
        ok = ok && (st = add_step(&t->pre, &t->pre_count, STEP_BYTE_LEVEL));
        if (ok) st->use_regex = digits == 0;
        ok = ok && add_step(&t->dec, &t->dec_count, STEP_BYTE_LEVEL);
        ok = ok && pairs_alloc(t, spec->merge_count);
        for (uint32_t k = 0; ok && k < spec->merge_count; k++) {
            const SpgString *m = &spec->merges[k];
            const char *space = m->n ? (const char *)memchr(m->s, ' ', m->n) : NULL;
            if (!space) { tfail(error, sizeof error, "GGUF: a merge is not two tokens", NULL); ok = false; break; }
            size_t an = (size_t)(space - m->s);
            ok = merge_add(t, m->s, an, space + 1, m->n - an - 1, k, error, sizeof error);
        }
    }
    if (ok) {
        t->has_decoder = true;
        t->split_marks = marks_split(t);
        if (spec->add_bos && t->bos != NONE) t->before[t->before_count++] = t->bos;
        if (spec->add_eos && t->eos != NONE) t->after[t->after_count++] = t->eos;
        return t;
    }
    set_error(error[0] && strcmp(error, "out of memory") ? SPINGALETT_ERR_INVALID : SPINGALETT_ERR_ALLOC,
              error[0] ? error : "GGUF: out of memory");
    spingalett_tokenizer_free(t);
    return NULL;
}
