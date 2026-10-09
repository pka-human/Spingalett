/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * DigitPad: draw a digit with the mouse and watch a Spingalett network classify it as you draw.
 *
 *   DigitPad [--verbose] [model.slett]          interactive window
 *   DigitPad --classify image.pgm [model.slett] classify a binary PGM drawing and exit
 *
 * The model is looked up next to the executable (../share/digitpad/mnist.slett, then mnist.slett)
 * unless given on the command line or in $DIGITPAD_MODEL. The whole UI is rendered in software
 * into one framebuffer that SDL2 only presents, with glyphs from an embedded font atlas, so at
 * run time the program needs nothing beyond SDL2 and libspingalett.
 *
 * On Windows it is a GUI program: it opens no console of its own, but prints to the console it
 * was started from (--help, --classify, --verbose).
 */

#include "Digits.h"
#include "FontAtlas.h"
#include <SDL.h>
#include <Spingalett/Spingalett.Short.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define WIN_W 880
#define WIN_H 584
#define CANVAS 400
#define CANVAS_X 24
#define CANVAS_Y 84
#define PANEL_X 448
#define PANEL_Y CANVAS_Y
#define PANEL_W 408
#define PREVIEW_SCALE 4
#define BRUSH_RADIUS 17.0f
#define ERASER_RADIUS 26.0f
#define UNDO_DEPTH 32

/* palette */
#define COL_BG        0xFF1C1D22u
#define COL_PANEL     0xFF26282Fu
#define COL_BORDER    0xFF3A3D47u
#define COL_TEXT      0xFFE9EAEEu
#define COL_MUTED     0xFF9095A1u
#define COL_FAINT     0xFF5C606Cu
#define COL_ACCENT    0xFF4C9BFAu
#define COL_BAR       0xFF5F6472u
#define COL_TRACK     0xFF33363Fu
#define COL_BUTTON    0xFF33363Fu
#define COL_BUTTON_HI 0xFF41454Fu

/* ---------------------------------------------------------------- framebuffer drawing */

static uint32_t fb[WIN_W * WIN_H];

static inline uint32_t mix(uint32_t dst, uint32_t src, unsigned a) {
    unsigned r = ((src >> 16 & 255) * a + (dst >> 16 & 255) * (255 - a) + 127) / 255;
    unsigned g = ((src >> 8 & 255) * a + (dst >> 8 & 255) * (255 - a) + 127) / 255;
    unsigned b = ((src & 255) * a + (dst & 255) * (255 - a) + 127) / 255;
    return 0xFF000000u | r << 16 | g << 8 | b;
}

static inline void blend(int x, int y, uint32_t color, unsigned a) {
    if (a == 0 || x < 0 || y < 0 || x >= WIN_W || y >= WIN_H) return;
    uint32_t *p = &fb[(size_t)y * WIN_W + x];
    *p = a >= 255 ? color : mix(*p, color, a);
}

static void fill_rect(int x, int y, int w, int h, uint32_t color) {
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
    int x1 = x + w > WIN_W ? WIN_W : x + w, y1 = y + h > WIN_H ? WIN_H : y + h;
    for (int py = y0; py < y1; py++)
        for (int px = x0; px < x1; px++) fb[(size_t)py * WIN_W + px] = color;
}

/* Rectangle with anti-aliased corners of radius r. */
static void fill_round_rect(int x, int y, int w, int h, float r, uint32_t color) {
    if (w <= 0 || h <= 0) return;
    if (r > w / 2.0f) r = w / 2.0f;
    if (r > h / 2.0f) r = h / 2.0f;
    for (int py = y; py < y + h; py++)
        for (int px = x; px < x + w; px++) {
            float cx = px + 0.5f, cy = py + 0.5f;
            float qx = cx < x + r ? x + r - cx : cx > x + w - r ? cx - (x + w - r) : 0.0f;
            float qy = cy < y + r ? y + r - cy : cy > y + h - r ? cy - (y + h - r) : 0.0f;
            float cover = r - sqrtf(qx * qx + qy * qy) + 0.5f;
            if (qx == 0.0f || qy == 0.0f) cover = 1.0f;
            if (cover <= 0.0f) continue;
            blend(px, py, color, cover >= 1.0f ? 255u : (unsigned)(cover * 255.0f));
        }
}

static void stroke_rect(int x, int y, int w, int h, uint32_t color) {
    fill_rect(x, y, w, 1, color);
    fill_rect(x, y + h - 1, w, 1, color);
    fill_rect(x, y, 1, h, color);
    fill_rect(x + w - 1, y, 1, h, color);
}

/* ---------------------------------------------------------------- text */

typedef struct {
    const AtlasFont *atlas;
    uint8_t *alpha;                 /* decoded strip, width x height */
} Font;

static Font ui, title, big;

static bool font_decode(Font *font, const AtlasFont *atlas) {
    size_t n = (size_t)atlas->width * atlas->height, i = 0;
    font->atlas = atlas;
    font->alpha = calloc(n, 1);
    if (!font->alpha) return false;
    for (const char *const *piece = atlas->rle; *piece; piece++)
        for (const char *p = *piece; *p && i < n; p++) {
            if (*p >= 'A' && *p <= 'Z') i += (size_t)(*p - 'A' + 1);
            else font->alpha[i++] = (uint8_t)((*p - 'a' + 1) * 17);
        }
    return true;
}

static int glyph_index(const AtlasFont *f, unsigned char c) {
    if (f->chars[0]) {
        const char *p = c ? strchr(f->chars, c) : NULL;
        return p ? (int)(p - f->chars) : -1;
    }
    return c >= f->first && c < f->first + f->count ? c - f->first : -1;
}

static int text_width(const Font *font, const char *s) {
    int w = 0;
    for (; *s; s++) {
        int g = glyph_index(font->atlas, (unsigned char)*s);
        if (g >= 0) w += font->atlas->glyphs[g].advance;
    }
    return w;
}

/* Draws s with its top-left corner (of the line box) at x, y. */
static void draw_text(const Font *font, int x, int y, const char *s, uint32_t color) {
    const AtlasFont *f = font->atlas;
    for (; *s; s++) {
        int g = glyph_index(f, (unsigned char)*s);
        if (g < 0) continue;
        const AtlasGlyph *gl = &f->glyphs[g];
        for (int row = 0; row < f->height; row++) {
            const uint8_t *src = font->alpha + (size_t)row * f->width + gl->x;
            for (int col = 0; col < gl->width; col++)
                blend(x - f->pad + col, y + row, color, src[col]);
        }
        x += gl->advance;
    }
}

static void draw_text_right(const Font *font, int right, int y, const char *s, uint32_t color) {
    draw_text(font, right - text_width(font, s), y, s, color);
}

static void draw_text_center(const Font *font, int center, int y, const char *s, uint32_t color) {
    draw_text(font, center - text_width(font, s) / 2, y, s, color);
}

/* ---------------------------------------------------------------- canvas and model */

static float canvas[CANVAS * CANVAS];
static float *undo_stack;          /* ring buffer of UNDO_DEPTH canvas snapshots */
static int undo_head, undo_count;

static double *sat;                 /* scratch for digit_normalize */
static float input[DIGIT_PIXELS];   /* what the network sees */
static float probs[10];
static bool has_ink;

static void undo_push(void) {
    memcpy(undo_stack + (size_t)undo_head * CANVAS * CANVAS, canvas, sizeof canvas);
    undo_head = (undo_head + 1) % UNDO_DEPTH;
    if (undo_count < UNDO_DEPTH) undo_count++;
}

static bool undo_pop(void) {
    if (undo_count == 0) return false;
    undo_head = (undo_head + UNDO_DEPTH - 1) % UNDO_DEPTH;
    undo_count--;
    memcpy(canvas, undo_stack + (size_t)undo_head * CANVAS * CANVAS, sizeof canvas);
    return true;
}

/* Paints (or erases) a round-capped segment with an anti-aliased edge. */
static void paint_segment(float x0, float y0, float x1, float y1, float radius, bool erase) {
    float dx = x1 - x0, dy = y1 - y0, len2 = dx * dx + dy * dy;
    int bx0 = (int)floorf(fminf(x0, x1) - radius - 2), bx1 = (int)ceilf(fmaxf(x0, x1) + radius + 2);
    int by0 = (int)floorf(fminf(y0, y1) - radius - 2), by1 = (int)ceilf(fmaxf(y0, y1) + radius + 2);
    if (bx0 < 0) bx0 = 0;
    if (by0 < 0) by0 = 0;
    if (bx1 > CANVAS) bx1 = CANVAS;
    if (by1 > CANVAS) by1 = CANVAS;
    for (int y = by0; y < by1; y++)
        for (int x = bx0; x < bx1; x++) {
            float px = x + 0.5f - x0, py = y + 0.5f - y0;
            float t = len2 > 0 ? (px * dx + py * dy) / len2 : 0.0f;
            t = t < 0 ? 0 : t > 1 ? 1 : t;
            float ex = px - t * dx, ey = py - t * dy;
            float v = (radius - sqrtf(ex * ex + ey * ey)) / 1.5f + 0.5f;  /* 1.5 px soft edge */
            if (v <= 0) continue;
            if (v > 1) v = 1;
            float *c = &canvas[(size_t)y * CANVAS + x];
            if (erase) { if (1 - v < *c) *c = 1 - v; }
            else if (v > *c) *c = v;
        }
}

static int best_class(int *runner_up) {
    int best = 0, second = 1;
    for (int k = 1; k < 10; k++) {
        if (probs[k] > probs[best]) { second = best; best = k; }
        else if (k != best && probs[k] > probs[second]) second = k;
    }
    if (runner_up) *runner_up = second;
    return best;
}

static bool classify(NeuralNetwork *net, const float *img, int w, int h) {
    has_ink = digit_normalize(img, w, h, sat, input);
    if (!has_ink) return true;
    float *out = forward(.net = net, .input = input);
    if (!out) return false;
    memcpy(probs, out, sizeof probs);
    return true;
}

/* ---------------------------------------------------------------- UI */

typedef struct { int x, y, w, h; const char *label; } Button;

static Button btn_clear, btn_undo;
static int mouse_x = -1, mouse_y = -1;
static char model_info[256];

static bool inside(const Button *b, int x, int y) {
    return x >= b->x && x < b->x + b->w && y >= b->y && y < b->y + b->h;
}

static void draw_button(const Button *b, bool enabled) {
    fill_round_rect(b->x, b->y, b->w, b->h, 6.0f, enabled && inside(b, mouse_x, mouse_y) ? COL_BUTTON_HI : COL_BUTTON);
    draw_text_center(&ui, b->x + b->w / 2, b->y + (b->h - ui.atlas->height) / 2, b->label, enabled ? COL_TEXT : COL_FAINT);
}

static void render(void) {
    fill_rect(0, 0, WIN_W, WIN_H, COL_BG);

    /* header */
    draw_text(&title, 24, 18, "DigitPad", COL_TEXT);
    char subtitle[128];
    snprintf(subtitle, sizeof subtitle, "handwritten digit recognition with Spingalett %s", spingalett_version());
    draw_text(&ui, 24 + text_width(&title, "DigitPad") + 14, 18 + title.atlas->ascent - ui.atlas->ascent, subtitle, COL_MUTED);
    fill_rect(24, 64, WIN_W - 48, 1, COL_BORDER);

    /* canvas: white ink on black, like MNIST */
    for (int y = 0; y < CANVAS; y++)
        for (int x = 0; x < CANVAS; x++) {
            unsigned v = (unsigned)(canvas[(size_t)y * CANVAS + x] * 255.0f + 0.5f);
            fb[(size_t)(CANVAS_Y + y) * WIN_W + CANVAS_X + x] = 0xFF000000u | v << 16 | v << 8 | v;
        }
    stroke_rect(CANVAS_X - 1, CANVAS_Y - 1, CANVAS + 2, CANVAS + 2, COL_BORDER);
    if (!has_ink)
        draw_text_center(&ui, CANVAS_X + CANVAS / 2, CANVAS_Y + CANVAS / 2 - 9, "draw a digit here", COL_FAINT);

    /* buttons and hint under the canvas */
    draw_button(&btn_clear, has_ink);
    draw_button(&btn_undo, undo_count > 0);
    draw_text_right(&ui, PANEL_X + PANEL_W, btn_clear.y + (btn_clear.h - ui.atlas->height) / 2,
                    "left button draws, right button erases", COL_MUTED);

    /* result panel */
    const int px = PANEL_X, py = PANEL_Y;
    fill_round_rect(px, py, PANEL_W, CANVAS, 10.0f, COL_PANEL);
    draw_text(&ui, px + 20, py + 16, "PREDICTION", COL_MUTED);

    int second = 0, best = has_ink ? best_class(&second) : -1;
    char line[64];
    if (best >= 0) {
        char digit[2] = {(char)('0' + best), 0};
        draw_text(&big, px + 20, py + 38, digit, COL_TEXT);
        snprintf(line, sizeof line, "%.1f%%", 100.0 * (double)probs[best]);
        draw_text(&title, px + 104, py + 60, line, COL_TEXT);
        draw_text(&ui, px + 104, py + 90, "confidence", COL_MUTED);
        snprintf(line, sizeof line, "next: %d (%.1f%%)", second, 100.0 * (double)probs[second]);
        draw_text(&ui, px + 104, py + 112, line, COL_MUTED);
    } else {
        draw_text(&big, px + 20, py + 38, "?", COL_FAINT);
        draw_text(&title, px + 104, py + 60, "--", COL_FAINT);
        draw_text(&ui, px + 104, py + 90, "waiting for ink", COL_FAINT);
    }

    /* 28x28 network input */
    const int side = DIGIT_SIDE * PREVIEW_SCALE;
    const int vx = px + PANEL_W - 20 - side, vy = py + 20;
    for (int y = 0; y < side; y++)
        for (int x = 0; x < side; x++) {
            float f = has_ink ? input[(y / PREVIEW_SCALE) * DIGIT_SIDE + x / PREVIEW_SCALE] : 0.0f;
            unsigned v = (unsigned)(f * 255.0f + 0.5f);
            fb[(size_t)(vy + y) * WIN_W + vx + x] = 0xFF000000u | v << 16 | v << 8 | v;
        }
    stroke_rect(vx - 1, vy - 1, side + 2, side + 2, COL_BORDER);
    draw_text_center(&ui, vx + side / 2, vy + side + 6, "network input", COL_MUTED);

    fill_rect(px + 20, py + 172, PANEL_W - 40, 1, COL_BORDER);

    /* class probabilities */
    for (int k = 0; k < 10; k++) {
        int ry = py + 186 + k * 21;
        char label[2] = {(char)('0' + k), 0};
        draw_text(&ui, px + 22, ry + 1, label, k == best ? COL_TEXT : COL_MUTED);
        const int bx = px + 44, bw = PANEL_W - 44 - 92;
        fill_round_rect(bx, ry + 5, bw, 10, 5.0f, COL_TRACK);
        if (best >= 0) {
            int fill = (int)lroundf(probs[k] * (float)bw);
            if (fill > 0) fill_round_rect(bx, ry + 5, fill < 10 ? 10 : fill, 10, 5.0f, k == best ? COL_ACCENT : COL_BAR);
            snprintf(line, sizeof line, "%.1f%%", 100.0 * (double)probs[k]);
            draw_text_right(&ui, px + PANEL_W - 20, ry + 1, line, k == best ? COL_TEXT : COL_MUTED);
        }
    }

    /* footer */
    fill_rect(24, WIN_H - 44, WIN_W - 48, 1, COL_BORDER);
    draw_text(&ui, 24, WIN_H - 32, model_info, COL_FAINT);
    draw_text_right(&ui, WIN_W - 24, WIN_H - 32, "C clear   Ctrl+Z undo   Esc quit", COL_FAINT);
}

/* ---------------------------------------------------------------- model lookup */

static bool file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static bool find_model(const char *arg, char *out, size_t size) {
    if (arg) { snprintf(out, size, "%s", arg); return file_exists(out); }
    const char *env = getenv("DIGITPAD_MODEL");
    if (env && *env) { snprintf(out, size, "%s", env); return file_exists(out); }
    char *base = SDL_GetBasePath();
    const char *candidates[] = {"../share/digitpad/mnist.slett", "mnist.slett"};
    bool found = false;
    for (size_t i = 0; base && !found && i < sizeof candidates / sizeof *candidates; i++) {
        snprintf(out, size, "%s%s", base, candidates[i]);
        found = file_exists(out);
    }
    SDL_free(base);
    if (!found) snprintf(out, size, "mnist.slett");
    return found;
}

static const char *base_name(const char *path) {
    const char *name = path;
    for (const char *c = path; *c; c++)
        if (*c == '/' || *c == '\\') name = c + 1;
    return name;
}

static void read_model_info(const char *model) {
    char path[4200];
    snprintf(path, sizeof path, "%s.info", model);
    FILE *f = fopen(path, "r");
    if (f && fgets(model_info, sizeof model_info, f)) {
        model_info[strcspn(model_info, "\r\n")] = 0;
    } else {
        snprintf(model_info, sizeof model_info, "model: %.200s", base_name(model));
    }
    if (f) fclose(f);
}

/* ---------------------------------------------------------------- --classify */

/* Minimal binary PGM (P5, 8 or 16 bits) reader; the drawing may be dark-on-light or light-on-dark. */
static float *load_pgm(const char *path, int *w, int *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int fields[3] = {0}, n = 0, c = fgetc(f);
    bool ok = c == 'P' && fgetc(f) == '5';
    c = fgetc(f);
    while (ok && n < 3) {
        while (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '#') {
            if (c == '#') while (c != '\n' && c != EOF) c = fgetc(f);
            c = fgetc(f);
        }
        if (c < '0' || c > '9') { ok = false; break; }
        while (c >= '0' && c <= '9' && fields[n] < 100000) { fields[n] = fields[n] * 10 + (c - '0'); c = fgetc(f); }
        n++;
    }
    *w = fields[0]; *h = fields[1];
    const int maxval = fields[2], bytes = maxval > 255 ? 2 : 1;
    ok = ok && *w > 0 && *h > 0 && *w <= 8192 && *h <= 8192 && maxval > 0 && maxval <= 65535;
    size_t count = ok ? (size_t)*w * *h : 0;
    float *img = ok ? malloc(count * sizeof(float)) : NULL;
    unsigned char *raw = ok ? malloc(count * bytes) : NULL;
    ok = img && raw && fread(raw, (size_t)bytes, count, f) == count;
    fclose(f);
    if (ok) {
        double mean = 0;
        for (size_t i = 0; i < count; i++) {
            unsigned v = bytes == 2 ? (unsigned)raw[2 * i] << 8 | raw[2 * i + 1] : raw[i];
            img[i] = (float)(v > (unsigned)maxval ? 1.0 : (double)v / maxval);
            mean += img[i];
        }
        if (mean > 0.5 * (double)count)     /* mostly bright: dark ink on paper */
            for (size_t i = 0; i < count; i++) img[i] = 1.0f - img[i];
    }
    free(raw);
    if (!ok) { free(img); img = NULL; }
    return img;
}

static void print_prediction(void) {
    if (!has_ink) { printf("no ink\n"); fflush(stdout); return; }
    int second, best = best_class(&second);
    printf("prediction %d (%.1f%%), next %d (%.1f%%)\n", best, 100.0 * (double)probs[best], second,
           100.0 * (double)probs[second]);
    fflush(stdout);
}

static int classify_file(NeuralNetwork *net, const char *path) {
    int w, h;
    float *img = load_pgm(path, &w, &h);
    if (!img) { fprintf(stderr, "DigitPad: cannot read %s (expected a binary PGM)\n", path); return 1; }
    sat = malloc(((size_t)w + 1) * ((size_t)h + 1) * sizeof(double));
    bool ok = sat && classify(net, img, w, h);
    if (ok) print_prediction();
    free(img);
    free(sat);
    return ok ? 0 : 1;
}

/* ---------------------------------------------------------------- main */

static bool gui = true;              /* false for --classify: errors go to stderr only */

static void fatal(const char *message) {
    fprintf(stderr, "DigitPad: %s\n", message);
    if (gui) SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "DigitPad", message, NULL);
}

#if defined(_WIN32)
/* A GUI-subsystem program starts without standard streams unless its parent redirected them;
 * when it was started from a console, write to that console. */
static void attach_parent_console(void) {
    if (GetStdHandle(STD_OUTPUT_HANDLE) || !AttachConsole(ATTACH_PARENT_PROCESS)) return;
    if (!freopen("CONOUT$", "w", stdout) || !freopen("CONOUT$", "w", stderr)) return;
    printf("\n");                   /* the shell has already printed its next prompt */
}
#endif

int main(int argc, char **argv) {
#if defined(_WIN32)
    attach_parent_console();
#endif
    const char *model_arg = NULL, *classify_path = NULL;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--verbose")) verbose = true;
        else if (!strcmp(argv[i], "--classify") && i + 1 < argc) classify_path = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            const char *self = base_name(argv[0]);
            printf("usage: %s [--verbose] [model.slett]\n       %s --classify image.pgm [model.slett]\n", self, self);
            return 0;
        } else model_arg = argv[i];
    }

    gui = classify_path == NULL;
    spingalett_set_verbose(false);
    char model_path[4096];
    bool found = find_model(model_arg, model_path, sizeof model_path);
    NeuralNetwork *net = found ? load_spingalett(model_path) : NULL;
    if (!net) {
        char message[4300];
        snprintf(message, sizeof message, "cannot load the model %s", model_path);
        fatal(message);
        return 1;
    }
    if (classify_path) {
        int status = classify_file(net, classify_path);
        free_network(net);
        return status;
    }
    read_model_info(model_path);

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
    SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fatal(SDL_GetError()); free_network(net); return 1; }

    /* double the window on tall screens; the usable bounds are in window coordinates on every
     * platform (Windows reports scaled values to programs that are not DPI-aware) */
    int scale = 1;
    SDL_Rect usable;
    if (SDL_GetDisplayUsableBounds(0, &usable) == 0 && usable.h >= 2 * WIN_H + 300) scale = 2;
    SDL_Window *window = SDL_CreateWindow("DigitPad - Spingalett", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          WIN_W * scale, WIN_H * scale, SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Renderer *renderer = window ? SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) : NULL;
    if (window && !renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    SDL_Texture *texture = renderer ? SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIN_W, WIN_H) : NULL;
    undo_stack = malloc((size_t)UNDO_DEPTH * CANVAS * CANVAS * sizeof(float));
    sat = malloc((size_t)(CANVAS + 1) * (CANVAS + 1) * sizeof(double));
    if (!texture || !undo_stack || !sat || !font_decode(&ui, &font_ui) || !font_decode(&title, &font_title) ||
        !font_decode(&big, &font_big)) {
        fatal(texture ? "out of memory" : SDL_GetError());
        return 1;
    }
    SDL_RenderSetLogicalSize(renderer, WIN_W, WIN_H);
    SDL_SetWindowMinimumSize(window, WIN_W / 2, WIN_H / 2);

    const int by = CANVAS_Y + CANVAS + 16;
    btn_clear = (Button){CANVAS_X, by, text_width(&ui, "Clear") + 40, 32, "Clear"};
    btn_undo = (Button){btn_clear.x + btn_clear.w + 10, by, text_width(&ui, "Undo") + 40, 32, "Undo"};

    bool running = true, drawing = false, erasing = false, changed = false, dirty = true, report = false;
    float last_x = 0, last_y = 0;
    while (running) {
        SDL_Event e;
        if (!SDL_WaitEvent(&e)) break;
        do {
            switch (e.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT:
                dirty = true;
                break;
            case SDL_KEYDOWN: {
                SDL_Keycode key = e.key.keysym.sym;
                bool ctrl = (e.key.keysym.mod & KMOD_CTRL) != 0;
                if (key == SDLK_ESCAPE || (key == SDLK_q && ctrl)) running = false;
                else if (key == SDLK_c || key == SDLK_SPACE || key == SDLK_DELETE || key == SDLK_BACKSPACE) {
                    if (has_ink) { undo_push(); memset(canvas, 0, sizeof canvas); changed = true; }
                } else if ((key == SDLK_z && ctrl) || key == SDLK_u) {
                    changed |= undo_pop();
                }
                break;
            }
            case SDL_MOUSEBUTTONDOWN: {
                int x = e.button.x - CANVAS_X, y = e.button.y - CANVAS_Y;
                bool in_canvas = x >= 0 && y >= 0 && x < CANVAS && y < CANVAS;
                if (in_canvas && (e.button.button == SDL_BUTTON_LEFT || e.button.button == SDL_BUTTON_RIGHT)) {
                    undo_push();
                    drawing = true;
                    erasing = e.button.button == SDL_BUTTON_RIGHT;
                    last_x = (float)x; last_y = (float)y;
                    paint_segment(last_x, last_y, last_x, last_y, erasing ? ERASER_RADIUS : BRUSH_RADIUS, erasing);
                    changed = true;
                }
                break;
            }
            case SDL_MOUSEMOTION:
                if (e.motion.x != mouse_x || e.motion.y != mouse_y) {
                    bool hover = inside(&btn_clear, mouse_x, mouse_y) || inside(&btn_undo, mouse_x, mouse_y) ||
                                 inside(&btn_clear, e.motion.x, e.motion.y) || inside(&btn_undo, e.motion.x, e.motion.y);
                    mouse_x = e.motion.x; mouse_y = e.motion.y;
                    dirty |= hover;
                }
                if (drawing) {
                    float x = (float)(e.motion.x - CANVAS_X), y = (float)(e.motion.y - CANVAS_Y);
                    paint_segment(last_x, last_y, x, y, erasing ? ERASER_RADIUS : BRUSH_RADIUS, erasing);
                    last_x = x; last_y = y;
                    changed = true;
                }
                break;
            case SDL_MOUSEBUTTONUP:
                if (drawing) {
                    drawing = false;
                    report = verbose;           /* printed once this batch of events is classified */
                } else if (e.button.button == SDL_BUTTON_LEFT) {
                    if (inside(&btn_clear, e.button.x, e.button.y) && has_ink) {
                        undo_push(); memset(canvas, 0, sizeof canvas); changed = true;
                    } else if (inside(&btn_undo, e.button.x, e.button.y)) {
                        changed |= undo_pop();
                    }
                }
                break;
            default:
                break;
            }
        } while (SDL_PollEvent(&e));

        if (changed) {
            changed = false;
            dirty = true;
            if (!classify(net, canvas, CANVAS, CANVAS)) { fatal(spingalett_last_error_message()); running = false; }
        }
        if (report) {
            report = false;
            print_prediction();
        }
        if (dirty) {
            dirty = false;
            render();
            SDL_UpdateTexture(texture, NULL, fb, WIN_W * (int)sizeof(uint32_t));
            SDL_SetRenderDrawColor(renderer, 0x1C, 0x1D, 0x22, 255);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
        }
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    free(ui.alpha); free(title.alpha); free(big.alpha);
    free(undo_stack);
    free(sat);
    free_network(net);
    return 0;
}
