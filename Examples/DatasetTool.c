/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Converts data sets to the .slettd format and inspects .slettd files.
 *   DatasetTool idx <images> <labels> <out.slettd> [options]     IDX pair (MNIST format)
 *   DatasetTool csv <file.csv> <targets> <classes> <out.slettd> [options]
 *                                       last <targets> columns are targets; <classes> > 0 one-hot encodes them
 *   DatasetTool cifar <out.slettd> <batch.bin>... [--cifar100] [options]
 *                                       CIFAR-10 batches, or CIFAR-100 ones with both their fine and
 *                                       coarse labels (two sets of targets); class names are read
 *                                       from the .txt files next to the batches when present
 *   DatasetTool images <folder> <out.slettd> [--size WxH] [--gray|--rgb] [options]
 *                                       one subfolder per class (named after it), images in PNG,
 *                                       JPEG, BMP, TGA, GIF, PSD, HDR, PIC or PNM, resized to WxH
 *                                       (by default all must have the size of the first)
 *   DatasetTool info <file.slettd>
 *   DatasetTool verify <file.slettd>    decodes every chunk and checks every checksum
 * Options: --fp16, --bf16, --u8 (lossy input encodings; default: smallest lossless), --store (no compression).
 */

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-function"
#  pragma GCC diagnostic ignored "-Wsign-compare"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#include "ThirdParty/stb_image.h"
#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#endif

static const char *encoding_name(DatasetEncoding e) {
    static const char *names[] = {"auto", "float32", "fp16", "bfloat16", "u8 (q/255)", "u8 (per-feature affine)", "class index"};
    return (unsigned)e < DATASET_ENCODING_COUNT ? names[e] : "?";
}

static double now(void) {
#if defined(TIME_UTC)                    /* C11 timespec_get; some Windows C runtimes lack it */
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#elif defined(_WIN32)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static int info(const char *path) {
    DatasetReaderOptions o = {.no_prefetch = true};
    SpingalettDatasetReader *r = spingalett_dataset_open_ex(path, &o);
    if (!r) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        return 1;
    }
    SpingalettDatasetInfo i = spingalett_dataset_info(r);
    double raw = (double)i.count * ((double)i.input_size + i.target_size) * 4;
    printf("%s (format version %u)\n", path, i.format_version);
    printf("  samples      %u\n", i.count);
    printf("  inputs       %u per sample, %s", i.input_size, encoding_name(i.input_encoding));
    if (i.height || i.width || i.channels) printf(", shape %u x %u x %u", i.height, i.width, i.channels);
    printf("\n");
    for (uint32_t k = 0; k < i.target_set_count; k++) {
        const char *name = spingalett_dataset_target_set_name(r, k);
        uint32_t size = spingalett_dataset_target_set_size(r, k);
        printf("  targets %-4u %u per sample%s%s%s", k, size, name ? " (" : "", name ? name : "", name ? ")" : "");
        if (k == 0) printf(", %s", encoding_name(i.target_encoding));
        if (spingalett_dataset_class_name(r, k, 0)) {
            printf(", classes:");
            for (uint32_t c = 0; c < size && c < 12; c++) printf(" %s", spingalett_dataset_class_name(r, k, c));
            if (size > 12) printf(" ... (%u)", size);
        }
        printf("\n");
    }
    printf("  chunks       %u\n", i.chunk_count);
    printf("  file size    %llu bytes (%.1f%% of float32)\n", (unsigned long long)i.file_size, 100.0 * (double)i.file_size / raw);
    spingalett_dataset_close(r);
    return 0;
}

/* ---------------------------------------------------------------- CIFAR */

static int cifar(int argc, char **argv, int first, const char **out, SpingalettDataset *d, SpingalettTargetSet *coarse,
                 bool *two_sets, int *options) {
    *out = argv[first];
    bool hundred = false;
    const char **paths = (const char **)malloc((size_t)argc * sizeof(char *));
    uint32_t n = 0;
    int i = first + 1;
    for (; i < argc && strncmp(argv[i], "--", 2) != 0; i++) paths[n++] = argv[i];
    for (int k = i; k < argc; k++)
        if (!strcmp(argv[k], "--cifar100")) hundred = true;
    *options = i;
    if (n == 0) {
        free(paths);
        fprintf(stderr, "cifar: no batch files\n");
        return 2;
    }
    bool ok = spingalett_load_cifar(paths, n, hundred ? 100 : 10, d);
    *two_sets = false;
    if (ok && hundred) {
        SpingalettDataset c;
        ok = spingalett_load_cifar(paths, n, 20, &c);
        if (ok) {
            /* the coarse labels as a second set of targets (the data set keeps its arrays) */
            coarse->name = "coarse";
            coarse->size = c.target_size;
            coarse->targets = c.targets;
            coarse->class_names = (const char *const *)c.class_names;
            c.targets = NULL;
            c.class_names = NULL;
            spingalett_dataset_free(&c);
            *two_sets = true;
        }
    }
    free(paths);
    if (!ok) {
        fprintf(stderr, "cannot read the batches: %s\n", spingalett_last_error_message());
        return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------- image folders */

typedef struct {
    char **items;
    size_t count, cap;
} Names;

static void names_add(Names *n, const char *s) {
    if (n->count == n->cap) {
        n->cap = n->cap ? n->cap * 2 : 64;
        n->items = (char **)realloc(n->items, n->cap * sizeof(char *));
    }
    size_t len = strlen(s) + 1;
    n->items[n->count] = (char *)malloc(len);
    memcpy(n->items[n->count++], s, len);
}

static void names_free(Names *n) {
    for (size_t i = 0; i < n->count; i++) free(n->items[i]);
    free(n->items);
}

static int compare_names(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* The entries of a folder (subfolders or files), sorted by name. */
static bool list_folder(const char *path, bool folders, Names *out) {
#if defined(_WIN32)
    char pattern[4096];
    snprintf(pattern, sizeof pattern, "%s\\*", path);
    WIN32_FIND_DATAA f;
    HANDLE h = FindFirstFileA(pattern, &f);
    if (h == INVALID_HANDLE_VALUE) return false;
    do {
        if (f.cFileName[0] == '.') continue;
        bool dir = (f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        if (dir == folders) names_add(out, f.cFileName);
    } while (FindNextFileA(h, &f));
    FindClose(h);
#else
    DIR *dir = opendir(path);
    if (!dir) return false;
    struct dirent *e;
    while ((e = readdir(dir)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof full, "%s/%s", path, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0) continue;
        if (S_ISDIR(st.st_mode) == folders) names_add(out, e->d_name);
    }
    closedir(dir);
#endif
    qsort(out->items, out->count, sizeof(char *), compare_names);
    return true;
}

/* Resamples an image to w x h by averaging the source area under each output pixel. */
static void resize_area(const unsigned char *src, int sw, int sh, int c, unsigned char *dst, int w, int h) {
    double fx = (double)sw / w, fy = (double)sh / h;
    for (int y = 0; y < h; y++) {
        double y0 = y * fy, y1 = (y + 1) * fy;
        for (int x = 0; x < w; x++) {
            double x0 = x * fx, x1 = (x + 1) * fx;
            for (int k = 0; k < c; k++) {
                double sum = 0, area = 0;
                for (int sy = (int)y0; sy < sh && sy < y1; sy++) {
                    double wy = (sy + 1 < y1 ? sy + 1 : y1) - (sy > y0 ? sy : y0);
                    for (int sx = (int)x0; sx < sw && sx < x1; sx++) {
                        double wx = (sx + 1 < x1 ? sx + 1 : x1) - (sx > x0 ? sx : x0);
                        sum += wx * wy * src[((size_t)sy * sw + sx) * c + k];
                        area += wx * wy;
                    }
                }
                int v = (int)(sum / area + 0.5);
                dst[((size_t)y * w + x) * c + k] = (unsigned char)(v > 255 ? 255 : v);
            }
        }
    }
}

static int images(int argc, char **argv, const char **out, SpingalettDataset *d, int *options) {
    const char *root = argv[2];
    *out = argv[3];
    int w = 0, h = 0, channels = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--size") && i + 1 < argc) {
            if (sscanf(argv[++i], "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0) {
                fprintf(stderr, "images: --size takes WxH\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--gray")) {
            channels = 1;
        } else if (!strcmp(argv[i], "--rgb")) {
            channels = 3;
        }
    }
    *options = 4;
    Names classes = {0};
    if (!list_folder(root, true, &classes) || classes.count == 0) {
        fprintf(stderr, "images: %s has no class folders\n", root);
        names_free(&classes);
        return 1;
    }
    /* all files first: the data set's size */
    Names *files = (Names *)calloc(classes.count, sizeof(Names));
    size_t total = 0;
    for (size_t c = 0; c < classes.count; c++) {
        char dir[4096];
        snprintf(dir, sizeof dir, "%s/%s", root, classes.items[c]);
        list_folder(dir, false, &files[c]);
        total += files[c].count;
    }
    int rc = 0;
    unsigned char *pixels = NULL;
    size_t sample = 0, done = 0;
    for (size_t c = 0; c < classes.count && rc == 0; c++)
        for (size_t f = 0; f < files[c].count && rc == 0; f++) {
            char file[8192];
            snprintf(file, sizeof file, "%s/%s/%s", root, classes.items[c], files[c].items[f]);
            int iw, ih, ic;
            if (!stbi_info(file, &iw, &ih, &ic)) continue;     /* not an image */
            if (!channels) channels = ic >= 3 ? 3 : 1;
            unsigned char *img = stbi_load(file, &iw, &ih, &ic, channels);
            if (!img) continue;
            if (!w) { w = iw; h = ih; }
            if (!pixels) {
                sample = (size_t)w * h * channels;
                memset(d, 0, sizeof *d);
                d->input_size = (uint32_t)sample;
                d->target_size = (uint32_t)classes.count;
                d->inputs = (float *)malloc(total * sample * sizeof(float));
                d->targets = (float *)calloc(total * classes.count, sizeof(float));
                pixels = (unsigned char *)malloc(sample);
                if (!d->inputs || !d->targets || !pixels) {
                    fprintf(stderr, "images: out of memory\n");
                    rc = 1;
                }
            }
            if (rc == 0 && (iw != w || ih != h)) {
                bool sized = false;
                for (int i = 4; i < argc; i++) if (!strcmp(argv[i], "--size")) sized = true;
                if (!sized) {
                    fprintf(stderr, "images: %s is %d x %d, the first image %d x %d (use --size)\n", file, iw, ih, w, h);
                    rc = 1;
                } else {
                    resize_area(img, iw, ih, channels, pixels, w, h);
                }
            } else if (rc == 0) {
                memcpy(pixels, img, sample);
            }
            stbi_image_free(img);
            if (rc) break;
            float *x = d->inputs + done * sample;
            for (size_t k = 0; k < sample; k++) x[k] = (float)((double)pixels[k] / 255.0);
            d->targets[done * classes.count + c] = 1.0f;
            done++;
        }
    if (rc == 0 && done == 0) {
        fprintf(stderr, "images: no images under %s\n", root);
        rc = 1;
    }
    if (rc == 0) {
        d->count = (uint32_t)done;
        d->height = (uint32_t)h;
        d->width = (uint32_t)w;
        d->channels = (uint32_t)channels;
        spingalett_dataset_set_class_names(d, (const char *const *)classes.items, (uint32_t)classes.count);
        printf("%zu images of %d x %d x %d in %zu classes\n", done, w, h, channels, classes.count);
    } else if (pixels) {
        spingalett_dataset_free(d);
    }
    free(pixels);
    for (size_t c = 0; c < classes.count; c++) names_free(&files[c]);
    free(files);
    names_free(&classes);
    return rc;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv) {
    spingalett_set_verbose(false);
    spingalett_set_compute_mode(COMPUTE_OPENMP);        /* chunks encode and decode in parallel */
    if (argc >= 3 && !strcmp(argv[1], "info")) return info(argv[2]);
    if (argc >= 3 && !strcmp(argv[1], "verify")) {
        SpingalettDataset d;
        double t = now();
        if (!spingalett_load_dataset(argv[2], &d)) {
            fprintf(stderr, "%s: %s\n", argv[2], spingalett_last_error_message());
            return 1;
        }
        printf("%s: %u samples decoded and verified in %.2f s\n", argv[2], d.count, now() - t);
        spingalett_dataset_free(&d);
        return 0;
    }

    SpingalettDataset d = {0};
    SpingalettTargetSet coarse = {0};
    bool two_sets = false;
    const char *out = NULL;
    int first_option = 0, rc = -1;
    if (argc >= 5 && !strcmp(argv[1], "idx")) {
        rc = spingalett_load_idx(argv[2], argv[3], 0, &d) ? 0 : 1;
        out = argv[4];
        first_option = 5;
    } else if (argc >= 6 && !strcmp(argv[1], "csv")) {
        rc = spingalett_load_csv(argv[2], (uint32_t)strtoul(argv[3], NULL, 10), (uint32_t)strtoul(argv[4], NULL, 10), &d) ? 0 : 1;
        out = argv[5];
        first_option = 6;
    } else if (argc >= 4 && !strcmp(argv[1], "cifar")) {
        rc = cifar(argc, argv, 2, &out, &d, &coarse, &two_sets, &first_option);
    } else if (argc >= 4 && !strcmp(argv[1], "images")) {
        rc = images(argc, argv, &out, &d, &first_option);
    }
    if (rc < 0) {
        fprintf(stderr, "usage: %s idx <images> <labels> <out.slettd> [--fp16|--bf16|--u8] [--store]\n"
                        "       %s csv <file.csv> <target-columns> <classes> <out.slettd> [options]\n"
                        "       %s cifar <out.slettd> <batch.bin>... [--cifar100] [options]\n"
                        "       %s images <folder> <out.slettd> [--size WxH] [--gray|--rgb] [options]\n"
                        "       %s info|verify <file.slettd>\n", argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    if (rc != 0) {
        if (rc == 1 && (!strcmp(argv[1], "idx") || !strcmp(argv[1], "csv")))
            fprintf(stderr, "cannot read the input: %s\n", spingalett_last_error_message());
        return rc;
    }

    DatasetSaveOptions o = {0};
    if (two_sets) {
        o.target_name = "fine";
        o.extra_targets = &coarse;
        o.extra_target_count = 1;
    }
    for (int i = first_option; i < argc; i++) {
        if (!strcmp(argv[i], "--fp16")) o.input_encoding = DATASET_ENCODING_FP16;
        else if (!strcmp(argv[i], "--bf16")) o.input_encoding = DATASET_ENCODING_BFLOAT16;
        else if (!strcmp(argv[i], "--u8")) o.input_encoding = DATASET_ENCODING_U8_AFFINE;
        else if (!strcmp(argv[i], "--store")) o.no_compression = true;
        else if (!strcmp(argv[i], "--cifar100") || !strcmp(argv[i], "--gray") || !strcmp(argv[i], "--rgb")) continue;
        else if (!strcmp(argv[i], "--size")) i++;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    double t = now();
    bool ok = spingalett_save_dataset(&d, out, &o);
    spingalett_dataset_free(&d);
    free((void *)coarse.targets);
    free((void *)coarse.class_names);
    if (!ok) {
        fprintf(stderr, "cannot write %s: %s\n", out, spingalett_last_error_message());
        return 1;
    }
    printf("written in %.2f s\n", now() - t);
    /* the name may have had the extension appended */
    char path[4096];
    const char *dot = strrchr(out, '.'), *slash = strrchr(out, '/');
    snprintf(path, sizeof path, "%s%s", out, dot && (!slash || dot > slash) ? "" : SPINGALETT_DATASET_EXTENSION);
    return info(path);
}
