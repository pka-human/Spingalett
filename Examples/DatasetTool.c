/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Converts data sets to the .slettd format and inspects .slettd files.
 *
 *   DatasetTool idx <images> <labels> <out.slettd> [options]     IDX pair (MNIST format)
 *   DatasetTool csv <file.csv> <targets> <classes> <out.slettd> [options]
 *                                       last <targets> columns are targets; <classes> > 0 one-hot encodes them
 *   DatasetTool info <file.slettd>
 *   DatasetTool verify <file.slettd>    decodes every chunk and checks every checksum
 *
 * Options: --fp16, --bf16, --u8 (lossy input encodings; default: smallest lossless), --store (no compression).
 */

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *encoding_name(DatasetEncoding e) {
    static const char *names[] = {"auto", "float32", "fp16", "bfloat16", "u8 (q/255)", "u8 (per-feature affine)", "class index"};
    return (unsigned)e < DATASET_ENCODING_COUNT ? names[e] : "?";
}

static double now(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int info(const char *path) {
    SpingalettDatasetReader *r = spingalett_dataset_open(path, false);
    if (!r) {
        fprintf(stderr, "%s: %s\n", path, spingalett_last_error_message());
        return 1;
    }
    SpingalettDatasetInfo i = spingalett_dataset_info(r);
    double floats = (double)i.count * (i.input_size + i.target_size) * sizeof(float);
    printf("%s\n  samples      %u\n  inputs       %u per sample, %s\n  targets      %u per sample, %s\n"
           "  chunks       %u\n  file size    %llu bytes (%.1f%% of float32, %.1f bytes per sample)\n",
           path, i.count, i.input_size, encoding_name(i.input_encoding), i.target_size, encoding_name(i.target_encoding),
           i.chunk_count, (unsigned long long)i.file_size, 100.0 * (double)i.file_size / floats,
           (double)i.file_size / i.count);
    spingalett_dataset_close(r);
    return 0;
}

int main(int argc, char **argv) {
    spingalett_set_verbose(false);
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

    SpingalettDataset d;
    const char *out = NULL;
    int first_option = 0;
    bool loaded = false;
    if (argc >= 5 && !strcmp(argv[1], "idx")) {
        loaded = spingalett_load_idx(argv[2], argv[3], 0, &d);
        out = argv[4];
        first_option = 5;
    } else if (argc >= 6 && !strcmp(argv[1], "csv")) {
        loaded = spingalett_load_csv(argv[2], (uint32_t)strtoul(argv[3], NULL, 10), (uint32_t)strtoul(argv[4], NULL, 10), &d);
        out = argv[5];
        first_option = 6;
    } else {
        fprintf(stderr, "usage: %s idx <images> <labels> <out.slettd> [--fp16|--bf16|--u8] [--store]\n"
                        "       %s csv <file.csv> <target-columns> <classes> <out.slettd> [options]\n"
                        "       %s info|verify <file.slettd>\n", argv[0], argv[0], argv[0]);
        return 2;
    }
    if (!loaded) {
        fprintf(stderr, "cannot read the input: %s\n", spingalett_last_error_message());
        return 1;
    }

    DatasetSaveOptions o = {0};
    for (int i = first_option; i < argc; i++) {
        if (!strcmp(argv[i], "--fp16")) o.input_encoding = DATASET_ENCODING_FP16;
        else if (!strcmp(argv[i], "--bf16")) o.input_encoding = DATASET_ENCODING_BFLOAT16;
        else if (!strcmp(argv[i], "--u8")) o.input_encoding = DATASET_ENCODING_U8_AFFINE;
        else if (!strcmp(argv[i], "--store")) o.no_compression = true;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    double t = now();
    bool ok = spingalett_save_dataset(&d, out, &o);
    spingalett_dataset_free(&d);
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
