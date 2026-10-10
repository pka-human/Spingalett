/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* A package's runtime loads and names its kernels; given a model (the xor.slett test_package writes),
   it predicts with it. */

#include <Spingalett/Spingalett.Runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    printf("Spingalett runtime %s (%s kernels)", spingalett_version(), spingalett_cpu_kernels());
    if (argc < 2) {
        printf("\n");
        return 0;
    }
    static const float inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float outputs[4] = {-1, -1, -1, -1};
    spingalett_set_compute_mode(SPINGALETT_COMPUTE_OPENMP);
    SpingalettModel *model = spingalett_model_load(argv[1]);
    bool ok = model && spingalett_model_predict(model, inputs, 4, outputs);
    for (int i = 0; i < 4; i++) ok = ok && outputs[i] >= 0.0f && outputs[i] <= 1.0f;
    printf(": %s XOR %.2f %.2f %.2f %.2f\n", argv[1], outputs[0], outputs[1], outputs[2], outputs[3]);
    if (!model) fprintf(stderr, "%s\n", spingalett_last_error_message());
    spingalett_model_free(model);
    return ok ? 0 : 1;
}
