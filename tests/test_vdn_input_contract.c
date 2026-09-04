#include "h3.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        h3_free(ctx);                                                        \
        return 1;                                                            \
    }                                                                        \
} while (0)

static int ignore_frame(const h3_frame *frame, void *opaque) {
    (void)frame;
    (void)opaque;
    return 0;
}

static int fails_with(h3_ctx *ctx, const char *prompt, h3_params *params,
                      const char *expected) {
    h3_result *result = h3_generate(ctx, prompt, params);
    if (result) {
        h3_result_free(result);
        return 0;
    }
    const char *error = h3_last_error(ctx);
    return error && strstr(error, expected) != NULL;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT.safetensors\n",
                argv[0]);
        return 2;
    }
    h3_ctx *ctx = h3_load_vdn_dir(argv[1], argv[2], 0);
    if (!ctx) {
        fprintf(stderr, "cannot load VDN metadata: %s\n", h3_last_error(NULL));
        return 1;
    }
    const h3_model_info *model = h3_model(ctx);
    CHECK(model && model->vdn.enabled);

    h3_params params = H3_PARAMS_DEFAULT;
    params.steps = model->vdn.num_steps;
    CHECK(fails_with(ctx, "raw prompt", &params,
                     "raw VDN prompt encoding is not available"));
    CHECK(fails_with(ctx, NULL, &params,
                     "VDN generation requires prompt_embeddings"));

    params.prompt_embeddings = argv[3];
    params.first_frame = "not-opened-first.png";
    CHECK(fails_with(ctx, NULL, &params,
                     "OpenVDN does not define first/last-frame"));
    params.first_frame = NULL;
    params.last_frame = "not-opened-last.png";
    CHECK(fails_with(ctx, NULL, &params,
                     "OpenVDN does not define first/last-frame"));
    params.last_frame = NULL;

    h3_reference reference = {
        H3_REFERENCE_IMAGE, "not-opened-reference.png", NULL, 0
    };
    params.references = &reference;
    params.reference_count = 1;
    CHECK(fails_with(ctx, NULL, &params,
                     "OpenVDN does not define first/last-frame"));
    params.references = NULL;
    params.reference_count = 0;

    params.preview_denoise = 1;
    params.on_frame = ignore_frame;
    CHECK(fails_with(ctx, NULL, &params,
                     "VDN denoising preview is not implemented"));

    puts("VDN stable input contract fail-fast tests passed");
    h3_free(ctx);
    return 0;
}
