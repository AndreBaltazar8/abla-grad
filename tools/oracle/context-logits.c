// Write every vocabulary logit for every supplied ID as little-endian float32 rows.
// Compile against each pinned build's llama.h and libllama. CUDA uses one sequence.
#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: context-logits MODEL IDS POSITIONS CHUNK > logits.f32\n");
        return 2;
    }
    int positions = atoi(argv[3]), chunk = atoi(argv[4]);
    if (positions < 1 || chunk < 1 || chunk > positions) return 2;
    FILE *ids = fopen(argv[2], "r");
    if (!ids) { perror(argv[2]); return 2; }
    const char *plugins = getenv("GGML_BACKEND_PATH");
    if (plugins && *plugins) ggml_backend_load_all_from_path(plugins);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = positions;
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.swa_full = false;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.no_perf = true;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    struct llama_batch_ext *batch = llama_batch_ext_init(ctx);
    if (!batch) return 1;
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (int p = 0; p < positions; p += chunk) {
        const int n = positions - p < chunk ? positions - p : chunk;
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; ++i) {
            int id;
            llama_pos pos = p + i;
            if (fscanf(ids, "%d", &id) != 1 || id < 0 || id >= vocab) return 2;
            int row = llama_batch_ext_add_token(batch, 0, id);
            if (row != i || !llama_batch_ext_set_pos(batch, row, &pos) ||
                    !llama_batch_ext_set_output_logits(batch, row, true)) return 1;
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) return 1;
        for (int i = 0; i < n; ++i) {
            const float *row = llama_get_logits_ith(ctx, i);
            if (!row || fwrite(row, sizeof(float), vocab, stdout) != (size_t)vocab) return 1;
        }
    }
    char trailing;
    if (fscanf(ids, " %c", &trailing) != EOF || ferror(ids) || fflush(stdout)) return 1;
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fclose(ids);
    return 0;
}
