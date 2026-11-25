// Greedy continuation from a fixed sequence of token IDs with the pinned llama.cpp.
#include "llama.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message) {
    fprintf(stderr, "context-greedy: %s\n", message);
    return 1;
}

static int read_token(FILE *ids, const struct llama_vocab *vocab, int *token) {
    if (fscanf(ids, "%d", token) != 1 || *token < 0 || *token >= llama_vocab_n_tokens(vocab))
        return fail("invalid or missing input token");
    return 0;
}

static int pieces(const struct llama_vocab *vocab, FILE *ids) {
    int token;
    while (fscanf(ids, "%d", &token) == 1) {
        if (token < 0 || token >= llama_vocab_n_tokens(vocab)) return fail("invalid token");
        if (llama_vocab_is_eog(vocab, token)) return 0;
        char piece[1024];
        int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        if (n < 0 || fwrite(piece, 1, n, stdout) != (size_t)n) return fail("could not write token piece");
    }
    return feof(ids) ? 0 : fail("expected token IDs");
}

static int add_token(struct llama_batch_ext *batch, int token, llama_pos pos, bool output) {
    int row = llama_batch_ext_add_token(batch, 0, token);
    return row >= 0 && llama_batch_ext_set_pos(batch, row, &pos) &&
        llama_batch_ext_set_output_logits(batch, row, output);
}

int main(int argc, char **argv) {
    const bool decode = argc == 4 && strcmp(argv[3], "text") == 0;
    if (!decode && argc != 8)
        return fail("usage: context-greedy MODEL IDS text | MODEL IDS f16|q8 CONTEXT PROMPT_COUNT NEW_TOKENS CHUNK");
    FILE *ids = fopen(argv[2], "r");
    if (!ids) return fail("could not open IDs");
    const char *plugins = getenv("GGML_BACKEND_PATH");
    if (plugins && *plugins) ggml_backend_load_all_from_path(plugins);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = decode;
    mp.n_gpu_layers = decode ? 0 : 999;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return fail("model load failed");
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    if (decode) {
        int status = pieces(vocab, ids);
        llama_model_free(model);
        llama_backend_free();
        fclose(ids);
        return status;
    }

    const int context = atoi(argv[4]), count = atoi(argv[5]), generated = atoi(argv[6]), chunk = atoi(argv[7]);
    if (count < 1 || generated < 1 || chunk < 1 || context < count || context - count < generated)
        return fail("invalid prompt, generation budget or context");
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = chunk;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.swa_full = false;
    cp.no_perf = true;
    if (strcmp(argv[3], "f16") == 0) cp.type_k = cp.type_v = GGML_TYPE_F16;
    else if (strcmp(argv[3], "q8") == 0) {
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    } else return fail("unknown KV format");
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return fail("context allocation failed");
    struct llama_batch_ext *batch = llama_batch_ext_init(ctx);
    if (!batch) return fail("batch allocation failed");

    for (int start = 0; start < count;) {
        int end = count - start < chunk ? count : start + chunk;
        llama_batch_ext_clear(batch);
        for (int pos = start; pos < end; ++pos) {
            int token;
            if (read_token(ids, vocab, &token)) return 1;
            if (!add_token(batch, token, pos, pos == count - 1)) return fail("could not append prompt token");
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return fail("prompt decode failed");
        start = end;
    }
    char trailing;
    if (fscanf(ids, " %c", &trailing) != EOF || ferror(ids)) return fail("extra prompt data");
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < generated; ++step) {
        const float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) return fail("missing logits");
        int token = 0;
        for (int id = 1; id < n_vocab; ++id) if (logits[id] > logits[token]) token = id;
        if (llama_vocab_is_eog(vocab, token)) break;
        printf("%d\n", token);
        if (step + 1 < generated) {
            llama_batch_ext_clear(batch);
            if (!add_token(batch, token, count + step, true) || llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch))
                return fail("generation failed");
        }
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fclose(ids);
    return fflush(stdout) ? 1 : 0;
}
