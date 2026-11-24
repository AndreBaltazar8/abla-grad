#define _POSIX_C_SOURCE 200809L

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void fail(const char *message) {
    fprintf(stderr, "abla-grad-oracle: %s\n", message);
    exit(EXIT_FAILURE);
}

static FILE *open_file(const char *path, const char *mode) {
    FILE *file = fopen(path, mode);
    if (!file) {
        perror(path);
        exit(EXIT_FAILURE);
    }
    return file;
}

static FILE *open_output_mode(const char *directory, const char *name, const char *mode) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length < 0 || (size_t) length >= sizeof(path)) fail("output path too long");
    return fopen(path, mode);
}

static FILE *open_output(const char *directory, const char *name) {
    FILE *file = open_output_mode(directory, name, "wb");
    if (!file) { perror(name); exit(EXIT_FAILURE); }
    return file;
}

static void close_file(FILE *file) {
    if (ferror(file) || fclose(file) != 0) fail("I/O error");
}

static void *allocate(size_t bytes) {
    void *result = malloc(bytes ? bytes : 1);
    if (!result) fail("out of memory");
    return result;
}

static int argument(const char *text, int min, int max) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < min || value > max) fail("numeric argument out of range");
    return (int) value;
}

static int hex_value(char digit) {
    if (digit >= '0' && digit <= '9') return digit - '0';
    if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
    if (digit >= 'A' && digit <= 'F') return digit - 'A' + 10;
    fail("invalid input hex");
    return 0;
}

static void write_hex(FILE *file, const char *bytes, int32_t size) {
    static const char hex[] = "0123456789abcdef";
    for (int32_t i = 0; i < size; ++i) {
        unsigned byte = (unsigned char) bytes[i];
        fputc(hex[byte >> 4], file);
        fputc(hex[byte & 15], file);
    }
}

static llama_token *tokenize(const struct llama_vocab *vocab, const char *text, int32_t size,
                            bool add, bool parse, int32_t *count) {
    int32_t capacity = llama_tokenize(vocab, text, size, NULL, 0, add, parse);
    if (capacity == INT32_MIN || capacity > 0) fail("invalid token count");
    capacity = -capacity;
    llama_token *tokens = allocate((size_t) capacity * sizeof(*tokens));
    *count = llama_tokenize(vocab, text, size, tokens, capacity, add, parse);
    if (*count != capacity) fail("tokenization size changed");
    return tokens;
}

static void write_piece(FILE *file, const struct llama_vocab *vocab, llama_token token) {
    char local[128];
    int32_t length = llama_token_to_piece(vocab, token, local, sizeof(local), 0, true);
    if (length >= 0) {
        write_hex(file, local, length);
        return;
    }
    if (length == INT32_MIN) fail("token piece size overflow");
    const int32_t capacity = -length;
    char *piece = allocate((size_t) capacity);
    length = llama_token_to_piece(vocab, token, piece, capacity, 0, true);
    if (length != capacity) fail("token piece size changed");
    write_hex(file, piece, length);
    free(piece);
}

static void tokenizer_cases(const struct llama_vocab *vocab, const char *source, const char *destination) {
    FILE *input = open_file(source, "rb");
    FILE *output = open_file(destination, "wb");
    fputs("case\tadd_special\tparse_special\tinput_hex\ttoken_ids\tpieces_hex\n", output);

    char *line = NULL;
    size_t capacity = 0;
    ssize_t size;
    while ((size = getline(&line, &capacity, input)) >= 0) {
        if (size > 0 && line[size - 1] == '\n') line[--size] = '\0';
        if (size > 0 && line[size - 1] == '\r') line[--size] = '\0';
        char *fields[4] = {line};
        for (int field = 1; field < 4; ++field) {
            char *tab = strchr(fields[field - 1], '\t');
            if (!tab) fail("expected four corpus fields");
            *tab = '\0';
            fields[field] = tab + 1;
        }
        if (!*fields[0]) fail("empty case name");
        int add = argument(fields[1], 0, 1);
        int parse = argument(fields[2], 0, 1);
        size_t hex_size = strlen(fields[3]);
        if (hex_size % 2 || hex_size / 2 > INT32_MAX) fail("invalid corpus byte count");

        char *text = allocate(hex_size / 2);
        for (size_t i = 0; i < hex_size; i += 2) {
            text[i / 2] = (char) (16 * hex_value(fields[3][i]) + hex_value(fields[3][i + 1]));
        }
        int32_t count;
        llama_token *tokens = tokenize(vocab, text, (int32_t) (hex_size / 2), add, parse, &count);
        fprintf(output, "%s\t%d\t%d\t%s\t", fields[0], add, parse, fields[3]);
        for (int32_t i = 0; i < count; ++i) fprintf(output, "%s%d", i ? "," : "", tokens[i]);
        fputc('\t', output);
        for (int32_t i = 0; i < count; ++i) {
            if (i) fputc(',', output);
            write_piece(output, vocab, tokens[i]);
        }
        fputc('\n', output);
        free(tokens);
        free(text);
    }
    free(line);
    close_file(input);
    close_file(output);
}

static void dump_row(FILE *full, FILE *top, const float *logits, int32_t vocab_size,
                     int32_t position, int top_k, llama_token *ranking) {
    int ranked = 0;
    for (llama_token id = 0; id < vocab_size; ++id) {
        const float value = logits[id];
        if (isnan(value) || value == INFINITY) fail("non-finite raw logit");
        if (ranked == top_k && value <= logits[ranking[top_k - 1]]) continue;

        int index = ranked < top_k ? ranked++ : top_k - 1;
        while (index > 0 && value > logits[ranking[index - 1]]) {
            ranking[index] = ranking[index - 1];
            --index;
        }
        ranking[index] = id;
    }
    if (fwrite(logits, sizeof(*logits), (size_t) vocab_size, full) != (size_t) vocab_size) {
        fail("could not write logits vector");
    }
    for (int rank = 0; rank < top_k; ++rank) {
        llama_token id = ranking[rank];
        fprintf(top, "%d\t%d\t%d\t%.9g\n", position, rank, id, logits[id]);
    }
}

static void next_token(FILE *file, const struct llama_vocab *vocab, int32_t position,
                       const char *phase, llama_token token) {
    fprintf(file, "%d\t%s\t%d\t", position, phase, token);
    write_piece(file, vocab, token);
    fputc('\n', file);
}

struct tensor_dump {
    const char *directory;
    FILE *index;
    unsigned int sequence;
};

static bool save_tensor(struct ggml_tensor *tensor, bool ask, void *context) {
    const char *name = ggml_get_name(tensor);
    const bool supported = *name && (tensor->type == GGML_TYPE_F32 ||
        tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16);
    if (ask) return supported;
    if (!supported) return true;
    struct tensor_dump *dump = context;
    char filename[GGML_MAX_NAME + 32];
    int length = 0;
    ++dump->sequence;
    for (size_t i = 0; name[i] && i < GGML_MAX_NAME; ++i) {
        unsigned char c = (unsigned char) name[i];
        bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        filename[length++] = safe ? (char) c : '_';
    }
    memcpy(filename + length, ".f32", 5);
    FILE *file = open_output_mode(dump->directory, filename, "wbx");
    if (!file && errno == EEXIST) {
        snprintf(filename + length, sizeof(filename) - (size_t) length, ".%06u.f32", dump->sequence);
        file = open_output_mode(dump->directory, filename, "wbx");
    }
    if (!file) { perror(filename); exit(EXIT_FAILURE); }
    size_t count = (size_t) ggml_nelements(tensor);
    unsigned char *bytes = allocate(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes, 0, ggml_nbytes(tensor));

    if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
        if (fwrite(bytes, sizeof(float), count, file) != count) fail("tensor write failed");
    } else {
        float *values = allocate(count * sizeof(*values));
        size_t index = 0;
        for (int64_t w = 0; w < tensor->ne[3]; ++w)
            for (int64_t z = 0; z < tensor->ne[2]; ++z)
                for (int64_t y = 0; y < tensor->ne[1]; ++y)
                    for (int64_t x = 0; x < tensor->ne[0]; ++x) {
                        const unsigned char *element = bytes + w * tensor->nb[3] +
                            z * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
                        if (tensor->type == GGML_TYPE_F32) {
                            memcpy(&values[index], element, sizeof(float));
                        } else if (tensor->type == GGML_TYPE_F16) {
                            ggml_fp16_t half;
                            memcpy(&half, element, sizeof(half));
                            values[index] = ggml_fp16_to_fp32(half);
                        } else {
                            ggml_bf16_t half;
                            memcpy(&half, element, sizeof(half));
                            values[index] = ggml_bf16_to_fp32(half);
                        }
                        ++index;
                    }
        if (fwrite(values, sizeof(*values), count, file) != count) fail("tensor write failed");
        free(values);
    }
    free(bytes);
    close_file(file);
    fprintf(dump->index, "%s\t%lld,%lld,%lld,%lld\t%s\t%s\t%s\n", name,
        (long long) tensor->ne[0], (long long) tensor->ne[1], (long long) tensor->ne[2],
        (long long) tensor->ne[3], ggml_op_name(tensor->op), ggml_type_name(tensor->type), filename);
    return true;
}

static struct llama_context *open_context(struct llama_model *model, struct tensor_dump *dump) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.no_perf = true;
    if (dump) {
        params.cb_eval = save_tensor;
        params.cb_eval_user_data = dump;
    }
    struct llama_context *context = llama_init_from_model(model, params);
    if (!context) fail("could not initialize context");
    return context;
}

static const float *decode_position(struct llama_context *context, struct llama_batch_ext *batch,
                                    llama_token token, llama_pos position) {
    llama_batch_ext_clear(batch);
    int32_t index = llama_batch_ext_add_token(batch, 0, token);
    if (index < 0 || !llama_batch_ext_set_pos(batch, index, &position) ||
        !llama_batch_ext_set_output_logits(batch, index, true) ||
        llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) fail("token decode failed");
    const float *logits = llama_get_logits_ith(context, index);
    if (!logits) fail("missing logits");
    return logits;
}

static uint32_t read_ids(const char *path, llama_token tokens[512], int32_t vocab_size) {
    FILE *file = open_file(path, "rb");
    char *line = NULL;
    size_t capacity = 0;
    ssize_t size;
    uint32_t count = 0;
    while ((size = getline(&line, &capacity, file)) >= 0) {
        if (size && line[size - 1] == '\n') line[--size] = '\0';
        if (size && line[size - 1] == '\r') line[--size] = '\0';
        if (count == 512 || strlen(line) != (size_t) size) fail("invalid ids file");
        tokens[count++] = argument(line, 0, vocab_size - 1);
    }
    free(line);
    close_file(file);
    if (!count) fail("empty ids file");
    return count;
}

static void dump_logits(struct llama_model *model, const char *path, const char *output,
                        const char *decoded_prompt) {
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    uint32_t vocab_size = (uint32_t) llama_vocab_n_tokens(vocab);
    llama_token tokens[512];
    uint32_t count = read_ids(path, tokens, (int32_t) vocab_size);
    struct llama_context *context = open_context(model, NULL);
    struct llama_batch_ext *batch = llama_batch_ext_init(context);
    if (!batch) fail("could not allocate decode batch");

    FILE *file = open_file(output, "wb");
    const uint32_t header[] = {1, vocab_size, count};
    if (fwrite("GRDL", 1, 4, file) != 4 || fwrite(header, sizeof(uint32_t), 3, file) != 3 ||
        fwrite(tokens, sizeof(uint32_t), count, file) != count) fail("GRDL header write failed");
    for (uint32_t position = 0; position < count; ++position) {
        const float *logits = decode_position(context, batch, tokens[position], (llama_pos) position);
        if (fwrite(logits, sizeof(float), vocab_size, file) != vocab_size) fail("GRDL logit write failed");
    }
    close_file(file);
    llama_batch_ext_free(batch);
    llama_free(context);

    int32_t required = llama_detokenize(vocab, tokens, (int32_t) count, NULL, 0, true, false);
    if (required == INT32_MIN || required > 0) fail("invalid detokenization length");
    char *text = allocate((size_t) -required);
    int32_t bytes = llama_detokenize(vocab, tokens, (int32_t) count, text, -required, true, false);
    if (bytes != -required) fail("detokenization size changed");
    file = open_file(decoded_prompt, "wb");
    if (fwrite(text, 1, (size_t) bytes, file) != (size_t) bytes) fail("prompt write failed");
    close_file(file);
    free(text);
}

static void dump_tensors(struct llama_model *model, const char *path, const char *directory) {
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token tokens[512];
    read_ids(path, tokens, llama_vocab_n_tokens(vocab));
    if (mkdir(directory, 0755) != 0) {
        perror(directory);
        exit(EXIT_FAILURE);
    }
    struct tensor_dump dump = {directory, open_output(directory, "index.txt"), 0};
    fputs("name\tdims\top\tsource_type\tfile\n", dump.index);
    struct llama_context *context = open_context(model, &dump);
    struct llama_batch_ext *batch = llama_batch_ext_init(context);
    if (!batch) fail("could not allocate decode batch");
    decode_position(context, batch, tokens[0], 0);
    llama_batch_ext_free(batch);
    llama_free(context);
    if (!dump.sequence) fail("no intermediate tensors observed");
    close_file(dump.index);
}

static void decode(struct llama_model *model, const char *prompt_path, const char *directory,
                   int steps, int top_k, int gpu_layers) {
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t vocab_size = llama_vocab_n_tokens(vocab);
    if (top_k > vocab_size) fail("top-k exceeds vocabulary");
    if (mkdir(directory, 0755) != 0) {
        perror(directory);
        exit(EXIT_FAILURE);
    }
    FILE *prompt = open_file(prompt_path, "rb");
    char text[65537];
    size_t size = fread(text, 1, sizeof(text), prompt);
    if (size == sizeof(text)) fail("prompt exceeds 64 KiB");
    close_file(prompt);

    int32_t count;
    llama_token *tokens = tokenize(vocab, text, (int32_t) size, true, false, &count);
    if (count == 0 || count > 512 - steps) fail("prompt and generation must fit in 512 tokens");
    struct llama_context *context = open_context(model, NULL);
    struct llama_batch_ext *batch = llama_batch_ext_init(context);
    if (!batch) fail("could not allocate decode batch");
    llama_token *ranking = allocate((size_t) top_k * sizeof(*ranking));

    FILE *full = open_output(directory, "logits.f32le");
    FILE *top = open_output(directory, "topk.tsv");
    FILE *rows = open_output(directory, "rows.tsv");
    FILE *ids = open_output(directory, "tokens.tsv");
    FILE *exact_prompt = open_output(directory, "prompt.txt");
    if (fwrite(text, 1, size, exact_prompt) != size) fail("could not copy prompt");
    close_file(exact_prompt);
    fputs("row\trank\ttoken_id\tlogit\n", top);
    fputs("row\tinput_id\tnext_id\tnext_source\n", rows);
    fputs("position\tphase\ttoken_id\tpiece_hex\n", ids);

    for (int32_t position = 0; position < count; ++position) {
        next_token(ids, vocab, position, "prompt", tokens[position]);
    }

    int32_t generated = 0;
    int32_t position = 0;
    bool eog = false;
    llama_token token = tokens[0];
    while (generated < steps) {
        const float *logits = decode_position(context, batch, token, position);
        dump_row(full, top, logits, vocab_size, position, top_k, ranking);

        bool greedy = position + 1 >= count;
        llama_token next = greedy ? ranking[0] : tokens[position + 1];
        fprintf(rows, "%d\t%d\t%d\t%s\n", position, token, next, greedy ? "greedy" : "prompt");
        ++position;
        if (greedy) {
            next_token(ids, vocab, position, "generated", next);
            ++generated;
            eog = llama_vocab_is_eog(vocab, next);
            if (eog) break;
        }
        token = next;
    }

    close_file(full);
    close_file(top);
    close_file(rows);
    close_file(ids);
    FILE *metadata = open_output(directory, "meta.tsv");
    fprintf(metadata, "format\tablagrad-oracle-v1\nvocab_size\t%d\nrows\t%d\nprompt_tokens\t%d\n"
        "generated_tokens\t%d\nrequested_tokens\t%d\ntop_k\t%d\neog\t%d\n"
        "gpu_layers_requested\t%d\ncontext_size\t%u\nadd_special\t1\nparse_special\t0\n",
        vocab_size, position, count, generated, steps, top_k, eog, gpu_layers, llama_n_ctx(context));
    close_file(metadata);
    free(ranking);
    free(tokens);
    llama_batch_ext_free(batch);
    llama_free(context);
}

static void read_exact(FILE *file, void *data, size_t size, size_t count) {
    if (fread(data, size, count, file) != count) fail("truncated GRDL input");
}

static void read_header(FILE *file, uint32_t header[3]) {
    char magic[4];
    read_exact(file, magic, 1, sizeof(magic));
    read_exact(file, header, sizeof(uint32_t), 3);
    if (memcmp(magic, "GRDL", 4) != 0 || header[0] != 1 || header[1] == 0 ||
        header[1] > 1000000 || header[2] == 0 || header[2] > 512) fail("invalid GRDL header");
}

static void compare_logits(const char *baseline, const char *other) {
    FILE *cpu = open_file(baseline, "rb");
    FILE *cuda = open_file(other, "rb");
    uint32_t a_header[3], b_header[3];
    read_header(cpu, a_header);
    read_header(cuda, b_header);
    if (memcmp(a_header, b_header, sizeof(a_header)) != 0) fail("GRDL shapes differ");
    uint32_t vocab = a_header[1], rows = a_header[2];
    uint32_t ids[512], other_ids[512];
    read_exact(cpu, ids, sizeof(uint32_t), rows);
    read_exact(cuda, other_ids, sizeof(uint32_t), rows);
    if (memcmp(ids, other_ids, rows * sizeof(uint32_t)) != 0) fail("GRDL token IDs differ");
    float *a = allocate(vocab * sizeof(float));
    float *b = allocate(vocab * sizeof(float));
    double overall_abs = 0, overall_kl = 0, sum_kl = 0;
    uint32_t matched = 0;

    for (uint32_t position = 0; position < rows; ++position) {
        read_exact(cpu, a, sizeof(float), vocab);
        read_exact(cuda, b, sizeof(float), vocab);
        uint32_t arg_a = 0, arg_b = 0;
        double max_delta = 0;
        for (uint32_t id = 0; id < vocab; ++id) {
            if (isnan(a[id]) || isnan(b[id]) || a[id] == INFINITY || b[id] == INFINITY) {
                fail("invalid logit in comparison");
            }
            if (a[id] > a[arg_a]) arg_a = id;
            if (b[id] > b[arg_b]) arg_b = id;
            double delta = a[id] == b[id] ? 0 : fabs((double) a[id] - b[id]);
            if (delta > max_delta) max_delta = delta;
        }
        if (!isfinite(a[arg_a]) || !isfinite(b[arg_b])) fail("logits give no finite probability");
        matched += arg_a == arg_b;
        const double max_a = a[arg_a], max_b = b[arg_b];
        double sum_a = 0, sum_b = 0, weighted = 0;
        for (uint32_t id = 0; id < vocab; ++id) {
            double la = a[id] - max_a, lb = b[id] - max_b;
            double probability = exp(la);
            sum_a += probability;
            sum_b += exp(lb);
            if (probability > 0) weighted += probability * (la - lb);
        }
        double kl = fmax(0, weighted / sum_a + log(sum_b) - log(sum_a));
        if (max_delta > overall_abs) overall_abs = max_delta;
        if (kl > overall_kl) overall_kl = kl;
        sum_kl += kl;
    }

    if (fgetc(cpu) != EOF || fgetc(cuda) != EOF) fail("trailing bytes in GRDL input");
    printf("%u\t%.9g\t%.9g\t%.9g\t%u\n", rows, overall_abs, overall_kl, sum_kl / rows, matched);
    free(a);
    free(b);
    close_file(cpu);
    close_file(cuda);
}

static void tensor_census(const char *path) {
    struct gguf_context *model = gguf_init_from_file(path,
        (struct gguf_init_params) { .no_alloc = true, .ctx = NULL });
    if (!model) fail("could not read GGUF header");
    uint64_t counts[GGML_TYPE_COUNT] = {0}, bytes[GGML_TYPE_COUNT] = {0};
    for (int64_t i = 0; i < gguf_get_n_tensors(model); ++i) {
        const enum ggml_type type = gguf_get_tensor_type(model, i);
        ++counts[type];
        bytes[type] += gguf_get_tensor_size(model, i);
    }
    puts("type\ttensors\tstored_bytes");
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        if (counts[type]) printf("%s\t%llu\t%llu\n", ggml_type_name((enum ggml_type) type),
            (unsigned long long) counts[type], (unsigned long long) bytes[type]);
    }
    gguf_free(model);
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "census") == 0) {
        tensor_census(argv[2]);
        return EXIT_SUCCESS;
    }
    if (argc == 4 && strcmp(argv[1], "noise") == 0) {
        compare_logits(argv[2], argv[3]);
        return EXIT_SUCCESS;
    }
    const bool vocab_only = argc == 5 && strcmp(argv[1], "tokenize") == 0;
    const bool generate = argc == 8 && strcmp(argv[1], "decode") == 0;
    const bool logits = argc == 7 && strcmp(argv[1], "logits") == 0;
    const bool tensors = argc == 6 && strcmp(argv[1], "tensors") == 0;
    if (!vocab_only && !generate && !logits && !tensors) {
        fprintf(stderr, "usage: %s tokenize MODEL CORPUS.tsv OUTPUT.tsv\n"
            "       %s decode MODEL PROMPT OUT_DIR STEPS TOP_K GPU_LAYERS\n", argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    _Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24, "requires float32");
    uint32_t one = 1;
    if (*(const unsigned char *) &one != 1) fail("requires a little-endian host");

    int gpu_layers = vocab_only ? 0 : argument(argv[argc - 1], 0, INT_MAX);
    llama_backend_init();
    struct llama_model_params params = llama_model_default_params();
    params.vocab_only = vocab_only;
    params.n_gpu_layers = gpu_layers;
    struct llama_model *model = llama_model_load_from_file(argv[2], params);
    if (!model) fail("could not load model");

    if (vocab_only) tokenizer_cases(llama_model_get_vocab(model), argv[3], argv[4]);
    else if (logits) dump_logits(model, argv[3], argv[4], argv[5]);
    else if (tensors) dump_tensors(model, argv[3], argv[4]);
    else decode(model, argv[3], argv[4], argument(argv[5], 1, 512),
                argument(argv[6], 1, INT_MAX), gpu_layers);
    llama_model_free(model);
    llama_backend_free();
    return EXIT_SUCCESS;
}
