// Score streamed vocabulary logits against the next input token and save each row's argmax.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { batch_size = 16 };

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: score-context IDS POSITIONS VOCAB ARGMAX_OUT < logits.f32\n");
        return 2;
    }
    int count = atoi(argv[2]), vocab = atoi(argv[3]);
    if (count < 2 || vocab < 1) return 2;
    FILE *ids = fopen(argv[1], "r");
    FILE *out = fopen(argv[4], "w");
    int *tokens = calloc(count, sizeof(*tokens));
    float *batch = malloc((size_t)batch_size * vocab * sizeof(*batch));
    if (!ids || !out || !tokens || !batch) { perror("score-context"); return 1; }
    for (int row = 0; row < count; ++row) {
        if (fscanf(ids, "%d", &tokens[row]) != 1 || tokens[row] < 0 || tokens[row] >= vocab) {
            fprintf(stderr, "invalid input token at row %d\n", row);
            return 1;
        }
    }
    int extra;
    if (fscanf(ids, "%d", &extra) != EOF || ferror(ids)) return 1;
    fclose(ids);
    double total_loss = 0;
    for (int start = 0; start < count; start += batch_size) {
        int rows = count - start < batch_size ? count - start : batch_size;
        size_t elements = (size_t)rows * vocab;
        if (fread(batch, sizeof(float), elements, stdin) != elements) {
            fprintf(stderr, "incomplete logit batch at position %d\n", start);
            return 1;
        }
        double loss[batch_size];
        int choices[batch_size];
        #pragma omp parallel for
        for (int r = 0; r < rows; ++r) {
            const float *logits = batch + (size_t)r * vocab;
            int best = 0;
            for (int i = 1; i < vocab; ++i) if (logits[i] > logits[best]) best = i;
            choices[r] = best;
            double max = logits[best], sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < vocab; ++i) sum += exp((double)logits[i] - max);
            int row = start + r;
            loss[r] = !isfinite(sum) ? NAN : (row + 1 == count ? 0 : log(sum) + max - logits[tokens[row + 1]]);
        }
        for (int r = 0; r < rows; ++r) {
            if (!isfinite(loss[r]) || fprintf(out, "%d\n", choices[r]) < 0) return 1;
            total_loss += loss[r];
        }
    }
    if (fgetc(stdin) != EOF || ferror(stdin) || fclose(out) != 0) return 1;
    double mean = total_loss / (count - 1);
    printf("predictions %d mean_nll %.12f perplexity %.12f\n", count - 1, mean, exp(mean));
    free(tokens);
    free(batch);
    return 0;
}
