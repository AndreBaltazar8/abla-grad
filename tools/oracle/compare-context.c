// Read full float32 rows in bounded memory; '-' accepts a candidate stream.
// Output is one TSV record: positions, rows over KL .04, argmax differences,
// worst KL, its zero-based position, maximum absolute logit difference.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { batch = 16 };
struct metrics { double kl, max_abs; int miss; };

static struct metrics compare_row(const float *a, const float *b, int vocab) {
    int ia = 0, ib = 0;
    double max_abs = 0;
    for (int i = 0; i < vocab; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            fprintf(stderr, "nonfinite logit at vocabulary entry %d\n", i);
            exit(1);
        }
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
        max_abs = fmax(max_abs, fabs((double)a[i] - b[i]));
    }
    const double ma = a[ia], mb = b[ib];
    double sa = 0, sb = 0, weighted = 0;
    #pragma omp simd reduction(+:sa,sb,weighted)
    for (int i = 0; i < vocab; ++i) {
        double p = exp((double)a[i] - ma);
        sa += p;
        sb += exp((double)b[i] - mb);
        weighted += p * ((double)a[i] - b[i]);
    }
    return (struct metrics){mb - ma + log(sb / sa) + weighted / sa, max_abs, ia != ib};
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: compare-context REFERENCE CANDIDATE POSITIONS VOCAB\n");
        return 2;
    }
    int rows = atoi(argv[3]), vocab = atoi(argv[4]);
    if (rows <= 0 || vocab <= 0) return 2;
    FILE *a = fopen(argv[1], "rb");
    FILE *b = !strcmp(argv[2], "-") ? stdin : fopen(argv[2], "rb");
    float *ra = malloc((size_t)batch * vocab * sizeof(float));
    float *rb = malloc((size_t)batch * vocab * sizeof(float));
    if (!a || !b || !ra || !rb) { perror("compare-context"); return 1; }
    int over = 0, misses = 0, worst_position = 0;
    double worst = -INFINITY, max_abs = 0;
    for (int start = 0; start < rows; start += batch) {
        int n = rows - start < batch ? rows - start : batch;
        size_t elements = (size_t)n * vocab;
        if (fread(ra, sizeof(float), elements, a) != elements || fread(rb, sizeof(float), elements, b) != elements) {
            fprintf(stderr, "short input at row batch starting at %d\n", start);
            return 1;
        }
        struct metrics result[batch];
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) result[i] = compare_row(ra + (size_t)i*vocab, rb + (size_t)i*vocab, vocab);
        for (int i = 0; i < n; ++i) {
            struct metrics m = result[i];
            if (!isfinite(m.kl)) return 1;
            if (m.kl > worst) { worst = m.kl; worst_position = start + i; }
            max_abs = fmax(max_abs, m.max_abs);
            over += m.kl > .04;
            misses += m.miss;
        }
    }
    if (fgetc(a) != EOF || fgetc(b) != EOF || ferror(a) || ferror(b)) {
        fprintf(stderr, "input length or I/O error after %d rows\n", rows);
        return 1;
    }
    printf("%d\t%d\t%d\t%.12g\t%d\t%.9g\n", rows, over, misses, worst, worst_position, max_abs);
    free(ra); free(rb);
    fclose(a); fclose(b);
    return 0;
}
