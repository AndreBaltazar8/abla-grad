extern "C" __global__ void abla_fuse_reordered(const unsigned *p0, const float *p1, const float *p2, const unsigned char *p3, float *out) {
    constexpr unsigned cols = 128u;
    constexpr unsigned ff = 9u;
    constexpr unsigned used = 1u;
    constexpr unsigned experts = 2u;
    constexpr unsigned long long rowBytes = (cols / 64u) * 36ull;
    constexpr unsigned long long expertStrideBytes = 2ull * ff * rowBytes;
    const unsigned row = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    const unsigned slot = blockIdx.y;
    if (row >= ff || slot >= used) return;

    const unsigned expertId = p0[slot];
    const unsigned long long output = static_cast<unsigned long long>(slot) * ff + row;
    if (expertId >= experts) {
        if ((threadIdx.x & 31u) == 0) out[output] = 0.0f;
        return;
    }
    const unsigned char *expert = p3 + expertId * expertStrideBytes;
    const float gate = nvfp4_dot(expert + row * rowBytes, p2, cols);
    const float up = nvfp4_dot(expert + (ff + row) * rowBytes, p2, cols);
    if ((threadIdx.x & 31u) == 0) {
        const float scale = p1 ? p1[expertId] : 1.0f;
        out[output] = grad_situ(gate * scale, up * scale, 4.0f, 0.0f);
    }
}
