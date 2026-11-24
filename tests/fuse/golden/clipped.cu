extern "C" __global__ void abla_fuse_clipped(const unsigned char *p0, const float *p1, const unsigned *p2, const float *p3, float *out) {
    constexpr unsigned cols = 256u;
    constexpr unsigned ff = 8u;
    constexpr unsigned used = 4u;
    constexpr unsigned experts = 16u;
    constexpr unsigned long long rowBytes = (cols / 64u) * 36ull;
    constexpr unsigned long long expertStrideBytes = 2ull * ff * rowBytes;
    const unsigned row = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    const unsigned slot = blockIdx.y;
    if (row >= ff || slot >= used) return;

    const unsigned expertId = p2[slot];
    const unsigned long long output = static_cast<unsigned long long>(slot) * ff + row;
    if (expertId >= experts) {
        if ((threadIdx.x & 31u) == 0) out[output] = 0.0f;
        return;
    }
    const unsigned char *expert = p0 + expertId * expertStrideBytes;
    const float gate = nvfp4_dot(expert + row * rowBytes, p1, cols);
    const float up = nvfp4_dot(expert + (ff + row) * rowBytes, p1, cols);
    if ((threadIdx.x & 31u) == 0) {
        const float scale = p3 ? p3[expertId] : 1.0f;
        out[output] = grad_situ(gate * scale, up * scale, 2.5f, 7.125f);
    }
}
