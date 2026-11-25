extern "C" __global__ void abla_fuse_controller_norm_down(const float *p0, const float *p1, const float *p2, float *out) {
    constexpr unsigned rank = 64u;
    constexpr unsigned n = 2304u;
    __shared__ float norm[2u * n];
    __shared__ float partial[256u];
    const unsigned half = threadIdx.x / 128u;
    const unsigned lane = threadIdx.x % 128u;
    rmsnorm_2304(half == 0 ? p0 : p1, nullptr, norm + half * n,
        half == 0 ? 0.000001f : 0.000001f, partial + half * 128u, lane);
    __syncthreads();
    const unsigned row = blockIdx.x * 8u + threadIdx.x / 32u;
    if (row >= rank) return;
    const float sum = grad_f32_dot(p2 + static_cast<unsigned long long>(row) * (2u * n),
        norm, 2u * n);
    if ((threadIdx.x & 31u) == 0) out[row] = sum;
}
