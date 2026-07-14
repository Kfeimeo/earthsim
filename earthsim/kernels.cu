// ============================================================
// EarthSim CUDA kernels
// Scalar advection + diffusion on a latitude/longitude grid.
// Longitude wraps periodically; latitude uses clamped boundaries.
// ============================================================
extern "C" {

#define ADV_BLOCK_X 16
#define ADV_BLOCK_Y 16

__device__ __forceinline__ int wrap(int j, int n) {
    return (j + n) % n;
}

__device__ __forceinline__ int clampi(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

__device__ __forceinline__ float load_clamped_wrapped(
    const float* __restrict__ F, int i, int j, int nlat, int nlon)
{
    return F[clampi(i, nlat) * nlon + wrap(j, nlon)];
}

// F += dt * (-u dF/dx - v dF/dy + K * lap(F)).
// The Python wrapper launches 16x16 blocks, so the shared tile is 18x18.
__global__ void adv_diff(
    const float* __restrict__ F,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ invdx,   // [nlat]
    float invdy, float K, float dt,
    float* __restrict__ out,           // F + dF
    int nlat, int nlon)
{
    __shared__ float tile[ADV_BLOCK_Y + 2][ADV_BLOCK_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx;  // lon
    int i = blockIdx.y * blockDim.y + ty;  // lat
    int sj = tx + 1;
    int si = ty + 1;

    tile[si][sj] = load_clamped_wrapped(F, i, j, nlat, nlon);

    if (tx == 0) {
        tile[si][0] = load_clamped_wrapped(F, i, j - 1, nlat, nlon);
    }
    if (tx == blockDim.x - 1) {
        tile[si][sj + 1] = load_clamped_wrapped(F, i, j + 1, nlat, nlon);
    }
    if (ty == 0) {
        tile[0][sj] = load_clamped_wrapped(F, i - 1, j, nlat, nlon);
    }
    if (ty == blockDim.y - 1) {
        tile[si + 1][sj] = load_clamped_wrapped(F, i + 1, j, nlat, nlon);
    }

    __syncthreads();

    if (i >= nlat || j >= nlon) return;

    int idx = i * nlon + j;
    float f  = tile[si][sj];
    float fw = tile[si][sj - 1], fe = tile[si][sj + 1];
    float fs = tile[si - 1][sj], fn = tile[si + 1][sj];
    float uu = u[idx], vv = v[idx];
    float idx_ = invdx[i];

    float dfdx = (uu > 0.f) ? (f - fw) * idx_ : (fe - f) * idx_;
    float dfdy = (vv > 0.f) ? (f - fs) * invdy : (fn - f) * invdy;

    float lap = (fw + fe - 2.f * f) * idx_ * idx_
              + (fn + fs - 2.f * f) * invdy * invdy;

    out[idx] = f + dt * (-uu * dfdx - vv * dfdy + K * lap);
}

// Apply every zonal 1-2-1 pass inside one block. Each block owns one complete
// latitude ring, allowing synchronization between passes without a new launch.
// Leading dimensions are treated as a batch of [nlat, nlon] fields.
__global__ void polar_filter(
    const float* __restrict__ F,
    const float* __restrict__ weights,  // [nlat]
    float* __restrict__ out,
    int nlat, int nlon, int rings, int passes)
{
    int ring = blockIdx.x;
    if (ring >= rings) return;

    extern __shared__ float buffers[];
    float* current = buffers;
    float* next = buffers + nlon;
    int base = ring * nlon;

    for (int j = threadIdx.x; j < nlon; j += blockDim.x) {
        current[j] = F[base + j];
    }
    __syncthreads();

    for (int pass = 0; pass < passes; ++pass) {
        for (int j = threadIdx.x; j < nlon; j += blockDim.x) {
            int west = j == 0 ? nlon - 1 : j - 1;
            int east = j + 1 == nlon ? 0 : j + 1;
            next[j] = 0.25f * current[west]
                    + 0.50f * current[j]
                    + 0.25f * current[east];
        }
        __syncthreads();
        float* swap = current;
        current = next;
        next = swap;
    }

    float weight = weights[ring % nlat];
    for (int j = threadIdx.x; j < nlon; j += blockDim.x) {
        float original = F[base + j];
        out[base + j] = original + weight * (current[j] - original);
    }
}

#undef ADV_BLOCK_X
#undef ADV_BLOCK_Y

} // extern "C"
