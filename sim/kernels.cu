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
    const float* __restrict__ F, int base, int i, int j, int nlat, int nlon)
{
    return F[base + clampi(i, nlat) * nlon + wrap(j, nlon)];
}

// F += dt * (-u dF/dx - v dF/dy + K * lap(F)).
// The Python wrapper launches one 16x16 grid per leading-dimension plane.
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
    int base = blockIdx.z * nlat * nlon;
    int sj = tx + 1;
    int si = ty + 1;

    tile[si][sj] = load_clamped_wrapped(F, base, i, j, nlat, nlon);

    if (tx == 0) {
        tile[si][0] = load_clamped_wrapped(F, base, i, j - 1, nlat, nlon);
    }
    if (tx == blockDim.x - 1) {
        tile[si][sj + 1] = load_clamped_wrapped(F, base, i, j + 1,
                                                nlat, nlon);
    }
    if (ty == 0) {
        tile[0][sj] = load_clamped_wrapped(F, base, i - 1, j, nlat, nlon);
    }
    if (ty == blockDim.y - 1) {
        tile[si + 1][sj] = load_clamped_wrapped(F, base, i + 1, j,
                                                nlat, nlon);
    }

    __syncthreads();

    if (i >= nlat || j >= nlon) return;

    int idx = base + i * nlon + j;
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

// Centred horizontal gradient for a 2-D field or a batch of fields.
__global__ void gradient(
    const float* __restrict__ F,
    const float* __restrict__ invdx,   // [nlat]
    float invdy,
    float* __restrict__ out_x,
    float* __restrict__ out_y,
    int nlat, int nlon)
{
    __shared__ float tile[ADV_BLOCK_Y + 2][ADV_BLOCK_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;
    int base = blockIdx.z * nlat * nlon;
    int sj = tx + 1;
    int si = ty + 1;

    tile[si][sj] = load_clamped_wrapped(F, base, i, j, nlat, nlon);
    if (tx == 0) {
        tile[si][0] = load_clamped_wrapped(F, base, i, j - 1, nlat, nlon);
    }
    if (tx == blockDim.x - 1) {
        tile[si][sj + 1] = load_clamped_wrapped(F, base, i, j + 1,
                                                nlat, nlon);
    }
    if (ty == 0) {
        tile[0][sj] = load_clamped_wrapped(F, base, i - 1, j, nlat, nlon);
    }
    if (ty == blockDim.y - 1) {
        tile[si + 1][sj] = load_clamped_wrapped(F, base, i + 1, j,
                                                nlat, nlon);
    }
    __syncthreads();

    if (i >= nlat || j >= nlon) return;
    int idx = base + i * nlon + j;
    out_x[idx] = (tile[si][sj + 1] - tile[si][sj - 1])
               * (0.5f * invdx[i]);
    out_y[idx] = (tile[si + 1][sj] - tile[si - 1][sj])
               * (0.5f * invdy);
}

// Spherical horizontal divergence:
// du/dx + d(v*cos(lat))/dy / max(cos(lat), cos_clamp).
__global__ void divergence(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ invdx,       // [nlat]
    const float* __restrict__ coslat,      // [nlat]
    const float* __restrict__ invcoslat,   // [nlat]
    float invdy,
    float* __restrict__ out,
    int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nlat || j >= nlon) return;

    int base = blockIdx.z * nlat * nlon;
    int west = j == 0 ? nlon - 1 : j - 1;
    int east = j + 1 == nlon ? 0 : j + 1;
    int south = i == 0 ? 0 : i - 1;
    int north = i + 1 == nlat ? nlat - 1 : i + 1;

    float dudx = (u[base + i * nlon + east]
                 - u[base + i * nlon + west]) * (0.5f * invdx[i]);
    float dvcdy = (v[base + north * nlon + j] * coslat[north]
                  - v[base + south * nlon + j] * coslat[south])
                 * (0.5f * invdy);
    out[base + i * nlon + j] = dudx + dvcdy * invcoslat[i];
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
