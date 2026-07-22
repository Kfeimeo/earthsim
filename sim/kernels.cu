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

// Flux-form transport in terrain-following sigma coordinates. Each thread
// owns one [level, latitude, longitude] cell and applies both adjacent
// interface fluxes directly, avoiding a Python loop over vertical interfaces.
__global__ void mass_transport(
    const float* __restrict__ field,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ mu_old,          // [nlat, nlon]
    const float* __restrict__ mu_new,          // [nlat, nlon]
    const float* __restrict__ sigma_thickness, // [nz]
    const float* __restrict__ div_mass,        // [nz, nlat, nlon]
    const float* __restrict__ interface_flux,  // [nz+1, nlat, nlon]
    const float* __restrict__ invdx,           // [nlat]
    float invdy, float diffusivity, float dt,
    float* __restrict__ out,
    int nz, int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (k >= nz || i >= nlat || j >= nlon) return;

    int cells = nlat * nlon;
    int col = i * nlon + j;
    int idx = k * cells + col;
    int west = j == 0 ? nlon - 1 : j - 1;
    int east = j + 1 == nlon ? 0 : j + 1;
    int south = i == 0 ? 0 : i - 1;
    int north = i + 1 == nlat ? nlat - 1 : i + 1;

    float f = field[idx];
    float fw = field[k * cells + i * nlon + west];
    float fe = field[k * cells + i * nlon + east];
    float fs = field[k * cells + south * nlon + j];
    float fn = field[k * cells + north * nlon + j];
    float uu = u[idx];
    float vv = v[idx];
    float idx_ = invdx[i];

    float dfdx = uu > 0.0f ? (f - fw) * idx_ : (fe - f) * idx_;
    float dfdy = vv > 0.0f ? (f - fs) * invdy : (fn - f) * invdy;
    float advective = -(uu * dfdx + vv * dfdy);
    float ds = sigma_thickness[k];
    float mu0 = mu_old[col];
    float numerator = mu0 * ds * f
                    + dt * ds * (-f * div_mass[idx] + mu0 * advective);

    if (k < nz - 1) {
        float flux = interface_flux[(k + 1) * cells + col];
        float upstream = flux >= 0.0f ? field[idx + cells] : f;
        numerator += dt * flux * upstream;
    }
    if (k > 0) {
        float flux = interface_flux[k * cells + col];
        float upstream = flux >= 0.0f ? f : field[idx - cells];
        numerator -= dt * flux * upstream;
    }

    float result = numerator / fmaxf(mu_new[col] * ds, 1.0f);
    if (diffusivity != 0.0f) {
        float lap = (fw + fe - 2.0f * f) * idx_ * idx_
                  + (fn + fs - 2.0f * f) * invdy * invdy;
        result += dt * diffusivity * lap;
    }
    out[idx] = result;
}

// Hydrostatic integration in a sigma column. One thread owns one horizontal
// column and walks from the surface to the model top.
__global__ void hydrostatic_column(
    const float* __restrict__ temperature,       // [nz, nlat, nlon]
    const float* __restrict__ humidity,          // [nz, nlat, nlon]
    const float* __restrict__ surface_pressure,  // [nlat, nlon]
    const float* __restrict__ sigma_interfaces,  // [nz+1]
    const float* __restrict__ surface_geopotential,
    float top_pressure, float gas_constant,
    float* __restrict__ pressure_interfaces,
    float* __restrict__ pressure_layers,
    float* __restrict__ geopotential_layers,
    float* __restrict__ geopotential_interfaces,
    float* __restrict__ virtual_temperature,
    int nz, int cells)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cells) return;

    float mu = surface_pressure[col] - top_pressure;
    float phi_bottom = surface_geopotential[col];
    pressure_interfaces[col] =
        top_pressure + sigma_interfaces[0] * mu;
    geopotential_interfaces[col] = phi_bottom;

    for (int k = 0; k < nz; ++k) {
        int idx = k * cells + col;
        float p_bottom = top_pressure + sigma_interfaces[k] * mu;
        float p_upper = top_pressure + sigma_interfaces[k + 1] * mu;
        float p_mid = 0.5f * (p_bottom + p_upper);
        float tv = temperature[idx] * (1.0f + 0.608f * humidity[idx]);
        float coefficient = gas_constant * tv;
        float phi_mid = phi_bottom + coefficient * logf(p_bottom / p_mid);
        float phi_upper = phi_bottom + coefficient * logf(p_bottom / p_upper);

        pressure_layers[idx] = p_mid;
        virtual_temperature[idx] = tv;
        geopotential_layers[idx] = phi_mid;
        pressure_interfaces[(k + 1) * cells + col] = p_upper;
        geopotential_interfaces[(k + 1) * cells + col] = phi_upper;
        phi_bottom = phi_upper;
    }
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
