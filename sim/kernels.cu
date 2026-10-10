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

__device__ __forceinline__ float tvd_slope(
    float backward, float forward, int limiter)
{
    if (backward * forward <= 0.0f) return 0.0f;
    if (limiter == 0) {  // minmod
        return copysignf(fminf(fabsf(backward), fabsf(forward)), backward);
    }
    if (limiter == 1) {  // van Leer
        return 2.0f * backward * forward / (backward + forward);
    }
    // monotonized central (MC)
    float centred = 0.5f * (backward + forward);
    float magnitude = fminf(fabsf(centred),
                            fminf(2.0f * fabsf(backward),
                                  2.0f * fabsf(forward)));
    return copysignf(magnitude, centred);
}

__device__ __forceinline__ float slope_x_at(
    const float* F, int base, int i, int j, int nlat, int nlon, int limiter)
{
    float west = load_clamped_wrapped(F, base, i, j - 1, nlat, nlon);
    float centre = load_clamped_wrapped(F, base, i, j, nlat, nlon);
    float east = load_clamped_wrapped(F, base, i, j + 1, nlat, nlon);
    return tvd_slope(centre - west, east - centre, limiter);
}

__device__ __forceinline__ float slope_y_at(
    const float* F, int base, int i, int j, int nlat, int nlon, int limiter)
{
    float south = load_clamped_wrapped(F, base, i - 1, j, nlat, nlon);
    float centre = load_clamped_wrapped(F, base, i, j, nlat, nlon);
    float north = load_clamped_wrapped(F, base, i + 1, j, nlat, nlon);
    return tvd_slope(centre - south, north - centre, limiter);
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

// Five-point horizontal Laplacian for a 2-D field or a batch of fields.
__global__ void laplacian(
    const float* __restrict__ F,
    const float* __restrict__ invdx,   // [nlat]
    float invdy,
    float* __restrict__ out,
    int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nlat || j >= nlon) return;
    int base = blockIdx.z * nlat * nlon;

    float f = F[base + i * nlon + j];
    float fw = load_clamped_wrapped(F, base, i, j - 1, nlat, nlon);
    float fe = load_clamped_wrapped(F, base, i, j + 1, nlat, nlon);
    float fs = load_clamped_wrapped(F, base, i - 1, j, nlat, nlon);
    float fn = load_clamped_wrapped(F, base, i + 1, j, nlat, nlon);
    out[base + i * nlon + j] = (fw + fe - 2.f * f) * invdx[i] * invdx[i]
                             + (fs + fn - 2.f * f) * invdy * invdy;
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

// Limited vertical slope of a layer field (zero at the two boundary layers).
__device__ __forceinline__ float vertical_slope(
    const float* __restrict__ field, int k, int nz, int cells, int col,
    int limiter)
{
    if (limiter < 0 || k <= 0 || k >= nz - 1) return 0.0f;
    float below = field[(k - 1) * cells + col];
    float centre = field[k * cells + col];
    float above = field[(k + 1) * cells + col];
    return tvd_slope(centre - below, above - centre, limiter);
}

// Net vertical exchange for cell k from the two shared interface fluxes.
// Positive interface flux is downward; face values are upstream-biased with
// half a limited slope (first-order upstream when limiter < 0).
__device__ __forceinline__ float vertical_exchange(
    const float* __restrict__ field,
    const float* __restrict__ interface_flux,
    int k, int nz, int cells, int col, float dt, int limiter)
{
    float f = field[k * cells + col];
    float total = 0.0f;
    if (k < nz - 1) {
        float flux = interface_flux[(k + 1) * cells + col];
        float face = flux >= 0.0f
            ? field[(k + 1) * cells + col]
              - 0.5f * vertical_slope(field, k + 1, nz, cells, col, limiter)
            : f + 0.5f * vertical_slope(field, k, nz, cells, col, limiter);
        total += dt * flux * face;
    }
    if (k > 0) {
        float flux = interface_flux[k * cells + col];
        float face = flux >= 0.0f
            ? f - 0.5f * vertical_slope(field, k, nz, cells, col, limiter)
            : field[(k - 1) * cells + col]
              + 0.5f * vertical_slope(field, k - 1, nz, cells, col, limiter);
        total -= dt * flux * face;
    }
    return total;
}

// Flux-form transport in hybrid sigma-pressure layers. Each thread owns one
// [level, latitude, longitude] cell and applies both adjacent interface
// fluxes directly, avoiding a Python loop over vertical interfaces.
__global__ void mass_transport(
    const float* __restrict__ field,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ dp_old,          // [nz, nlat, nlon]
    const float* __restrict__ dp_new,          // [nz, nlat, nlon]
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
    float dp0 = dp_old[idx];
    float numerator = dp0 * f
                    + dt * (-f * div_mass[idx] + dp0 * advective);
    numerator += vertical_exchange(field, interface_flux, k, nz, cells, col,
                                   dt, -1);

    float result = numerator / fmaxf(dp_new[idx], 1.0f);
    if (diffusivity != 0.0f) {
        float lap = (fw + fe - 2.0f * f) * idx_ * idx_
                  + (fn + fs - 2.0f * f) * invdy * invdy;
        result += dt * diffusivity * lap;
    }
    out[idx] = result;
}

// Hydrostatic integration in a hybrid sigma-pressure column. One thread owns
// one horizontal column and walks from the surface to the model top.
__global__ void hydrostatic_column(
    const float* __restrict__ temperature,       // [nz, nlat, nlon]
    const float* __restrict__ humidity,          // [nz, nlat, nlon]
    const float* __restrict__ surface_pressure,  // [nlat, nlon]
    const float* __restrict__ hyai,              // [nz+1], Pa
    const float* __restrict__ hybi,              // [nz+1]
    const float* __restrict__ surface_geopotential,
    float gas_constant,
    float* __restrict__ pressure_interfaces,
    float* __restrict__ pressure_layers,
    float* __restrict__ geopotential_layers,
    float* __restrict__ geopotential_interfaces,
    float* __restrict__ virtual_temperature,
    int nz, int cells)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cells) return;

    float ps = surface_pressure[col];
    float phi_bottom = surface_geopotential[col];
    pressure_interfaces[col] = hyai[0] + hybi[0] * ps;
    geopotential_interfaces[col] = phi_bottom;

    for (int k = 0; k < nz; ++k) {
        int idx = k * cells + col;
        float p_bottom = hyai[k] + hybi[k] * ps;
        float p_upper = hyai[k + 1] + hybi[k + 1] * ps;
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

// Backward-Euler vertical diffusion. One thread owns one column and solves
// its tridiagonal system with the Thomas algorithm; exchange[k] couples the
// layers k and k+1. `diag` is [nz, nlat, nlon] scratch for the eliminated
// diagonal, `out` holds the right-hand side until back substitution.
__global__ void vertical_diffusion_column(
    const float* __restrict__ field,       // [nz, nlat, nlon]
    const float* __restrict__ layer_mass,  // [nz, nlat, nlon]
    const float* __restrict__ exchange,    // [nz-1, nlat, nlon]
    float dt,
    float* __restrict__ diag,
    float* __restrict__ out,
    int nz, int cells)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cells) return;

    float mass = layer_mass[col];
    float d_prev = mass + dt * exchange[col];
    float r_prev = mass * field[col];
    diag[col] = d_prev;
    out[col] = r_prev;
    for (int k = 1; k < nz; ++k) {
        int idx = k * cells + col;
        float below = dt * exchange[idx - cells];
        float above = (k < nz - 1) ? dt * exchange[idx] : 0.0f;
        float w = -below / d_prev;
        mass = layer_mass[idx];
        d_prev = mass + below + above + w * below;
        r_prev = mass * field[idx] - w * r_prev;
        diag[idx] = d_prev;
        out[idx] = r_prev;
    }

    float x = r_prev / d_prev;
    out[(nz - 1) * cells + col] = x;
    for (int k = nz - 2; k >= 0; --k) {
        int idx = k * cells + col;
        x = (out[idx] + dt * exchange[idx] * x) / diag[idx];
        out[idx] = x;
    }
}

// Enthalpy-conserving dry convective adjustment. One thread sweeps one
// column upward `passes` times, mixing each statically unstable layer pair.
__global__ void dry_adjustment_column(
    const float* __restrict__ temperature,  // [nz, nlat, nlon]
    const float* __restrict__ humidity,     // [nz, nlat, nlon] if has_q
    const float* __restrict__ layer_mass,   // [nz, nlat, nlon]
    const float* __restrict__ exner,        // [nz, nlat, nlon]
    float threshold, int passes, int has_q,
    float* __restrict__ out_T,
    float* __restrict__ out_q,
    int nz, int cells)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cells) return;

    for (int k = 0; k < nz; ++k) {
        int idx = k * cells + col;
        out_T[idx] = temperature[idx];
        if (has_q) out_q[idx] = humidity[idx];
    }
    for (int pass = 0; pass < passes; ++pass) {
        for (int k = 0; k < nz - 1; ++k) {
            int lo = k * cells + col;
            int hi = lo + cells;
            float t_lo = out_T[lo], t_hi = out_T[hi];
            float ex_lo = exner[lo], ex_hi = exner[hi];
            if (!(t_lo / ex_lo > t_hi / ex_hi + threshold)) continue;
            float m_lo = layer_mass[lo], m_hi = layer_mass[hi];
            float theta = (m_lo * t_lo + m_hi * t_hi)
                        / (m_lo * ex_lo + m_hi * ex_hi);
            out_T[lo] = theta * ex_lo;
            out_T[hi] = theta * ex_hi;
            if (has_q) {
                float q_mixed = (m_lo * out_q[lo] + m_hi * out_q[hi])
                              / (m_lo + m_hi);
                out_q[lo] = q_mixed;
                out_q[hi] = q_mixed;
            }
        }
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

// Second-order-in-space MUSCL/TVD transport of an advected scalar.  Face
// fluxes use centred velocities and limited left/right reconstructed states.
__global__ void muscl_adv_diff(
    const float* __restrict__ F,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ invdx,
    const float* __restrict__ coslat,
    const float* __restrict__ invcoslat,
    float invdy, float K, float dt, int limiter,
    float* __restrict__ out,
    int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int plane = blockIdx.z;
    if (i >= nlat || j >= nlon) return;

    int cells = nlat * nlon;
    int base = plane * cells;
    int west = j == 0 ? nlon - 1 : j - 1;
    int east = j + 1 == nlon ? 0 : j + 1;
    int south = i == 0 ? 0 : i - 1;
    int north = i + 1 == nlat ? nlat - 1 : i + 1;
    int idx = base + i * nlon + j;
    int idx_w = base + i * nlon + west;
    int idx_e = base + i * nlon + east;
    int idx_s = base + south * nlon + j;
    int idx_n = base + north * nlon + j;

    float f = F[idx], fw = F[idx_w], fe = F[idx_e];
    float fs = F[idx_s], fn = F[idx_n];
    float sx = slope_x_at(F, base, i, j, nlat, nlon, limiter);
    float sx_w = slope_x_at(F, base, i, j - 1, nlat, nlon, limiter);
    float sx_e = slope_x_at(F, base, i, j + 1, nlat, nlon, limiter);
    float sy = slope_y_at(F, base, i, j, nlat, nlon, limiter);
    float sy_s = slope_y_at(F, base, i - 1, j, nlat, nlon, limiter);
    float sy_n = slope_y_at(F, base, i + 1, j, nlat, nlon, limiter);

    float ue = 0.5f * (u[idx] + u[idx_e]);
    float uw = 0.5f * (u[idx_w] + u[idx]);
    float state_e = ue >= 0.0f ? f + 0.5f * sx : fe - 0.5f * sx_e;
    float state_w = uw >= 0.0f ? fw + 0.5f * sx_w : f - 0.5f * sx;

    float vn = i + 1 < nlat
        ? 0.5f * (v[idx] * coslat[i] + v[idx_n] * coslat[north]) : 0.0f;
    float vs = i > 0
        ? 0.5f * (v[idx_s] * coslat[south] + v[idx] * coslat[i]) : 0.0f;
    float state_n = vn >= 0.0f ? f + 0.5f * sy : fn - 0.5f * sy_n;
    float state_s = vs >= 0.0f ? fs + 0.5f * sy_s : f - 0.5f * sy;

    float scalar_div = (ue * state_e - uw * state_w) * invdx[i]
        + (vn * state_n - vs * state_s) * invdy * invcoslat[i];
    float velocity_div = (ue - uw) * invdx[i]
        + (vn - vs) * invdy * invcoslat[i];
    float lap = (fw + fe - 2.0f * f) * invdx[i] * invdx[i]
              + (fn + fs - 2.0f * f) * invdy * invdy;
    out[idx] = f + dt * (-scalar_div + f * velocity_div + K * lap);
}

// Mass-consistent MUSCL/TVD transport in hybrid sigma-pressure layers.
__global__ void muscl_mass_transport(
    const float* __restrict__ field,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ dp_old,
    const float* __restrict__ dp_new,
    const float* __restrict__ div_mass,
    const float* __restrict__ interface_flux,
    const float* __restrict__ invdx,
    const float* __restrict__ coslat,
    const float* __restrict__ invcoslat,
    float invdy, float diffusivity, float dt, int limiter,
    float* __restrict__ out,
    int nz, int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (k >= nz || i >= nlat || j >= nlon) return;

    int cells = nlat * nlon;
    int base = k * cells;
    int col = i * nlon + j;
    int west = j == 0 ? nlon - 1 : j - 1;
    int east = j + 1 == nlon ? 0 : j + 1;
    int south = i == 0 ? 0 : i - 1;
    int north = i + 1 == nlat ? nlat - 1 : i + 1;
    int col_w = i * nlon + west;
    int col_e = i * nlon + east;
    int col_s = south * nlon + j;
    int col_n = north * nlon + j;
    int idx = base + col;
    int idx_w = base + col_w;
    int idx_e = base + col_e;
    int idx_s = base + col_s;
    int idx_n = base + col_n;

    float f = field[idx], fw = field[idx_w], fe = field[idx_e];
    float fs = field[idx_s], fn = field[idx_n];
    float sx = slope_x_at(field, base, i, j, nlat, nlon, limiter);
    float sx_w = slope_x_at(field, base, i, j - 1, nlat, nlon, limiter);
    float sx_e = slope_x_at(field, base, i, j + 1, nlat, nlon, limiter);
    float sy = slope_y_at(field, base, i, j, nlat, nlon, limiter);
    float sy_s = slope_y_at(field, base, i - 1, j, nlat, nlon, limiter);
    float sy_n = slope_y_at(field, base, i + 1, j, nlat, nlon, limiter);

    float mass_e = 0.5f * (dp_old[idx] * u[idx]
                            + dp_old[idx_e] * u[idx_e]);
    float mass_w = 0.5f * (dp_old[idx_w] * u[idx_w]
                            + dp_old[idx] * u[idx]);
    float state_e = mass_e >= 0.0f ? f + 0.5f * sx : fe - 0.5f * sx_e;
    float state_w = mass_w >= 0.0f ? fw + 0.5f * sx_w : f - 0.5f * sx;

    float mass_n = i + 1 < nlat
        ? 0.5f * (dp_old[idx] * v[idx] * coslat[i]
                  + dp_old[idx_n] * v[idx_n] * coslat[north]) : 0.0f;
    float mass_s = i > 0
        ? 0.5f * (dp_old[idx_s] * v[idx_s] * coslat[south]
                  + dp_old[idx] * v[idx] * coslat[i]) : 0.0f;
    float state_n = mass_n >= 0.0f ? f + 0.5f * sy : fn - 0.5f * sy_n;
    float state_s = mass_s >= 0.0f ? fs + 0.5f * sy_s : f - 0.5f * sy;

    float scalar_div = (mass_e * state_e - mass_w * state_w) * invdx[i]
        + (mass_n * state_n - mass_s * state_s) * invdy * invcoslat[i];
    float face_mass_div = (mass_e - mass_w) * invdx[i]
        + (mass_n - mass_s) * invdy * invcoslat[i];
    float numerator = dp_old[idx] * f + dt
        * (-f * div_mass[idx] - scalar_div + f * face_mass_div);
    numerator += vertical_exchange(field, interface_flux, k, nz, cells, col,
                                   dt, limiter);

    float result = numerator / fmaxf(dp_new[idx], 1.0f);
    if (diffusivity != 0.0f) {
        float lap = (fw + fe - 2.0f * f) * invdx[i] * invdx[i]
                  + (fn + fs - 2.0f * f) * invdy * invdy;
        result += dt * diffusivity * lap;
    }
    out[idx] = result;
}

#undef ADV_BLOCK_X
#undef ADV_BLOCK_Y

} // extern "C"
