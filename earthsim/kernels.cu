// ============================================================
// EarthSim CUDA kernels
// 球面经纬网格上的一阶迎风平流 + 拉普拉斯扩散 (融合 kernel)
// 由 cuda_kernels.py 通过 CuPy RawModule 编译加载。
// 经度方向周期边界, 纬度方向边界复制。
// ============================================================
extern "C" {

__device__ __forceinline__ int wrap(int j, int n) {
    return (j + n) % n;
}
__device__ __forceinline__ int clampi(int i, int n) {
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

// 对标量场 F 计算  dF = dt * ( -u dF/dx - v dF/dy + K * lap(F) )
// invdx: 每一行(纬度)的 1/dx, invdy: 常数 1/dy
__global__ void adv_diff(
    const float* __restrict__ F,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ invdx,   // [nlat]
    float invdy, float K, float dt,
    float* __restrict__ out,           // F + dF
    int nlat, int nlon)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // lon
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // lat
    if (i >= nlat || j >= nlon) return;

    int idx = i * nlon + j;
    int jm = wrap(j - 1, nlon), jp = wrap(j + 1, nlon);
    int im = clampi(i - 1, nlat), ip = clampi(i + 1, nlat);

    float f  = F[idx];
    float fw = F[i * nlon + jm], fe = F[i * nlon + jp];
    float fs = F[im * nlon + j], fn = F[ip * nlon + j];

    float uu = u[idx], vv = v[idx];
    float idx_ = invdx[i];

    // 一阶迎风
    float dfdx = (uu > 0.f) ? (f - fw) * idx_ : (fe - f) * idx_;
    float dfdy = (vv > 0.f) ? (f - fs) * invdy : (fn - f) * invdy;

    // 扩散
    float lap = (fw + fe - 2.f * f) * idx_ * idx_
              + (fn + fs - 2.f * f) * invdy * invdy;

    out[idx] = f + dt * (-uu * dfdx - vv * dfdy + K * lap);
}

} // extern "C"
