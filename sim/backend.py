"""后端选择: CUDA (CuPy) 或 CPU (NumPy)。

物理代码全部用数组算子编写, 在两种后端下运行同一份代码;
GPU 下热点(迎风平流)另有手写 CUDA kernel (kernels.cu) 融合加速。
"""
import numpy as np

_cupy = None
_backend = "cpu"


def init_backend(mode: str = "auto"):
    """mode: auto | cuda | cpu. 返回 (xp, backend_name)"""
    global _cupy, _backend
    if mode in ("auto", "cuda"):
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            _cupy, _backend = cp, "cuda"
            return cp, "cuda"
        except Exception as e:  # 无 GPU / 未安装 cupy
            if mode == "cuda":
                raise RuntimeError(f"要求 CUDA 后端但初始化失败: {e}")
    _backend = "cpu"
    return np, "cpu"


def get_xp():
    return _cupy if _backend == "cuda" else np


def backend_name():
    return _backend


def to_cpu(a):
    """把数组搬回主机内存 (numpy)。"""
    if _backend == "cuda" and _cupy is not None and isinstance(a, _cupy.ndarray):
        return _cupy.asnumpy(a)
    return np.asarray(a)
