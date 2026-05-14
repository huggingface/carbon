import ctypes
import importlib.util
import os


_CUDNN_LIBS = (
    "libcudnn.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_graph.so.9",
)
_CUDNN_PRELOADED = False


def preload_cudnn_libraries():
    global _CUDNN_PRELOADED
    if _CUDNN_PRELOADED:
        return

    spec = importlib.util.find_spec("nvidia.cudnn")
    if spec is None or spec.submodule_search_locations is None:
        return

    libdir = os.path.join(next(iter(spec.submodule_search_locations)), "lib")
    for name in _CUDNN_LIBS:
        path = os.path.join(libdir, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    _CUDNN_PRELOADED = True
