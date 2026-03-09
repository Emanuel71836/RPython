import importlib
import os
import sys

try:
    import rpython_jit
except ImportError:
    lib_path = os.path.join(os.path.dirname(__file__), "__rust__", "target", "release")
    sys.path.append(lib_path)
    rpython_jit = importlib.import_module("rpython_jit")


def compile_native(code: str) -> str:
    return rpython_jit.compile_to_native(code)
