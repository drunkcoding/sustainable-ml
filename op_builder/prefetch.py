from .builder import OpBuilder
import distutils
import subprocess
import glob


class BackendBuilder(OpBuilder):
    BUILD_VAR = "MOE_BUILD_BACKEND"
    NAME = "backend"

    def __init__(self):
        super().__init__(name=self.NAME)
    
    def absolute_name(self):
        return f'edgeml.ops.backend.{self.NAME}_op'

    def sources(self):
        return [
            'csrc/backend/delegate_mul.cpp',
            'csrc/python/py_backend.cpp',
        ]

    def include_paths(self):
        return ['csrc']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        return [
            '-g',
            '-Wall',
            '-O2',
            '-std=c++17',
            '-shared',
            '-fPIC',
            '-Wno-reorder',
            CPU_ARCH,
            '-fopenmp',
            SIMD_WIDTH,
            '-I/usr/local/cuda/include',
            '-L/usr/local/cuda/lib64',
            '-lcuda',
            '-lcudart',
            '-lcublas',
            '-lpthread',
        ]

    def extra_ldflags(self):
        return []

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
