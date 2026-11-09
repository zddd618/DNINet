from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sysconfig

curand_include = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "curand", "include")

setup(
    name='BpOps',
    ext_modules=[
        CUDAExtension('BpOps',
                      [
                          'bp_cuda.cpp',
                          'bp_cuda_kernel.cu',
                      ],
                      include_dirs=[curand_include],
                      extra_compile_args={'cxx': ['-g'],
                                          'nvcc': ['-O3']}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
